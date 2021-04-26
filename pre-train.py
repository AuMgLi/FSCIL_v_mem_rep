import os
import time
import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from args import ArgumentManager
from utils.data_manager import DataManager, de_normalize_image
from utils.mdl_opt_util import create_encoder, create_decoder, create_optim
from utils.avgmeter import AverageMeter
from utils.iotools import save_checkpoint

USE_GPU = torch.cuda.is_available()


# torch.autograd.set_detect_anomaly(True)


class PreTrain:

    def __init__(self):
        self.args = ArgumentManager().get_args(parser_type='pre-train')
        torch.manual_seed(self.args.seed)
        print('==============\nArgs:{}\n=============='.format(self.args))

        if USE_GPU:
            print('Currently using GPU: {}-{}'.format(
                torch.cuda.current_device(), torch.cuda.get_device_name()))
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            print('Currently using CPU (GPU is highly recommended)')

        self.dsae = True  # True default

        # === Data
        self.dataset = self.args.dataset
        if self.dataset in ('miniImageNet', 'cnbc-face', 'cub-200'):
            self.decoder_output_size = 6 if self.dsae else 84
            self.decoder_scale_factor = 3
        elif self.dataset == 'cifar100':
            self.decoder_output_size = 4 if self.dsae else 32
            self.decoder_scale_factor = 2
        elif self.dataset == 'casia-face':
            self.decoder_output_size = 8 if self.dsae else 128
            self.decoder_scale_factor = 3
        else:
            raise NotImplementedError
        print('Preparing data {}...'.format(self.dataset))
        data_manager = DataManager(self.args, use_gpu=USE_GPU)
        self.train_loader, self.eval_loader = data_manager.get_dataloaders()
        print('len of train loader:', len(self.train_loader))  # 1673

        # === Models
        self.encoder = create_encoder(self.args, is_snail=False)
        print('Encoder:', self.encoder)

        if self.dsae:
            self.decoder = create_decoder(self.args, out_dim=256, fm_level=3)
        else:
            self.decoder = create_decoder(self.args, out_dim=3, fm_level=-1)
        print('Decoder:', self.decoder)

        # Load checkpoint
        if self.args.dataset == 'cnbc-face':
            print('Loading checkpoint...')
            ckp = torch.load(r'')
            del ckp['encoder_state_dict']['downsample.0.weight'], ckp['encoder_state_dict']['classifier.weight']
            self.encoder.load_state_dict(ckp['encoder_state_dict'], strict=False)
            self.decoder.load_state_dict(ckp['decoder_state_dict'], strict=False)

        self.trainable_params = chain(self.encoder.parameters(), self.decoder.parameters())
        self.lr = self.args.lr
        self.optimizer = create_optim(args=self.args, params=self.trainable_params)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.lr_decay_steps, gamma=0.1)
        if USE_GPU:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(log_dir=os.path.join('runs', 'pretrain', current_time))
        self.gamma = 0.5

    def run(self):
        start_time = time.time()
        train_time = 0
        best_epoch = 0
        best_acc = self._eval_epoch(epoch=0)
        # === Train
        for epoch in range(1, self.args.max_epoch + 1):
            self.lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning rate', self.lr, global_step=epoch)

            epoch_time_start = time.time()
            self._train_epoch(epoch)
            train_time += round(time.time() - epoch_time_start)

            # === Evaluation
            if epoch == 1 or epoch == self.args.max_epoch or epoch % self.args.eval_per_epoch == 0:
                acc = self._eval_epoch(epoch)
                is_best = acc > best_acc
                if is_best:
                    best_acc = acc
                    best_epoch = epoch
                if is_best or epoch % self.args.save_per_epoch == 0:
                    state = {
                        'encoder_state_dict': self.encoder.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    file_path = os.path.join(
                        self.args.save_dir, self.dataset, 'pretrain', self.args.encoder,
                        'ckp_ep' + str(epoch) + '.ckp')
                    save_checkpoint(state, is_best, file_path)
                print('==> Test best accuracy {:.2%}, achieved at epoch {}'.format(
                    best_acc, best_epoch))
            torch.cuda.empty_cache()
        elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
        train_time = str(datetime.timedelta(seconds=train_time))
        print('Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}'.format(elapsed, train_time))
        print("==========\nArgs:{}\n==========".format(self.args))

    def _train_epoch(self, epoch):
        self.encoder.train()
        self.decoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_clsf = AverageMeter()
        losses_recons = AverageMeter()
        accs = AverageMeter()

        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            if USE_GPU:
                inputs = inputs.cuda()
                targets = targets.cuda()

            bs = inputs.size(0)
            # === Forward
            outputs, fms = self.encoder(inputs, return_fm=True)
            img_recons = self.decoder(fms[3], scale_factor=self.decoder_scale_factor,
                                      out_size=self.decoder_output_size)
            if self.dsae:
                loss_recons = self.criterion_mse(img_recons, fms[2])  # reconstruction loss
            else:
                loss_recons = self.criterion_mse(img_recons, inputs)

            losses_recons.update(loss_recons.item(), bs)

            loss_clsf = self.criterion_ce(outputs, targets)  # classification loss
            _, preds = outputs.max(dim=1)
            acc = preds.eq(targets).sum().float() / bs
            accs.update(acc.item(), bs)
            losses_clsf.update(loss_clsf.item(), bs)

            # === Backward
            loss_all = self.gamma * loss_clsf + (1 - self.gamma) * loss_recons
            self.optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(self.trainable_params, max_norm=5., norm_type=2)
            self.optimizer.step()

            # print(batch_idx, '; loss:', loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            # Release CUDA memory
            torch.cuda.empty_cache()
        self.scheduler.step()

        acc_avg = accs.avg
        loss_c_avg = losses_clsf.avg
        loss_r_avg = losses_recons.avg
        self.writer.add_scalar('Loss/train/Classification', loss_c_avg, global_step=epoch)
        self.writer.add_scalar('Loss/train/Reconstruction', loss_r_avg, global_step=epoch)
        print(
            '-Train- Epoch: {}, Lr: {:.5f}, Time: {:.1f}s, Data: {:.1f}s, '
            'Loss(C|R): {:.4f} | {:.4f}, Acc: {:.2%}'.format(
                epoch, self.lr, batch_time.sum, data_time.sum, loss_c_avg, loss_r_avg, acc_avg))

    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.encoder.eval()
        self.decoder.eval()

        accs = AverageMeter()
        losses_clsf = AverageMeter()
        losses_recons = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(self.eval_loader):
            # print('inputs:', inputs.shape)
            bs = inputs.size(0)
            if USE_GPU:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs, fms = self.encoder(inputs, return_fm=True)
            # print('fm4:', fms[3].shape, '; fm3:', fms[2].shape)
            loss_clsf = self.criterion_ce(outputs, targets)
            _, preds = outputs.max(dim=1)
            acc = preds.eq(targets).sum().float() / bs
            accs.update(acc.item(), bs)
            losses_clsf.update(loss_clsf.item(), bs)

            img_recons = self.decoder(fms[3], scale_factor=self.decoder_scale_factor,
                                      out_size=self.decoder_output_size)
            # print('img recon shape:', img_recons.shape)
            if self.dsae:
                loss_recons = self.criterion_mse(img_recons, fms[2])  # reconstruction loss
            else:
                loss_recons = self.criterion_mse(img_recons, inputs)  # reconstruction loss
            losses_recons.update(loss_recons, bs)

        acc_avg = accs.avg
        loss_c_avg = losses_clsf.avg
        loss_r_avg = losses_recons.avg
        self.writer.add_scalar('Loss/eval/Classification', loss_c_avg, global_step=epoch)
        self.writer.add_scalar('Loss/eval/Reconstruction', loss_r_avg, global_step=epoch)
        self.writer.add_scalar('Accuracy/eval', acc_avg, global_step=epoch)
        print('-Eval- Epoch: {}, Loss(C|R): {:.4f} | {:.4f}, Accuracy: {:.2%}'.format(
            epoch, loss_c_avg, loss_r_avg, acc_avg))

        return acc_avg


if __name__ == '__main__':
    # main()
    pre_train = PreTrain()
    pre_train.run()
