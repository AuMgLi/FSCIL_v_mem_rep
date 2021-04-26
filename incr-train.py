import os.path as osp
import time
import datetime
import math
import copy
from collections import Counter
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import normal
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from args import ArgumentManager
from utils.data_manager import DataManager, de_normalize_image
from utils.mdl_opt_util import create_encoder, create_decoder, create_optim, adjust_lr
from utils.avgmeter import AverageMeter
from utils.iotools import save_checkpoint
from memory import MemoryK, Memory

USE_GPU = torch.cuda.is_available()

# torch.autograd.set_detect_anomaly(True)


class IncrementalLearning:

    def __init__(self):
        self.args = ArgumentManager().get_args(parser_type='incremental')

        self.dsae = True

        # === Data
        self.dataset = self.args.dataset
        if self.dataset == 'miniImageNet' or 'cnbc-face' or 'cub-200':
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
        torch.manual_seed(self.args.seed)
        print('==============\nArgs:{}\n=============='.format(self.args))

        if USE_GPU:
            print('Currently using GPU: {}-{}'.format(
                torch.cuda.current_device(), torch.cuda.get_device_name()))
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            print('Currently using CPU (GPU is highly recommended)')

        # === Encoder & Decoder
        self.encoder = create_encoder(self.args, use_avgpool=True, is_snail=False)
        print('Encoder:', self.encoder)

        if self.dsae:
            self.decoder = create_decoder(self.args, out_dim=256, fm_level=3)
        else:
            self.decoder = create_decoder(self.args, out_dim=3, fm_level=-1)
        print('Decoder:', self.decoder)

        if self.args.load:  # Loading pre-trained checkpoint
            ckp_path = osp.join(self.args.load_dir, self.dataset, 'pretrain', self.args.encoder, 'best_model.ckp')
            print('Loading checkpoint from {}'.format(ckp_path))
            ckp = torch.load(ckp_path)
            encoder_state_dict = ckp['encoder_state_dict']
            decoder_state_dict = ckp['decoder_state_dict']
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
            self.decoder.load_state_dict(decoder_state_dict, strict=True)
        # === MemoryK
        self.m_sz = self.args.memory_size  # 1024
        self.m_key_dim = self.args.memory_key_dim  # 128
        self.m_K = self.args.memory_K
        self.need_norm = self.args.need_norm
        self.memory = Memory(mem_size=self.m_sz, key_dim=self.m_key_dim, tau=0.95, need_norm=self.need_norm)

        self.trainable_params = chain(
            self.encoder.parameters(), self.memory.parameters(), self.decoder.parameters())
        self.optimizer_all = create_optim(
            optim_name='sgd', lr=1e-3, params=self.trainable_params)
        self.mse_loss = nn.MSELoss(reduction='mean')
        # self.scheduler_encoder = MultiStepLR(
        #     self.optimizer_encoder, milestones=self.args.lr_decay_steps, gamma=0.1)
        self.gamma = 0.2
        self.param_frozen = False  # False default
        self.cur_session = None

        if USE_GPU:
            self.encoder = self.encoder.cuda()
            self.memory = self.memory.cuda()
            self.decoder = self.decoder.cuda()

        self.eval_dataloaders = []
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(log_dir=osp.join('runs', 'incremental', current_time))
        self.data_manager = DataManager(self.args, use_gpu=USE_GPU)

    def get_dataloader(self, session):
        if session == 0:
            train_loader, eval_loader = self.data_manager.get_dataloaders()
        else:  # Incremental sessions
            if self.dataset == 'miniImageNet':
                train_loader = self.data_manager.get_dataloaders(session=session, is_fewshot=True)
                eval_loader = self.data_manager.get_dataloaders(session=session, is_fewshot=False)
            elif self.dataset in ('cifar100', 'cub-200', 'casia-face', 'cnbc-face'):
                train_loader, eval_loader = self.data_manager.get_dataloaders(
                    session=session, is_fewshot=True)
            else:
                raise NotImplementedError
        return train_loader, eval_loader

    def run(self, start_session=0, end_session=8):
        if start_session > 0:  # Load
            load_dir = osp.join(self.args.save_dir, self.dataset, 'incr', self.args.encoder,
                                'session' + str(start_session - 1), 'best_model.ckp')
            print('Start session > 0, loading checkpoint from:', load_dir)
            ckp = torch.load(load_dir)
            self.encoder.load_state_dict(ckp['encoder_state_dict'])
            self.memory.load_state_dict(ckp['memory_state_dict'])
            self.decoder.load_state_dict(ckp['decoder_state_dict'])
            # Evaluate seen classes
            for passed_session in range(start_session):
                # if passed_session == 0:
                #     _, eval_loader = self.data_manager.get_dataloaders()
                # else:
                #     eval_loader = self.data_manager.get_dataloaders(session=passed_session)
                _, eval_loader = self.get_dataloader(passed_session)
                self.eval_dataloaders.append(eval_loader)
            # self._eval_session(start_session - 1)
        for sess in range(start_session, end_session + 1):
            if sess > 0 and not self.param_frozen:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.encoder.layer4.parameters():
                    param.requires_grad = True
                self.trainable_params = chain(
                    self.encoder.parameters(), self.memory.parameters(), self.decoder.parameters())
                self.optimizer_all = create_optim(
                    optim_name='sgd', lr=1e-3,
                    params=filter(lambda p: p.requires_grad, self.trainable_params))
                self.param_frozen = True
                print('Encoder frozen.')
            self._train_session(sess, use_centroid=True)
            self._eval_session(sess, use_centroid=True)

    def _train_session(self, session, use_centroid=False):
        # assert session in range(0, 9)
        print('Training session {}'.format(session))
        self.cur_session = session
        if session > 0:
            self.memory.del_noisy_slots()
            memory_vals = self.memory.m_vals.cpu().numpy()
            memory_vals_counter = Counter(memory_vals)
            print('memory val:', len(memory_vals_counter), memory_vals_counter.most_common())

        # === Data
        print('Preparing data {} with session {}...'.format(self.dataset, session))
        train_loader, eval_loader = self.get_dataloader(session)
        if session == 0:
            # train_loader, eval_loader = self.data_manager.get_dataloaders()
            max_epoch = self.args.max_epoch_sess0
            m_replays = None
        else:
            # if self.dataset == 'miniImageNet':
            #     train_loader = self.data_manager.get_dataloaders(session=session, is_fewshot=True)
            #     eval_loader = self.data_manager.get_dataloaders(session=session, is_fewshot=False)
            # elif self.dataset == 'cifar100' or 'cub-200' or 'casia-face':
            #     train_loader, eval_loader = self.data_manager.get_dataloaders(
            #         session=session, is_fewshot=True)
            # else:
            #     raise NotImplementedError

            # Memory replay data
            m_keys, m_targets = self._get_nonempty_memory_slots()
            # m_keys = m_keys.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)  # [nonempty, key_dim, 3, 3]
            m_replays = (m_keys, m_targets)

            max_epoch = 20  # Todo
        print('Num of batches of train loader:', len(train_loader))
        self.eval_dataloaders.append(eval_loader)

        start_time = time.time()
        train_time = 0
        best_epoch = 0
        # best_acc = self._eval_epoch(epoch=0, eval_loader=eval_loader)
        best_acc = -1
        best_state = None
        # === Train
        for epoch in range(1, max_epoch + 1):
            if session == 0:
                if epoch == 1:
                    adjust_lr(self.optimizer_all, 1e-3)
                if epoch > 10:
                    adjust_lr(self.optimizer_all, 1e-4)
            else:
                if epoch == 1:
                    adjust_lr(self.optimizer_all, 1e-4)
                if epoch > 15:
                    adjust_lr(self.optimizer_all, 1e-5)
            cur_lr = self.optimizer_all.param_groups[0]['lr']
            self.writer.add_scalar('Learning rate', cur_lr, global_step=epoch)

            epoch_time_start = time.time()
            self._train_epoch(epoch, cur_lr, train_loader, m_replays)
            train_time += round(time.time() - epoch_time_start)

            # === Eval on current session's dataloader only
            if epoch == self.args.max_epoch_sess0 or epoch % self.args.eval_per_epoch == 0:
                if use_centroid:
                    self.memory.upd_centroids()
                acc = self._eval_epoch(epoch, eval_loader, use_centroid=use_centroid)

                # === Save checkpoint
                is_best = acc > best_acc
                if is_best or epoch == 2 or epoch % self.args.save_per_epoch == 0:
                    state = {
                        'encoder_state_dict': self.encoder.state_dict(),
                        'memory_state_dict': self.memory.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'acc': acc,
                        'session': session,
                        'epoch': epoch,
                    }
                    file_path = osp.join(
                        self.args.save_dir, self.dataset, 'incr', self.args.encoder,
                        'session' + str(session), 'ckp_ep' + str(epoch) + '.ckp')
                    if epoch == 2:
                        pass
                        # save_checkpoint(state, False, file_path)
                    else:
                        save_checkpoint(state, is_best, file_path)
                    if is_best:
                        best_acc = acc
                        best_epoch = epoch
                        best_state = copy.deepcopy(state)
                print('==> Test best accuracy {:.2%}, achieved at epoch {}'.format(
                    best_acc, best_epoch))
            torch.cuda.empty_cache()
        # Load best checkpoint
        self.encoder.load_state_dict(best_state['encoder_state_dict'])
        self.memory.load_state_dict(best_state['memory_state_dict'])
        self.decoder.load_state_dict(best_state['decoder_state_dict'])

        elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
        train_time = str(datetime.timedelta(seconds=train_time))
        print('Session {} finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}'.format(
            session, elapsed, train_time))
        # print("==========\nArgs:{}\n==========".format(self.args))

        memory_vals = self.memory.m_vals.cpu().numpy()
        memory_vals_counter = Counter(memory_vals)
        print('memory val:', len(memory_vals_counter), memory_vals_counter.most_common())

    def _train_epoch(self, epoch, lr, train_loader, memory_replay=None, use_reparam=False):
        self.encoder.train()
        self.memory.train()
        self.decoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_clsf = AverageMeter()
        losses_recons = AverageMeter()
        accs = AverageMeter()

        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # print('targets:', len(targets), targets[targets > 54])
            data_time.update(time.time() - end)
            if USE_GPU:
                inputs = inputs.cuda()
                targets = targets.cuda()

            bs = inputs.shape[0]
            if len(inputs.shape) == 5:  # episode, batch size = 1
                inputs = inputs.squeeze(0)  # [25, 3, 84, 84]
                targets = targets.squeeze(0)
                bs = inputs.shape[0]

            # === Encoder & Decoder forward
            outputs, fms = self.encoder(inputs, return_fm=True)
            img_recons = self.decoder(fms[3], scale_factor=self.decoder_scale_factor,
                                      out_size=self.decoder_output_size)
            if self.dsae:
                loss_recons = self.mse_loss(img_recons, fms[2])  # reconstruction loss
            else:
                loss_recons = self.mse_loss(img_recons, inputs)

            # === MemoryK forward
            # loss_e = torch.tensor(0.)
            preds, loss_memory = self.memory(outputs, targets)
            acc = preds.eq(targets).sum().float() / bs
            accs.update(acc.item(), bs)

            loss_all = self.gamma * loss_memory + (1 - self.gamma) * loss_recons
            self.optimizer_all.zero_grad()
            loss_all.backward()
            self.optimizer_all.step()

            if batch_idx % 90 == 0:
                print(batch_idx, '; memory loss:', loss_memory.item(),
                      '; decoder loss:', loss_recons.item())
            losses_clsf.update(loss_memory.item(), bs)
            losses_recons.update(loss_recons.item(), bs)
            batch_time.update(time.time() - end)
            end = time.time()

        # memory_replay = None
        if memory_replay is not None:
            with torch.no_grad():
                m_inputs, m_targets = memory_replay  # [nonempty, key_dim]
                # m_inputs = m_inputs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)  # [nonempty, key_dim, 3, 3]
                m_inputs_aug = []
                m_targets_aug = []
                n_classes = 60 + self.args.n_novel * (self.cur_session - 1)

                rand_C_classes = np.random.choice(n_classes, self.args.n_novel, replace=False)
                for v in rand_C_classes:  # rand_C_classes:  # range(n_classes) for all classes
                    m_inputs_v = m_inputs[torch.eq(m_targets, v)]
                    # print('m_inputs_v:', m_inputs_v.shape)

                    if use_reparam:  # re-parameterize
                        m_mean_v = m_inputs_v.mean(dim=0)
                        m_std_v = m_inputs_v.std(dim=0)
                        for i in range(self.args.n_shot * 2):
                            v_aug = torch.normal(mean=m_mean_v, std=m_std_v)
                            # print('v_aug:', v_aug.shape)
                            m_inputs_aug.append(v_aug)
                            m_targets_aug.append(v)
                    else:  # random sample
                        n_v = m_inputs_v.size(0)
                        if n_v == 0:
                            continue
                        for i in range(self.args.n_shot):
                            rand_idxs = np.random.choice(n_v, 3, replace=True)
                            rand_w = F.normalize(torch.rand([3]), p=1, dim=0)
                            v_aug = (rand_w[0] * m_inputs_v[rand_idxs[0]] +
                                     rand_w[1] * m_inputs_v[rand_idxs[1]] + rand_w[2] * m_inputs_v[rand_idxs[2]])
                            m_inputs_aug.append(v_aug)
                            m_targets_aug.append(v)
                m_inputs = torch.stack(m_inputs_aug, dim=0)
                m_inputs = m_inputs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)
                if self.need_norm:
                    m_inputs = F.normalize(m_inputs, p=2, dim=1)
                m_targets = torch.tensor(m_targets_aug, dtype=torch.long)
                # Shuffle
                sfl_idxs = torch.randperm(m_inputs.size(0))
                m_inputs = m_inputs[sfl_idxs]
                m_targets = m_targets[sfl_idxs]
                print('Memory replay size:', m_inputs.size(0))

                m_inputs = self.decoder(m_inputs, scale_factor=self.decoder_scale_factor,
                                        out_size=self.decoder_output_size)
                batch_size = 128
                n_sample = m_targets.size(0)
                n_batch = math.ceil(n_sample / batch_size)
                inputs = m_inputs.chunk(chunks=n_batch, dim=0)
                targets = m_targets.chunk(chunks=n_batch, dim=0)
                print('After chunk, inputs:', inputs[0].shape, '; targets:', targets[0].shape)
                m_train_loader = list(zip(inputs, targets))
            for batch_idx, (inputs, targets) in enumerate(m_train_loader):
                data_time.update(time.time() - end)
                if USE_GPU:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # === Encoder & Decoder forward
                outputs = self.encoder(inputs, return_fm=False, feed_fm=self.dsae)
                img_recons = self.decoder(outputs, scale_factor=self.decoder_scale_factor,
                                          out_size=self.decoder_output_size)
                loss_recons = self.mse_loss(img_recons, inputs)

                # === MemoryK forward
                preds, loss_memory = self.memory(outputs, targets, upd_memory=False)
                loss_all = self.gamma * loss_memory + (1 - self.gamma) * loss_recons
                self.optimizer_all.zero_grad()
                loss_all.backward()
                self.optimizer_all.step()
        acc_avg = accs.avg
        loss_c_avg = losses_clsf.avg
        loss_r_avg = losses_recons.avg
        self.writer.add_scalar('Loss/train/Classification', loss_c_avg, global_step=epoch)
        self.writer.add_scalar('Loss/train/Reconstruction', loss_r_avg, global_step=epoch)
        print(
            '-Train- Epoch: {}, Lr: {:.6f}, Time: {:.1f}s, Data: {:.1f}s, '
            'Loss(C|R): {:.4f} | {:.4f}, Acc: {:.2%}'.format(
                epoch, lr, batch_time.sum, data_time.sum, loss_c_avg, loss_r_avg, acc_avg))

    @torch.no_grad()
    def _eval_session(self, session, use_centroid=False):
        assert len(self.eval_dataloaders) == session + 1

        if use_centroid:
            self.memory.upd_centroids()
        accuracies = []
        for sess in range(session + 1):
            eval_loader_sess = self.eval_dataloaders[sess]
            acc_sess = self._eval_epoch(epoch=None, eval_loader=eval_loader_sess, use_centroid=use_centroid)
            accuracies.append(acc_sess)
        acc_sum = AverageMeter()
        for sess in range(session + 1):
            acc = accuracies[sess]
            if sess == 0:
                n_cls = 60  # self.args.n_class
            else:
                n_cls = self.args.n_novel
            acc_sum.update(acc, n_cls)
        print('Session {} Evaluation. Overall Acc.: {}'.format(session, acc_sum.avg))

    @torch.no_grad()
    def _eval_epoch(self, epoch, eval_loader, use_centroid=False):
        self.encoder.eval()
        self.memory.eval()
        self.decoder.eval()

        accs = AverageMeter()
        losses_clsf = AverageMeter()
        losses_recons = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            if USE_GPU:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs, fms = self.encoder(inputs, return_fm=True)
            # print('outputs:', outputs.shape)

            # Decoder
            img_recons = self.decoder(fms[3], scale_factor=self.decoder_scale_factor,
                                      out_size=self.decoder_output_size)
            if self.dsae:
                loss_recons = self.mse_loss(img_recons, fms[2])  # reconstruction loss
            else:
                loss_recons = self.mse_loss(img_recons, inputs)

            preds, loss_memory = self.memory(outputs, targets, use_centroid=use_centroid)
            losses_clsf.update(loss_memory.item(), targets.size(0))
            losses_recons.update(loss_recons.item(), targets.size(0))
            acc = preds.eq(targets).sum().float() / targets.size(0)
            accs.update(acc.item(), targets.size(0))

        acc_avg = accs.avg
        loss_c_avg = losses_clsf.avg
        loss_r_avg = losses_recons.avg
        if epoch is not None:
            self.writer.add_scalar('Loss/eval/Classification', loss_c_avg, global_step=epoch)
            self.writer.add_scalar('Loss/eval/Reconstruction', loss_r_avg, global_step=epoch)
            self.writer.add_scalar('Accuracy/eval', acc_avg, global_step=epoch)
            print('-Eval- Epoch: {}, Loss(C|R): {:.4f} | {:.4f}, Accuracy: {:.2%}'.format(
                epoch, loss_c_avg, loss_r_avg, acc_avg))

        return acc_avg

    @torch.no_grad()
    def _get_nonempty_memory_slots(self):
        nonempty_idxs = torch.where(self.memory.m_vals != -1)
        m_keys = self.memory.m_keys[nonempty_idxs]  # [nonempty, key_dim]
        # m_keys = m_keys.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)  # [nonempty, key_dim, 3, 3]
        m_vals = self.memory.m_vals[nonempty_idxs]  # [nonempty]

        return m_keys, m_vals


if __name__ == '__main__':
    incremental_learner = IncrementalLearning()
    incremental_learner.run(start_session=1, end_session=8)
