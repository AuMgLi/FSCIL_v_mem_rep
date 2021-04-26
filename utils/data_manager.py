import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import *
from PIL import Image

USE_GPU = torch.cuda.is_available()


def de_normalize_image(images: torch.Tensor, mean=None, std=None):
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    if len(images.size()) == 3:
        images = images.unsqueeze(0)
    n_image = images.size(0)
    for i in range(n_image):
        for c in range(3):
            images[i][c] = images[i][c] * std[c] + mean[c]

    return images


def _image_from_array(array):
    return Image.fromarray(array)


class DataManager:

    def __init__(self, args, use_gpu=USE_GPU, n_workers=4):
        super().__init__()

        self.args = args
        self.dataset_name = args.dataset
        self.use_gpu = use_gpu
        self.pin_memory = True if use_gpu else False
        self.n_workers = n_workers

    def get_dataloaders(self, session=0, is_fewshot=False):
        args = self.args
        # dataloader_train = None
        # dataloader_eval = None
        if self.dataset_name == 'miniImageNet':
            data_pool = MiniImageNetDataPool()
            transform_train = T.Compose([
                T.Lambda(_image_from_array),
                T.RandomCrop(84, padding=8, padding_mode='reflect'),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # T.RandomErasing(p=0.5)
            ])
            transform_eval = T.Compose([
                T.Lambda(_image_from_array),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            if is_fewshot:
                dataset = MiniImageNetEpisode(
                    data_pool=data_pool, session=session, episode_size=args.n_train_ep,
                    n_novel=args.n_novel, n_shot=args.n_shot, n_eval_per_cls=0,
                    transform=transform_train)
                dataloader_train = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=True,
                    drop_last=False,
                    num_workers=self.n_workers,
                    pin_memory=self.pin_memory,
                )
                return dataloader_train
            else:
                dataset_eval = MiniImageNet(
                    data_pool=data_pool, session=session, partition='eval', transform=transform_eval)
                dataloader_eval = DataLoader(
                    dataset_eval,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.n_workers,
                    pin_memory=self.pin_memory,
                )
                if session == 0:
                    dataset_train = MiniImageNet(
                        data_pool=data_pool, session=0, partition='train',
                        transform=transform_train)
                    dataloader_train = DataLoader(
                        dataset_train,
                        batch_size=self.args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.n_workers,
                        pin_memory=self.pin_memory,
                    )
                    return dataloader_train, dataloader_eval
                return dataloader_eval  # session > 0


def debug():
    from args import ArgumentManager

    args = ArgumentManager().get_args(parser_type='incremental')
    dm = DataManager(args, n_workers=4)
    train_loader, eval_loader = dm.get_dataloaders(session=1, is_fewshot=True)
    print('train loader:', len(train_loader))
    print('eval loader:', len(eval_loader))
    for batch_index, data in enumerate(train_loader):
        print('batch', batch_index)
        # print(len(data))
        img_spt, lbl_spt = data
        print(img_spt.shape, img_spt.squeeze(0).shape)
        print(lbl_spt.shape, lbl_spt)
        break
    for batch_index, (inputs, targets) in enumerate(eval_loader):
        print('eval batch:', batch_index)
        print(inputs.shape)
        print(targets.shape, targets)
        break


if __name__ == '__main__':
    debug()
