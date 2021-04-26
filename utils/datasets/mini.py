import os.path as osp
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset

_BASE_LABELS = list(range(0, 60))
_NOVEL_LABELS = list(range(60, 100))
_DATA_ROOT = 'E:/python_projects/datasets/miniImagenet/'


def load_data(file):
    with open(file, 'rb') as f:
        try:
            data = pickle.load(f, encoding='latin1')
        except EOFError:
            return None
    return data


def buildLabelIndex(labels):
    label2index = {}
    for idx, label in enumerate(labels):
        if label not in label2index:
            label2index[label] = []
        label2index[label].append(idx)
    return label2index


def get_pair(data, labels):
    assert (data.shape[0] == len(labels))
    data_pair = []
    for i in range(data.shape[0]):
        data_pair.append((data[i], labels[i]))
    return data_pair


class MiniImageNetDataPool:
    def __init__(self, dataset_dir=_DATA_ROOT):
        self._dataset_dir = dataset_dir
        # Train
        train_train_path = osp.join(
            self._dataset_dir, 'miniImageNet_category_split_train_phase_train.pickle')
        train_val_path = osp.join(
            self._dataset_dir, 'miniImageNet_category_split_train_phase_val.pickle')
        train_test_path = osp.join(
            self._dataset_dir, 'miniImageNet_category_split_train_phase_test.pickle')
        data_train_train = load_data(train_train_path)
        data_train_val = load_data(train_val_path)
        data_train_test = load_data(train_test_path)

        self.image_train_train = data_train_train['data']  # 38400
        self.label_train_train = data_train_train['labels']  # [0 ~ 63]
        self.image_train_val = data_train_val['data']  # 18748
        self.label_train_val = data_train_val['labels']
        self.image_train_test = data_train_test['data']  # 19200
        self.label_train_test = data_train_test['labels']

        # Val & Test
        val_path = osp.join(self._dataset_dir, 'miniImageNet_category_split_val.pickle')
        test_path = osp.join(self._dataset_dir, 'miniImageNet_category_split_test.pickle')
        data_val = load_data(val_path)  # [64 ~ 79]
        data_test = load_data(test_path)  # [80 ~ 99]

        self.image_val = data_val['data']
        self.label_val = data_val['labels']
        self.image_test = data_test['data']
        self.label_test = data_test['labels']


class MiniImageNet(Dataset):
    """
    Dataset for pretrain phase.
    Dataset statistics:
        -Base(60)
            --Base_train
            --Base_eval
        -Novel(40, 5 classes * 8 session)
    """

    def __init__(self, data_pool: MiniImageNetDataPool, session, partition, n_novel=5,
                 transform=None, dataset_dir=_DATA_ROOT):
        super().__init__()
        assert partition in ('train', 'eval')
        self._data_pool = data_pool
        self._dataset_dir = dataset_dir
        self._session = session
        assert session in range(0, 9)  # [0 ~ 8]
        if session == 0:
            self._session_labels = _BASE_LABELS
        else:
            assert partition == 'eval'
            self._session_labels = _NOVEL_LABELS[(session - 1) * n_novel: session * n_novel]
        self._partition = partition
        self._transform = transform

        image_train_train = data_pool.image_train_train  # 38400
        label_train_train = data_pool.label_train_train  # [0~63]
        if self._partition == 'train':
            # images_all = np.concatenate([image_train_train, image_train_val], axis=0)
            # labels_all = np.concatenate([label_train_train, label_train_val], axis=0)
            images_all = image_train_train
            labels_all = np.asarray(label_train_train)
        else:  # eval
            # Pre-training
            image_train_val = data_pool.image_train_val  # 18748
            label_train_val = data_pool.label_train_val
            image_train_test = data_pool.image_train_test  # 19200
            label_train_test = data_pool.label_train_test  # [0~63]
            # Incremental training
            image_val = data_pool.image_val
            label_val = data_pool.label_val  # [64~79]
            image_test = data_pool.image_test
            label_test = data_pool.label_test  # [80~99]

            # images_all = np.concatenate([image_train_test, image_val, image_test], axis=0)
            # labels_all = np.concatenate([label_train_test, label_val, label_test], axis=0)
            images_all = np.concatenate(
                [image_train_val, image_train_test, image_val, image_test], axis=0)
            labels_all = np.concatenate(
                [label_train_val, label_train_test, label_val, label_test], axis=0)
        select_idxs = np.where(np.isin(labels_all, self._session_labels))[0]
        self._images = images_all[select_idxs]
        self._labels = labels_all[select_idxs].tolist()

    def __getitem__(self, index):
        image = self._images[index]
        if self._transform is not None:
            image = self._transform(image)
        return image, self._labels[index]

    def __len__(self):
        return len(self._labels)

    def print_stats(self):
        unique_labels = np.unique(self._labels)
        print(unique_labels)
        n_cats = len(unique_labels)
        n_samples = len(self._labels)
        print('Dataset statistics:')
        print('-' * 60)
        print('# PART.: Base-{} | # CATS.: {} | # IMAGES: {}'.format(self._partition, n_cats, n_samples))
        print('-' * 60)


class MiniImageNetEpisode(Dataset):
    """
    Dataset for incremental phase.
    Dataset statistics:
        -Base(60)
            --Base_train
            --Base_eval
        -Novel(40, 5 classes * 8 session)
    """
    def __init__(self, data_pool: MiniImageNetDataPool, session, n_novel=5, n_shot=5, episode_size=1,
                 n_eval_per_cls=0, transform=None, dataset_dir=_DATA_ROOT):
        super().__init__()

        assert session in range(0, 9)  # [0~8]
        self._session = session
        if session != 0:  # else in _sample_episode()
            self._session_labels = _NOVEL_LABELS[(session - 1) * n_novel: session * n_novel]
        self._epoch_size = episode_size
        self._n_novel = n_novel
        self._n_shot = n_shot
        self._n_eval_per_cls = n_eval_per_cls
        self._transform = transform
        self._dataset_dir = dataset_dir

        if session == 0:  # [0 ~ 63]
            image_train_train = data_pool.image_train_train  # 38400
            label_train_train = data_pool.label_train_train

            self._images_all = image_train_train
            self._labels_all = np.asarray(label_train_train)
        else:
            image_val = data_pool.image_val
            label_val = data_pool.label_val  # [64 ~ 79]
            image_test = data_pool.image_test
            label_test = data_pool.label_test  # [80 ~ 99]

            self._images_all = np.concatenate([image_val, image_test], axis=0)
            self._labels_all = np.concatenate([label_val, label_test], axis=0)
            if session == 1:  # need [60 ~ 63] here
                # train_train_path = osp.join(
                #     self._dataset_dir, 'miniImageNet_category_split_train_phase_train.pickle')
                # data_train_train = load_data(train_train_path)  # [0~63]
                # image_train_train = data_train_train['data']  # 38400
                # label_train_train = data_train_train['labels']
                image_train_train = data_pool.image_train_train  # 38400
                label_train_train = data_pool.label_train_train

                self._images_all = np.concatenate([image_train_train, self._images_all], axis=0)
                self._labels_all = np.concatenate([label_train_train, self._labels_all], axis=0)
        # print(np.unique(labels_all))

    def _sample_episode(self):
        if self._session == 0:
            self._session_labels = np.random.choice(_BASE_LABELS, self._n_novel, replace=False)
        sess_idxs = np.where(np.isin(self._labels_all, self._session_labels))
        self._images_ep = self._images_all[sess_idxs]
        self._labels_ep = self._labels_all[sess_idxs]

        # Sample episode
        self._images_spt = []
        self._labels_spt = []
        self._images_qry = []
        self._labels_qry = []
        for i in self._session_labels:
            sess_idxs = np.where(self._labels_ep == i)[0]
            sess_idxs = np.random.choice(sess_idxs, self._n_shot + self._n_eval_per_cls, replace=False)

            spt_idxs = sess_idxs[:self._n_shot]
            np.random.shuffle(spt_idxs)
            self._images_spt.append(self._images_ep[spt_idxs])
            self._labels_spt.append(self._labels_ep[spt_idxs])

            qry_idxs = sess_idxs[self._n_shot:]
            self._images_qry.append(self._images_ep[qry_idxs])
            self._labels_qry.append(self._labels_ep[qry_idxs])
        self._images_spt = np.asarray(self._images_spt)
        self._labels_spt = np.asarray(self._labels_spt)
        self._images_spt = self._images_spt.reshape(
            (self._n_novel * self._n_shot, *self._images_spt.shape[2:]))  # (25, 84, 84, 3)
        self._labels_spt = self._labels_spt.reshape(self._n_novel * self._n_shot)  # (25,)

        self._images_qry = np.array(self._images_qry)
        self._labels_qry = np.array(self._labels_qry)
        self._images_qry = self._images_qry.reshape(
            (self._n_novel * self._n_eval_per_cls, *self._images_qry.shape[2:]))
        self._labels_qry = self._labels_qry.reshape(self._n_novel * self._n_eval_per_cls)

        # Shuffle support set
        spt_idxs = list(range(len(self._labels_spt)))
        np.random.shuffle(spt_idxs)
        self._images_spt = self._images_spt[spt_idxs]
        self._labels_spt = self._labels_spt[spt_idxs]

    def __getitem__(self, index):
        self._sample_episode()
        if self._transform is not None:
            images_spt = None
            images_qry = None
            for i in range(len(self._images_spt)):
                image_transformed = self._transform(self._images_spt[i]).unsqueeze(0)
                if images_spt is None:
                    images_spt = image_transformed  # [1, 3, 84, 84]
                else:
                    images_spt = torch.cat([images_spt, image_transformed], dim=0)
            for i in range(len(self._images_qry)):
                image_transformed = self._transform(self._images_qry[i]).unsqueeze(0)
                if images_qry is None:
                    images_qry = image_transformed
                else:
                    images_qry = torch.cat([images_qry, image_transformed], dim=0)
            # self._images_spt = torch.from_numpy(np.asarray(images_spt))
            self._images_spt = images_spt
            self._images_qry = images_qry

        self._labels_spt = torch.from_numpy(self._labels_spt)
        if self._images_qry is None:
            return self._images_spt, self._labels_spt
        self._labels_qry = torch.from_numpy(self._labels_qry)
        return self._images_spt, self._labels_spt, self._images_qry, self._labels_qry

    def __len__(self):
        return self._epoch_size

    def print_stats(self):
        unique_labels = np.unique(self._labels_all)
        n_cats = len(unique_labels)
        print('Dataset statistics:')
        print('-' * 58)
        print('# PART.: Novel | # CATS.: {} | # SPT IMG.: {} | QRY IMG.: {}'.format(
            n_cats, self._n_shot, self._n_eval_per_cls))
        print('# SESS: {} | LABELS: {}'.format(self._session, self._session_labels))  # TODO
        print('-' * 58)


if __name__ == '__main__':
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF
    from PIL import Image

    transform_train = T.Compose([
        lambda x: Image.fromarray(x),
        T.RandomCrop(84, padding=8),
        # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5)
    ])

    transform_eval = T.Compose([
        lambda x: Image.fromarray(x),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dp = MiniImageNetDataPool()
    # mi = MiniImageNet(dp, session=1, partition='eval', transform=transform_eval)
    # mi.print_stats()
    # exit()

    mini_imagenet_novel = MiniImageNetEpisode(dp, session=1, episode_size=1, transform=transform_eval)
    print(len(mini_imagenet_novel))
    exit()
    for ii in range(50):
        img_spt, label_spt, img_qry, label_qry = mini_imagenet_novel[ii]
        print(len(img_spt), img_spt[0].shape, len(img_qry))
        _labels = np.unique(label_spt)
        print(_labels)
        print(label_spt)
        print(label_qry)
        for j, im in enumerate(img_spt):
            # if label_spt[j] == labels[0]:
            if label_spt[j] == 0:
                im = TF.to_pil_image(im)
                # print(type(im))
                im.show()
