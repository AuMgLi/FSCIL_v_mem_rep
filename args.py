import argparse


class ArgumentManager:

    def __init__(self):
        self._parser = argparse.ArgumentParser()

        self._parser.add_argument('--dataset', type=str, default='miniImageNet')
        self._parser.add_argument('-E', '--encoder', type=str, default='resnet18')
        self._parser.add_argument('-D', '--decoder', type=str, default='resnet18')
        self._parser.add_argument('--height', type=int, default=84,
                                  help="height of an image (default: 84)")
        self._parser.add_argument('--width', type=int, default=84,
                                  help="width of an image (default: 84)")
        self._parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
        self._parser.add_argument('--pretrain', type=bool)

        self._parser.add_argument('-C', '--n_novel', type=int, default=5)
        self._parser.add_argument('-K', '--n_shot', type=int, default=5)

        self._parser.add_argument('--save_per_epoch', type=int, default=20)
        self._parser.add_argument('--eval_per_epoch', type=int, default=1)
        self._parser.add_argument('--seed', type=int, default=97)

    def get_args(self, parser_type):
        assert parser_type in ('pre-train', 'incremental')
        if parser_type == 'pre-train':
            self._parser.add_argument('--use_trainval', default=False, help='use trainval dataset')
            self._parser.add_argument('--max_epoch', default=60, type=int, help="maximum epochs to run")
            self._parser.add_argument('--lr', default=1e-1, type=float, help="initial learning rate")
            self._parser.add_argument('--lr_decay_steps', default=[40, 50, ],
                                      help='multi-step milestones to decay learning rate')
            self._parser.add_argument('--batch_size', default=128, type=int, help="batch size")
            self._parser.add_argument('--n_class', type=int, default=60)
            self._parser.add_argument('--out_channels', type=int, default=512)
            self._parser.add_argument('--save_dir', type=str, default='./ckp/')

            args = self._parser.parse_args()
            args.pretrain = True
        elif parser_type == 'incremental':
            self._parser.add_argument('--max_epoch_sess0', default=20, type=int,
                                      help="maximum epochs to run")
            # self._parser.add_argument('--use_avgpool', default=False)
            self._parser.add_argument('--lr', default=1e-2, type=float, help="initial learning rate")
            self._parser.add_argument('--lr_decay_steps', default=[10, ],
                                      help='multi-step milestones to decay learning rate')
            self._parser.add_argument('--batch_size', default=128, type=int, help="batch size")
            self._parser.add_argument('--n_train_ep', type=int, default=1)
            self._parser.add_argument('--load', type=bool, default=True)
            self._parser.add_argument('--need_norm', type=bool, default=True)
            self._parser.add_argument('--load_dir', type=str, default='./ckp/')
            self._parser.add_argument('--save_dir', type=str, default='./ckp/')

            self._parser.add_argument('--memory_size', type=int, default=1024)
            self._parser.add_argument('--memory_key_dim', type=int, default=512)
            self._parser.add_argument('--memory_K', type=int, default=9)

            args = self._parser.parse_args()
            args.pretrain = False
        else:
            raise NotImplementedError('Parser type "{}" not found.'.format(parser_type))
        return args
