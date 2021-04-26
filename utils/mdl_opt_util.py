from torch.optim import SGD, Adam
from torch.optim.adamw import AdamW
from torch.optim.rmsprop import RMSprop
from models.encoders.resnet import resnet12, resnet18
from models.decoders import res_decoder18


def create_encoder(args, use_avgpool=True, is_snail=False):
    if args.pretrain:
        n_cls = args.n_class
        out_channels = args.out_channels
    else:  # incremental
        n_cls = None
        out_channels = args.memory_key_dim
    model_name = args.encoder
    print('Preparing encoder {}...'.format(model_name))
    if model_name == 'resnet12':
        model = resnet12(n_classes=n_cls, use_avgpool=use_avgpool,
                         out_channels=out_channels, is_snail=is_snail)
    elif model_name == 'resnet18':
        model = resnet18(n_classes=n_cls, use_avgpool=use_avgpool,
                         out_channels=out_channels, is_snail=is_snail)
    else:
        raise NotImplementedError('Invalid encoder name: {}.'.format(args.encoder))
    return model


def create_decoder(args, out_dim, fm_level=-1):
    if args.pretrain:
        out_channels = args.out_channels
    else:  # incremental
        out_channels = args.memory_key_dim
    model_name = args.decoder
    print('Preparing decoder {}...'.format(model_name))
    if model_name == 'resnet18':
        model = res_decoder18(in_dim=out_channels, out_dim=out_dim, fm_level=fm_level)
    else:
        raise NotImplementedError('Invalid decoder name: {}.'.format(args.decoder))
    return model


def create_optim(params, args=None, optim_name=None, lr=None):
    if args is None:
        assert optim_name is not None and lr is not None
    else:
        assert optim_name is None and lr is None
        optim_name = args.optim
        lr = args.lr
    if optim_name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif optim_name == 'adam':
        return Adam(params, lr=lr, weight_decay=0)
    elif optim_name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=1e-2)
    elif optim_name == 'amsgrad':
        return Adam(params, lr=lr, weight_decay=0, amsgrad=True)
    elif optim_name == 'rmsprop':
        return RMSprop(params, lr=lr, momentum=0.9, weight_decay=0)
    else:
        raise NotImplementedError('Unsupported optimizer_memory: {}'.format(optim_name))


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
