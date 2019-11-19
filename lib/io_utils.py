import argparse

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append("resnext-101")
model_names.append("efficientNet-b7")


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default='./data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--warmup-epoch', default=20, type=int,
                        metavar='E', help='warmup epoch (default: 20)')
    parser.add_argument('--warmup-multiplier', default=16, type=int,
                        metavar='E', help='warmup multiplier (default: 16)')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save-freq', default=50, type=int,
                        metavar='S', help='save frequency (default: 50)')
    parser.add_argument('--model-dir', type=str, metavar='PATH',
                        help='path to save and log models')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='checkpoint / number or best_model')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()
    vars(args)['output_dir'] = './output/' + args.model_dir + '/'
    vars(args)['save_dir'] = './output/' + args.model_dir + '/checkpoints'
    vars(args)['log_dir'] = './output/' + args.model_dir + '/tensorboard'

    if args.resume != '':
        vars(args)['resume'] = './output/' + args.model_dir + '/checkpoints/' + args.resume + '.tar'

    return args
