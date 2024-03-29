import os
import time
import warnings
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from efficientnet_pytorch import EfficientNet

from lib.io_utils import parse_args
from lib.utils import check_dir, AverageMeter, ProgressMeter
from lib.utils import GradualWarmupScheduler
from lib.dataset import get_loader
from lib.model import se_resnext101_32x48d, wide_se_resnext101_32x32d
from lib.mixup import mixup_data, mixup_criterion
from lib.loss import LabelSmoothingLoss


best_acc1 = 0


def train(train_loader, model, criterion, optimizer, scheduler, epoch, summary_writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.mixup:
            images, target_a, target_b, lam = mixup_data(images, target, args.alpha)
            output = model(images)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        step = epoch * len(train_loader) + i

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(step)

        # log
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
        summary_writer.add_scalar('train_acc1', acc1, step)
        summary_writer.add_scalar('train_loss', loss, step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, summary_writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # log
            step = epoch * len(val_loader) + i
            summary_writer.add_scalar('val_acc1', acc1, step)
            summary_writer.add_scalar('val_loss', loss, step)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def test(test_loader, model, args):
    # switch to evaluate mode
    model.eval()

    results = pd.DataFrame(test_loader.dataset.images, columns=['Id', 'Category'])
    results['Id'] = results['Id'].apply(lambda x: x.split('/')[1])

    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            output = model.forward(images)
            pred = output.argmax(dim=-1)
            results.iloc[i*args.batch_size:(i+1)*args.batch_size, 1] = pred.cpu().numpy()

    resultsName = args.output_dir + "results.csv"
    results.to_csv(resultsName, index=False)
    print(f'=> save results to ' + resultsName)


def extract_features(loader, model, args):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        if args.evaluate:
            images = np.array(loader.dataset.images)
            results = pd.DataFrame(index=range(images.shape[0]), columns=range(101))
            results.iloc[:, 0] = images[:, 0]
            for i, images in enumerate(tqdm(loader)):
                images = images.cuda(non_blocking=True)
                # compute output
                output = model.forward(images)

                if i == len(loader)-1:
                    results.iloc[i*args.batch_size:i*args.batch_size +
                                 output.shape[0], 1:101] = output.cpu().numpy()[:, :100]
                else:
                    results.iloc[i*args.batch_size:(i+1)*args.batch_size, 1:101] = output.cpu().numpy()[:, :100]
        else:
            images = np.array(loader.dataset.images)
            results = pd.DataFrame(index=range(images.shape[0]), columns=range(102))
            results.iloc[:, 0] = images[:, 0]
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                # compute output
                output = model.forward(images)

                if i == len(loader)-1:
                    results.iloc[i*args.batch_size:i*args.batch_size +
                                 output.shape[0], 1:101] = output.cpu().numpy()[:, :100]
                    results.iloc[i*args.batch_size:i*args.batch_size + target.shape[0], 101] = target.cpu().numpy()
                else:
                    results.iloc[i*args.batch_size:(i+1)*args.batch_size, 1:101] = output.cpu().numpy()[:, :100]
                    results.iloc[i*args.batch_size:(i+1)*args.batch_size, 101] = target.cpu().numpy()

    results = results.dropna()
    if args.evaluate:
        outputName = args.output_dir + "test_output.csv"
    else:
        outputName = args.output_dir + "output.csv"
    results.to_csv(outputName, index=False)
    print(f'=> save results to ' + outputName)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_scheduler(optimizer, n_iter_per_epoch, args):
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer, eta_min=0.000001,
        T_max=(args.epochs - args.start_epoch - args.warmup_epoch) * n_iter_per_epoch)
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=args.warmup_multiplier,
        total_epoch=args.warmup_epoch * n_iter_per_epoch,
        after_scheduler=cosine_scheduler)
    return scheduler


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    global best_acc1

    summary_writer = SummaryWriter(args.log_dir)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if(args.arch == "efficientNet-b7"):
            model = EfficientNet.from_pretrained('efficientnet-b7')
            model = nn.DataParallel(model)
        elif(args.arch == "resnext101"):
            model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
            model = nn.DataParallel(model)
        elif(args.arch == "se_resnet101"):
            model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet101', num_classes=100)
        elif(args.arch == "se_resnext101"):
            model = se_resnext101_32x48d(num_classes=100)
        elif(args.arch == "wide_se_resnext101"):
            model = wide_se_resnext101_32x32d(num_classes=100)
            model = nn.DataParallel(model)
        else:
            model = models.__dict__[args.arch]()

    model = model.cuda()

    # Data loading code
    if args.features:
        train_loader = get_loader(args.data, 'data/train.txt', args.batch_size, args.workers, False)
    elif args.augment:
        train_loader = get_loader(args.data, 'data/train.txt', args.batch_size, args.workers, True)
    else:
        train_loader = get_loader(args.data, 'data/train.txt', args.batch_size, args.workers, False)
    val_loader = get_loader(args.data, 'data/val.txt', args.batch_size, args.workers, False)
    test_loader = get_loader(args.data, 'data/test.txt', args.batch_size, args.workers, False)

    # define loss function (criterion), optimizer and scheduler
    if args.label_smoothing > 0.0:
        criterion = LabelSmoothingLoss(label_smoothing=0.1, tgt_size=1000, keep_index=100).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint from '{}'".format(args.resume))
            # loc = 'cuda' + os.environ["CUDA_VISIBLE_DEVICES"]
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.features:
        if args.evaluate:
            extract_features(test_loader, model, args)
        else:
            extract_features(train_loader, model, args)

        return

    if args.evaluate:
        test(test_loader, model, args)
        return

    # for epoch in range(args.start_epoch, args.epochs):
    for epoch in range(args.epochs - args.start_epoch):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, summary_writer, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, summary_writer, args)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

        if (epoch + 1) % args.save_freq == 0 or ((epoch + 1) == args.epochs - args.start_epoch):
            state = {
                'epoch': args.start_epoch + epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }
            if (epoch + 1) == args.epochs:
                filename = os.path.join(check_dir(args.save_dir), 'best_model.tar')
            else:
                filename = os.path.join(check_dir(args.save_dir), f'{epoch}.tar')
            print(f'=> saving checkpoint to {filename}')
            torch.save(state, filename)

    test(test_loader, model, args)


if __name__ == '__main__':
    main()
