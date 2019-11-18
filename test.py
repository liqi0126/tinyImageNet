import os
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models

from lib.io_utils import parse_args
from lib.dataset import get_loader

best_acc1 = 0


def test(test_loader, model, args):
    # switch to evaluate mode
    model.eval()

    results = pd.DataFrame(test_loader.dataset.images, columns=['Id', 'Category'])
    results['Id'] = results['Id'].apply(lambda x: x.split('/')[1])

    with torch.no_grad():
        for i, images in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model.forward(images)
            pred = output.argmax(dim=-1)
            results.iloc[i*args.batch_size:(i+1)*args.batch_size, 1] = pred.cpu().numpy()

    results.to_csv("results.csv", index=False)


def main():
    args = parse_args()

    if args.gpu is not None:
        print("Use GPU: {} for testresuing".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("you need to specific the resume dir to get model")
        return

    cudnn.benchmark = True

    # Data loading code
    test_loader = get_loader(args.data, 'data/test.txt', args.batch_size, args.workers, False)

    test(test_loader, model, args)


if __name__ == '__main__':
    main()
