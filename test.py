import os
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

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
            images = images.cuda(non_blocking=True)

            # compute output
            output = model.forward(images)
            pred = output.argmax(dim=-1)
            results.iloc[i*args.batch_size:(i+1)*args.batch_size, 1] = pred.cpu().numpy()

    resultsName = args.output_dir + "results.csv"
    results.to_csv(resultsName, index=False)
    print(f'=> save results to ' + resultsName)


def main():
    args = parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if(args.arch == "efficientNet-b7"):
            model = EfficientNet.from_pretrained('efficientnet-b7')
            model = nn.DataParallel(model)
        elif(args.arch == "resnext-101"):
            model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
            model = nn.DataParallel(model)
        else:
            model = models.__dict__[args.arch]()

    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint from '{}'".format(args.resume))
        loc = 'cuda' + os.environ["CUDA_VISIBLE_DEVICES"]
        checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("you need to specific the resume dir to get model")
        return

    cudnn.benchmark = True

    # Data loading code
    test_loader = get_loader(args.data, 'data/test.txt', args.batch_size, args.workers, False)

    test(test_loader, model, args)

# login


# logout

if __name__ == '__main__':
    main()
