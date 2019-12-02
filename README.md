# tinyImageNet

This code is modified from [PyTorch ImageNet classification example](https://github.com/pytorch/examples/tree/master/imagenet). We support more models like [efficientNet-b7](https://arxiv.org/abs/1905.11946), [resnext101](https://pytorch.org/hub/pytorch_vision_resnext/) and models with [Squeeze-and-Excitation attention](https://arxiv.org/abs/1709.01507). we also add many regularization tricks borrowed like [mixup](https://arxiv.org/abs/1710.09412), [labelsmoothing](https://arxiv.org/pdf/1701.06548.pdf). 


## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/)
 - [tensorboard](https://www.tensorflow.org/tensorboard)

## Getting started
* put the all data into `./data`

## Train
Run
```python ./main.py --arch [ARCHTECHTURE] --model-dir [DIRTOSAVE] [--OPTIONARG]```

For example, run `python ./main.py  --arch efficientNet-b7 --model-dir efficientNet_mixup --lr 0.07`  
Commands below follow this example, and please refer to Usage below for additional options.


## Prediction
- After training, the prediction be automatically generated for you in `./output/your-model-dir/results.csv` 
- If you want to test model by yourself, just add `--evaluate` in train command.
```python ./test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.

### Usage

```usage: main.py [-h] [--data DIR] [--arch ARCH] [-j N] [--epochs N]
               [--start-epoch N] [-b N] [--lr LR] [--momentum m] [--wd W]
               [--mixup MIXUP] [--alpha ALPHA] [--augment AUGMENT]
               [--label-smoothing LABEL_SMOOTHING] [--warmup-epoch E]
               [--warmup-multiplier E] [-e] [-x] [-p N] [--save-freq S]
               [--model-dir PATH] [--resume PATH] [--pretrained] [--seed SEED]
               [--using-AdaBoost USING_ADABOOST]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  --arch ARCH           model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn | wide_resnet101_2 |
                        wide_resnet50_2 | resnext101 | efficientNet-b7 |
                        se_resnet101 | se_resnext101 | wide_se_resnext101
                        (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum m          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --mixup MIXUP         whether to use mixup
  --alpha ALPHA         alpha used for mix up
  --augment AUGMENT     whether to use data augment
  --label-smoothing LABEL_SMOOTHING
                        label smoothing ratio
  --warmup-epoch E      warmup epoch (default: 20)
  --warmup-multiplier E
                        warmup multiplier (default: 16)
  -e, --evaluate        evaluate model on validation set
  -x, --extract-features
                        extract features on train set
  -p N, --print-freq N  print frequency (default: 10)
  --save-freq S         save frequency (default: 10)
  --model-dir PATH      path to save and log models
  --resume PATH         checkpoint / number or best_model
  --pretrained          use pre-trained model
  --seed SEED           seed for initializing training.
  --using-AdaBoost USING_ADABOOST
                        using AdaBoost to manage training data
```
