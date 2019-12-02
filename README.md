# tinyImageNet

This code is modified from [PyTorch ImageNet classification example](https://github.com/pytorch/examples/tree/master/imagenet). But many new features are added.


## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/)
 - [tensorboard](https://www.tensorflow.org/tensorboard)

## Getting started
* put the all data into `./data`

## Train
Run
```python ./main.py --arch [ARCHTECHTURE] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

For example, run `python ./train.py --dataset miniImagenet --model Conv4 --method baseline --train_aug`  
Commands below follow this example, and please refer to io_utils.py for additional options.
