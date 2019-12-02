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

For example, run `python ./main.py --dataset miniImagenet --model Conv4 --method baseline --train_aug`  
Commands below follow this example, and please refer to `lib/io_utils.py` for additional options.


## Test
the train process will automatically do test for you, but if you want to do test by yourself, just
Run
```python ./test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.
