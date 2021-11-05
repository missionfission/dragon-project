## Quick Start
To get the worst case (#chip scale by 2) message traffic for Illusion scaleup on Resnet18, run
```
python sweep_CNN_message.py -m resnet -r -b BasicBlock -s 2
```




## Usage
```
usage: sweep_CNN_message.py [-h] [-m {bert,resnet}] [-r]
                            [-b {BasicBlock,Bottleneck}] [-s CHIP_SCALE]

optional arguments:
  -h, --help            show this help message and exit
  -m {bert,resnet}, --model {bert,resnet}
                        model to be swept
  -r, --real_model      use real model
  -b {BasicBlock,Bottleneck}, --blocktype {BasicBlock,Bottleneck}
                        resnet block type
  -s CHIP_SCALE, --chip_scale CHIP_SCALE
                        #chip scaling factor
```