import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--node",default=3)
parser.add_argument("--workload",default='resnet18')
parser.add_argument("--pitch", default="1um")
parser.add_argument("--backprop",default=False)
parser.add_argument("--mapping",default="edp")
parser.add_argument("",default="")

args = parser.parse_args()

