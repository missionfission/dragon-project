import argparse
from collections import namedtuple

import torch
import torch.nn as nn

from ir.handlers import handlers
from ir.trace import get_backprop_memory, trace
from plugins.darts.cnn.operations import *
from utils.logger import create_logger
from utils.visualizer import *
from utils.visualizer import plot_descent


class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        self.drop_path_prob = 0
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


def dart_graph():
    Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

    PRIMITIVES = [
        "none",
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5",
    ]

    NASNet = Genotype(
        normal=[
            ("sep_conv_5x5", 1),
            ("sep_conv_3x3", 0),
            ("sep_conv_5x5", 0),
            ("sep_conv_3x3", 0),
            ("avg_pool_3x3", 1),
            ("skip_connect", 0),
            ("avg_pool_3x3", 0),
            ("avg_pool_3x3", 0),
            ("sep_conv_3x3", 1),
            ("skip_connect", 1),
        ],
        normal_concat=[2, 3, 4, 5, 6],
        reduce=[
            ("sep_conv_5x5", 1),
            ("sep_conv_7x7", 0),
            ("max_pool_3x3", 1),
            ("sep_conv_7x7", 0),
            ("avg_pool_3x3", 1),
            ("sep_conv_5x5", 0),
            ("skip_connect", 3),
            ("avg_pool_3x3", 2),
            ("sep_conv_3x3", 2),
            ("max_pool_3x3", 1),
        ],
        reduce_concat=[4, 5, 6],
    )

    AmoebaNet = Genotype(
        normal=[
            ("avg_pool_3x3", 0),
            ("max_pool_3x3", 1),
            ("sep_conv_3x3", 0),
            ("sep_conv_5x5", 2),
            ("sep_conv_3x3", 0),
            ("avg_pool_3x3", 3),
            ("sep_conv_3x3", 1),
            ("skip_connect", 1),
            ("skip_connect", 0),
            ("avg_pool_3x3", 1),
        ],
        normal_concat=[4, 5, 6],
        reduce=[
            ("avg_pool_3x3", 0),
            ("sep_conv_3x3", 1),
            ("max_pool_3x3", 0),
            ("sep_conv_7x7", 2),
            ("sep_conv_7x7", 0),
            ("avg_pool_3x3", 1),
            ("max_pool_3x3", 0),
            ("max_pool_3x3", 1),
            ("conv_7x1_1x7", 0),
            ("sep_conv_3x3", 5),
        ],
        reduce_concat=[3, 4, 6],
    )

    DARTS_V1 = Genotype(
        normal=[
            ("sep_conv_3x3", 1),
            ("sep_conv_3x3", 0),
            ("skip_connect", 0),
            ("sep_conv_3x3", 1),
            ("skip_connect", 0),
            ("sep_conv_3x3", 1),
            ("sep_conv_3x3", 0),
            ("skip_connect", 2),
        ],
        normal_concat=[2, 3, 4, 5],
        reduce=[
            ("max_pool_3x3", 0),
            ("max_pool_3x3", 1),
            ("skip_connect", 2),
            ("max_pool_3x3", 0),
            ("max_pool_3x3", 0),
            ("skip_connect", 2),
            ("skip_connect", 2),
            ("avg_pool_3x3", 0),
        ],
        reduce_concat=[2, 3, 4, 5],
    )
    DARTS_V2 = Genotype(
        normal=[
            ("sep_conv_3x3", 0),
            ("sep_conv_3x3", 1),
            ("sep_conv_3x3", 0),
            ("sep_conv_3x3", 1),
            ("sep_conv_3x3", 1),
            ("skip_connect", 0),
            ("skip_connect", 0),
            ("dil_conv_3x3", 2),
        ],
        normal_concat=[2, 3, 4, 5],
        reduce=[
            ("max_pool_3x3", 0),
            ("max_pool_3x3", 1),
            ("skip_connect", 2),
            ("max_pool_3x3", 1),
            ("max_pool_3x3", 0),
            ("skip_connect", 2),
            ("skip_connect", 2),
            ("max_pool_3x3", 1),
        ],
        reduce_concat=[2, 3, 4, 5],
    )

    DARTS = DARTS_V2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="../data/imagenet/",
        help="location of the data corpus",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="init learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="weight decay")
    parser.add_argument(
        "--report_freq", type=float, default=100, help="report frequency"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--epochs", type=int, default=250, help="num of training epochs"
    )
    parser.add_argument(
        "--init_channels", type=int, default=48, help="num of init channels"
    )
    parser.add_argument("--layers", type=int, default=14, help="total number of layers")
    parser.add_argument(
        "--auxiliary", action="store_true", default=False, help="use auxiliary tower"
    )
    parser.add_argument(
        "--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss"
    )
    parser.add_argument(
        "--drop_path_prob", type=float, default=0, help="drop path probability"
    )
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--arch", type=str, default="DARTS", help="which architecture to use"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5.0, help="gradient clipping"
    )
    parser.add_argument(
        "--label_smooth", type=float, default=0.1, help="label smoothing"
    )
    parser.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    parser.add_argument(
        "--decay_period",
        type=int,
        default=1,
        help="epochs between two learning rate decays",
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False, help="data parallelism"
    )
    args = parser.parse_args([])
    CLASSES = 1000
    from torch.autograd import Variable

    model = NetworkImageNet(
        args.init_channels, CLASSES, args.layers, args.auxiliary, DARTS
    )

    inputs = torch.randn(1, 3, 224, 224)
    inputs = Variable(inputs)

    dart_graph = trace(model, inputs)
    return dart_graph
