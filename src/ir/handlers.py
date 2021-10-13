import numpy as np

__all__ = ["handlers"]

#  Augment these with memory handling capabilities
# ComputeExpense, read_access,write_access
def addmm(node):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p, n * p + n * m + m * p, 0


def addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    return (n * m, n * m + n + m, 0)


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return (b * n * m * p, b * n * m + b * m * p, 0)


def matmul(node):
    # print(node.inputs[0].shape, node.inputs[1].shape)
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        return (n, n, 0)
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        return n * m, n * m, 0
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        return n * m, 0, 0
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p, m * p + n * m, 0
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        return np.prod(b) * n * m, np.prod(b) * n * m, 0
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        return np.prod(b) * n * m, 0, 0
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return np.prod(b) * n * m * p, np.prod(b) * n * m + np.prod(b) * m * p, 0


def mul(node):
    # print(node.outputs[0].shape, node.inputs[0].shape, node.inputs[1].shape)
    os = node.outputs[0].shape
    # print("used")
    return np.prod(os), np.prod(node.inputs[0].shape), 0


def convolution(node):
    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
        oc, ic, *ks = node.inputs[1].shape
    else:
        ic, oc, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    return np.prod(os) * ic * np.prod(ks), np.prod(ks) * ic * oc, 0


def batch_norm(node):
    # TODO: provide an option to not fuse `batch_norm` into `linear` or `conv`
    return 0, 0, 0


def instance_norm_or_layer_norm(node):
    os = node.outputs[0].shape

    return np.prod(os), 0, 0


def avg_pool_or_mean(node):
    os = node.outputs[0].shape
    return np.prod(os), 0, 0


def lstm(node):
    os = node.outputs[0].shape
    inp = node.inputs[0].shape
    print(node.inputs[1], os)
    return np.prod(inp), np.prod(os), 0

def linear(node):
    os = node.outputs[0].shape
    inp = node.inputs[0].shape
    return np.prod(inp), np.prod(os), 0
    
handlers = (
    (("aten::linear","aten:flatten",), linear),
    ("aten::lstm", lstm),
    ("aten::addmm", addmm),
    ("aten::addmv", addmv),
    ("aten::bmm", bmm),
    ("aten::matmul", matmul),
    (("aten::mul", "aten::mul_",), mul),
    ("aten::_convolution", convolution),
    ("aten::batch_norm", batch_norm),
    (("aten::instance_norm", "aten::layer_norm",), instance_norm_or_layer_norm),
    (
        (
            "aten::adaptive_avg_pool1d",
            "aten::adaptive_avg_pool2d",
            "aten::adaptive_avg_pool3d",
            "aten::avg_pool1d",
            "aten::avg_pool2d",
            "aten::avg_pool3d",
            "aten::mean",
        ),
        avg_pool_or_mean,
    ),
    (
        (
            "aten::adaptive_max_pool1d",
            "aten::adaptive_max_pool2d",
            "aten::adaptive_max_pool3d",
            "aten::add",
            "aten::add_",
            "aten::alpha_dropout",
            "aten::cat",
            "aten::chunk",
            "aten::clone",
            "aten::constant_pad_nd",
            "aten::contiguous",
            "aten::div",
            "aten::div_",
            "aten::dropout",
            "aten::dropout_",
            "aten::embedding",
            "aten::eq",
            "aten::feature_dropout",
            "aten::flatten",
            "aten::gt",
            "aten::hardtanh_",
            "aten::int",
            "aten::lt",
            "aten::log_softmax",
            "aten::max_pool1d",
            "aten::max_pool1d_with_indices",
            "aten::max_pool2d",
            "aten::max_pool2d_with_indices",
            "aten::max_pool3d",
            "aten::max_pool3d_with_indices",
            "aten::max_unpool1d",
            "aten::max_unpool2d",
            "aten::max_unpool3d",
            "aten::ne",
            "aten::reflection_pad1d",
            "aten::reflection_pad2d",
            "aten::reflection_pad3d",
            "aten::relu",
            "aten::relu_",
            "aten::replication_pad1d",
            "aten::replication_pad2d",
            "aten::replication_pad3d",
            "aten::rsub",
            "aten::select",
            "aten::sigmoid",
            "aten::size",
            "aten::slice",
            "aten::softmax",
            "aten::softshrink",
            "aten::squeeze",
            "aten::sub",
            "aten::sum",
            "aten::t",
            "aten::tanh",
            "aten::threshold",
            "aten::transpose",
            "aten::view",
            "aten::zeros",
            "prim::constant",
            "prim::listconstruct",
            "prim::listunpack",
            "prim::numtotensor",
        ),
        None,
    ),
)
