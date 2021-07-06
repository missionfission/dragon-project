import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from models.ssd_rs34 import SSD_R34
from models.rnnt import RNNT
from dlrm.dlrm_s_pytorch import DLRM_Net, dash_separated_floats, dash_separated_ints
from ir.trace import trace
from models.Unet import Generic_UNet

# from transformers import (
#     DPRConfig,
#     DPRContextEncoder,
#     DPRQuestionEncoder,
#     DPRReader,
#     DPRReaderTokenizer,
# )


def dlrm_graph():
    import dlrm.dlrm_data_pytorch as dp

    # ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="dataset"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    args = parser.parse_args([])

    # if args.mlperf_logging:
    #     print("command line args: ", json.dumps(vars(args)))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    # ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if args.data_generation == "dataset":

        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    # ### construct the neural network specified above ###
    # # WARNING: to obtain exactly the same initialization for
    # # the weights we need to start from the same random seed.
    # # np.random.seed(args.numpy_rand_seed)
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
    )
    for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
        Z = dlrm(X, lS_o, lS_i)
    dlrm_graph = trace(dlrm, (X, lS_o, lS_i))
    return dlrm_graph


def resnet_18_graph():
    for name, model in models.__dict__.items():
        #     print(name)
        if not name.islower() or name.startswith("__") or not callable(model):
            continue
        if "resnet18" in name:
            print(name)
            model = model().eval()
            inputs = torch.randn(1, 3, 224, 224)
            resnet_graph = trace(model, inputs)
            break
    return resnet_graph

def resnet_50_graph():
    for name, model in models.__dict__.items():
        #     print(name)
        if not name.islower() or name.startswith("__") or not callable(model):
            continue
        if "resnet50" in name:
            print(name)
            model = model().eval()
            inputs = torch.randn(1, 3, 224, 224)
            resnet_graph = trace(model, inputs)
            break
    return resnet_graph

def vggnet_graph():
    for name, model in models.__dict__.items():
        #     print(name)
        if not name.islower() or name.startswith("__") or not callable(model):
            continue
        if "vgg11" in name and "vgg11_bn" not in name:
            inputs = torch.randn(1, 3, 224, 224)
            vgg11_graph = trace(model().eval(), inputs)
            # print(vgg11_graph)
            break
    return vgg11_graph


def bert_graph():
    from transformers import BertModel, BertConfig

    # model.configMM

    # tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'tokenizer', 'bert-base-cased', do_basic_tokenize=False)

    # Tokenized input
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    # tokenized_text = tokenizer.tokenize(text)
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens)
    ### Get the hidden states computed by `bertModel`
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    indexed_tokens = [
        101,
        2627,
        1108,
        3104,
        1124,
        15703,
        136,
        102,
        3104,
        1124,
        15703,
        1108,
        170,
        16797,
        8284,
        102,
    ]

    # Convert inputs to PyTorch tensors
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])

    # model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'model', 'bert-base-cased')
    configuration = BertConfig()
    model = BertModel(configuration)
    model.eval()

    model(tokens_tensor)
    bert_graph = trace(model, tokens_tensor)
    return bert_graph


def dpr_graph():

    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    indexed_tokens = [
        101,
        2627,
        1108,
        3104,
        1124,
        15703,
        136,
        102,
        3104,
        1124,
        15703,
        1108,
        170,
        16797,
        8284,
        102,
    ]

    # Convert inputs to PyTorch tensors
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])
    configuration = DPRConfig()
    contextencoder_graph = trace(DPRContextEncoder(configuration).eval(), tokens_tensor)
    questionencoder_graph = trace(
        DPRQuestionEncoder(configuration).eval(), tokens_tensor
    )
    reader_graph = trace(DPRReader(configuration).eval(), tokens_tensor)
    return contextencoder_graph, questionencoder_graph, reader_graph


def gpt2_graph():
    from transformers import GPT2Model, GPT2Config

    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    indexed_tokens = [
        101,
        2627,
        1108,
        3104,
        1124,
        15703,
        136,
        102,
        3104,
        1124,
        15703,
        1108,
        170,
        16797,
        8284,
        102,
    ]

    # Convert inputs to PyTorch tensors
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])
    configuration = GPT2Config()
    model = GPT2Model(configuration)
    model.eval()

    model(tokens_tensor)
    gpt2_graph = trace(model, tokens_tensor)
    return gpt2_graph


def alexnet_graph():
    import torchvision.models as models

    for name, model in models.__dict__.items():
        #             print(name)
        if not name.islower() or name.startswith("__") or not callable(model):
            #             print(name.islower())
            continue
        #         print(name)
        if "alexnet" in name:
            model = model().eval()
            inputs = torch.randn(1, 3, 224, 224)
            alexnet_graph = trace(model, inputs)
            break
    return alexnet_graph


class LSTMTagger(nn.Module):
    def __init__(
        self, embedding_dim=24, hidden_dim=2048, vocab_size=32768, tagset_size=1024
    ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def langmodel_graph():

    # torch.manual_seed(1)
    # lstm = nn.LSTM(793470, 1024)  # Input dim is 3, output dim is 3
    # inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
    # # initialize the hidden state.
    # hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
    # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    # hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    # # out, hidden = lstm(inputs, hidden)

    # LAYERS["embed1"] = FCLayer(1, 793470, 1)
    # LAYERS["act_1"] = FCLayer(2048, 32768, 1)
    # LAYERS["proj_1"] = FCLayer(8192, 1024, 1)
    inputs = torch.randn(2048)  # make a sequence of length 5
    model1 = nn.Sequential(nn.Linear(2048, 32768))
    model2 = nn.Sequential(nn.Linear(8192, 1024))

    langmod_graph1 = trace(model1.eval(), inputs)
    inputs = torch.randn(8192)  # make a sequence of length 5
    langmod_graph2 = trace(model2.eval(), inputs)
    return langmod_graph1, langmod_graph2


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def Unet():
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d

    inputs = torch.randn(1, 160, 224, 224)

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    net_num_pool_op_kernel_sizes = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    net_conv_kernel_sizes =  [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
     # loaded automatically from plans_file
    model = Generic_UNet(160, 24, 16,len(net_num_pool_op_kernel_sizes),1, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,dropout_op_kwargs,net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    unet_Graph = trace(model.eval(), inputs)
    # return unet_Graph
    return 0


# featurizer_config = dict({sample_rate : 16000, window_size : 0.02, window_stride : 0.01, window : "hann", features : 80, n_fft : 512,frame_splicing : 3, dither : 0.00001, feat_type : "logfbank", normalize_transcripts : true, trim_silence : true, pad_to : 0})

# rnn_config = {rnn_type : "lstm", encoder_n_hidden : 1024, encoder_pre_rnn_layers : 2, encoder_stack_time_factor : 2, encoder_post_rnn_layers : 3, pred_n_hidden : 320, pred_rnn_layers : 2, forget_gate_bias : 1.0, joint_n_hidden : 512, dropout:0.32}
import toml
def speech2text_model():
    config = toml.load("rnnt.toml")
    featurizer_config = config["input_eval"]
    model = RNNT(
            feature_config=featurizer_config,
            rnnt= config["rnnt"],
            num_classes=29
        )
    seq_length, batch_size, feature_length = 157, 1, 240
    inp = torch.randn([seq_length, batch_size, feature_length])
    feature_length = torch.LongTensor([seq_length])
    x_padded, x_lens = model.encoder(inp, feature_length)
    speech2text_graph = trace(model.eval(), (inp, feature_length))
    return speech2text_graph

def objectdetection_model():
    model = SSD_R34().model
    inputs = torch.randn(1, 3, 1200, 1200)
    objectdetection_graph = trace(model.eval(), inputs)
    return objectdetection_graph


# 224x224
# 1200x1200
# 224x224x160,16
# bert max_seq_len=384