#! /usr/bin/env python3
#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

import models
import logging
"""The script for family tree or general graphs experiments."""
import scheduler
import math
import pickle
from IPython.core.debugger import Pdb
import copy
import collections
import functools
import os
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import random as py_random
import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from difflogic.cli import format_args
from difflogic.dataset.graph import GraphOutDegreeDataset, \
    GraphConnectivityDataset, GraphAdjacentDataset, FamilyTreeDataset, NQueensDataset, FutoshikiDataset, TowerDataset, SudokuDataset

from difflogic.nn.baselines import MemoryNet
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.rl.reinforce import REINFORCELoss
from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from difflogic.train import TrainerBase
import utils

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.data.dataloader import JacDataLoader
from jactorch.optim.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.train.env import TrainerEnv
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor
import time
import datetime
import warnings
from torch.distributions.categorical import Categorical

warnings.simplefilter('once')
torch.set_printoptions(linewidth=150)

TASKS = [
    'outdegree', 'connectivity', 'adjacent', 'adjacent-mnist', 'has-father',
    'has-sister', 'grandparents', 'uncle', 'maternal-great-uncle', 'nqueens', 'futoshiki', 'tower', 'sudoku']

parser = JacArgumentParser()

parser.add_argument('--upper-limit-on-grad-norm', type=float, default=1000,
                            metavar='M', help='skip optim step if grad beyond this number')

parser.add_argument('--solution-count', type=int, default=5,
                            metavar='M', help='number at which to cap target-set')
parser.add_argument(
    '--model',
    default='nlm',
    choices=['nlm', 'memnet', 'satnet', 'rrn'],
    help='model choices, nlm: Neural Logic Machine, memnet: Memory Networks')

# NLM parameters, works when model is 'nlm'
nlm_group = parser.add_argument_group('Neural Logic Machines')
LogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 4,
        'breadth': 2,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)
nlm_group.add_argument(
    '--nlm-nullary-dim',
    type=int,
    default=16,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

nlm_latent_group = parser.add_argument_group('NLM Latent Model Args')
LogicMachine.make_nlm_parser(
    nlm_latent_group, {
        'depth': 4,
        'breadth': 2,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='latent')

nlm_latent_group.add_argument(
    '--latent-attributes',
    type=int,
    default=10,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

nlm_latent_group.add_argument('--warmup-epochs', type=int,
                              default=1, metavar='N', help='#of iterations with z = 0')
nlm_latent_group.add_argument('--selector-model', type=int,
                              default=0, metavar='N', help='Use latent model as selector for soft min loss')

nlm_latent_group.add_argument('--latent-model', type=str, choices=['conv', 'nlm','rrn','mlp'],
                              default='rrn', metavar='S', help='which latent model to use when model is rrn')


satnet_latent_group = parser.add_argument_group('SATNET Latent Model Args')
satnet_latent_group.add_argument('--latent-bn', type=int,
                                 default=0, metavar='N', help='batch norm for latent mlp in case of satnet')

satnet_latent_group.add_argument('--latent-do', type=float,
                                 default=0, metavar='N', help='dropout for latent mlp in case of satnet')

satnet_latent_group.add_argument('--latent-hidden-list', type=int, nargs='*',
                                 default=[],  help="list of hidden units for MLP in latent model for satnet")


satnet_latent_group.add_argument('--latent-wt-decay', type=float,
                                 default=0, metavar='N', help='wt decay for latent optimizer')

# MemNN parameters, works when model is 'memnet'
memnet_group = parser.add_argument_group('Memory Networks')
MemoryNet.make_memnet_parser(memnet_group, {}, prefix='memnet')

satnet_group = parser.add_argument_group('Satnet model specific args')
satnet_group.add_argument(
    '--satnet-m',
    type=int,
    default=50,
    metavar='N',
    help='low rank of the SDP matrix'
)
satnet_group.add_argument(
    '--satnet-aux',
    type=int,
    default=0,
    metavar='N',
    help='number of aux variables to be introduced'
)

rrn_group = parser.add_argument_group('rrn model specific args')
rrn_group.add_argument('--sudoku-num-steps', type=int,
                       default=32, metavar='N', help='num steps')
rrn_group.add_argument('--sudoku-embed-size', type=int,
                       default=16, metavar='N', help='embed size')
rrn_group.add_argument('--sudoku-hidden-dim', type=int,
                       default=96, metavar='N', help='sudoku hidden dim')
rrn_group.add_argument('--sudoku-do', type=float, default=0.1,
                       metavar='N', help='dropout for msg passing')

rrn_group.add_argument('--latent-sudoku-num-steps',
                       type=int, default=4, metavar='N', help='num steps')
#rrn_group.add_argument('--latent-sudoku-embed-size',type=int,default = 16,metavar = 'N',help='embed size')
#rrn_group.add_argument('--latent-sudoku-hidden-dim',type=int,default = 96,metavar = 'N',help='sudoku hidden dim')
rrn_group.add_argument('--latent-sudoku-do', type=float,
                       default=0.1, metavar='N', help='dropout for msg passing')
rrn_group.add_argument('--latent-sudoku-input-type', type=str, default='pae',
                       choices=['pae', 'dif', 'cat'], help='whats the input to the latent model?')
rrn_group.add_argument('--latent-sudoku-input-prob', type=int, default=0,
                       help='latent model should use probability of target or do argmax')

# task related
task_group = parser.add_argument_group('Task')
task_group.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')
task_group.add_argument(
    '--train-number',
    type=int,
    default=10,
    metavar='N',
    help='size of training instances')
task_group.add_argument(
    '--adjacent-pred-colors', type=int, default=4, metavar='N')
task_group.add_argument('--outdegree-n', type=int, default=2, metavar='N')
task_group.add_argument(
    '--connectivity-dist-limit', type=int, default=4, metavar='N')

data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
    '--gen-graph-method',
    default='edge',
    choices=['dnc', 'edge'],
    help='method use to generate random graph')
data_gen_group.add_argument(
    '--gen-graph-pmin',
    type=float,
    default=0.0,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-pmax',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-colors',
    type=int,
    default=4,
    metavar='N',
    help='number of colors in adjacent task')

data_gen_group.add_argument(
    '--gen-directed', action='store_true', help='directed graph')

data_gen_group.add_argument('--num-missing-queens', type=int, default=1,
                            metavar='M', help='number of missing queens in the query')
data_gen_group.add_argument('--num-constraints', type=int, default=0,
                            metavar='M', help='number of constraints in FutoshikiDataset')

data_gen_group.add_argument('--train-data-size', type=int, default=-1,
                            metavar='M', help='size of training data in FutoshikiDataset')


data_gen_group.add_argument(
    '--regime', action='store_true', help='whether to run training regime or not')
data_gen_group.add_argument('--pretrain-phi', type=int,
                            default=0, help='whether to pretrain phi network  or not')
data_gen_group.add_argument('--min-loss', type=int, default=0,
                            help='compute minimum of loss over possible solutions')
data_gen_group.add_argument('--hot-min-loss', type=int, default=0,
                            help='compute minimum of loss over possible solutions in hot phase')
data_gen_group.add_argument('--arbit-solution', type=int, default=0,
                            help='pick an arbitrary solution from the list of possible solutions')
data_gen_group.add_argument(
    '--train-file', type=str, help="train data file", default='data/nqueens_data_10_5.pkl')
data_gen_group.add_argument('--test-file', type=str, help="test data file")
data_gen_group.add_argument('--hot-data-sampling', type=str,
                            default="rs", help="data sampling strategy when hot")
data_gen_group.add_argument('--warmup-data-sampling', type=str,
                            default="rs", help="data sampling strategy when in warmup phase")

train_group = parser.add_argument_group('Train')
train_group.add_argument(
    '--incomplete-targetset',
    type=int,
    default=0,
    metavar='MISSING TARGETS',
    help='is target set incomplete?')
train_group.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='seed of jacinle.random')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')

train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')

train_group.add_argument(
    '--lr-hot',
    type=float,
    default=0.001,
    metavar='F',
    help='initial learning rate for hot mode')

train_group.add_argument(
    '--lr-latent',
    type=float,
    default=0.0,
    metavar='F',
    help='initial learning rate for hot mode')


train_group.add_argument(
    '--wt-decay',
    type=float,
    default=0.0,
    metavar='F',
    help='weight decay of learning rate per lesson')

train_group.add_argument(
    '--grad-clip',
    type=float,
    default=1000.0,
    metavar='F',
    help='value at which gradients need to be clipped')

train_group.add_argument(
    '--lr-decay',
    type=float,
    default=1.0,
    metavar='F',
    help='exponential decay of learning rate per lesson')

train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient for batches (default: 1)')
train_group.add_argument(
    '--ohem-size',
    type=int,
    default=0,
    metavar='N',
    help='size of online hard negative mining')
train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for training')

train_group.add_argument(
    '--max-batch-size',
    type=int,
    default=8,
    metavar='N',
    help='batch size for training')

train_group.add_argument(
    '--test-batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for testing')
train_group.add_argument(
    '--early-stop-loss-thresh',
    type=float,
    default=1e-5,
    metavar='F',
    help='threshold of loss for early stop')

train_group.add_argument(
    '--reduce-lr',
    type=int,
    default=1,
    metavar='N',
    help='should reduce lr and stop early?')


train_group.add_argument('--latent-reg-wt', type=float, default=0.0,
                         help="weight of unique examples latent regularizer")
train_group.add_argument('--latent-margin-fraction', type=float,
                         default=1.0, help=" margin for the hinge loss")
train_group.add_argument('--latent-margin-min', type=float,
                         default=0.0, help=" margin for the hinge loss")

train_group.add_argument('--latent-aux-loss', type=str, nargs='*',
                         default=[],  help="the kind of negative loss: max, margin or cosine")
train_group.add_argument('--latent-aux-loss-factor', type=float,
                         nargs='*', default=[1.0], help=" relative wt of the aux loss")
train_group.add_argument('--latent-dis-prob', type=str, default='softmax',
                         choices={'softmax', 'inverse'}, help=" how to convert distance to prob")
train_group.add_argument('--rl-exploration', type=int, default=0,
                         help="whether to do exploration based on rl agent's policy or choose greedy action")
train_group.add_argument('--exploration-eps', type=float, default=0,
                         help="epsilon for epsilon greedy policy")
train_group.add_argument('--rl-reward', type=str, default='acc', choices={
                         'acc', 'count'}, help="rl reward. discrete accuracy or pointwise correct count")
train_group.add_argument('--skip-warmup', type=int, default=0,
                         help="whether to skip warmup if checkkpoint is also given.")
train_group.add_argument('--copy-back-frequency', type=int,
                         default=0, help="frequency at which static model to be updated")
train_group.add_argument('--no-static', type=int,
                         default=0, help="no static model.")
train_group.add_argument('--bl-saturate', type=int,
                         default=0, help="whether a saturated baseline is used or not")

# Note that nr_examples_per_epoch = epoch_size * batch_size
TrainerBase.make_trainer_parser(
    parser, {
        'epochs': 20,
        'epoch_size': 250,
        'test_epoch_size': 1000,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 10,
    })

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', type=str, default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--load-checkpoint',
    type=str,
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=200,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-begin-epoch',
    type=int,
    default=0,
    metavar='N',
    help='the interval(number of epochs) after which test starts')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')

logger = get_logger(__file__)

glogger = logging.getLogger("grad")
glogger.setLevel(logging.INFO)

args = parser.parse_args()


assert not (args.min_loss and (args.warmup_data_sampling ==
                               "unique" or args.hot_data_sampling == "unique"))

assert (len(args.latent_aux_loss_factor) == 1) or (
    len(args.latent_aux_loss_factor) == len(args.latent_aux_loss))

if len(args.latent_aux_loss_factor) == 1:
    factor = args.latent_aux_loss_factor[0]
    args.latent_aux_loss_factor = [factor for _ in args.latent_aux_loss]

args.latent_aux_loss_factor = dict(
    zip(args.latent_aux_loss, args.latent_aux_loss_factor))


if args.selector_model:
    assert 'rl' in args.latent_aux_loss_factor
    assert len(args.latent_aux_loss_factor) == 1
    assert args.latent_dis_prob == 'softmax'
    #assert 'margin' not in args.latent_aux_loss_factor
    #assert 'cosine' not in args.latent_aux_loss_factor

if args.lr_latent == 0.0:
    args.lr_latent = args.lr_hot


args.use_gpu = args.use_gpu and torch.cuda.is_available()

if args.dump_dir is not None:
    io.mkdir(args.dump_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)

    grad_handle = logging.FileHandler(os.path.join(args.dump_dir, 'grad.csv'))
    glogger.addHandler(grad_handle)
    glogger.propagate = False
    glogger.info(
        'epoch,iter,loss,latent_loss,grad_norm_before_clip,grad_norm_after_clip,param_norm_before_clip,lgrad_norm_before_clip,lgrad_norm_after_clip,lparam_norm_before_clip')
else:
    args.checkpoints_dir = None
    args.summary_file = None

if args.seed is not None:
    import jacinle.random as random
    random.reset_global_seed(args.seed)

args.task_is_outdegree = args.task in ['outdegree']
args.task_is_connectivity = args.task in ['connectivity']
args.task_is_adjacent = args.task in ['adjacent', 'adjacent-mnist']
args.task_is_family_tree = args.task in [
    'has-father', 'has-sister', 'grandparents', 'uncle', 'maternal-great-uncle'
]
args.task_is_mnist_input = args.task in ['adjacent-mnist']
args.task_is_1d_output = args.task in [
    'outdegree', 'adjacent', 'adjacent-mnist', 'has-father', 'has-sister', 'nqueens', 'futoshiki', 'tower']

args.task_is_nqueens = args.task in ['nqueens']
args.task_is_futoshiki = args.task in ['futoshiki']
args.task_is_tower = args.task in ['tower']
args.task_is_sudoku = args.task in ['sudoku']


def make_dataset(n, epoch_size, is_train):
    pmin, pmax = args.gen_graph_pmin, args.gen_graph_pmax
    if args.task_is_outdegree:
        return GraphOutDegreeDataset(
            args.outdegree_n,
            epoch_size,
            n,
            pmin=pmin,
            pmax=pmax,
            directed=args.gen_directed,
            gen_method=args.gen_graph_method)
    elif args.task_is_connectivity:
        nmin, nmax = n, n
        if is_train and args.nlm_recursion:
            nmin = 2
        return GraphConnectivityDataset(
            args.connectivity_dist_limit,
            epoch_size,
            nmin,
            pmin,
            nmax,
            pmax,
            directed=args.gen_directed,
            gen_method=args.gen_graph_method)
    elif args.task_is_adjacent:
        return GraphAdjacentDataset(
            args.gen_graph_colors,
            epoch_size,
            n,
            pmin=pmin,
            pmax=pmax,
            directed=args.gen_directed,
            gen_method=args.gen_graph_method,
            is_train=is_train,
            is_mnist_colors=args.task_is_mnist_input)
    elif args.task_is_nqueens:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        #
        return NQueensDataset(epoch_size=epoch_size, n=n,
                              num_missing=args.num_missing_queens,
                              random_seed=args.seed,
                              min_loss=args.min_loss,
                              arbit_solution=args.arbit_solution,
                              train_dev_test=0 if is_train else 2,
                              data_file=data_file,
                              data_sampling=args.warmup_data_sampling)
    elif args.task_is_futoshiki:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        return FutoshikiDataset(epoch_size=epoch_size, n=n,
                                num_missing=args.num_missing_queens,
                                num_constraints=args.num_constraints,
                                data_size=args.train_data_size if is_train else -1,
                                arbit_solution = args.arbit_solution,
                                random_seed=args.seed,
                                min_loss=args.min_loss,
                                train_dev_test=0 if is_train else 2,
                                data_file=data_file,
                                data_sampling=args.warmup_data_sampling,
                                args=args)
    elif args.task_is_tower:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        return TowerDataset(epoch_size=epoch_size, n=n,
                            num_missing=args.num_missing_queens,
                            random_seed=args.seed,
                            train_dev_test=0 if is_train else 2,
                            data_file=data_file,
                            data_sampling=args.warmup_data_sampling,
                            )
    elif args.task_is_sudoku:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        return SudokuDataset(epoch_size=epoch_size,
                             data_size=args.train_data_size if is_train else -1,
                             arbit_solution=args.arbit_solution,
                             train_dev_test=0 if is_train else 2,
                             data_file=data_file,
                             data_sampling=args.warmup_data_sampling, args=args
                             )

    else:
        return FamilyTreeDataset(args.task, epoch_size, n, p_marriage=1.0)


def default_reduce_func(k, v):
    return v.mean()


def hook(grad):
    print("grad z latent", grad)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def collapse_tensor(t, count):
    # inverse of expand_tensor
    # same elements are repeated in t. we just have to pick the first one
    orig_batch_size = count.size(0)
    cum_sum = count.cumsum(0)
    return t[cum_sum - 1]


def expand_tensor(t, count):
    # just repeat t[i] count[i] times.
    orig_batch_size = t.size(0)
    rv = []
    final_batch_size = count.sum()
    for i in range(orig_batch_size):
        this_count = count[i]
        rv.append(t[i].unsqueeze(0).expand(this_count, *t[i].size()))
    #
    rv = torch.cat(rv)
    assert rv.size(0) == final_batch_size
    return rv


def unravel_tensor(target_set, count):
    # assuming pad_dim as 1
    pad_dim = 1
    orig_batch_size = target_set.size(0)
    final_batch_size = count.sum()
    rv = []
    for i in range(orig_batch_size):
        this_count = count[i]
        rv.append(target_set[i][:this_count])
    #
    rv = torch.cat(rv)
    assert rv.size(0) == final_batch_size
    return rv


def ravel_and_pad_tensor(target_set, count):
    # TODO: this is padding by 0 and not repeating the last element! couldn't figure out.
    pad_dim = 1
    max_count = count.max()
    cum_sum = count.cumsum(0)
    orig_batch_size = count.size(0)
    rv = []
    for i in range(orig_batch_size):
        start_ind = 0 if i == 0 else cum_sum[i-1]
        end_ind = cum_sum[i]
        rv.append(F.pad(target_set[start_ind:end_ind].transpose(
            0, -1).unsqueeze(0), (0, (max_count - count[i]))).squeeze(0).transpose(0, -1))
    #
    return torch.stack(rv)


def select_feed_dict(feed_dict, select_indices):
    keys = ['mask', 'gtlt', 'n', 'query', 'count', 'is_ambiguous',
            'qid', 'relations', 'target', 'target_set']

    selected_feed_dict = {}
    for key in keys:
        if key in feed_dict:
            selected_feed_dict[key] = feed_dict[key][select_indices]

    return selected_feed_dict


def get_log_prob_from_dis(dis2):
    eps = 0.00001
    if args.latent_dis_prob == 'softmax':
        return F.log_softmax(-1.0*dis2, dim=0)
    else:
        inverse_dis2 = 1.0/(dis2 + eps)
        return torch.log(inverse_dis2/inverse_dis2.sum())


def get_prob_from_dis(dis2):
    eps = 0.00001
    if args.latent_dis_prob == 'softmax':
        return F.softmax(-1.0*dis2, dim=0)
    else:
        inverse_dis2 = 1.0/(dis2 + eps)
        return inverse_dis2/inverse_dis2.sum()

"""
def rl_sampling(weights):
    # give weights 1-eps and eps respectively to top2 indices
    top_2 = torch.topk(weights,k=2,dim=1)[1]
    dist = F.one_hot(top_2[:,0],num_classes=weights.shape[1]).float()*(1-args.exploration_eps) + F.one_hot(top_2[:,1],num_classes=weights.shape[1]).float()*(args.exploration_eps)
    # handle cases for unique where there is no second index
    dist = dist*(weights>0).float()
    dist = dist/dist.sum(dim=1,keepdim=True)
    dist = Categorical(dist)
    return weights.fill_(0.0).scatter_(1,dist.sample().unsqueeze(-1),1.0)
"""

def rl_sampling(weights):
    # give weights 1-eps and eps respectively to top2 indices
    #Pdb().set_trace()
    eps = args.exploration_eps
    probs = eps*weights 
    probs[torch.arange(weights.size(0)), weights.argmax(dim=1)] += 1.0 - probs.sum(dim=1)   
    #handling unique case with a hack, assuming that weights[i,0] == 1 whenever ith is unique 
    probs[weights[:,0] == 1] = 0.0
    probs[weights[:,0] == 1,0] = 1.0

    if len(torch.nonzero(torch.abs(probs.sum(dim=1)-1)>eps)):
        print(probs)
        print(weights)
    dist = Categorical(probs)
    return weights.fill_(0.0).scatter_(1,dist.sample().unsqueeze(-1),1.0)

def distributed_iter(selected_feed_dict, start_index, end_index):
    for s, e in zip(start_index, end_index):
        yield_feed_dict = {}
        for k in selected_feed_dict:
            yield_feed_dict[k] = selected_feed_dict[k][s:e]
        yield yield_feed_dict


def update_output(output_dict, this_output_dict):
    if output_dict is None:
        output_dict = this_output_dict
        return output_dict
    for k in this_output_dict:
        if torch.is_tensor(this_output_dict[k]):
            output_dict[k] = torch.cat(
                [output_dict[k], this_output_dict[k]], dim=0)
        else:
            output_dict[k] = output_dict[k].extend(this_output_dict)
    return output_dict


def update_monitors(monitors, this_monitors, count):
    # Pdb().set_trace()
    if monitors is None:
        return this_monitors
    else:
        for k in this_monitors:
            monitors[k] = (count*monitors[k] + this_monitors[k])/(count+1)

    return this_monitors


class MyTrainer(TrainerBase):
    
    def reset_test(self):
        self.pred_dump = []
        self.errors = []
        for i in self.error_distribution:
            self.error_distribution[i]=0
    
    def step(self, feed_dict, reduce_func=default_reduce_func, cast_tensor=False):
        assert self._model.training, 'Step a evaluation-mode model.'
        self.num_iters += 1
        self.trigger_event('step:before', self)
        loss_latent = 0.0
        if cast_tensor:
            feed_dict = as_tensor(feed_dict)

        begin = time.time()

        self.trigger_event('forward:before', self, feed_dict)

        rl_loss = 0.0
        distributed = False
        # Pdb().set_trace()
        if self.mode == 'warmup':
            loss, monitors, output_dict = self._model(feed_dict)
        elif self.mode == "phi-training":
            # forward pass to obtain intermediate y's for computation of latent variable
            with torch.no_grad():
                #y_hat = self._static_model(feed_dict)['pred'].detach()
                loss, monitors, output_dict = self._model(feed_dict)
                y_hat = output_dict["pred"].detach()
                reward = 1-torch.clamp_max(-output_dict["reward"].detach(), 1)

            # expand each data point for each possible target
            # copy the mentioned keys for each y
            #keys = ['n', 'query', 'count', 'is_ambiguous', 'qid', 'relations','target_set']
            keys = ['mask', 'n', 'query', 'count', 'is_ambiguous',
                    'qid', 'target_set', 'relations', 'gtlt']

            expanded_feed_dict = {}
            for key in keys:
                if key in feed_dict:
                    expanded_feed_dict[key] = expand_tensor(
                        feed_dict[key], feed_dict["count"])
            #
            # unravel target set to obtain different targets
            expanded_feed_dict["target"] = unravel_tensor(
                feed_dict["target_set"], feed_dict["count"])
            # copy interemediate y for each target
            y_hat = expand_tensor(y_hat, feed_dict["count"])

            # compute latent variable
            z_latent = self._latent_model(
                expanded_feed_dict, y_hat)['latent_z']

            # start index and end index are markers for start and end indices
            # of each query in the expanded feed dict
            start_index = torch.cumsum(
                feed_dict["count"], 0) - feed_dict["count"]
            end_index = torch.cumsum(feed_dict["count"], 0)

            # loop over each query
            mask_count = 0
            for i, (s, e) in enumerate(zip(start_index, end_index)):
                if (e-s).item() > 1 and reward[i].sum().item() > 0:
                    mask = 1
                else:
                    mask = 0
                mask_count += mask
                if args.selector_model:
                    dis2 = z_latent[s:e].squeeze(1)
                else:
                    #err_1 = z_latent[s:e] - self._static_model.learnable_z
                    err_1 = z_latent[s:e] - self._model.learnable_z
                    dis2 = (err_1**2).sum(dim=1)
                probs = get_log_prob_from_dis(dis2)
                loss_latent += - \
                    (probs*reward[i][:feed_dict["count"][i]]).sum()*mask
            if mask_count > 0:
                loss_latent /= mask_count
                # Pdb().set_trace()
        else:
            # forward pass to obtain intermediate y's for computation of latent variable
            # with torch.no_grad():
            #    y_hat = self._static_model(feed_dict)['pred'].detach()
            #

            # expand each data point for each possible target
            # copy the mentioned keys for each y
            #keys = ['n', 'query', 'count', 'is_ambiguous', 'qid', 'relations','target_set']

            if args.no_static:
                loss, monitors, output_dict = self._model(
                    feed_dict, return_loss_matrix=True)
                y_hat = output_dict['pred'].detach()
            else:
                with torch.no_grad():
                    #y_hat = self._static_model(feed_dict)['pred'].detach()
                    static_model_output = self._static_model(feed_dict)
                    if isinstance(static_model_output, dict):
                        y_hat = static_model_output['pred'].detach()
                    else:
                        y_hat = static_model_output[2]['pred'].detach()

            keys = ['mask', 'n', 'query', 'count', 'is_ambiguous',
                    'qid', 'target_set', 'relations', 'gtlt']

            expanded_feed_dict = {}
            for key in keys:
                if key in feed_dict:
                    expanded_feed_dict[key] = expand_tensor(
                        feed_dict[key], feed_dict["count"])
            #
            # unravel target set to obtain different targets
            expanded_feed_dict["target"] = unravel_tensor(
                feed_dict["target_set"], feed_dict["count"])
            # copy interemediate y for each target
            y_hat = expand_tensor(y_hat, feed_dict["count"])

            # compute latent variable
            z_latent = self._latent_model(
                expanded_feed_dict, y_hat)['latent_z']

            # start index and end index are markers for start and end indices
            # of each query in the expanded feed dict
            start_index = torch.cumsum(
                feed_dict["count"], 0) - feed_dict["count"]
            end_index = torch.cumsum(feed_dict["count"], 0)

            max_reg_loss = 0.0
            min_reg_loss = 0.0
            min_indices = []
            action_prob = []
            #rl_weights = []
            weights = []
            min_margin = args.latent_margin_min
            #min_margin = min_margin + args.latent_margin_fraction* ((self._static_model.learnable_z**2).sum())
            min_margin = min_margin + args.latent_margin_fraction * \
                ((self._model.learnable_z**2).sum())

            # loop over each query
            # Pdb().set_trace()
            for s, e in zip(start_index, end_index):
                if args.selector_model:
                    dis2 = z_latent[s:e].squeeze(1)
                else:
                    # compute distance from z* for each z_i
                    #err_1 = z_latent[s:e] - self._static_model.learnable_z
                    err_1 = z_latent[s:e] - self._model.learnable_z
                    dis2 = (err_1**2).sum(dim=1)
                    # find min distance z
                    this_min, this_argmin = dis2.min(dim=0)
                    min_reg_loss += this_min
                    min_indices.append(this_argmin + s)
                    # for ambiguous cases
                    if (e - s) > 0:
                        if 'margin' in args.latent_aux_loss_factor:
                            max_reg_loss += args.latent_aux_loss_factor['margin']*torch.clamp(min_margin - (
                                err_1**2).sum(dim=1), min=0)[torch.arange(err_1.size(0)).cuda() != this_argmin].sum()

                        if 'cosine' in args.latent_aux_loss_factor:
                            #max_reg_loss += args.latent_aux_loss_factor['cosine']*torch.clamp(F.cosine_similarity(z_latent[s:e], self._static_model.learnable_z.unsqueeze(0).expand(e-s,-1),dim=1,eps=1e-4),min=0)[torch.arange(err_1.size(0)).cuda() != this_argmin].sum()
                            max_reg_loss += args.latent_aux_loss_factor['cosine']*torch.clamp(F.cosine_similarity(z_latent[s:e], self._model.learnable_z.unsqueeze(
                                0).expand(e-s, -1), dim=1, eps=1e-4), min=0)[torch.arange(err_1.size(0)).cuda() != this_argmin].sum()

                probs = get_prob_from_dis(dis2)
                weights.append(F.pad(
                    probs, (0, feed_dict['target_set'].size(1) - probs.size(0)), "constant", 0))
            #
            min_reg_loss = min_reg_loss/feed_dict['count'].size(0)
            if feed_dict['is_ambiguous'].sum() != 0:
                max_reg_loss = max_reg_loss / \
                    (feed_dict['is_ambiguous'].sum().float())


            # if using rl then pass the entire expanded feed dict along with rl weights
            # otherwise use only min indices
            if 'rl' not in args.latent_aux_loss_factor:
                selected_feed_dict = select_feed_dict(
                    expanded_feed_dict, torch.stack(min_indices))
            else:
                selected_feed_dict = feed_dict
                # TODO why not torch.cat(rl_weights) ? it will automatically be on cuda as get_prob_from_dis returns cuda tensors
                #selected_feed_dict["weights"] = torch.tensor(rl_weights).cuda()
                #Pdb().set_trace()
                if args.rl_exploration:
                    selected_feed_dict["weights"] = rl_sampling(torch.stack(weights).detach().clone())
                else:
                    selected_feed_dict["weights"] = torch.stack(weights).detach().clone()
                    
            #output_dict = None
            #monitors = None
            loss = 0
            # Pdb().set_trace()
            if not args.no_static:
                # Pdb().set_trace()
                loss, monitors, output_dict = self._model(selected_feed_dict)
            else:
                # take weighted avg of loss_matrix in output_dict
                if 'rl' in args.latent_aux_loss_factor:
                    loss = (output_dict['loss_matrix']*selected_feed_dict['weights']
                            ).sum()/selected_feed_dict['weights'].sum()
                else:
                    raise
            # Pdb().set_trace()
#
#            with torch.no_grad():
#                ind = (output_dict['loss_matrix'].masked_fill((feed_dict['mask']<1),float('inf')).min(dim=1)[1] != feed_dict['weights'].nonzero()[:,1]).nonzero()[:,0]
#                if len(ind)>0:
#                    selected_loss = (output_dict['loss_matrix']*feed_dict['weights']).sum(dim=1)[ind]
#
#                    selected_ind = feed_dict['weights'][ind].nonzero()[:,1]
#                    min_loss, min_loss_ind = output_dict['loss_matrix'].masked_fill((feed_dict['mask']<1),float('inf')).min(dim=1)
#                    print('min loss reward: ',output_dict['reward'][ind,min_loss_ind[ind]])
#                    print('selected reward: ', output_dict['reward'][ind,selected_ind])
#                    print('min loss loss: ', loss_matrix[ind,min_loss_ind[ind]])
#                    print('selected loss: ', loss_matrix[ind, selected_ind])
#                    Pdb().set_trace()
#                    print(output_dict['reward'][ind])
            if (args.selector_model or ('rl' in args.latent_aux_loss_factor)) and (feed_dict['is_ambiguous'].sum() > 0):
                #log_prob = torch.log(torch.stack(rl_weights))
                #prob = torch.stack(rl_weights).detach()
                #print(output_dict['reward'])
                #print(torch.stack(weights))
                #print(selected_feed_dict['count'])
                if self.debugger_mode=="hot":
                    pass 
                    #Pdb().set_trace()
                    
                if not args.rl_exploration:
                    avg_reward = ((output_dict['reward']*(feed_dict['mask'].float())).sum(
                    dim=1)/(feed_dict['mask'].sum(dim=1).float())).unsqueeze(-1)
                    #avg_reward = (output_dict['reward']*(feed_dict['mask'].float())).sum()/(feed_dict['mask'].sum().float())
                    rewards = (output_dict['reward'] -
                           avg_reward)*(feed_dict['mask'].float())
                    rl_loss = -1.0*(rewards*torch.stack(weights)).sum()/feed_dict['is_ambiguous'].sum()
                else:
                    #use selected_feed_dict['weights']. rewards should be only for non zero samples. 
                    #Also, now we use REINFORCE : maximize : reward*log(p_action)
                    rl_loss = -1.0*((output_dict['reward']+0.5)*selected_feed_dict['weights']*torch.log(torch.stack(weights) + 1.0 - selected_feed_dict['weights'])).sum()/feed_dict['is_ambiguous'].sum().float() 
                #rewards = (output_dict['reward'] )*(feed_dict['mask'].float())
                #rewards = 100.0
                #max_reg_loss += -1.0*args.latent_aux_loss_factor['rl'] * (prob*log_prob*rewards).sum()/feed_dict['is_ambiguous'].sum()
            loss_latent = args.latent_reg_wt * \
                (rl_loss + (max_reg_loss + min_reg_loss) /
                 self._model.learnable_z.shape[0])

            self.z_latent_sum += z_latent.data.cpu().sum(dim=0)
            self.z_latent2_sum += (z_latent.data.cpu()
                                   * z_latent.data.cpu()).sum(dim=0)
            self.z_count += len(z_latent)

        self.trigger_event('forward:after', self, feed_dict,
                           loss, monitors, output_dict)

        if not distributed:
            loss = reduce_func('loss', loss)
            loss_f = as_float(loss)
        else:
            loss_f = loss

        monitors = {k: reduce_func(k, v) for k, v in monitors.items()}
        if self.mode == 'hot':
            monitors['loss_latent'] = loss_latent
            monitors['loss_margin'] = args.latent_reg_wt * \
                max_reg_loss/self._model.learnable_z.shape[0]
            monitors['loss_rl'] = rl_loss
        if self.mode == 'phi-training':
            monitors['loss_latent'] = loss_latent
        monitors_f = as_float(monitors)

        self._optimizer.zero_grad()
        if self.mode in ['hot', 'phi-training']:
            if torch.is_tensor(loss_latent):
                loss_latent = reduce_func('loss_latent', loss_latent)
            #
            self._latent_optimizer.zero_grad()

        self.trigger_event('backward:before', self,
                           feed_dict, loss, monitors, output_dict)

        if (not distributed) and loss.requires_grad:
            loss.backward()

        if self.mode in ['hot', 'phi-training']:
            if torch.is_tensor(loss_latent):
                loss_latent.backward()
                # print("Grad:",self._latent_model.digit_embed.weight.grad[2,:2],self._latent_model.atn_across_steps.grad)
                # Pdb().set_trace()
                #print('Latent: ',self.digit_embed.weight.data[2,:4], self.row_embed.weight.data[2,:4])
                #print('Atn over steps: ',self.atn_across_steps)

        self.trigger_event('backward:after', self, feed_dict,
                           loss, monitors, output_dict)

        loss_latent_f = loss_latent.item() if torch.is_tensor(loss_latent) else loss_latent
        grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip, lgrad_norm_before_clip, lgrad_norm_after_clip, lparam_norm_before_clip = 0, 0, 0, -1, -1, 0

        if (not distributed) and loss.requires_grad:
            grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip = utils.gradient_normalization(
                self._model, grad_norm=args.grad_clip)
            #glogger.info(','.join(map(lambda x: str(round(x,6)),[self.current_epoch, self.num_iters, loss_f, loss_latent_f, grad_norm_before_clip.item(), grad_norm_after_clip.item(), param_norm_before_clip.item()])))
            if grad_norm_before_clip <= args.upper_limit_on_grad_norm:
                self._optimizer.step()
            else:
                self.num_bad_updates += 1
                logger.info('not taking optim step. Grad too high {}. Num bad updates: {}'.format(round(grad_norm_before_clip,2), self.num_bad_updates))

            #self._optimizer.step()

        if self.mode in ['hot', 'phi-training']:
            lgrad_norm_before_clip, lgrad_norm_after_clip, lparam_norm_before_clip = utils.gradient_normalization(
                self._latent_model, grad_norm=args.grad_clip)
            self._latent_optimizer.step()

        glogger.info(','.join(map(lambda x: str(round(x, 6)), [self.current_epoch, self.num_iters, loss_f, loss_latent_f, grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip,lgrad_norm_before_clip, lgrad_norm_after_clip, lparam_norm_before_clip ])))
        end = time.time()

        self.trigger_event('step:after', self)

        return loss_f, monitors_f, output_dict, {'time/gpu': end - begin}

    def save_checkpoint(self, name):

        if args.checkpoints_dir is not None:
            checkpoint_file = os.path.join(args.checkpoints_dir,
                                           'checkpoint_{}.pth'.format(name))

            model = self._model
            if self._latent_model:
                latent_model = self._latent_model
                if not args.no_static:
                    static_model = self._static_model
            else:
                latent_model = None
                if not args.no_static:
                    static_model = None

            if isinstance(model, nn.DataParallel):
                model = model.module
                if latent_model:
                    latent_model = latent_model.module
                    if not args.no_static:
                        static_model = static_model.module

            state = {
                'model': as_cpu(model.state_dict()),
                'optimizer': as_cpu(self._optimizer.state_dict()),
                'extra': {'name': name}
            }

            if latent_model:
                state["latent_model"] = as_cpu(latent_model.state_dict())
                if not args.no_static:
                    state["static_model"] = as_cpu(static_model.state_dict())
                state["latent_optimizer"] = as_cpu(
                    self._latent_optimizer.state_dict())
            try:
                torch.save(state, checkpoint_file)
                logger.info('Checkpoint saved: "{}".'.format(checkpoint_file))
            except Exception:
                logger.exception(
                    'Error occurred when dump checkpoint "{}".'.format(checkpoint_file))

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            model = self._model
            if self._latent_model:
                latent_model = self._latent_model
                if not args.no_static:
                    static_model = self._static_model
            else:
                latent_model = None
                if not args.no_static:
                    static_model = None

            if isinstance(model, nn.DataParallel):
                model = model.module
                if latent_model:
                    latent_model = latent_model.module
                    if not args.no_static:
                        static_model = static_model.module
            try:
                checkpoint = torch.load(filename)
                model.load_state_dict(checkpoint['model'])
                if "latent_model" in checkpoint:
                    latent_model.load_state_dict(checkpoint["latent_model"])
                    if not args.no_static:
                        static_model.load_state_dict(
                            checkpoint["static_model"])

                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception(
                    'Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning(
                'No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def _dump_meters(self, meters, mode):
        if args.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            meters_kv['mode'] = mode
            meters_kv['time'] = time.time()
            meters_kv['htime'] = str(datetime.datetime.now())
            meters_kv['config'] = args.dump_dir
            meters_kv['lr'] = self._optimizer.param_groups[0]['lr']
            if mode == 'train':
                meters_kv['epoch'] = self.current_epoch
                meters_kv['data_file'] = args.train_file
            else:
                meters_kv['epoch'] = -1
                meters_kv['data_file'] = args.test_file
                meters_kv['error distribution'] = "-".join(
                    [str(self.error_distribution[k]) for k in sorted(self.error_distribution.keys())])
            with open(args.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv))
                f.write('\n')

    data_iterator = {}
    datasets = {}

    def _prepare_dataset(self, epoch_size, mode):
        assert mode in ['train', 'test']

        if mode == 'train':
            batch_size = args.batch_size
            number = args.train_number
        else:
            batch_size = args.test_batch_size
            number = self.test_number

        # The actual number of instances in an epoch is epoch_size * batch_size.
        #
        if mode in self.datasets:
            dataset = self.datasets[mode]
        else:
            # TODO why are we always passing mode = train in make_dataset??
            dataset = make_dataset(number, epoch_size *
                                   batch_size, mode == 'train')
            self.datasets[mode] = dataset

        dataloader = JacDataLoader(
            dataset,
            shuffle=(mode == 'train'),
            batch_size=batch_size,
            num_workers=min(epoch_size, 0))
        self.data_iterator[mode] = dataloader.__iter__()

    def ravel_feed_dict(self, feed_dict):
        ret_dict = {}
        pad_count = len(feed_dict["target_set"][0])
        tile_keys = set(feed_dict.keys())
        tile_keys = tile_keys.difference(set(["target_set", "target", "qid"]))

        for key in tile_keys:
            ret_dict[key] = tile(feed_dict[key], 0, pad_count)

        ret_dict["target"] = feed_dict["target_set"].view(
            -1, feed_dict["target_set"].shape[2])
        return ret_dict

    def _get_data(self, index, meters, mode):
        #
        # Pdb().set_trace()
        feed_dict = self.data_iterator[mode].next()

        meters.update(number=feed_dict['n'].data.numpy().mean())
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)
        return feed_dict

    #used in _test
    def _get_result(self, index, meters, mode):
        feed_dict = self._get_data(index, meters, mode)

        # sample latent
        z_latent = None
        # if self.z_list is not None:
        #    z_latent = torch.stack(py_random.sample(self.z_list,feed_dict['query'].size(0))).cuda()
        #    z_latent = z_latent * feed_dict['is_ambiguous'].float().unsqueeze(-1).expand_as(z_latent)

        # provide  ambiguity information about test
        # print(feed_dict['qid'])
        output_dict = self.model(feed_dict, z_latent)
        if not isinstance(output_dict,dict):
            output_dict = output_dict[2]
        if args.test_only:
            self.pred_dump.append(dict(feed_dict=as_cpu(
                feed_dict), output_dict=as_cpu(output_dict)))
        # if mode=="test":
        #    feed_dict["query"] = output_dict["pred"].unsqueeze(-1)
        #    output_dict = self.model(feed_dict)
        target = feed_dict['target']
        if args.task_is_adjacent:
            target = target[:, :, :args.adjacent_pred_colors]
        result, errors, _ = instance_accuracy(
            target, output_dict['pred'], return_float=True, feed_dict=feed_dict, task=args.task, args=args)
        succ = result['accuracy'] == 1.0

        meters.update(succ=succ)
        meters.update(result, n=target.size(0))
        message = '> {} iter={iter}, accuracy={accuracy:.4f}'.format(
            mode, iter=index, **meters.val)

        if mode == "test":
            self.dump_errors(errors)

        return message, dict(succ=succ, feed_dict=feed_dict)

    def dump_errors(self, errors=None, force=False):
        if errors is not None:
            self.errors.extend(errors)
            for num in errors:
                if num in self.error_distribution:
                    self.error_distribution[num] += 1
                else:
                    self.error_distribution[num] = 1
        if force:
            print("Called with force")
            self.error_distribution = dict(Counter(self.errors))

    def dump_latent_samples(self, z_dumpfile):
        # Pdb().set_trace()
        meters = GroupMeters()
        self._prepare_dataset(self.epoch_size, mode='train')

        latent_z_list = []
        for i in tqdm(range(self.epoch_size)):
            feed_dict = self._get_data(i, meters, "train")
            y_hat = self._static_model(feed_dict)['pred'].detach()
            z_latent = self._latent_model(feed_dict, y_hat)['latent_z']
            # keep z_latent only for ambiguous cases
            #z_latent = z_latent[feed_dict['is_ambiguous'] == 1]
            #
            z_latent = z_latent * \
                feed_dict['is_ambiguous'].float(
                ).unsqueeze(-1).expand_as(z_latent)
            latent_z_list.extend(z_latent)

        logger.info("Dumping {} samples to {}".format(
            len(latent_z_list), z_dumpfile))
        with open(z_dumpfile, "wb") as f:
            pickle.dump(latent_z_list, f)
        return meters

    def load_latent_samples(self, z_dumpfile):
        self.z_list = None
        self.filtered_z_list = None
        if os.path.exists(z_dumpfile):
            logger.info("Loading latent dump file from: {}".format(z_dumpfile))
            with open(z_dumpfile, 'rb') as f:
                self.z_list = pickle.load(f)
                self.filtered_z_list = list(filter(torch.sum, self.z_list))
        else:
            logger.info("latent dump file not found.. ")

    def _get_train_data(self, index, meters):
        return self._get_data(index, meters, mode='train')

    def _train_epoch(self, epoch_size, is_last=False):
        meters = super()._train_epoch(epoch_size)
        # Pdb().set_trace()
#        print(self._latent_model.avg_weight)
        if self.mode == "phi-training":
            return meters, None

        logger.info("Best Dev Accuracy: {}".format(self.best_accuracy))
        i = self.current_epoch

        if self.mode == "hot" and args.copy_back_frequency > 0 and i % args.copy_back_frequency == 0:

            if not args.no_static:
                logger.info("Copying updated parameters to static model")
                self._static_model = copy.deepcopy(self._model)
                self._static_model.train()
                #self._static_model.training = True
                self._static_model.add_to_targetset = args.incomplete_targetset

        if args.save_interval is not None and i % args.save_interval == 0:
            self.save_checkpoint(str(i))
        test_meters = None
        if (self.mode != 'phi-training') and (is_last or (args.test_interval is not None and i % args.test_interval == 0 and i > args.test_begin_epoch)):
            for i in self.error_distribution:
                self.error_distribution[i]=0
            self.reset_test()
            test_meters = self.test()
            if self.best_accuracy < test_meters[0].avg["corrected accuracy"]:
                self.best_accuracy = test_meters[0].avg["corrected accuracy"]
                if self.checkpoint_mode == "warmup":
                    self.save_checkpoint("best_warmup")
                    self.save_checkpoint("best")
                else:
                    self.save_checkpoint("best")
        return meters, test_meters

         

    def _early_stop(self, meters):
        return meters.avg['loss'] < args.early_stop_loss_thresh

    def train(self, start_epoch=1, num_epochs=0):
        self.early_stopped = False
        meters = None

        for i in range(start_epoch, start_epoch + num_epochs):
            self.current_epoch = i

            self.z_latent_sum = torch.zeros(args.nlm_nullary_dim)
            self.z_latent2_sum = torch.zeros(args.nlm_nullary_dim)
            self.z_count = 0
            meters, test_meters = self._train_epoch(
                self.epoch_size, (self.current_epoch == (start_epoch + num_epochs-1)))

            if self.z_count > 0:
                z_mean = self.z_latent_sum/self.z_count
                z2_mean = self.z_latent2_sum/self.z_count
                logger.info("z_std")
                logger.info(torch.sqrt((z2_mean) - (z_mean*z_mean)))
                logger.info("z_mean")
                logger.info(z_mean)
            logger.info("learnable_z")
            logger.info(self._model.learnable_z)

            if self._early_stop(meters):
                self.early_stopped = True
                break

            # Pdb().set_trace()

            if args.reduce_lr and test_meters is not None:
                metric = test_meters[0].avg["corrected accuracy"]
                self.my_lr_scheduler.step(1.0-1.0*metric)
                if self.my_lr_scheduler.shouldStopTraining():
                    logger.info("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(
                        self.my_lr_scheduler.unconstrainedBadEpochs, self.my_lr_scheduler.maxPatienceToStopTraining))
                    break

        return meters, test_meters



def test_at_end(trainer):
    logger.info("++++++++ START RUNNING TEST AT END -------")
    test_files = {}
    if args.task_is_sudoku:
        test_files = {'data/sudoku_9_val_e_big_amb.pkl':'val_e_big_amb','data/sudoku_9_val_e_big_unique.pkl':'val_e_big_unq', 'data/sudoku_9_val_d.pkl':'val_d','data/sudoku_9_val_a.pkl': 'val_a','data/sudoku_9_val_e.pkl': 'val_e', 'data/sudoku_9_test_e.pkl': 'test_e_all', 'data/sudoku_9_test_e_big_amb.pkl': 'test_e_big_amb', 'data/sudoku_9_test_e_big_unique.pkl': 'test_e_big_unq'}
    #
    if args.task_is_nqueens:
        test_files = {'data/nqueens_11_6_test.pkl': 'test_11_6','data/nqueens_11_6_val.pkl': 'val_11_6'}

    if args.task_is_futoshiki:
        test_files = {'data/futo_6_18_5_test.pkl': 'test_6_18','data/futo_6_18_5_val.pkl': 'val_6_18'}
    args.test_only = 1
    for tf in test_files:
        logger.info("Testing for: {}".format(tf))
        args.test_file = tf
        if 'test' in trainer.datasets:
            del trainer.datasets['test']
        if 'test' in trainer.data_iterator:
            del trainer.data_iterator['test'] 
        trainer.reset_test()
        if args.task_is_nqueens or args.task_is_futoshiki:
            args.test_number = int(test_files[tf].split('_')[-2])
            trainer.test_number_begin = args.test_number
            trainer.test_number_end = args.test_number
        
        rv = trainer.test()
        #with open(os.path.join(args.current_dump_dir, test_files[tf]+"_pred_dump.pkl"), "wb") as f:
        #    pickle.dump(trainer.pred_dump, f)
        with open(os.path.join(args.current_dump_dir, 'results.out'), "a") as f:
            print(tf,test_files[tf],rv[0].avg['corrected accuracy'], file=f)


def main(run_id):
    if args.dump_dir is not None:
        if args.runs > 1:
            args.current_dump_dir = os.path.join(args.dump_dir,
                                                 'run_{}'.format(run_id))
            io.mkdir(args.current_dump_dir)
        else:
            args.current_dump_dir = args.dump_dir

        args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')
        args.checkpoints_dir = os.path.join(
            args.current_dump_dir, 'checkpoints')
        io.mkdir(args.checkpoints_dir)

    exp_fh = open(os.path.join(args.current_dump_dir,'exp.sh'),'a')
    print('jac-run {}'.format(' '.join(sys.argv)),file=exp_fh)
    exp_fh.close()

    logger.info('jac-run {}'.format(' '.join(sys.argv))) 
    logger.info(format_args(args))
    print(args.solution_count)
    model = models.get_model(args)

    if args.use_gpu:
        model.cuda()
    optimizer = get_optimizer(args.optimizer, model,
                              args.lr, weight_decay=args.wt_decay)
    if args.accum_grad > 1:
        optimizer = AccumGrad(optimizer, args.accum_grad)


    trainer = MyTrainer.from_args(model, optimizer, args)
    trainer.num_iters = 0
    trainer.num_bad_updates = 0
    trainer.test_batch_size = args.test_batch_size
    trainer.mode = 'warmup'
    trainer.checkpoint_mode = "warmup"
    trainer._latent_model = None
    trainer._static_model = None

    skip_warmup = False
    if args.load_checkpoint is not None:
        extra = trainer.load_checkpoint(args.load_checkpoint)
        #skip_warmup = extra is not None and (extra['name'] == 'last_warmup')
        skip_warmup = args.skip_warmup

    my_lr_scheduler = scheduler.CustomReduceLROnPlateau(trainer._optimizer, {'mode': 'min', 'factor': 0.2, 'patience': math.ceil(
        7/args.test_interval), 'verbose': True, 'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0.01*args.lr, 'eps': 0.0000001}, maxPatienceToStopTraining=math.ceil(20/args.test_interval))
    
    trainer.my_lr_scheduler = my_lr_scheduler
    
    if args.test_only:
        #
        # trainer.load_latent_samples(os.path.join(
        # args.current_dump_dir, "latent_z_samples.pkl"))
        trainer.pred_dump = []
        trainer.reset_test()
        rv = trainer.test()
        #with open(os.path.join(args.current_dump_dir, "pred_dump.pkl"), "wb") as f:
        #    pickle.dump(trainer.pred_dump, f)
        trainer.dump_errors(force=True)
        with open(os.path.join(args.current_dump_dir, 'results.out'), "w") as f:
            print(rv[0].avg['corrected accuracy'], file=f)

        test_at_end(trainer)
        return None, rv

    # training in four phases
    # first phase train model with unique data and also train z*, freeze z*
    # enforce phi network to output z* on unique samples
    # enforce phi network to output z* on at  one and -z* at another  ambiguous samples
    # train final model using
    if not skip_warmup:
        # setup add to target set
        trainer.model.add_to_targetset = args.min_loss and args.incomplete_targetset
        print(trainer.model.learnable_z)
        warmup_meters, warmup_test_meters = trainer.train(
            1, args.warmup_epochs)
        print(trainer.model.learnable_z)
        trainer.save_checkpoint('last_warmup')
    else:
        logger.info("Skipping warmup")

    if args.epochs > 0:
        # define latent model
        # clone the main model
        # set the optimizer
        if skip_warmup:
            trainer._prepare_dataset(args.epoch_size, 'train')
        #
        trainer.checkpoint_mode = "hot"
        trainer.best_accuracy = -1
        args.min_loss = args.hot_min_loss
        if args.regime:
            logger.info("Regime training.")
            trainer.mode = "warmup"
            trainer._model.add_to_targetset = args.min_loss and args.incomplete_targetset
            logger.info("In regime training. trainer._model.add_to_targetset: {}".format(
                trainer._model.add_to_targetset))
        else:
            if not args.selector_model:
                trainer.model.learnable_z.requires_grad = False

            trainer._latent_model = models.get_latent_model(
                args, trainer.model)
            trainer._latent_model.train()
            if not args.no_static:
                trainer._static_model = copy.deepcopy(trainer._model)
            trainer._latent_optimizer = get_optimizer(
                args.optimizer, trainer._latent_model, args.lr_latent, weight_decay=args.latent_wt_decay)

            """
            if args.pretrain_phi > 0:
                logger.info("Pretraining phi")
                trainer.datasets['train'].data_sampling = 'ambiguous'
                #trainer.mode = "phi-training"
                trainer._model.add_to_targetset = args.incomplete_targetset
                logger.info("In phi training. trainer._model.add_to_targetset: {}".format(
                    trainer._model.add_to_targetset))
                _ = trainer.train(args.warmup_epochs+1, args.pretrain_phi)
            """
            trainer.mode = "hot"

            # switch off training mode only after pretraining phi
            # since pretraining phi requires training statistics
            if not args.no_static:
                trainer._static_model.eval()
                #trainer._static_model.training = True
                trainer._static_model.add_to_targetset = args.incomplete_targetset
                trainer._model.add_to_targetset = False
                logger.info("In hot training using a static model. trainer._static_model.add_to_targetset: {}, trainer._model.add_to_targetset: {}".format(
                    trainer._static_model.add_to_targetset, trainer._model.add_to_targetset))

            else:
                trainer._model.add_to_targetset = args.incomplete_targetset
                logger.info("In hot training without a static model. trainer._model.add_to_targetset: {}".format(
                    trainer._model.add_to_targetset))

        #
        # if skip_warmup:
        #    extra = trainer.load_checkpoint(args.load_checkpoint)
        trainer.datasets['train'].reset_sampler(args.hot_data_sampling)
        #trainer.datasets["train"].data_sampling = args.hot_data_sampling
   
        if not args.no_static:
            trainer._static_model.train()
        if args.pretrain_phi > 0:
            trainer.debugger_mode = "phi-training"
            my_lr_scheduler.maxPatienceToStopTraining = 10000
            for x in trainer._optimizer.param_groups:
                x['lr'] = 0.0
            _ = trainer.train(args.warmup_epochs+1, args.pretrain_phi)
        trainer.debugger_mode = "hot"
        trainer.best_accuracy = -1

        trainer._optimizer = get_optimizer(
            args.optimizer, trainer.model, args.lr_hot, weight_decay=args.wt_decay)

        my_lr_scheduler = scheduler.CustomReduceLROnPlateau(trainer._optimizer, {'mode': 'min', 'factor': 0.2, 'patience': math.ceil(
            7/args.test_interval), 'verbose': True, 'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0.01*args.lr_hot, 'eps': 0.0000001}, maxPatienceToStopTraining=math.ceil(25/args.test_interval))
        trainer.my_lr_scheduler = my_lr_scheduler

        final_meters = trainer.train(
            args.warmup_epochs+args.pretrain_phi+1, args.epochs)
        print(trainer.model.learnable_z)
        trainer.save_checkpoint('last')

    trainer.load_checkpoint(os.path.join(
        args.checkpoints_dir, 'checkpoint_best.pth'))
    logger.info("Best Dev Accuracy: {}".format(trainer.best_accuracy))

    # trainer.dump_latent_samples(os.path.join(
    #    args.current_dump_dir, "latent_z_samples.pkl"))

    # trainer.load_latent_samples(os.path.join(
    #    args.current_dump_dir, "latent_z_samples.pkl"))
    
    trainer.reset_test()
    ret = trainer.test()
    trainer.dump_errors(force=True)
    with open(os.path.join(args.current_dump_dir, 'results.out'), "w") as f:
        print(trainer.best_accuracy, ret[0].avg['corrected accuracy'], file=f)

    test_at_end(trainer)
    return trainer.early_stopped, ret


if __name__ == '__main__':
    stats = []
    nr_graduated = 0

    for i in range(args.runs):
        graduated, test_meters = main(i)
        logger.info('run {}'.format(i + 1))

        if test_meters is not None:
            for j, meters in enumerate(test_meters):
                if len(stats) <= j:
                    stats.append(GroupMeters())
                stats[j].update(
                    number=meters.avg['number'], test_acc=meters.avg['accuracy'])

            for meters in stats:
                logger.info('number {}, test_acc {}'.format(meters.avg['number'],
                                                            meters.avg['test_acc']))

        if not args.test_only:
            nr_graduated += int(graduated)
            logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
            if graduated:
                for j, meters in enumerate(test_meters):
                    stats[j].update(grad_test_acc=meters.avg['accuracy'])
            if nr_graduated > 0:
                for meters in stats:
                    logger.info('number {}, grad_test_acc {}'.format(
                        meters.avg['number'], meters.avg['grad_test_acc']))
