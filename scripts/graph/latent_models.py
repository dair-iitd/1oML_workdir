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
    GraphConnectivityDataset, GraphAdjacentDataset, FamilyTreeDataset, NQueensDataset, FutoshikiDataset, TowerDataset 
from difflogic.nn.baselines import MemoryNet
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.rl.reinforce import REINFORCELoss
from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from difflogic.train import TrainerBase

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


import dgl
from rrn.sudoku import SudokuNN 
import rrn.sudoku_data as sd 

from rrn.rrn import RRN


class SudokuConvNet(nn.Module):
    def __init__(self,args):
        super(SudokuConvNet,self).__init__()
        self.args = args 
        
        self.layers = args.latent_hidden_list
        self.add_module("conv_0",nn.Conv2d(32,self.layers[0], kernel_size=3, padding=1))
        for i in range(1,len(self.layers)):
            self.add_module("conv_{}".format(i), nn.Conv2d(self.layers[i-1], self.layers[i], kernel_size=3, padding=1))
        self.add_module("conv_{}".format(len(self.layers)),nn.Conv2d(self.layers[-1], 1, kernel_size=3, padding=1))

        if args.selector_model:
            self.linear = nn.Linear(81,1)
        else:
            self.linear = nn.Linear(81,args.nlm_nullary_dim)

        if args.use_gpu:
            self = self.cuda()

    def forward(self,feed_dict, y_hat,additional_info=None):
        feed_dict = GView(feed_dict)
        target = feed_dict["target"].long()
        # y_hat has shape exp_batch_size x 10 x 81 x num_steps
        # x has shape exp_batch_size x 81 x num_steps
        if self.args.latent_sudoku_input_prob:
            x = torch.gather(y_hat.softmax(dim=1),dim=1,index=target.unsqueeze(-1).expand(len(y_hat),81,self.args.sudoku_num_steps).unsqueeze(1)).squeeze(1)
        else:
            x = y_hat.argmax(dim=1).long()
            x = (x==target.unsqueeze(-1).expand(len(y_hat),81,self.args.sudoku_num_steps)).float()

        # shuffle dimensions to make it exp_batch_size x num_steps x 81 
        # reshape it to exp_batch_size x num_steps x 9 x 9 
        x = x.transpose(1,2).view(-1,self.args.sudoku_num_steps,9,9)

        for i in range(len(self.layers)+1):
            x = torch.relu(self._modules["conv_{}".format(i)](x))
        x = x.view(-1,81)
        return {'latent_z':self.linear(x)}

class SudokuMLPNet(nn.Module):
    def __init__(self,args):
        super(SudokuMLPNet,self).__init__()
        self.args = args 

        self.flinear = nn.Linear(81,32)
        if args.selector_model:
            self.linear = nn.Linear(32,1)
        else:
            self.linear = nn.Linear(32,args.nlm_nullary_dim)
        
        self.atn_over_steps = nn.Parameter(torch.ones(1,1,args.sudoku_num_steps))        
        if args.use_gpu:
            self = self.cuda()

    def forward(self,feed_dict, y_hat, additional_info = None):
        feed_dict = GView(feed_dict)
        target = feed_dict["target"].long()
        # y_hat has shape exp_batch_size x 10 x 81 x num_steps
        # x has shape exp_batch_size x 81 x num_steps
        if self.args.latent_sudoku_input_prob:
            x = torch.gather(y_hat.softmax(dim=1),dim=1,index=target.unsqueeze(-1).expand(len(y_hat),81,self.args.sudoku_num_steps).unsqueeze(1)).squeeze(1)
        else:
            x = y_hat.argmax(dim=1).long()
            x = (x==target.unsqueeze(-1).expand(len(y_hat),81,self.args.sudoku_num_steps)).float()
        x = (x*self.atn_over_steps).sum(dim=2)
        
        
        x = torch.relu(self.flinear(x))
        #return {'latent_z':(err*self.weight).sum(dim=1,keepdim=True)}
        return {'latent_z':self.linear(x)}


class SudokuRRNLatent(nn.Module):
    def __init__(self, args, base_model):
        super(SudokuRRNLatent, self).__init__()
        self.args = args
        
        embed_size=args.sudoku_embed_size
        hidden_dim= args.sudoku_hidden_dim
        edge_drop= args.latent_sudoku_do
        
        self.num_steps = args.latent_sudoku_num_steps 
        self.basic_graph = sd._basic_sudoku_graph()
        
        self.sudoku_indices = torch.arange(0, 81)
        
        if args.use_gpu:
            self.sudoku_indices = self.sudoku_indices.cuda()
        
        self.rows = self.sudoku_indices // 9
        self.cols = self.sudoku_indices % 9
        
        self.row_embed = nn.Embedding(9, embed_size)
        self.col_embed = nn.Embedding(9, embed_size)
   
        #Pdb().set_trace()
        self.row_embed.weight.data = base_model.sudoku_solver.row_embed.weight.data.clone().detach()
        self.col_embed.weight.data = base_model.sudoku_solver.col_embed.weight.data.clone().detach()
        
        if  args.latent_sudoku_input_type in [ 'dif','cat']:
            self.digit_embed = nn.Embedding(10, embed_size)
            self.digit_embed.weight.data = base_model.sudoku_solver.digit_embed.weight.data.clone().detach() 
    

        input_dim = 2*embed_size + 10 if args.latent_sudoku_input_type == 'pae' else 3*embed_size
        if args.latent_sudoku_input_type == 'cat':
            input_dim = 4*embed_size
        elif args.latent_sudoku_input_type == 'pae':
            input_dim = 2*embed_size + 10
        elif args.latent_sudoku_input_type == 'dif':
            input_dim = 3*embed_size

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lstm = nn.LSTMCell(hidden_dim*2, hidden_dim, bias=False)

        msg_layer = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rrn = RRN(msg_layer, self.node_update_func, self.num_steps, edge_drop)

        if args.selector_model:
            self.output_layer = nn.Linear(hidden_dim, 1)
        else:
            self.output_layer = nn.Linear(hidden_dim, args.nlm_nullary_dim)

        """
        atn_across_steps = torch.rand(args.latent_sudoku_num_steps)
        atn_across_steps = atn_across_steps/atn_across_steps.sum()
        atn_across_nodes = torch.rand(81)
        atn_across_nodes = atn_across_nodes/atn_across_nodes.sum()
        atn_across_steps = torch.zeros(args.latent_sudoku_num_steps).fill_(1.0/args.latent_sudoku_num_steps)
        atn_across_nodes = torch.zeros(81).fill_(1.0/81.0)

        self.atn_across_steps = nn.Parameter(atn_across_steps)
        self.atn_across_nodes = nn.Parameter(atn_across_nodes)
#        self.avg_weight = nn.Parameter(torch.rand(1))
        """
    def node_update_func(self, nodes):
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'], nodes.data['rnn_c']
        new_h, new_c = self.lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}



    def collate_fn(self, feed_dict, y_hat):
        graph_list = []
        #Pdb().set_trace()
        for i in range(len(feed_dict['query'])):
            #@TODO may have to change dtype of q. keep an eye
            #q = feed_dict['query'][i]
            graph = copy.deepcopy(self.basic_graph)
            if self.args.latent_sudoku_input_type == 'pae':
                oht = y_hat[i].transpose(0,1) - torch.scatter(torch.zeros_like(y_hat[i]),0, feed_dict['target'][i].flatten().unsqueeze(0).long(), 1).transpose(0,1)
                graph.ndata['err'] = oht  # q means question
            else:
                graph.ndata['prob'] = y_hat[i].transpose(0,1)
                graph.ndata['a'] = feed_dict['target'][i].long()

            graph.ndata['row'] = self.rows
            graph.ndata['col'] = self.cols
            graph_list.append(graph)
        batch_graph = dgl.batch(graph_list)
        return batch_graph


    def forward(self, feed_dict, y_hat,additional_info=None):
        feed_dict = GView(feed_dict)
#        err = (81-(feed_dict["target"].float()==y_hat.argmax(dim=1).float())).sum(dim=1).float().unsqueeze(-1)
#        err = err*self.avg_weight 
#        return {'latent_z':err}
        #convert it to graph
        g = self.collate_fn(feed_dict, y_hat)
        
        if self.args.latent_sudoku_input_type == 'pae':
            input_emb = g.ndata.pop('err')
        else:
            ##ALTERNATIVE - USING TYPE_EMB AS WEIGHTS OF A LINEAR LAYER
            #type_context_scores = F.linear(context_representation_bag.squeeze(-1),self.type_embeddings.weight[:num_types]).view(-1,bag_size, num_types)
            avg_emb = torch.mm(g.ndata.pop('prob'),self.digit_embed.weight)
            #print('Latent: ',self.digit_embed.weight.data[2,:4], self.row_embed.weight.data[2,:4])
            #print('Atn over steps: ',self.atn_across_steps)
            if self.args.latent_sudoku_input_type == 'dif':
                input_emb = avg_emb - self.digit_embed(g.ndata.pop('a'))
            else:
                input_emb = torch.cat([avg_emb,self.digit_embed(g.ndata.pop('a'))], -1)
        #
        #input_digits = self.digit_embed(g.ndata['q'])
        rows = self.row_embed(g.ndata.pop('row'))
        cols = self.col_embed(g.ndata.pop('col'))
        x = self.input_layer(torch.cat([input_emb, rows, cols], -1))
            
        g.ndata['x'] = x
        g.ndata['h'] = x
        g.ndata['rnn_h'] = torch.zeros_like(x, dtype=torch.float)
        g.ndata['rnn_c'] = torch.zeros_like(x, dtype=torch.float)

        outputs = self.rrn(g, True)[-1]
        outputs = outputs.view(-1,81,outputs.size(-1))
        max_pool_output,_  = outputs.max(dim=1)
        #Pdb().set_trace()
        attn_wts = F.softmax(torch.bmm(outputs,max_pool_output.unsqueeze(-1))/float(outputs.size(-1)),dim=1)
        outputs = (outputs*attn_wts.expand_as(outputs)).sum(dim=1)
        logits = self.output_layer(outputs)
        
        
        #logits = self.output_layer(outputs)
        #logits : of shape : args.latent_sudoku_num_steps x batchsize*81 x nullary_dim         
        
        #logits = (self.atn_across_steps.unsqueeze(-1).unsqueeze(-1).expand_as(logits)*logits).sum(dim=0)
        #shape:  batchsize*81 x nullary_dim
        
        #if self.args.selector_model:
        #    logits = logits.view(-1,81,1)
        #else:    
        #    logits = logits.view(-1,81,self.args.nlm_nullary_dim)
        #shape: batchsize x 81 x nullary_dim

        #logits = (self.atn_across_nodes.unsqueeze(0).unsqueeze(-1).expand_as(logits)*logits).sum(dim=1)
        #shape: batchsize x nullary_dim

        return {'latent_z': logits}

