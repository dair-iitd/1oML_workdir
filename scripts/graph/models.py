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
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference, LogicSoftmaxInference
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
from latent_models import *
import utils
try:
    import satnet
except:
    pass



logger = get_logger(__file__)

def get_model(args):
    if args.model == 'nlm':
        rmodel = Model(args)
    elif args.model == 'satnet':
        rmodel = get_satnet_model(args)
    elif args.model == 'rrn':
        rmodel = get_rrn_model(args)
    if args.use_gpu:
        rmodel = rmodel.cuda()
    return rmodel

def get_latent_model(args,base_model=None):
    if args.latent_model == 'eg':
        rmodel = EpsilonGreedyLatentModel(args)
    elif args.latent_model == 'det':
        rmodel = DeterministicLatentModel(args)
    elif args.model == 'nlm':
        rmodel = LatentModel(args) 
    elif args.model == 'satnet':
        rmodel = get_satnet_model_latent(args)
    elif args.model == 'rrn':
        rmodel = get_rrn_model_latent(args,base_model)
    if args.use_gpu:
        rmodel = rmodel.cuda()
    return rmodel

def get_rrn_model_latent(args, base_model):
    if args.latent_model=="conv":
        return SudokuConvNet(args)
    if args.latent_model=="rrn":
        return SudokuRRNLatent(args, base_model)
    if args.latent_model=="mlp":
        return SudokuMLPNet(args)
    if args.latent_model=="nlm":
        return LatentModel(args)

def get_satnet_model(args):
    return SatnetFutoSolver(args)

def get_satnet_model_latent(args):
    #import satnet
    if args.latent_model == 'conv':
        return ConvNet(args)  
    elif args.latent_model == 'nlm':
        return LatentModel(args)

def get_rrn_model(args):
    return SudokuRRNNet(args)


class SatnetFutoSolver(nn.Module):
    def __init__(self, args):
        super(SatnetFutoSolver, self).__init__()
        self.args = args
        if args.task_is_futoshiki:
            self.n = (args.train_number**3)+2*(args.train_number**2)
        elif args.task_is_nqueens:
            self.n = args.train_number**2
        elif args.task_is_tower:
            self.n = (args.train_number**3)+4*(args.train_number**2)
        
        #
        self.sat = satnet.SATNet(self.n, args.satnet_m, args.satnet_aux, weight_normalize=True)
        
        self.base_loss = nn.BCELoss()
        self.wt_base_loss = nn.BCELoss(reduction='none')
        self.learnable_z = torch.randn(3)

        def loss_aggregator(pred, target, count, target_mask,  weights=None):
            # if pred and target have same dimension then simply compute loss
            if pred.dim()==target.dim():
                if weights is not None:
                    loss = (weights*self.wt_base_loss(pred,target).sum(dim=1)).sum()/weights.sum()
                    return loss
                if self.args.task_is_nqueens:
                    return self.wt_base_loss(pred,target).sum(dim=-1).mean()
                return self.base_loss(pred, target)
            #Pdb().set_trace() 
            # if pred and target are not of same dimension then compute loss wrt each element in target set
            # return a (batchsize x targetset size) vector 
            #target.shape = batch_size x target size x num_variables
            #pred.shape: batch_size x num variables
            loss_tensor = self.wt_base_loss(pred.unsqueeze(dim=1).expand_as(target),target)*target_mask.unsqueeze(-1).expand_as(target).float()
            loss_tensor = loss_tensor.mean(dim=-1)
            #shape = batch_size x target_size s
            if self.args.min_loss:
                #return has shape: batch_size 
                loss_tensor  = loss_tensor.masked_fill((1-target_mask.byte()),float('inf')).min(dim=1)[0]
            #
            return loss_tensor

        self.loss = loss_aggregator

    # target_mask: shape = batch size x target size
    
    # input_mask: corresponds to variables given in the query, should include > and < as well in addition to other 
    # variables that have been instantiated. shape: batch size x #variables

    # query_mask: corresponds to data variables for which loss should be computed. 
    # should exclude > and < only. shape: batch size x #variables. used to compute loss. should be same for all elements in the batch.
    def forward(self, feed_dict, z_latent=None):
        feed_dict = GView(feed_dict)
        if self.args.task_is_futoshiki:
            sat_input = torch.cat((feed_dict.query[:,:,0],feed_dict.gtlt),dim=1).float()
            sat_mask  = torch.cat((feed_dict.query[:,:,0],torch.ones_like(feed_dict.gtlt)),dim =1 ).int()
            sat_pred = self.sat(sat_input, sat_mask)
            pred = sat_pred[:,:feed_dict.query.size(1)]     
        elif self.args.task_is_nqueens:
            sat_input = feed_dict.query[:,:,0].float()
            sat_mask = feed_dict.query[:,:,0].int()
            pred = self.sat(sat_input, sat_mask)
        
        elif self.args.task_is_tower:
            sat_input = feed_dict.query[:,:,0].float()
            sat_mask = torch.cat((torch.ones((len(feed_dict.n),4*feed_dict.n[0]**2)),torch.zeros((len(feed_dict.n),feed_dict.n[0]**3))),dim=1).int().cuda()
            pred = self.sat(sat_input, sat_mask)
        
        if self.training:
            monitors = dict()
            target = feed_dict.target.float()
            count = None
            #Pdb().set_trace()
            if self.args.min_loss or 'weights' in feed_dict:
                target = feed_dict.target_set.float()
                count = feed_dict.count.int()
           
            this_meters,_,reward,new_targets = instance_accuracy(
                 feed_dict.target.float(), pred, return_float= False, feed_dict=feed_dict, task = self.args.task,args=self.args)
           
            #logger.info("Reward: ")
            #logger.info(reward)
            monitors.update(this_meters)
            #def loss_aggregator(pred, target, count, query_mask, target_mask,  weights=None):
            loss = self.loss(pred, target, count,feed_dict.mask)

            if self.args.min_loss:
                loss = loss.mean()
            elif 'weights' in feed_dict:
                loss = (feed_dict.weights*loss).sum()/feed_dict.weights.sum()
                
            return loss, monitors, dict(pred=pred,reward=reward)
        else:
            return dict(pred=pred)



class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = jacnn.Conv2dLayer(
            1, 10, kernel_size=5, batch_norm=True, activation='relu')
        self.conv2 = jacnn.Conv2dLayer(
            10,
            20,
            kernel_size=5,
            batch_norm=True,
            dropout=False,
            activation='relu')
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_sudoku_relations():
    relations = np.zeros((81,81,3))
    
    for i in range(81):
        row = i//9
        col = i%9
        
        for j in range(9):
            relations[i,row*9+j,0]=1
            relations[i,j*9+col,1]=1
        
        b_row = (row//3)*3
        b_col = (col//3)*3
        
        for j in range(b_row,b_row+3):
            for k in range(b_col,b_col+3):
                relations[i,j*9+k,2]=1
    return torch.tensor(relations).float().cuda()


class DeterministicLatentModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.dummy_parameter = nn.Parameter(torch.ones(1))
        logger.info("Returning deterministic latent model")
    def forward(self, feed_dict, y_hat,additional_info=None):
        constant = (self.dummy_parameter + 100)/(self.dummy_parameter + 100)
        return dict(latent_z=constant*self.args.latent_annealing*feed_dict['query'].size(1)*feed_dict['loss'].unsqueeze(1))

class EpsilonGreedyLatentModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.dummy_parameter = nn.Parameter(torch.ones(1))
        logger.info("Returning deterministic epsilon greedy latent model")
    def forward(self, feed_dict, y_hat,additional_info=None):
        constant = (self.dummy_parameter + 100)/(self.dummy_parameter + 100)
        #Pdb().set_trace()
        return dict(latent_z=constant*feed_dict['minloss_eg_prob'].unsqueeze(1))


        
class LatentModel(nn.Module):
    """ The model for latent variable """

    def __init__(self,args):
        super().__init__()
        self.args = args
        self.feature_axis = 0
        # inputs
        input_dims = [0 for _ in range(args.latent_breadth + 1)]
        if args.task_is_nqueens:
            input_dims[1] = 1
            input_dims[2] = 4
        elif args.task_is_futoshiki:
            input_dims[1]=3
            input_dims[2]=3
        elif args.task_is_tower:
            input_dims[1]=5
            input_dims[2]=3
        elif args.task_is_sudoku:
            input_dims[1]=10
            input_dims[2]=3
            self.relations = get_sudoku_relations()

        self.features = LogicMachine.from_args(
            input_dims, args.latent_attributes, args, prefix='latent')
        output_dim = self.features.output_dims[self.feature_axis]

        target_dim = args.nlm_nullary_dim
        if args.selector_model:
            target_dim=1
        # nothing but MLP with sigmoid
        self.pred = LogitsInference(output_dim, target_dim, [])
        self.latent_breadth = args.latent_breadth
        self.task_is_futoshiki = args.task_is_futoshiki
        self.task_is_tower = args.task_is_tower
        self.task_is_sudoku = args.task_is_sudoku 

    def forward(self, feed_dict, y_hat,additional_info=None):
        feed_dict = GView(feed_dict)
        #Pdb().set_trace()
        if self.task_is_sudoku:
            relations = self.relations.expand(len(feed_dict.target),81,81,3)
        else:
            relations = feed_dict.relations.float()
        
        batch_size, nr = relations.size()[:2]

        #states = feed_dict.query.float()
        # @TODO : should we give x as input as well?
        if self.task_is_futoshiki:
            states = torch.stack([y_hat - feed_dict.target.float(),feed_dict.query[:,:,1].float(),feed_dict.query[:,:,2].float()],2)
        elif self.task_is_tower:
            states = torch.stack([y_hat - feed_dict.target.float()]+[feed_dict.query[:,:,i].float() for i in range(1,5)],2)
        elif self.task_is_sudoku:
            states = y_hat.transpose(1,2) - torch.nn.functional.one_hot(feed_dict.target.long(),10).float()
        else:
            states = (y_hat - feed_dict.target.float()).unsqueeze(2)
        #

        inp = [None for _ in range(self.latent_breadth + 1)]
        inp[1] = states
        inp[2] = relations

        depth = None
        feature = self.features(inp, depth=None)[self.feature_axis]

        latent_z = self.pred(feature)
        return dict(latent_z=latent_z)




class Model(nn.Module):

    def __init__(self,args):
        super().__init__()

        # inputs
        self.args = args
        if args.task_is_nqueens:
            binary_dim = 4 
            unary_dim = 1 
        elif args.task_is_futoshiki:
            binary_dim = 3
            unary_dim = 3
        elif args.task_is_tower:
            binary_dim = 3
            unary_dim = 5
        elif args.task_is_sudoku:
            binary_dim = 3
            unary_dim = 10
            self.relations = get_sudoku_relations()

        self.feature_axis = 1 if (args.task_is_1d_output or args.task_is_sudoku) else 2
        input_dims = [0 for _ in range(args.nlm_breadth + 1)]
        input_dims[0] = args.nlm_nullary_dim
        input_dims[1] = unary_dim
        input_dims[2] = binary_dim
        #Pdb().set_trace()
        self.features = LogicMachine.from_args(
                input_dims, args.nlm_attributes, args, prefix='nlm')
        
        output_dim = self.features.output_dims[self.feature_axis]
        if args.task_is_sudoku:
            target_dim = 10
            self.pred = LogicSoftmaxInference(output_dim, target_dim, [])
        else:
            target_dim = 1
            self.pred = LogicInference(output_dim, target_dim, [])
        
        # learnable z parameter
        learnable_z = torch.randn(args.nlm_nullary_dim)
        self.learnable_z = nn.Parameter(learnable_z)

        # losses
        if not args.task_is_sudoku:
            self.base_loss = nn.BCELoss()
            self.wt_base_loss = nn.BCELoss(reduction='none')
        else:
            self.base_loss = nn.CrossEntropyLoss()
            self.wt_base_loss = nn.CrossEntropyLoss(reduction='none')
    
        def loss_aggregator(pred, target, count, weights=None):
            # if pred and target have same dimension then simply compute loss
            #Pdb().set_trace()
            if pred.dim()==target.dim():
                # if weights are not none then weigh each datapoint appropriately else simply average them
                if weights is not None:
                    loss = (weights*self.wt_base_loss(pred,target).sum(dim=1)).sum()/weights.sum()
                    return loss
                return self.base_loss(pred, target)
            
            if self.args.cc_loss:
                #loss = log(sum_prob)
                #first compute probability of each of the targets
                #pred.shape = BS X N
                #target.shape = BS x num targets x N
                target_prob  = pred.unsqueeze(1).expand_as(target) 
                target_prob = (target_prob*target.float() + (  1 - target_prob) * ( 1 - target.float())).prod(dim=-1)
                #now target_prob is of shape: batchsize x number of targets.
                batch_loss = []
                for i in range(len(target_prob)):
                    total_prob = target_prob[i][:count[i]].sum()
                    batch_loss.append(-1*torch.log(total_prob)/target.size(-1))
                return torch.stack(batch_loss)
            #
            else:
                # if pred and target are not of same dimension then compute loss wrt each element in target set
                # return a (batchsize x targetset size) vector 
                batch_loss = []
                for i in range(len(pred)):
                    x = pred[i]
                    instance_loss = []
                    for y in target[i][:count[i]]:
                        instance_loss.append(self.base_loss(x, y))
                    if self.args.min_loss:
                        batch_loss.append(torch.min(torch.stack(instance_loss)))
                    elif self.args.naive_pll_loss:
                        batch_loss.append(torch.mean(torch.stack(instance_loss)))
                    else:
                        batch_loss.append(F.pad(torch.stack(instance_loss),(0,len(target[i])-count[i]),"constant",0))
                return torch.stack(batch_loss)

        self.loss = loss_aggregator


    def distributed_pred(self,inp,depth):
        #if args.max_batch_size == -1 or args.max_batch_size >= inp[1].shape[0]:
        if True: #args.max_batch_size == -1 or args.max_batch_size >= inp[1].shape[0]:
            feature =  self.features(inp, depth=depth)[self.feature_axis]
            pred = self.pred(feature)
            if self.args.task_is_sudoku:
                pred = pred.transpose(1,2)
            else:
                pred = pred.squeeze(-1)
            return pred
        else:
            #Pdb().set_trace()
            cat_output= []
            inp_size = inp[1].shape[0]
            num_batches = int(math.ceil((1.0*inp_size) / self.args.max_batch_size))
            for i in range(num_batches):
                start_i = self.args.max_batch_size*i
                end_i = self.args.max_batch_size*(i+1)
                temp_inp = [x if x is None else x[start_i:end_i] for x in inp]
                
                cat_output.append(self.pred(self.features(temp_inp, depth=depth)[self.feature_axis]))
            #
            return torch.cat(cat_output,dim=0).squeeze(-1)



    def forward(self, feed_dict, z_latent=None,return_loss_matrix = False):
        feed_dict = GView(feed_dict)

        # properties
        if self.args.task_is_adjacent:
            states = feed_dict.states.float()
        else:
            states = None
        
        # relations
        if self.args.task_is_sudoku:
            relations = self.relations.expand(len(feed_dict.target),81,81,3)
            states = F.one_hot(feed_dict.query).float()
        else:
            relations = feed_dict.relations.float()
            states = feed_dict.query.float()
        batch_size, nr = relations.size()[:2]
        inp = [None for _ in range(self.args.nlm_breadth + 1)]
        if z_latent is None:
            inp[0] = torch.sigmoid(self.learnable_z).unsqueeze(0).expand(batch_size,-1)
        else:
            inp[0] = torch.sigmoid(z_latent)

        inp[1] = states
        inp[2] = relations
        depth = None
        if self.args.nlm_recursion:
            depth = 1
            while 2**depth + 1 < nr:
                depth += 1
            depth = depth * 2 + 1
        
        
        pred = self.distributed_pred(inp, depth=depth)

        if self.training:
            monitors = dict()
            target = feed_dict.target
            if self.args.task_is_sudoku:
                target = target.long()
            else:
                target = target.float()
            count = None
            if self.args.cc_loss or self.args.min_loss or self.args.naive_pll_loss or  'weights' in feed_dict or return_loss_matrix:
                target = feed_dict.target_set
                if self.args.task_is_sudoku:
                    target = target.long()
                else:
                    target = target.float()
                count = feed_dict.count.int()
            if self.args.task_is_adjacent:
                target = target[:, :, :self.args.adjacent_pred_colors]
           
            this_meters,_,reward,new_targets = instance_accuracy(
                 feed_dict.target.float(), pred, return_float= False, feed_dict=feed_dict, task = self.args.task,args=self.args)
           
            #logger.info("Reward: ")
            #logger.info(reward)
            monitors.update(this_meters)
            loss_matrix = self.loss(pred, target, count)

            if self.args.min_loss or self.args.cc_loss or self.args.naive_pll_loss:
                loss = loss_matrix.mean()
            elif 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum()/feed_dict.weights.sum()
            else:
                loss  = loss_matrix

            return loss, monitors, dict(pred=pred,reward=reward,loss_matrix=loss_matrix)
        else:
            return dict(pred=pred)


class SudokuRRNNet(nn.Module):
    def __init__(self, args):
        super(SudokuRRNNet, self).__init__()
        self.args = args
        self.num_steps = args.sudoku_num_steps 
        learnable_z = torch.randn(args.nlm_nullary_dim)
        self.learnable_z = nn.Parameter(learnable_z)
        self.sudoku_solver = SudokuNN(
                 num_steps= args.sudoku_num_steps,
                 embed_size=args.sudoku_embed_size,
                 hidden_dim=args.sudoku_hidden_dim,
                 edge_drop=args.sudoku_do,
                 learnable_z = self.learnable_z)
 
        self.basic_graph = sd._basic_sudoku_graph()
        self.sudoku_indices = torch.arange(0, 81)
        if args.use_gpu:
            self.sudoku_indices = self.sudoku_indices.cuda()
        self.rows = self.sudoku_indices // 9
        self.cols = self.sudoku_indices % 9
        self.wt_base_loss = torch.nn.CrossEntropyLoss(reduction='none')
        def loss_aggregator(pred, target, target_mask):
            # if pred and target have same dimension then simply compute loss
            # pred.shape == BS x 10 x 81 x  32
            #target.shape = BS X Target size x 81
            #if pred.dim()==self.args.sudoku_num_steps*target.dim():
            #    return self.base_loss(pred, target)
            #Pdb().set_trace() 
            # if pred and target are not of same dimension then compute loss wrt each element in target set
            # return a (batchsize x targetset size) vector 
            #target.shape = batch_size x target size x num_variables
            #pred.shape: batch_size x num variables
            #Pdb().set_trace()
            batch_size, target_size, num_variables = target.size()
            num_steps = pred.size(-1)
            #target= torch.stack([target.transpose(1,2)]*num_steps,dim=-1).transpose(-1,-2) 
            if self.args.cc_loss:
                #Pdb().set_trace()
                log_pred_prob = F.log_softmax(pred,dim=1)
                epsilon = 1e-10
                #pred_prob = pred_prob.unsqueeze(-1).expand(*pred_size(),target_size)
                target= target.unsqueeze(-1).expand(*target.size(),num_steps) 
                #Pdb().set_trace()
                log_target_prob = torch.gather(log_pred_prob,dim=1,index=target.long()).sum(dim=-2)
                #multiply the probability accross cells. dim = -2. 
                #target_prob.shape  = BS x Target Size x num cells x num steps
                #target_prob = target_prob.prod(dim=-2)
                #log_target_prob = torch.log(target_prob).sum(dim=-2)
                expanded_mask= target_mask.float().unsqueeze(-1).expand_as(log_target_prob) 
                #mask_log_target_prob = log_target_prob*(target_mask.float().unsqueeze(-1).expand_as(log_target_prob))
                #add probabilities for all targets 
                #target_prob = expanded_mask*(epsilon + torch.exp(log_target_prob)) 
                #Pdb().set_trace()
                log_max_prob,max_prob_index = log_target_prob.max(dim=1)
                #log_max_prob =  log_target_prob.gather(dim=1,index=max_prob_index.unsqueeze(1)).squeeze(1)
                #log(sum) = log(sum*max/max) = log(max) + log(sum/max) = log(max) + log(sum(exp(log(p_i) - log(p_max)))
                
                log_total_prob = log_max_prob + torch.log((expanded_mask*torch.exp(log_target_prob - log_max_prob.unsqueeze(dim=1))).sum(dim=1)) 
                
                #log_total_prob = log_max_prob + torch.log((target_prob.sum(dim=1))/max_prob)
                #log_total_prob = log_target_prob[:,0,:] + torch.log(
                #                1.0 + (target_prob[:,1:,:].sum(dim=1)/target_prob[:,0,:]))
                loss_tensor = (-1.0*log_total_prob).mean(dim=-1)/num_variables
                #total_target_prob.shape = BS x Num steps
                #loss_tensor = (-1.0*torch.log(total_target_prob + epsilon )).mean(dim=-1)/num_variables
            else:
                pred = pred.unsqueeze(-1).expand(*pred.size(),target_size)
                target= target.transpose(1,2).unsqueeze(-1).expand(batch_size, num_variables, target_size,num_steps).transpose(-1,-2) 
                loss_tensor = self.wt_base_loss(pred, target.long())
                loss_tensor = loss_tensor.mean(dim=list(range(1,loss_tensor.dim()-1)))*target_mask.float()
                
                #shape = batch_size x target_size
                if self.args.min_loss:
                    #return has shape: batch_size 
                    #loss_tensor  = loss_tensor.masked_fill((1-target_mask.byte()),float('inf')).min(dim=1)[0]
                    loss_tensor  = loss_tensor.masked_fill((target_mask<1),float('inf')).min(dim=1)[0]
                elif self.args.naive_pll_loss:
                    loss_tensor = loss_tensor.sum(dim=1)/target_mask.sum(dim=1).float()

            return loss_tensor
        self.loss_func = nn.CrossEntropyLoss()
        self.loss = loss_aggregator
        self.add_to_targetset = False


    def collate_fn(self, feed_dict):
        graph_list = []
        for i in range(len(feed_dict['query'])):
            #@TODO may have to change dtype of q. keep an eye
            q = feed_dict['query'][i]
            graph = copy.deepcopy(self.basic_graph)
            graph.ndata['q'] = q  # q means question
            #graph.ndata['a'] = feed_dict['target'][i].long()
            graph.ndata['row'] = self.rows
            graph.ndata['col'] = self.cols
            graph_list.append(graph)
        batch_graph = dgl.batch(graph_list)
        return batch_graph


    def forward(self, feed_dict, z_latent=None,return_loss_matrix = False, can_break = False):
        #Pdb().set_trace()
        feed_dict = GView(feed_dict)
        #convert it to graph
        bg = self.collate_fn(feed_dict)
        #logits : of shape : args.sudoku_num_steps x batchsize*81 x 10 if training
        #logits: of shape : batch_size*81 x 10 if not training
        logits = self.sudoku_solver(bg,z_latent,self.training)
        
        
        if self.training:
            #testing
            """
            labelsa = bg.ndata['a']
            labelsb = torch.stack([labelsa]*self.num_steps, 0)
            labels = labelsb.view([-1])
            labels1 = feed_dict.target.flatten().unsqueeze(0).expand(self.num_steps,-1).flatten().long()
            gl = dgl.unbatch(bg)
            gl[0].ndata['q']
            gl[1].ndata['q']
            Pdb().set_trace()
            print((labels != labels1).sum())
            loss = self.loss_func(logits.view([-1,10]), labels)
            #
            """
            logits = logits.transpose(1,2)
            logits = logits.transpose(0,2)
        else:
            logits = logits.unsqueeze(-1)
        #shape of logits now : BS*81 x 10 x 32 if self.training ,  otherwise BS*81 x 10 x 1
        logits = logits.view(-1, 81, logits.size(-2), logits.size(-1)) 
        #shape of logits now : BS x  81 x 10 x 32(1)
        logits = logits.transpose(1,2)
        #shape of logits now : BS x  10 x 81 x 32(1)
        #pred = logits[:,:,:,-1].argmax(dim=1)
        pred = logits 

        if self.training or self.add_to_targetset:
            #Pdb().set_trace()
            this_meters,_,reward, new_targets = instance_accuracy(feed_dict.target.float(), pred, return_float= False, feed_dict=feed_dict, task = self.args.task, args=self.args)
          
            
            if self.add_to_targetset:
                #Pdb().set_trace()
                utils.add_missing_target(feed_dict,new_targets,reward)

            monitors = dict()
            target = feed_dict.target.float()
            count = None
            #Pdb().set_trace()
            loss_matrix = None
            if self.args.cc_loss or self.args.naive_pll_loss or self.args.min_loss or 'weights' in feed_dict or return_loss_matrix:
                loss_matrix = self.loss(logits, feed_dict.target_set,feed_dict.mask)
            else:
                loss_matrix = self.loss(logits, target.unsqueeze(1),feed_dict.mask[:,0].unsqueeze(-1))
            #Pdb().set_trace()
            if 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum()/feed_dict.weights.sum()

            else:
                loss = loss_matrix.mean()
        
            #loss = loss_ch
            #print(loss,loss_ch)
           
            #logger.info("Reward: ")
            #logger.info(reward)
            monitors.update(this_meters)
            #logits = logits.view([, 10])
            #labels = labels.view([-1])

            #loss_matrix of size: batch_size x target set size 
            # when in training mode return prediction for all steps
            return loss, monitors, dict(pred=pred,reward=reward,loss_matrix = loss_matrix)
        else:
            return dict(pred=pred)


