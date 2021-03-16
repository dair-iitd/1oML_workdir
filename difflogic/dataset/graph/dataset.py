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
"""Implement datasets classes for graph and family tree tasks."""

import numpy as np
import itertools
from torch.utils.data.dataset import Dataset
#from torchvision import datasets
import torch
import jacinle.random as random
import pickle

from .family import randomly_generate_family
from ...envs.graph import get_random_graph_generator
import math
from .nqueens import NQueenSolution  
from IPython.core.debugger import Pdb
import copy
from jacinle.logging import get_logger, set_output_file

from torch.distributions.categorical import Categorical

TRAIN = 0
DEV = 1
TEST = 2

logger = get_logger(__file__)

__all__ = [
        'GraphOutDegreeDataset', 'GraphConnectivityDataset', 'GraphAdjacentDataset',
        'FamilyTreeDataset','NQueensDataset', 'FutoshikiDataset','TowerDataset','SudokuDataset'
]


class GraphDatasetBase(Dataset):
    """Base dataset class for graphs.

    Args:
        epoch_size: The number of batches for each epoch.
        nmin: The minimal number of nodes in the graph.
        pmin: The lower bound of the parameter p of the graph generator.
        nmax: The maximal number of nodes in the graph,
                the same as $nmin in default.
        pmax: The upper bound of the parameter p of the graph generator,
                the same as $pmin in default.
        directed: Generator directed graph if directed=True.
        gen_method: Controlling the graph generation method.
                If gen_method='dnc', use the similar way as in DNC paper.
                Else using Erdos-Renyi algorithm (each edge exists with prob).
    """

    def __init__(self,
                             epoch_size,
                             nmin,
                             pmin,
                             nmax=None,
                             pmax=None,
                             directed=False,
                             gen_method='dnc'):
        self._epoch_size = epoch_size
        self._nmin = nmin
        self._nmax = nmin if nmax is None else nmax
        assert self._nmin <= self._nmax
        self._pmin = pmin
        self._pmax = pmin if pmax is None else pmax
        assert self._pmin <= self._pmax
        self._directed = directed
        self._gen_method = gen_method

    def _gen_graph(self, item):
        n = self._nmin + item % (self._nmax - self._nmin + 1)
        p = self._pmin + random.rand() * (self._pmax - self._pmin)
        gen = get_random_graph_generator(self._gen_method)
        return gen(n, p, directed=self._directed)

    def __len__(self):
        return self._epoch_size


class GraphOutDegreeDataset(GraphDatasetBase):
    """The dataset for out-degree task in graphs."""

    def __init__(self,
                             degree,
                             epoch_size,
                             nmin,
                             pmin,
                             nmax=None,
                             pmax=None,
                             directed=False,
                             gen_method='dnc'):
        super().__init__(epoch_size, nmin, pmin, nmax, pmax, directed, gen_method)
        self._degree = degree

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        # The goal is to predict whether out-degree(x) == self._degree for all x.
        return dict(
                n=graph.nr_nodes,
                relations=np.expand_dims(graph.get_edges(), axis=-1),
                target=(graph.get_out_degree() == self._degree).astype('float'),
        )


class GraphConnectivityDataset(GraphDatasetBase):
    """The dataset for connectivity task in graphs."""

    def __init__(self,
                             dist_limit,
                             epoch_size,
                             nmin,
                             pmin,
                             nmax=None,
                             pmax=None,
                             directed=False,
                             gen_method='dnc'):
        super().__init__(epoch_size, nmin, pmin, nmax, pmax, directed, gen_method)
        self._dist_limit = dist_limit

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        # The goal is to predict whether (x, y) are connected within a limited steps
        # I.e. dist(x, y) <= self._dist_limit for all x, y.
        return dict(
                n=graph.nr_nodes,
                relations=np.expand_dims(graph.get_edges(), axis=-1),
                target=graph.get_connectivity(self._dist_limit, exclude_self=True),
        )


class GraphAdjacentDataset(GraphDatasetBase):
    """The dataset for adjacent task in graphs."""

    def __init__(self,
                             nr_colors,
                             epoch_size,
                             nmin,
                             pmin,
                             nmax=None,
                             pmax=None,
                             directed=False,
                             gen_method='dnc',
                             is_train=True,
                             is_mnist_colors=False,
                             mnist_dir='../data'):

        super().__init__(epoch_size, nmin, pmin, nmax, pmax, directed, gen_method)
        self._nr_colors = nr_colors
        self._is_mnist_colors = is_mnist_colors
        # When taking MNIST digits as inputs, fetch MNIST dataset.
        if self._is_mnist_colors:
            assert nr_colors == 10
            self.mnist = datasets.MNIST(
                    mnist_dir, train=is_train, download=True, transform=None)

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        n = graph.nr_nodes
        if self._is_mnist_colors:
            m = self.mnist.__len__()
            digits = []
            colors = []
            for i in range(n):
                x = random.randint(m)
                digit, color = self.mnist.__getitem__(x)
                digits.append(np.array(digit)[np.newaxis])
                colors.append(color)
            digits, colors = np.array(digits), np.array(colors)
        else:
            colors = random.randint(self._nr_colors, size=n)
        states = np.zeros((n, self._nr_colors))
        adjacent = np.zeros((n, self._nr_colors))
        # The goal is to predict whether there is a node with desired color
        # as adjacent node for each node x.
        for i in range(n):
            states[i, colors[i]] = 1
            adjacent[i, colors[i]] = 1
            for j in range(n):
                if graph.has_edge(i, j):
                    adjacent[i, colors[j]] = 1
        if self._is_mnist_colors:
            states = digits
        return dict(
                n=n,
                relations=np.expand_dims(graph.get_edges(), axis=-1),
                states=states,
                colors=colors,
                target=adjacent,
        )


class FamilyTreeDataset(Dataset):
    """The dataset for family tree tasks."""

    def __init__(self,
                             task,
                             epoch_size,
                             nmin,
                             nmax=None,
                             p_marriage=0.8,
                             balance_sample=False):
        super().__init__()
        self._task = task
        self._epoch_size = epoch_size
        self._nmin = nmin
        self._nmax = nmin if nmax is None else nmax
        assert self._nmin <= self._nmax
        self._p_marriage = p_marriage
        self._balance_sample = balance_sample
        self._data = []

    def _gen_family(self, item):
        n = self._nmin + item % (self._nmax - self._nmin + 1)
        return randomly_generate_family(n, self._p_marriage)

    def __getitem__(self, item):
        #Pdb().set_trace()
        while len(self._data) == 0:
            family = self._gen_family(item)
            relations = family.relations[:, :, 2:]
            if self._task == 'has-father':
                target = family.has_father()
            elif self._task == 'has-daughter':
                target = family.has_daughter()
            elif self._task == 'has-sister':
                target = family.has_sister()
            elif self._task == 'parents':
                target = family.get_parents()
            elif self._task == 'grandparents':
                target = family.get_grandparents()
            elif self._task == 'uncle':
                target = family.get_uncle()
            elif self._task == 'maternal-great-uncle':
                target = family.get_maternal_great_uncle()
            else:
                assert False, '{} is not supported.'.format(self._task)

            if not self._balance_sample:
                return dict(n=family.nr_people, relations=relations, target=target)

            # In balance_sample case, the data format is different. Not used.
            def get_positions(x):
                return list(np.vstack(np.where(x)).T)

            def append_data(pos, target):
                states = np.zeros((family.nr_people, 2))
                states[pos[0], 0] = states[pos[1], 1] = 1
                self._data.append(dict(n=family.nr_people,
                                                             relations=relations,
                                                             states=states,
                                                             target=target))

            positive = get_positions(target == 1)
            if len(positive) == 0:
                continue
            negative = get_positions(target == 0)
            np.random.shuffle(negative)
            negative = negative[:len(positive)]
            for i in positive:
                append_data(i, 1)
            for i in negative:
                append_data(i, 0)

        return self._data.pop()

    def __len__(self):
        return self._epoch_size


class NQueensDataset(Dataset):
    """The dataset for nqueens tasks."""
    def __init__(self,
                             epoch_size,
                             n=10,
                             num_missing = 1,
                             random_seed = 42,
                             min_loss = False,
                             arbit_solution = False,
                             train_dev_test = TRAIN,
                             data_file = None,
                             data_sampling='rs'):
        super().__init__()

        self._epoch_size = epoch_size
        self._n = n
        self.num_missing = num_missing
        self.min_loss = min_loss
        self.arbit_solution = arbit_solution 
        self.mode = train_dev_test
        self.data_sampling = data_sampling 

        self.nqueen_solver = NQueenSolution() 
        self.relations  = self.nqueen_solver.get_relations(n)
        print("In constructor. Size: {}".format(n))
        if data_file is None:
            outfile = "data/nqueens_data_"+str(self._n)+"_"+str(self.num_missing)+".pkl"
        else:
            outfile = data_file 
        #
        with open(outfile,"rb") as f:
            self.dataset = pickle.load(f)
        
        self.max_count = 0
        self.unique_indices = []
        self.ambiguous_indices = []
        for i,data in enumerate(self.dataset):
            self.max_count = max(self.max_count, data["count"])
            if data["count"]==1:
                self.unique_indices.append(i)
            else:
                self.ambiguous_indices.append(i)

        np.random.seed(random_seed)
        self.reset_sampler(data_sampling) 
    
    
    def reset_sampler(self,data_sampling):
        self.data_sampling = data_sampling
        if data_sampling == 'rsxy':
            logger.info("Sampling uniformly from (x,y) tuples")
            self.sampler = Categorical(probs = torch.tensor([x['count'] for x in self.dataset]).float())
        else:
            self.sampler = Categorical(probs = torch.tensor([1.0 for _ in self.dataset]).float())


    def pad_set(self,target_set):
        pad_counter = self.max_count - len(target_set)
        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def sample_imbalance(self, imbalance_ratio):
        if np.random.rand()<imbalance_ratio:
            ind  =  np.random.choice(self.ambiguous_indices)
        else:
            ind  =  np.random.choice(self.unique_indices)
        return ind

        
    def __getitem__(self, item):
        #ind = np.random.randint(0,len(self.dataset))
        ind = self.sampler.sample().item()
        if self.mode==TRAIN:
            if self.data_sampling=="unique":
                ind = self.sample_imbalance(0)
            elif self.data_sampling=="ambiguous":
                ind = self.sample_imbalance(1)
            elif self.data_sampling=="one-one":
                ind = self.sample_imbalance(0.5)
            elif self.data_sampling=="two-one":
                ind = self.sample_imbalance(0.33)
            elif self.data_sampling=="three-one":
                ind = self.sample_imbalance(0.25)
            elif self.data_sampling=="four-one":
                ind = self.sample_imbalance(0.20)
        else:
            ind = item%len(self.dataset)

        data = self.dataset[ind]
        
        if len(data["query"].shape)==1:    
            data["query"] = np.expand_dims(data["query"],1)
        if self.mode==TRAIN and self.arbit_solution:
            data["target"] = data["target_set"][0]
        else:
            data["target"] = data["target_set"][np.random.randint(len(data["target_set"]))]
        # 
        data["target_set"] = self.pad_set(data["target_set"])
        data['mask'] = np.array([1 for _ in range(data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])]) 
        #Pdb().set_trace() 
        data["relations"] = self.relations
        data['ind'] = ind
        if isinstance(data["qid"],tuple):
            data["qid"] = np.array([data["qid"][0]]+list(data["qid"][1]))
        return data

    def __len__(self):
        if self.mode==TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)


class FutoshikiDataset(Dataset):
    """The dataset for futoshiki tasks."""
    def __init__(self,
                             epoch_size,
                             n=10,
                             num_missing = 1,
                             num_constraints = 0,
                             data_size = -1,
                             random_seed = 42,
                             min_loss = False,
                             arbit_solution = False,
                             train_dev_test = TRAIN,
                             data_file = None,
                             data_sampling='rs',args=None):
        super().__init__()
        self.args = args
        self._epoch_size = epoch_size
        self._n = n
        self.num_missing = num_missing
        self.min_loss = min_loss
        self.arbit_solution = arbit_solution 
        self.mode = train_dev_test
        self.data_sampling = data_sampling 

        self.relations  = self.get_relations()
        print("In constructor. Size: {}".format(n))
        if train_dev_test == TRAIN:
            mode = 'train'
        elif train_dev_test ==  DEV:
            mode = 'val'
        elif train_dev_test == TEST:
            mode = 'test'

        if data_file is None:
            outfile = "data/futo_{}_{}_{}_{}.pkl".format(self._n, num_missing, num_constraints, mode)
        else:
            outfile = data_file 
        #
        logger.info("data file : {}".format(outfile))
        #Pdb().set_trace()
        with open(outfile,"rb") as f:
            self.dataset = pickle.load(f)
        
        if data_size != -1:
            self.dataset= self.dataset[:data_size]
        #
        self.max_count = 0
        self.unique_indices = []
        self.ambiguous_indices = []
        for i,data in enumerate(self.dataset):
            if 'count'  in data:
                this_count = data['count']
            else:
                this_count = data['target_set'].shape[0]
                data['count'] = this_count
            self.max_count = max(self.max_count, this_count)
            if this_count == 1:
                self.unique_indices.append(i)
            else:
                self.ambiguous_indices.append(i)
        np.random.seed(random_seed)
        self.reset_sampler(data_sampling)

    def reset_sampler(self,data_sampling):
        self.data_sampling = data_sampling
        if data_sampling == 'rsxy':
            logger.info("Sampling uniformly from (x,y) tuples")
            self.sampler = Categorical(probs = torch.tensor([x['count'] for x in self.dataset]).float())
        else:
            self.sampler = Categorical(probs = torch.tensor([1.0 for _ in self.dataset]).float())


    def get_relations(self):
        n = self._n
        n2 = self._n**2
        n3 = self._n**3
        relations = np.zeros((n3, n3,3))
     
        for x in range(n3):
            row = int(x/n2)
            col = int((x%n2)/n)
            num = int(x%n2)%n
         
            for y in range(n):
                # cell constraints
                relations[x][row*n2+col*n+y][0]=1
             
                # row constraints
                relations[x][y*n2+col*n+num][1]=1
             
                # column constraints
                relations[x][row*n2+y*n+num][2]=1
        return relations
    
    def pad_set(self,target_set):
        pad_counter = self.max_count - len(target_set)
        if pad_counter < 0:
            return target_set[:self.max_count]

        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def sample_imbalance(self, imbalance_ratio):
        if np.random.rand()<imbalance_ratio:
            ind  =  np.random.choice(self.ambiguous_indices)
        else:
            ind  =  np.random.choice(self.unique_indices)
        return ind

        
    def __getitem__(self, item):
        #Pdb().set_trace()
        #ind = np.random.randint(0,len(self.dataset))
        ind = self.sampler.sample().item()
        #print(ind)
        if self.mode==TRAIN:
            if self.data_sampling=="unique":
                ind = self.sample_imbalance(0)
            elif self.data_sampling=="ambiguous":
                ind = self.sample_imbalance(1)
            elif self.data_sampling=="one-one":
                ind = self.sample_imbalance(0.5)
            elif self.data_sampling=="two-one":
                ind = self.sample_imbalance(0.33)
            elif self.data_sampling=="three-one":
                ind = self.sample_imbalance(0.25)
            elif self.data_sampling=="four-one":
                ind = self.sample_imbalance(0.20)
        else:
            ind = item%len(self.dataset)

        data = self.dataset[ind]
        
        if self.mode==TRAIN and self.arbit_solution:
            data["target"] = data["target_set"][0]
        else:
            data["target"] = data["target_set"][np.random.randint(data['count'])]
        
        data["target_set"] = self.pad_set(data["target_set"])
        
        data['n'] = self._n
        data['is_ambiguous'] =  int(data['count'] > 1)
        data['qid'] = np.array([ind])
        data['ind'] = ind
        data['mask'] = np.array([1 for _ in range(data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])]) 
       
        if self.args.model != 'satnet' or self.args.latent_model == 'nlm': 
            data["relations"] = self.relations
        if self.args.model == 'satnet':
            data['gtlt'] = np.concatenate((data['query'][::self._n,1], data['query'][::self._n,2]),axis=0)                                                                           
        return data

    def __len__(self):
        if self.mode==TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)

class SudokuDataset(Dataset):
    """The dataset for sudoku tasks."""
    def __init__(self,
                             epoch_size,
                             data_size = -1,
                             arbit_solution = False,
                             train_dev_test = TRAIN,
                             data_file = None,
                             data_sampling='rs',args=None):
        super().__init__()
        self.args = args
        self._epoch_size = epoch_size
        self.arbit_solution = arbit_solution 
        self.mode = train_dev_test
        self.data_sampling = data_sampling 
        self._n = 81
        print("In constructor.  {}".format(args.task))
        if train_dev_test == TRAIN:
            mode = 'train'
        elif train_dev_test ==  DEV:
            mode = 'val'
        elif train_dev_test == TEST:
            mode = 'test'

        outfile = data_file 
        #
        logger.info("data file : {}".format(outfile))
        #Pdb().set_trace()
        with open(outfile,"rb") as f:
            self.dataset = pickle.load(f)
        
        if data_size != -1:
            self.dataset= self.dataset[:data_size]
        #
        np.random.seed(args.seed)
        self.max_count = args.solution_count
        self.unique_indices = []
        self.ambiguous_indices = []
        for i,data in enumerate(self.dataset):
            data['query'] = (data['query']).astype(int)
            if len(data["target_set"])>self.max_count:
                self.dataset[i]["target_set"] = data["target_set"][:self.max_count]
                self.dataset[i]["count"]=self.max_count
            if 'count'  in data:
                this_count = data['count']
            else:
                this_count = data['target_set'].shape[0]
                self.dataset[i]['count'] = this_count
            if this_count == 1:
                self.unique_indices.append(i)
            else:
                self.ambiguous_indices.append(i)
        self.max_count += 1
        self.reset_sampler(data_sampling) 


    def reset_sampler(self,data_sampling):
        self.data_sampling = data_sampling
        if data_sampling == 'rsxy':
            logger.info("Sampling uniformly from (x,y) tuples")
            self.sampler = Categorical(probs = torch.tensor([x['count'] for x in self.dataset]).float())
        else:
            self.sampler = Categorical(probs = torch.tensor([1.0 for _ in self.dataset]).float())


    def pad_set(self,target_set):
        pad_counter = self.max_count - len(target_set)
        if pad_counter < 0:
            return target_set[:self.max_count]

        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def sample_imbalance(self, imbalance_ratio):
        if np.random.rand()<imbalance_ratio:
            ind  =  np.random.choice(self.ambiguous_indices)
        else:
            ind  =  np.random.choice(self.unique_indices)
        return ind

        
    def __getitem__(self, item):
        #Pdb().set_trace()
        #ind = np.random.randint(0,len(self.dataset))
        ind = self.sampler.sample().item()
        #print(ind)
        if self.mode==TRAIN:
            if self.data_sampling=="unique":
                ind = self.sample_imbalance(0)
            elif self.data_sampling=="ambiguous":
                ind = self.sample_imbalance(1)
            elif self.data_sampling=="one-one":
                ind = self.sample_imbalance(0.5)
            elif self.data_sampling=="two-one":
                ind = self.sample_imbalance(0.33)
            elif self.data_sampling=="three-one":
                ind = self.sample_imbalance(0.25)
            elif self.data_sampling=="four-one":
                ind = self.sample_imbalance(0.20)
        else:
            ind = item%len(self.dataset)

        data = self.dataset[ind]
        
        if self.mode==TRAIN and self.arbit_solution:
            data["target"] = data["target_set"][0]
        else:
            data["target"] = data["target_set"][np.random.randint(data['count'])]
        
        data["target_set"] = self.pad_set(data["target_set"])
        
        data['n'] = self._n
        data['is_ambiguous'] =  int(data['count'] > 1)
        data['qid'] = np.array([ind])
        data['ind'] = ind
        data['mask'] = np.array([1 for _ in range(data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])]) 
       
        return data

    def __len__(self):
        if self.mode==TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)

class TowerDataset(Dataset):
    """The dataset for towers tasks."""
    def __init__(self,
                 epoch_size,
                 n=3,
                 num_missing = 4,
                 random_seed = 42,
                 arbit_solution = False,
                 train_dev_test = TRAIN,
                 data_file = None,
                 data_sampling='rs'):
        
        super().__init__()

        self._epoch_size = epoch_size
        self._n = n
        self.num_missing = num_missing
        self.arbit_solution = arbit_solution 
        self.mode = train_dev_test
        self.data_sampling = data_sampling 

        self.unary_relations, self.relations  = self.get_relations(n)
        
        outfile = data_file 
        with open(outfile,"rb") as f:
            self.dataset = pickle.load(f)
        
        self.max_count = 0
        self.unique_indices = []
        self.ambiguous_indices = []
        for i,data in enumerate(self.dataset):
            data["query"] = self.vectorize_query(data["query"])
            data["target_set"] = [np.concatenate((np.zeros(4*self._n**2),self.get_one_hot(target)))  
                                                 for target in data["target_set"]]
	    #data["target_set"] = [self.get_one_hot(target) for target in data["target_set"]]
            data["count"] = len(data["target_set"])
            data["is_ambiguous"] = (data["count"]>1)
            self.max_count = max(self.max_count, data["count"])
            if data["count"]==1:
                self.unique_indices.append(i)
            else:
                self.ambiguous_indices.append(i)

        np.random.seed(random_seed)
    
    def get_one_hot(self,grid):
        grid = grid.flatten()
        expand_grid = np.zeros((grid.size, self._n+1))
        expand_grid[np.arange(grid.size),grid] = 1
        expand_grid = expand_grid[:,1:]
        expand_grid = expand_grid.flatten()
        return expand_grid

    
    def vectorize_query(self,query):
        n3 = self._n**3
        exp_query = np.concatenate((self.get_one_hot(query), np.zeros(n3)))
        return np.stack([exp_query]+self.unary_relations).T
        
    
    def get_relations(self,n):
        n2 = n**2
        n3 = n**3
        vector_dim = n3+4*n2
        
        left_tower_numbers = np.array([1]*n2+[0]*(vector_dim-n2))
        up_tower_numbers = np.array([0]*n2+[1]*n2+[0]*(vector_dim-2*n2))
        right_tower_numbers = np.array([0]*(2*n2)+[1]*n2+[0]*(vector_dim-3*n2))
        down_tower_numbers = np.array([0]*(3*n2)+[1]*n2+[0]*(vector_dim-4*n2))
        
        unary_relations = [left_tower_numbers, up_tower_numbers, right_tower_numbers, down_tower_numbers]
        
        relations = np.zeros((vector_dim, vector_dim,3))
        prefix = 4*n2
        for x in range(n3):
            row = int(x/n2)
            col = int((x%n2)/n)
            num = int(x%n2)%n
         
            for y in range(n):
                # cell constraints
                relations[prefix+x][prefix+row*n2+col*n+y][0]=1
                
                # row constraints
                relations[prefix+x][prefix+y*n2+col*n+num][1]=1
             
                # column constraints
                relations[prefix+x][prefix+row*n2+y*n+num][2]=1
            
            for y in range(n):
                relations[prefix+x][row*n+y][1]=1
                relations[row*n+y][prefix+x][1]=1
                
                relations[prefix+x][2*n2+row*n+y][1]=1
                relations[2*n2+row*n+y][prefix+x][1]=1
                
                relations[prefix+x][n2+col*n+y][2]=1
                relations[n2+col*n+y][prefix+x][2]=1
                
                relations[prefix+x][3*n2+col*n+y][2]=1
                relations[3*n2+col*n+y][prefix+x][2]=1
        
        for x in range(n2):
            row = int(x/n)
            cell = int(x%n)
            
            for y in range(n):
                relations[x][row*n+y][0]=1
                relations[n2+x][n2+row*n+y][0]=1
                relations[2*n2+x][2*n2+row*n+y][0]=1
                relations[3*n2+x][3*n2+row*n+y][0]=1
        return unary_relations,relations
    
    def pad_set(self,target_set):
        pad_counter = self.max_count - len(target_set)
        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def sample_imbalance(self, imbalance_ratio):
        if np.random.rand()<imbalance_ratio:
            ind  =  np.random.choice(self.ambiguous_indices)
        else:
            ind  =  np.random.choice(self.unique_indices)
        return ind

        
    def __getitem__(self, item):
        ind = np.random.randint(0,len(self.dataset))
        if self.mode==TRAIN:
            if self.data_sampling=="unique":
                ind = self.sample_imbalance(0)
            elif self.data_sampling=="ambiguous":
                ind = self.sample_imbalance(1)
            elif self.data_sampling=="one-one":
                ind = self.sample_imbalance(0.5)
            elif self.data_sampling=="two-one":
                ind = self.sample_imbalance(0.33)
            elif self.data_sampling=="three-one":
                ind = self.sample_imbalance(0.25)
            elif self.data_sampling=="four-one":
                ind = self.sample_imbalance(0.20)
        else:
            ind = item%len(self.dataset)

        data = self.dataset[ind]
        
        if self.mode==TRAIN and self.arbit_solution:
            data["target"] = data["target_set"][0]
        else:
            data["target"] = data["target_set"][np.random.randint(len(data["target_set"]))]
       # 
        data['n']=self._n
        data['qid'] = np.array([ind])
        data["target_set"] = self.pad_set(data["target_set"])
        data["mask"] = np.array([1 for _ in range(data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])]) 
        data["relations"] = self.relations
        
        return data

    def __len__(self):
        if self.mode==TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)



class FutoshikiDatasetDynamic(Dataset):
    """The dataset for Futoshiki tasks."""
    def __init__(self,
                 epoch_size,
                 n=5,
                 num_missing = 1,
                 num_constraints = 0,
                 random_seed = 42,
                 min_loss = False,
                 train_dev_test = TRAIN,
                 data_file = None,
                 data_sampling='rs'):
     
        super().__init__()

        self._epoch_size = epoch_size
        self._n = n
        self.num_missing = num_missing
        self.num_constraints = num_constraints
        self.min_loss = min_loss
        self.mode = train_dev_test
        self.data_sampling = data_sampling
        self.relations = self.get_relation()
        self.max_count = 2*num_missing
        if data_sampling=="unique":
            self.max_count=1

        outfile = data_file
     
        with open(outfile,"rb") as f:
            self.dataset = pickle.load(f)
        if train_dev_test!=TRAIN:
            self._epoch_size = len(self.dataset)
         
        np.random.seed(random_seed)
    
    def check_validity(self,grid, constraints=None):
        for x in range(len(grid)):
            row = set(grid[x])
            if len(row)!=len(grid):
                return False
            col = set(grid[:,x])
            if len(col)!=len(grid):
                return False
        if constraints is None:
            return True
        gt = zip(*np.nonzero(constraints[0]))
        for ind in gt:
            next_ind = (ind[0],ind[1]+1)
            if grid[next_ind]>grid[ind]:
                return False
        lt = zip(*np.nonzero(constraints[1]))
        for ind in lt:
            next_ind = (ind[0],ind[1]+1)
            if grid[next_ind]<grid[ind]:
                return False
        return True
    
    def get_relation(self):
        n = self._n
        n2 = self._n**2
        n3 = self._n**3
        relations = np.zeros((n3, n3,3))
     
        for x in range(n3):
            row = int(x/n2)
            col = int((x%n2)/n)
            num = int(x%n2)%n
         
            for y in range(n):
                # cell constraints
                relations[x][row*n2+col*n+y][0]=1
             
                # row constraints
                relations[x][y*n2+col*n+num][1]=1
             
                # column constraints
                relations[x][row*n2+y*n+num][2]=1
        return relations
    
    def get_one_hot(self,grid):
        grid = grid.flatten()
        expand_grid = np.zeros((grid.size, self._n+1))
        expand_grid[np.arange(grid.size),grid] = 1
        expand_grid = expand_grid[:,1:]
        expand_grid = expand_grid.flatten()
        return expand_grid

    def find_solutions(self,query,zero_ind, constraints):
        size = self._n
        query_tight = query.reshape(size,size)
        full_set = set(range(size+1))

        fill_sets = []
        for ind in zero_ind:
            row_set = set(query_tight[int(ind/size)])
            col_set = set(query_tight[:,ind%size])
            fill_sets.append(list(full_set.difference(row_set.union(col_set))))

        solutions = []
        for sol in itertools.product(*fill_sets):
            solution = query_tight.flatten()
            solution[zero_ind] = sol
            solution = solution.reshape(size,size)
            if self.check_validity(solution, constraints):
#                 solutions.append(solution)
                solutions.append(self.get_one_hot(solution))
        return solutions
    
    def pad_set(self,target_set):
        target_set = target_set[:self.max_count]
        pad_counter = self.max_count - len(target_set)
        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)
    
    def get_constraints(self,grid):
        offset_grid = np.roll(grid,-1,axis=1)
        gt = grid>offset_grid
        gt[:,-1]=False
        lt = grid<offset_grid
        lt[:,-1]=False

        c = list(zip(*gt.nonzero()))
        idx = np.random.choice(range(len(c)),self.num_constraints,replace=True)
        gt_constraints = np.zeros_like(gt)
        for i in idx:
            gt_constraints[c[i]]=1

        c = list(zip(*lt.nonzero()))
        idx = np.random.choice(range(len(c)),self.num_constraints,replace=True)
        lt_constraints = np.zeros_like(lt)
        for i in idx:
            lt_constraints[c[i]]=1

        return np.stack([gt_constraints,lt_constraints])
        
    
    def generate_data(self, unique=False, ambiguous=False):
        for _ in range(100):
            ind = np.random.choice(range(len(self.dataset)))
            grid = self.dataset[ind]
            expanded_grid = self.get_one_hot(grid)
            board_dim = self._n
            
            constraints = self.get_constraints(grid)

            query = grid.flatten()
            zero_ind = np.random.choice(range(len(query)),self.num_missing,replace=False)
            mask = np.ones_like(query)
            mask[zero_ind]=0
            query = query*mask
            target_set = self.find_solutions(query,zero_ind,constraints)
            count = len(target_set)
            is_ambiguous = 1 if count>1 else 0
            if is_ambiguous and unique:
                continue
            if is_ambiguous==0 and ambiguous:
                continue
            #qid = np.array([ind]+list(zero_ind))
            qid = np.array([ind])
            #target = target_set[np.random.randint(len(target_set))]
            target = expanded_grid
            np.random.shuffle(target_set)
            target_set = self.pad_set(target_set)
            count = min(count,len(target_set))
            query = self.get_one_hot(query) 
#             return dict(n=board_dim, query=query.reshape(board_dim, board_dim), constraints=constraints, target_set = target_set, count=count, is_ambiguous=int(is_ambiguous),qid=qid)
            gt_constraints = constraints[0].flatten().repeat(self._n)
            lt_constraints = constraints[1].flatten().repeat(self._n)
            query = np.stack([query,gt_constraints,lt_constraints]).transpose()
            return dict(n=board_dim, query=query, target=target, target_set=target_set, count=count, is_ambiguous=int(is_ambiguous), qid=qid, relations=self.relations)

        raise
    
    def sample_imbalance(self,imbalance_ratio):
        if np.random.rand()<imbalance_ratio:
            return self.generate_data(ambiguous=True)
        else:
            return self.generate_data(unique=True)
    
    def __getitem__(self, item):
        if self.mode==TRAIN:
            if self.data_sampling=="unique":
                data = self.sample_imbalance(0)
            elif self.data_sampling=="ambiguous":
                data = self.sample_imbalance(1)
            elif self.data_sampling=="one-one":
                data = self.sample_imbalance(0.5)
            elif self.data_sampling=="two-one":
                data = self.sample_imbalance(0.33)
            elif self.data_sampling=="three-one":
                data = self.sample_imbalance(0.25)
            elif self.data_sampling=="four-one":
                data = self.sample_imbalance(0.20)
            elif self.data_sampling=="rs":
                data = self.generate_data()
        else:
            data = self.dataset[item]
            data['target_set'] = 1
        return data
    
    def __len__(self):
        return self._epoch_size 

class NQueensDatasetDynamic(Dataset):
    """The dataset for nqueens tasks."""
    def __init__(self,
                             epoch_size,
                             n=10,
                             num_missing = 1,
                             unique_solution = False,
                             random_seed = 42,
                             min_loss = False,
                             arbit_solution = False,
                             train_dev_test = TRAIN,
                             balance_sample=False):
        super().__init__()
        self._epoch_size = epoch_size
        self._n = n
        self._balance_sample = balance_sample
        self._data = []
        self.nqueen_solver = NQueenSolution() 
        self.nqueen_solver.solve(n)
        np.random.seed(random_seed)
        np.random.shuffle(self.nqueen_solver.solutions)
        #now use first 60% solutions as train, next 20% as dev and last 20% as test
        self.mode = train_dev_test

        if train_dev_test == TRAIN:
            start_index = 0
            end_index = int(0.8*len(self.nqueen_solver.solutions))
        elif train_dev_test == DEV:
            start_index =  int(0.6*len(self.nqueen_solver.solutions))
            end_index = int(0.8*len(self.nqueen_solver.solutions))
        elif train_dev_test == TEST:
            start_index =  int(0.8*len(self.nqueen_solver.solutions))
            end_index = len(self.nqueen_solver.solutions)
      
        #Pdb().set_trace()
        #self.complementary_solutions = copy.deepcopy(self.nqueen_solver.solutions[0:start_index] + self.nqueen_solver.solutions[end_index:])
        #self.nqueen_solver.solutions = self.nqueen_solver.solutions[start_index:end_index]
        
        self.num_missing = num_missing
        self.unique_solution = unique_solution 
        self.min_loss = min_loss
        self.arbit_solution = arbit_solution 
        _  = self.nqueen_solver.get_relations(n)

    def match_complementary_solutions(self,x):
        for grid,_ in self.complementary_solutions:
            solution = grid.flatten()
            if np.sum(np.abs(solution-x))==self.num_missing:
                return True 
        #
        return False

    def get_arbit_solution(self,x):
        for grid,_ in self.nqueen_solver.solutions:
            solution = grid.flatten()
            if np.sum(np.abs(solution-x))==self.num_missing:
                return torch.tensor(solution) 
        #
        raise 
 

    def is_ambiguous(self,x):
        flag = False 
        for grid,_ in self.nqueen_solver.solutions:
            solution = grid.flatten()
            if np.sum(np.abs(solution-x))==self.num_missing:
                if not flag:
                    flag = True
                else:
                    return True 
        #
        return False


    def match_solutions(self,x):
        count=0
        possible_solutions = []
        for grid,_ in self.nqueen_solver.solutions:
            solution = grid.flatten()
            if np.sum(np.abs(solution-x))==self.num_missing:
                count+=1
                possible_solutions.append(solution)
        for x in range(count,5*self.num_missing):
            possible_solutions.append(possible_solutions[-1])

        possible_solutions = possible_solutions[:5*self.num_missing]
        return count, torch.tensor(possible_solutions)

    def __getitem__(self, item):
        for _ in range(100):
            ind = np.random.randint(0,len(self.nqueen_solver.solutions))
            solution,n = self.nqueen_solver.solutions[ind]
            solution = solution.flatten()
            relations = self.nqueen_solver.get_relations(n)
            query = copy.deepcopy(solution)
            positions = np.where(query == 1)[0]
            mask_at = sorted(np.random.choice(positions,size=self.num_missing,replace=False))
            mask_ind = 0
            for i in mask_at:
                mask_ind = mask_ind*self._n + i
            query[mask_at] = 0
            #if self.match_complementary_solutions(query):
            #    continue
            #
            if self.mode==TRAIN and self.unique_solution and self.match_solutions(query)[0]>1:
                #print("Multiple Solution Found")
                continue
            
            if self.mode==TRAIN and self.arbit_solution:
                solution = self.get_arbit_solution(query)

            ret = dict(n=n*n, relations=relations, query = np.expand_dims(query, axis=-1), target=solution, qid=(ind,mask_ind), is_ambiguous = int(self.is_ambiguous(query))) 
            solution_set = None
            if self.mode==TRAIN and self.min_loss:
                count, solution_set = self.match_solutions(query)
                ret = dict(n=n*n, relations=relations, query = np.expand_dims(query, axis=-1), target=solution, target_set=solution_set,count=count, qid=(ind,mask_ind))
            if not self._balance_sample:
                return ret
            raise
        raise 


    def __len__(self):
        return self._epoch_size

