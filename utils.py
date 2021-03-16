import torch
import pickle
import numpy as np

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from collections import OrderedDict
import copy
from sys import argv
import json
import pandas as pd
import argparse
from tqdm.auto import tqdm
from IPython.core.debugger import Pdb


def read_summary_table(exp_dir):
    if not os.path.exists(os.path.join(exp_dir, "summary.json")):
        print(exp_dir, "missing summary.json file!")
        return None
    with open(os.path.join(exp_dir, "summary.json")) as f:
        l = f.readlines()
        l = [json.loads(x) for x in l]
        res = pd.DataFrame(l)
        res.loc[res['epoch'] == -1, 'epoch'] = None
        res.loc[:, ['epoch']] = res.loc[:, ['epoch']].ffill()
        return res


def match_query(query, pred):
    mask = (query>0)
    return torch.equal(query[mask], pred[mask])

def check_validity(grid, constraints=None):
    grid = grid.cpu().numpy()
    constraints = constraints.cpu().numpy()
    grid = grid.argmax(axis=2)
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


def is_safe_futoshiki(grid,constraints):
    size = int(len(grid)**0.3334)
    grid = grid.reshape(size,size,size).float()
    gold = torch.ones(size,size)
    if torch.sum(torch.abs(grid.sum(dim=0)-gold))>0:
        return False
    if torch.sum(torch.abs(grid.sum(dim=1)-gold))>0:
        return False
    if torch.sum(torch.abs(grid.sum(dim=2)-gold))>0:
        return False
     
    constraints = constraints.transpose(0,1)
    constraints = constraints.reshape(2,size,size,size)
    constraints = constraints[:,:,:,0]
    return check_validity(grid,constraints)


def is_safe_sudoku(x,n=9):
    grid = x.cpu().numpy().astype(int)
    grid = grid.reshape(n,n)
    b_size = int(np.sqrt(n))
    for i in range(n):
        if len(set(grid[i]))<n:
            return False 
        if len(set(grid[:,i]))<n:
            return False 
        #
        b_row = i//b_size 
        b_col = i%b_size 
        if len(set(grid[b_size*b_row:b_size*(b_row+1),b_size*b_col:b_size*(b_col+1)].flatten()))<n:
            return False 
    return True

def is_safe_nqueens(grid):
    size = int(len(grid)**0.5)

    grid = grid.reshape(size, size)
    indices = torch.nonzero(grid)
    if len(indices) != size:
        return False
    for x in range(size):
        r1, c1 = indices[x]
        for y in range(x+1, size):
            r2, c2 = indices[y]
            if (r1 == r2) or (c1 == c2) or (torch.abs(r1-r2) == torch.abs(c1-c2)):
                return False
    return True 


compare_func_futo = lambda x,query: match_query(query[:,0].float(),x) and is_safe_futoshiki(x,query[:,1:])
compare_func_nqueens = lambda x,query: match_query(query[:,0].float(),x) and is_safe_nqueens(x)
compare_func_tower = lambda x,query: is_safe_towers(query[:,0].float(),x, args.test_number_begin if return_float else args.train_number) 
compare_func_sudoku = lambda x,query: match_query(query,x) and is_safe_sudoku(x, args.test_number_begin if return_float else args.train_number)


