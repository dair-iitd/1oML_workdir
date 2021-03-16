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


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    help='parent directory containing experiment logs')

def is_safe_sudoku(x,query,n):
    mask = (query>0)
    if not torch.equal(x[mask],query[mask]):
        return False
     
    grid = x.detach().cpu().numpy().astype(int)
    grid = grid.reshape(n,n)

    b_size = int(np.sqrt(n))

    for i in range(n):
        if len(set(grid[i]))<n:
            return False
        if len(set(grid[:,i]))<n:
            return False

        b_row = i//b_size
        b_col = i%b_size

        if len(set(grid[b_size*b_row:b_size*(b_row+1),b_size*b_col:b_size*(b_col+1)].flatten()))<n:
            return False
    return True

def extract_data_from_dump(dump):
    query = []
    target_set = []
    count = []
    raw_pred = []
    
    for z in dump:
        q,t,c,r = z["feed_dict"]["query"], z["feed_dict"]["target_set"], z["feed_dict"]["count"], z["output_dict"]["pred"]
        query.append(q)
        target_set.append(t)
        count.append(c)
        raw_pred.append(r)
    return torch.cat(query).int(),torch.cat(target_set).int(),torch.cat(count).int(),torch.cat(raw_pred).float()

        
def in_out_target(dir):
    with open(os.path.join(dir,"pred_dump.pkl"),"rb") as f:
        z = pickle.load(f)

    q,ts,c,rpred = extract_data_from_dump(z)

    pred = rpred.argmax(dim=1)

    intarget_count = 0
    outtarget_count = 0
    for i in range(len(q)):
        if is_safe_sudoku(pred[i].int(),q[i],9):
            if ((ts[i]==pred[i].int()).sum(dim=1)==81).sum()>0:
                intarget_count+=1
            else:
                outtarget_count+=1
    return {"correct_in_targetset":intarget_count, "correct_out_target_set":outtarget_count}


def get_params(exp_dir):
    all_params = exp_dir.split('_')
    all_params = dict([(x.split('-')[0], '.'.join(x.split('-')[1:]))
                       for x in all_params])
    return all_params


def collate(parent_dir):
    exp_dirs = os.listdir(parent_dir)

    df = []
    for exp_dir in tqdm(exp_dirs):
        dir_name = os.path.join(parent_dir, exp_dir)
        exp_dict = in_out_target(dir_name)
        exp_dict["config"]=exp_dir
        exp_dict.update(get_params(exp_dir))
        df.append(exp_dict)
    return pd.DataFrame(df)

def main(dir):
    df = collate(dir)
    df = df.set_index(["config"])
    df.sort_index(inplace=True)
    df = df.reset_index()
    return df


if __name__ == '__main__':
    args = parser.parse_args()
    df = main(args.dir)
