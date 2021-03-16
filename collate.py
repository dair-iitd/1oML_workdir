import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
import copy
from sys import argv
import json 
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='parent directory containing experiment logs')
parser.add_argument('--out', type=str,help='output file for collated_results')

def load_errors(error_dir):
    errors = []
    error_files = [os.path.join(error_dir,x) for x in os.listdir(error_dir) if x.find("errors")>-1]
    for f in error_files:
        errors.extend(pickle.load(open(f,'rb')))
    retval = dict(Counter(errors))
    retval = dict([(str(k)+"_sol_errors",v) for k,v in retval.items()])
    return retval

def read_summary_file(exp_dir):
    if not os.path.exists(os.path.join(exp_dir,"summary.json")):
        print(exp_dir,"missing summary.json file!")
        return []
    with open(os.path.join(exp_dir, "summary.json")) as f:
        l = f.readlines()
    l = [json.loads(x) for x in l[::-1]]
    test_files = set()
    test_stats = []
    for stat in l:
        if int(stat["epoch"])!=-1:
            continue 
        tf = os.path.basename(stat["data_file"])
        if tf in test_files:
            continue
        test_files.add(tf)
        test_stats.append({"test_file":tf, "test_acc":stat["corrected accuracy"]})
        if "error distribution" in stat:
            test_stats[-1].update({(str(k+1)+"_sol_error",v) for k,v in enumerate(stat["error distribution"].split("-"))})
    return test_stats 

def collate(parent_dir):
    exp_dirs = os.listdir(parent_dir)
    
    df = [] 
    for exp_dir in exp_dirs:
        dir_name = os.path.join(parent_dir, exp_dir)
        for expt_dict in read_summary_file(dir_name):
            expt_dict["model config"] = exp_dir 
            df.append(expt_dict)
    return pd.DataFrame(df)


if __name__ == '__main__':
    args = parser.parse_args()
    df = collate(args.dir)
    df = df.set_index(["test_file","model config"])
    df.sort_index(inplace=True)
    df.to_csv(args.out)
