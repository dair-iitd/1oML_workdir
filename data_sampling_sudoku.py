import pickle
import torch
import numpy as np
import os,sys
import argparse
import math

from collections import Counter,defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ifile', required=True, type=str)
    
    parser.add_argument('--ofile', required=True, type=str, help='path to the output file ')
    parser.add_argument('--ufile', required=False,default=None, type=str, help='path to the unique file ')
    
    parser.add_argument('--num_samples', default=10000, type=int,
                        help='num samples', required=True)
    
    args = parser.parse_args(sys.argv[1:])
    return args

#args.ifile = '~/nlm/data/sudoku_data/ambiguous_data/sudoku_9_train_e_bulk.pkl'
def run(args):
    np.random.seed(42)
    ifile = args.ifile
    dat = pickle.load(open(ifile,'rb'))
    for t in dat:
        t['givens']  = (t['query'] > 0).sum()
    # 
    givens2ind = defaultdict(list)
    for i,x in enumerate(dat):
        if x['count'] > 1:
            givens2ind[x['givens']].append(i)
    # 
    samples = defaultdict(list)
    num_samples = int(math.ceil((1.0*args.num_samples)/len(givens2ind)))
    for k in givens2ind:
        if len(givens2ind[k]) < num_samples:
            print('{}: sampling only {} instead of {}'.format(k,len(givens2ind[k]),num_samples))
        #
        samples[k] = np.random.choice(givens2ind[k],min(len(givens2ind[k]),num_samples),replace=False)
    # 
    samples_ind = []
    for k in samples:
        samples_ind.extend(samples[k])
    # 
    sampled = list(np.array(dat)[samples_ind])
   
    if args.ufile is not None and os.path.exists(args.ufile):
        unique = pickle.load(open(args.ufile,'rb'))
        unique = [x for x in unique if x['count'] == 1]
        for t in unique:
            t['givens']  = (t['query'] > 0).sum()
        #    
        sampled.extend(unique)
    elif args.ufile is not None:
        print('ufile doesnt exist')
        raise

    print('count counter:',Counter([x['count'] for x in sampled]))
    print('givens counter:',Counter([x['givens'] for x in sampled]))


    pickle.dump(sampled,open(args.ofile,'wb'))
    return sampled


if __name__ == '__main__':
    args = parse_args()
    run(args)

