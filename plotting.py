import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from IPython.core.debugger import Pdb
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--warmup-dir-list',  nargs = '*', default=[], type=str)
parser.add_argument('--hot-dir-list',  nargs = '*', default=[], type=str)
parser.add_argument('--output-file-list',  nargs = '*', default=[], type=str)
parser.add_argument('--task',default = 'futo',type=str)
parser.add_argument('--type',default = 'b',type=str)
parser.add_argument('--ts_acc',action="store_true")

args = parser.parse_args(sys.argv[1:])


def get_warmup_dir(params):
    return 'e-0_lrh-0.0005_min-{}_d-30_nul-16_nmc-{}_nm-14_s-{}_tsb-6_tse-6_ts-10000_tr-5_wds-{}_we-200'.format(params['min'],params['nmc'], 3120, params['wds'])

def get_warmup_dir_nqueens(params):
    run_dir_template ='e-0_hds-rs_hmin-0_lrh-0.0005_min-0_d-30_nm-5_s-{}_tesb-11_tese-11_trs-10_wds-{}_we-200'
    return run_dir_template.format(params['s'], params['wds'])

def get_warmup_dir_sudoku(params):
    return "arb-0_e-0_clip-5.0_add-{}_lr-0.001_min-{}_s-3120_tesb-9_tese-9_trs-9_wds-{}_we-200_wtd-0.0001".format(params['add'],params["min"],params["wds"])


get_warmup_dir_fn = get_warmup_dir
task_futoshiki = args.task == 'futo'
task_nqueens =  args.task == 'nqueens'
task_satfuto = args.task == 'satfuto'
if args.task == 'nqueens':
    get_warmup_dir_fn = get_warmup_dir_nqueens 

if args.task =="sudoku":
    get_warmup_dir_fn = get_warmup_dir_sudoku 


def get_params(exp_dir):
    all_params = exp_dir.split('_')
    all_params =  dict([(x.split('-')[0], '-'.join(x.split('-')[1:])) for x in all_params])
    return all_params


def read_summary(summary):
    with open(summary) as f:
        l = f.readlines()
        l = [json.loads(x) for x in l]
        res = pd.DataFrame(l)        
        res.loc[res['epoch'] == -1,'epoch'] = None
        res.loc[:,['epoch']] = res.loc[:,['epoch']].ffill()
        return res

acol = 'corrected accuracy'
if args.ts_acc:
    acol = 'target set accuracy'
#gby_cols = ['s','min','wds','arb','add']

gby_cols = ['s','arb','min','add','wds']
if task_nqueens or task_futoshiki:
    #gby_cols = ['hds','hmin','min','s','wds','e','we']
    gby_cols = ['s','arb','min','wds']
    #gby_cols = ['s','arb','min','wds','lam']

if task_satfuto:
    gby_cols = ['m','aux','we','e','min','wds']


#warmup_dir = 'futo_models/baselines'
#hot_dir = 'futo_models/rl_20200411'
START = 2
for warmup_dir,output_file in zip(args.warmup_dir_list,args.output_file_list):
    warmuplist = os.listdir(warmup_dir)
    params = list(map(get_params,warmuplist))
    params = pd.DataFrame(params)
    params['this_dir'] = warmuplist
    
    groups = params.groupby(gby_cols)
    warmup_data = {}
    pp = PdfPages(output_file)
    for name,group in groups:
        for _,row in group.iterrows():
            #Pdb().set_trace()
            this_dir = row['this_dir']
            summary = os.path.join(warmup_dir, this_dir, 'summary.json')
            if os.path.exists(summary):
                res = read_summary(summary)
                train = res[res['mode'] == 'train']
                test = res[res['mode'] == 'test']
                train = train[train['epoch'] >= START]
                test = test[test['epoch'] >= START]
                test = test.set_index('epoch',drop=False)
                train = train.set_index('epoch',drop=False) 
                fig = plt.figure()
                train[acol].plot(marker = 'o',label='B. Train')
                test[acol].plot(style = 'g',marker='x', label = 'B. Test')
                plt.legend(loc='center right')
                #df.plot(x='ni',y='f')
                train['lr'].plot(secondary_y=True,style = 'r',label = 'LR (Right)')
                plt.legend(loc = 'center right')
                #plt.title('seed:{} min: {} #constraints: {} warmup data: {}'.format(row['s'], row['min'], row['nmc'], row['wds']))
                plt.title(' '.join(['{}:{}'.format(x, row[x]) for x in gby_cols]),fontsize=9)
                pp.savefig(fig)
                plt.close()
                lcol = 'loss' 
                fig = plt.figure()
                train[lcol].plot(marker = 'o',label='Train Loss')
                plt.legend(loc='center right')
                #df.plot(x='ni',y='f')
                train[acol].plot(secondary_y=True,style = 'r',label = 'Train Accuracy (Right)')
                plt.legend(loc = 'center right')
                #plt.title('seed:{} min: {} #constraints: {} warmup data: {}'.format(row['s'], row['min'], row['nmc'], row['wds']))
                plt.title(' '.join(['{}:{}'.format(x, row[x]) for x in gby_cols]),fontsize  = 9)
                pp.savefig(fig)
                plt.close()
                


                warmup_data[this_dir] = (train, test)
    #
    pp.close()
    
if len(args.hot_dir_list):
    hot_dir  = args.hot_dir_list[0]
    #gby_cols = ['hds','lal','lalf','prob','s','wds']
    hotlist = os.listdir(hot_dir)
    params = list(map(get_params,hotlist))
    params = pd.DataFrame(params)
    params['this_dir'] = hotlist
    
    #gby_cols = ['nmc','s','min','wds','lal','lalf','hds']
    #gby_cols = ['nmc','s','min','wds','lal','hds']
    gby_cols = ['nmc','s', 'wds', 'min', 'lal', 'hds', 'phip', 'rlrc','cpf']
    if task_nqueens:
        gby_cols = ['wds','hds','min','lal','phip','rlrc','s']
    if args.task=="sudoku":
        gby_cols = ['wds','hds','min','lal','phip','rlrc','s','add']


    groups = params.groupby(gby_cols)
    pp = PdfPages('hot_plots_{}'.format(args.output_file_list[0]))
    for name,group in groups:
        for _,row in group.iterrows():
            this_dir = row['this_dir']
            #Pdb().set_trace()
            #Pdb().set_trace()
            summary = os.path.join(hot_dir, this_dir, 'summary.json')
            if os.path.exists(summary):
                res = read_summary(summary) 
                train = res[res['mode'] == 'train']
                test = res[res['mode'] == 'test']
                if (train.shape[0] > 5) and (test.shape[0] > 5):
                    warmup_train, warmup_test = warmup_data[get_warmup_dir_fn(dict(zip(gby_cols,name)))]
                    best_epoch = warmup_test[acol].iloc[:-3].idxmax() 
                    start_epoch = train['epoch'].min()
                    adjustment = best_epoch - start_epoch + 1
                    train.loc[:,'epoch'] += adjustment
                    test.loc[:,'epoch'] += adjustment
                    test = test.set_index('epoch',drop=False)
                    train = train.set_index('epoch',drop=False) 
                    
                    #train = warmup_train.append(train)
                    #test =  warmup_test.append(test)
                    fig = plt.figure()
                    warmup_train[acol].plot(marker = 'o',linestyle='-',label='Warmup Train')
                    warmup_test[acol].plot(style = 'g',marker='x',linestyle = '-', label = 'Warmup Test')
                    train[acol].plot(style='r',marker = 'o',linestyle='-', label = 'Hot Train')
                    test[acol].plot(style = 'o',marker='x',linestyle = '-', label = 'Hot Test')
                    plt.legend(loc='center right')
                    warmup_train['lr'].plot(secondary_y=True,label = 'Warmup LR (Right)')
                    train['lr'].plot(secondary_y=True,label = 'Hot LR (Right)')
                    #df.plot(x='ni',y='f')
                    #df.plot(x='ni',y='nv',secondary_y=True,style = 'g')
                    row_dict  = row.to_dict()
                    title = ' '.join(['{}:{}'.format(k,row[k]) for k in row_dict])
                    title = title[:len(title)//2] + '\n' + title[len(title)//2:]
                    plt.title(title,fontsize=9)
                    #plt.title('seed:{} min: {} #constraints: {} warmup data: {} \n hot data: {}, {}:{}'.format(row['s'], row['min'], row['nmc'], row['wds'], row['hds'], row['lal'], row['lalf']))
                    plt.legend(loc='center right')
                    pp.savefig(fig)
                    plt.close()
    #
    pp.close()



