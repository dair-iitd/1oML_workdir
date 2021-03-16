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


split_files = { "nqueens_11_6_test.pkl": (8332,"nqueens_11_6_test_unique.pkl","nqueens_11_6_test_amb.pkl"),
                "nqueens_11_6_val.pkl": (8335,"nqueens_11_6_val_unique.pkl","nqueens_11_6_val_amb.pkl"),
                "futo_6_18_5_test.pkl": (7505,"futo_6_18_5_test_unique.pkl","futo_6_18_5_test_amb.pkl"),
                "futo_6_18_5_val.pkl": (7544,"futo_6_18_5_val_unique.pkl","futo_6_18_5_val_amb.pkl")
            }


def exclude_rows(table,colname,values):
    return table[~table[colname].isin(set(values))]

def read_tables(exp_dir,dir_list):
    gdf = None
    for this_dir in dir_list:
        df = main(os.path.join(exp_dir, this_dir))
        df = to_numeric(df, df.columns)
        if gdf is None:
            gdf = df
        else:
            gdf = gdf.append(df)
            
    nan_columns = gdf.columns[gdf.isnull().values.sum(axis=0) > 0]
    print("filling nans in following column with 0:", nan_columns)
    gdf = gdf.fillna(value= 0)
    gdf = exclude_rows(gdf,'wds',{'four-one','four.one'})
    return gdf


def row_splitter(row):
    if row["test_file"] in split_files:
        unq, unq_file, amb_file = split_files[row["test_file"]]
        row_unq = copy.deepcopy(row)
        row_unq["test_file"] = unq_file 
        row_unq["test_acc"] = 1 - float(row["1_sol_error"])/unq 
        row_amb = copy.deepcopy(row)
        row_amb["test_file"] = amb_file 
        row_amb["test_acc"] = (row["test_acc"]*10000 - (unq-float(row["1_sol_error"])))/(10000-unq)

        return [row,row_unq,row_amb]
    else:
        return [row]


def load_errors(error_dir):
    errors = []
    error_files = [os.path.join(error_dir, x) for x in os.listdir(
        error_dir) if x.find("errors") > -1]
    for f in error_files:
        errors.extend(pickle.load(open(f, 'rb')))
    retval = OrderedDict(Counter(errors))
    retval = OrderedDict([(str(k)+"_sol_errors", v)
                          for k, v in retval.items()])
    return retval


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

def pointwise_accuracy_stats(query, target_set, count, raw_pred, prefix=''):
    pred = raw_pred.argmax(dim=1).int()
    #Pdb().set_trace()
    non_zero_ind = (query > 0)
    copy_accuracy = (query[non_zero_ind] ==
                     pred[non_zero_ind]).sum().float()/non_zero_ind.sum()
    copy_point_total = non_zero_ind.sum().item()

    unique_point_ind = (target_set[:, 0, :] == target_set[:, 1, :])
    unique_point_ind = unique_point_ind*(~non_zero_ind)
    unique_point_accuracy = (target_set[:, 0, :][unique_point_ind]
                             == pred[unique_point_ind]).sum().float()/unique_point_ind.sum()
    unique_point_total = unique_point_ind.sum().item()

    ambiguous_point_ind = ~(target_set[:, 0, :] == target_set[:, 1, :])
    ambiguous_point_accuracy = (
        target_set[:, 0, :][ambiguous_point_ind] == pred[ambiguous_point_ind]).sum()
    ambiguous_point_accuracy += (target_set[:, 1, :]
                                 [ambiguous_point_ind] == pred[ambiguous_point_ind]).sum()
    ambiguous_point_accuracy = ambiguous_point_accuracy.float() / \
        ambiguous_point_ind.sum()
    ambiguous_point_total = ambiguous_point_ind.sum().item()

    total_points = query.numel()
    #Pdb().set_trace()
    strict_acc_count = float((target_set[:, 0, :] == pred).all(dim=1).sum(
    ) + (((target_set[:, 1, :] == pred).all(dim=1).int())*(count-1)).sum())
    
    strict_accuracy = strict_acc_count / float(pred.size(0))
   
    lac = ((target_set[:,0,:] == pred) | (target_set[:,1,:] == pred)).all(dim=1).sum().float().item()
    la = lac / float(pred.size(0))
    

    lousy_accuracy = (target_set[:,0,:]==pred).int()+(target_set[:,1,:]==pred).int()*(count-1).unsqueeze(1).expand_as(target_set[:,0,:])
    lousy_accuracy = ((lousy_accuracy>0).sum(dim=1)==81)
    lousy_acc_count = lousy_accuracy.sum().float()
    lousy_accuracy = (lousy_acc_count/pred.shape[0]).item()


    corrected_accuracy = []
    for i,x in enumerate(pred):
        corrected_accuracy.append(is_safe_sudoku(x,query[i],9))
    corrected_accuracy = torch.tensor(corrected_accuracy).float().mean()


    if lac != lousy_acc_count:
        Pdb().set_trace()
    
    rv = OrderedDict()
    rv[prefix+'copy_acc'] = copy_accuracy.item()
    rv[prefix+'unique_pt_acc'] = unique_point_accuracy.item()
    rv[prefix+'amb_pt_acc'] = ambiguous_point_accuracy.item()
    rv[prefix+'lousy_acc'] = lousy_accuracy
    rv[prefix+'strict_acc'] = strict_accuracy
    rv[prefix+'total_pts'] = total_points
    rv[prefix+'copy_pts'] = copy_point_total
    rv[prefix+'unique_pts'] = unique_point_total
    rv[prefix+'amb_pts'] = ambiguous_point_total
    rv[prefix+'strict_count'] = strict_acc_count
    rv[prefix+'lousy_count'] = lac
    rv[prefix+'corrected_acc'] = corrected_accuracy.item()
    return rv
    # return copy_accuracy.item(), unique_point_accuracy.item(), ambiguous_point_accuracy.item(), lousy_accuracy.item(), total_points, copy_point_total, unique_point_total, ambiguous_point_total, lousy_acc_count


def extract_data_from_dump(exp_dir, analyze=False):
    if not analyze:
        return [] 
    if not os.path.exists(os.path.join(exp_dir, "pred_dump.pkl")):
        print(exp_dir, "pred dump doesn't exist")
        return []

    with open(os.path.join(exp_dir, "pred_dump.pkl"), "rb") as f:
        dump = pickle.load(f)
    query = []
    target_set = []
    count = []
    raw_pred = []

    for z in dump:
        q, t, c, r = z["feed_dict"]["query"], z["feed_dict"]["target_set"], z["feed_dict"]["count"], z["output_dict"]["pred"]
        query.append(q)
        target_set.append(t)
        count.append(c)
        raw_pred.append(r)

    count_arr = torch.cat(count).int()
    unique_ind = count_arr == 1
    amb_ind = count_arr > 1

    all_stats = pointwise_accuracy_stats(torch.cat(query).int(), torch.cat(
        target_set).int(), torch.cat(count).int(), torch.cat(raw_pred).float(), 'ALL ')

    unique_stats = pointwise_accuracy_stats(torch.cat(query).int()[unique_ind], torch.cat(
        target_set).int()[unique_ind], count_arr[unique_ind], torch.cat(raw_pred).float()[unique_ind], 'UN ')
    
    amb_stats = pointwise_accuracy_stats(torch.cat(query).int()[amb_ind], torch.cat(
        target_set).int()[amb_ind], count_arr[amb_ind], torch.cat(raw_pred).float()[amb_ind], 'AMB ')

    return [all_stats, unique_stats, amb_stats]

def read_summary_file(exp_dir, analyze=False, acc_type= "corrected accuracy"):
    if not os.path.exists(os.path.join(exp_dir, "summary.json")):
        print(exp_dir, "missing summary.json file!")
        return []
    with open(os.path.join(exp_dir, "summary.json")) as f:
        l = f.readlines()
    l = [json.loads(x) for x in l[::-1]]
    test_files = set()
    test_stats = []
    best_train = 0
    last_train = -1
    best_train_epoch = 0
    best_dev = 0
    last_dev = -1
    best_dev_epoch = 0
    current_epoch = 0
    total_epochs = 0
    accuracy_type = acc_type
    
    for stat in l:
        if (stat["mode"]) != "test":
            current_epoch = stat['epoch']
            if total_epochs == 0:
                total_epochs = current_epoch
            if last_train < 0:
                last_train = stat[accuracy_type]
            if best_train <= stat[accuracy_type]:
                best_train_epoch = stat['epoch']
                best_train = stat[accuracy_type]
            continue
        if (stat["data_file"].find("dev")) > 0:
            if last_dev < 0:
                last_dev = stat[accuracy_type]
            if (best_dev <= stat[accuracy_type]) and (stat['lr'] != 0):
                best_dev_epoch = current_epoch
                best_dev = stat[accuracy_type]

        tf = os.path.basename(stat["data_file"])
        if tf in test_files:
            continue
        test_files.add(tf)
        test_stats.append(OrderedDict(
            {"test_file": tf, "test_acc": stat['corrected accuracy'] if stat['data_file'].find('test') > 0 else  stat[accuracy_type]}))
        if "error distribution" in stat:
            test_stats[-1].update({(str(k+1)+"_sol_error", v)
                                   for k, v in enumerate(stat["error distribution"].split("-"))})

    if len(test_stats) == 0:
        print(exp_dir, "missing test results in summary.json file!")

    #copy_accuracy, unique_point_accuracy, ambiguous_point_accuracy, lousy_accuracy = extract_data_from_dump(
    #    exp_dir, analyze)
    #Pdb().set_trace() 
    addn_stats  = extract_data_from_dump(exp_dir, analyze)
    split_test_stats = []
    for row in test_stats:
        row["best_train"] = best_train
        row["last_train"] = last_train
        row["best_train_epoch"] = best_train_epoch
        row["best_dev"] = best_dev
        row["last_dev"] = last_dev
        row["best_dev_epoch"] = best_dev_epoch
        row['total_epochs'] = total_epochs
        
        for this_dict in addn_stats:
            for k in this_dict:
                row[k] = this_dict[k]

        split_test_stats.extend(row_splitter(row))
        
        #row['copy_acc'] = copy_accuracy
        #row['unique_point_acc'] = unique_point_accuracy
        #row['ambiguous_point_acc'] = ambiguous_point_accuracy
        #row['lousy_acc'] = lousy_accuracy

    return split_test_stats


def get_params(exp_dir):
    all_params = exp_dir.split('_')
    all_params = dict([(x.split('-')[0], '.'.join(x.split('-')[1:]))
                       for x in all_params])
    return all_params


def collate(parent_dir, analyze=False,acc_type='corrected accuracy'):
    exp_dirs = os.listdir(parent_dir)

    df = []
    for exp_dir in tqdm(exp_dirs):
        dir_name = os.path.join(parent_dir, exp_dir)
        for expt_dict in read_summary_file(dir_name, analyze,acc_type):
            expt_dict["config"] = exp_dir
            expt_dict.update(get_params(exp_dir))
            df.append(expt_dict)
    return pd.DataFrame(df)


def analysis(df):
    p = ['lal', 'wds', 'hds']
    best = df.loc[df.groupby(p)['test_acc'].idxmax()]
    best[['test_acc', 'hds', 'wds', 'lal', 'lalf', 'ldp']]
    p = ['lal', 'lalf']
    best = df.loc[df.groupby(p)['test_acc'].idxmax()]
    best[['test_acc', 'hds', 'wds', 'lal', 'lalf', 'ldp']]
    p = ['ldp', 'lal', 'lalf']
    best = df.loc[df.groupby(p)['test_acc'].idxmax()]
    best[['test_acc', 'hds', 'wds', 'lal', 'lalf', 'ldp']]
    values = ['test_acc']
    index = ['min', 'hmin', 'we', 'e', 'wds', 'hds']
    columns = ['s']
    values = 'test_acc'
    best.pivot_table(values, index, columns)
    best.pivot_table(values, index, columns).reset_index()

    p = ['nmc', 'min', 's', 'test_file', 'wds']
    best = df.loc[df.groupby(p)['test_acc'].idxmax()]
    best[p+['test_acc']]
    values = ['test_acc']
    index = ['nmc', 'min', 's', 'wds']
    columns = ['test_file']
    values = 'test_acc'
    best.pivot_table(values, index, columns)
    best.pivot_table(values, index, columns).reset_index()

    p = ['nmc', 's', 'min', 'test_file', 'lal', 'lalf', 'hds']
    best = df.loc[df.groupby(p)['test_acc'].idxmax()]
    best[p+['wds', 'test_acc']]
    values = ['test_acc']
    index = ['nmc', 's', 'min', 'wds', 'hds', 'lal']
    columns = ['test_file']
    values = 'test_acc'
    best.pivot_table(values, index, columns)
    best.pivot_table(values, index, columns).reset_index()


def analyze_bl():
    # run collate_pivot.py --dir futo_models/baselines/ --out tempfuto.csv
    dfbl = df
    pbl = ['nmc', 's', 'min', 'wds', 'test_file', 'ts']
    bestbl = dfbl.loc[dfbl.groupby(pbl)['test_acc'].idxmax()]
    bbl = bestbl[pbl + ['test_acc']]
    values = ['test_acc']
    index = ['ts', 'nmc', 's', 'min', 'wds']
    columns = ['test_file']
    values = 'test_acc'
    bbl.pivot_table(values, index, columns).reset_index()


def analyze_rl():
    # run collate_pivot.py --dir futo_models/rl_20200411 --out rlfutotemp.csv
    dfrl = df
    pbl = ['nmc', 's', 'min', 'test_file']
    prl = pbl+['lal', 'lalf', 'hds']
    bestrl = dfrl.loc[dfrl.groupby(prl)['test_acc'].idxmax()]
    brl = bestrl[prl + ['wds', 'test_acc']]
    index = ['nmc', 's', 'min', 'lal', 'lalf', 'hds']
    columns = ['test_file']
    values = 'test_acc'
    brl.pivot_table(values, index, columns).reset_index()


def compare_rl_bl_nqueens():
    # run collate_pivot.py --dir models/rl_seed_val --out rlfutotemp.csv
    dfrl = main('models/rl_seed_val')
    # run collate_pivot.py --dir models/regime_seed_val --out tempfuto.csv
    dfbl = main('models/regime_seed_val')

    dfsl = main('models/rl_select')

    indexbl = ['min', 'hmin', 'we', 'e', 'wds', 'hds']
    dfbl.pivot_table(index=indexbl, columns='test_file', values='test_acc', aggfunc=[
                     'max', 'mean', 'std', 'count']).to_csv('nqueens_bl.csv')

    indexrl = ['we', 'e', 'wds', 'hds', 'lal', 'lalf', 'phip', 'rlrc']
    dfrl.pivot_table(index=indexrl, columns='test_file', values='test_acc', aggfunc=[
                     'max', 'mean', 'std', 'count']).to_csv('nqueens_rl.csv')

    dfsl.pivot_table(index=indexrl, columns='test_file', values='test_acc', aggfunc=[
                     'max', 'mean', 'std', 'count']).to_csv('nqueens_rl_select.csv')

    dfbl.pivot_table(index=indexbl, columns='test_file',
                     values='test_acc', aggfunc='mean')
    dfbl.pivot_table(index=indexbl, columns='test_file',
                     values='test_acc', aggfunc='std')
    dfbl.pivot_table(index=indexbl, columns='test_file',
                     values='test_acc', aggfunc='count')

    pbl = ['s', 'test_file', 'min', 'hmin', 'we', 'e', 'wds', 'hds']
    bestbl = dfbl.loc[dfbl.groupby(pbl)['test_acc'].idxmax()]

    prl = ['s', 'test_file', 'we', 'e', 'wds',
           'hds', 'lal', 'lalf', 'phip', 'rlrc']
    #prl = pbl+['lal','lalf']
    bestrl = dfrl.loc[dfrl.groupby(prl)['test_acc'].idxmax()]

    dfrl.pivot_table(index=indexrl, columns='test_file',
                     values='test_acc', aggfunc='mean')
    dfrl.pivot_table(index=indexrl, columns='test_file',
                     values='test_acc', aggfunc='std')
    dfrl.pivot_table(index=indexrl, columns='test_file',
                     values='test_acc', aggfunc='count')

    brl = bestrl[prl + ['wds', 'hds', 'test_acc']]
    bbl = bestbl[pbl + ['wds', 'test_acc']]
    brl.join(bbl.set_index(['nmc', 's', 'min', 'test_file']), on=[
             'nmc', 's', 'min', 'test_file'], rsuffix='_bl')

    pbl = ['test_file', 'min', 'hmin', 'we', 'e', 'wds', 'hds']
    bestbl = dfbl.loc[dfbl.groupby(pbl)['test_acc'].idxmax()]
    bestbl[pbl+['test_acc', 's']]

    prl = ['test_file', 'we', 'e', 'wds', 'hds', 'lal', 'lalf']
    bestrl = dfrl.loc[dfrl.groupby(prl)['test_acc'].idxmax()]
    bestrl[prl + ['test_acc', 's']]

    indexbl = ['min', 'hmin', 'we', 'e', 'wds', 'hds']
    bestbl.pivot_table(index=indexbl, columns='test_file',
                       values='test_acc', aggfunc='mean')
    indexrl = ['we', 'e', 'wds', 'hds', 'lal', 'lalf']
    bestrl.pivot_table(index=indexrl, columns='test_file',
                       values='test_acc', aggfunc='mean')


def compare_rl_select_bl():
    # run collate_pivot.py --dir futo_models/rl_20200411 --out rlfutotemp.csv
    dfrl = main('futo_models/rl_select')
    # run collate_pivot.py --dir futo_models/baselines/ --out tempfuto.csv
    dfbl = main('futo_models/pretrain_chk')
    pbl = ['nmc', 's', 'min', 'test_file']
    bestbl = dfbl.loc[dfbl.groupby(pbl)['test_acc'].idxmax()]
    bestbl = bestbl[bestbl['s'] == '3120']
    prl = pbl+['phip', 'rlrc', 'lal', 'lalf']
    bestrl = dfrl.loc[dfrl.groupby(prl)['test_acc'].idxmax()]
    bestrl = bestrl[bestrl['s'] == '31']
    addn_cols = ['best_dev_epoch', 'best_train_epoch',
                 'total_epochs', 'best_train', 'last_train']
    rlcols = prl + ['wds', 'hds', 'test_acc']+addn_cols
    blcols = pbl + ['wds', 'test_acc'] + addn_cols
    brl = bestrl[prl + ['wds', 'hds', 'test_acc']+addn_cols]
    bbl = bestbl[pbl + ['wds', 'test_acc']+addn_cols]
    brl.join(bbl.set_index(['nmc', 'min', 'test_file']), on=[
             'nmc', 'min', 'test_file'], rsuffix='_bl')

    brl.join(bbl.set_index(['nmc', 'min', 'test_file']), on=[
             'nmc', 'min', 'test_file'], rsuffix='_bl').to_csv('yatincompare.csv', index=False)
    dfrl[rlcols].to_csv('yatindfrl.csv', index=False)
    dfbl[blcols].to_csv('yatindfbl.csv', index=False)


def compare_rl_bl():
    # run collate_pivot.py --dir futo_models/rl_20200411 --out rlfutotemp.csv
    dfrl = main('futo_models/rl_seed_val')
    # run collate_pivot.py --dir futo_models/baselines/ --out tempfuto.csv
    dfbl = main('futo_models/pretrain_chk')
    pbl = ['nmc', 's', 'min', 'test_file']
    bestbl = dfbl.loc[dfbl.groupby(pbl)['test_acc'].idxmax()]
    bestbl = bestbl[bestbl['s'] == '3120']
    prl = pbl+['phip', 'rlrc', 'lal', 'lalf']
    bestrl = dfrl.loc[dfrl.groupby(prl)['test_acc'].idxmax()]
    bestrl = bestrl[bestrl['s'] == '31']
    addn_cols = ['best_dev_epoch', 'best_train_epoch',
                 'total_epochs', 'best_train', 'last_train']
    rlcols = prl + ['wds', 'hds', 'test_acc']+addn_cols
    blcols = pbl + ['wds', 'test_acc'] + addn_cols
    brl = bestrl[prl + ['wds', 'hds', 'test_acc']+addn_cols]
    bbl = bestbl[pbl + ['wds', 'test_acc']+addn_cols]
    brl.join(bbl.set_index(['nmc', 'min', 'test_file']), on=[
             'nmc', 'min', 'test_file'], rsuffix='_bl')

    brl.join(bbl.set_index(['nmc', 'min', 'test_file']), on=[
             'nmc', 'min', 'test_file'], rsuffix='_bl').to_csv('yatincompare.csv', index=False)
    dfrl[rlcols].to_csv('yatindfrl.csv', index=False)
    dfbl[blcols].to_csv('yatindfbl.csv', index=False)


def to_numeric(df, numeric):
    for col in numeric:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            print("could not convert {} to numeric".format(col))
    return df


def futo_nostatic():
    df = main('futo_models/rl_nostatic')
    df = to_numeric(df, df.columns)
    df.describe()
    values = ['test_acc']
    columns = ['test_file', 's', 'nos', 'cpf']
    index = ['m', 'aux', 'rlrc', 'min', 'wds', 'hds']
    index = ['rlrc', 'min', 'wds', 'hds']
    df.pivot_table(index=index, columns=columns,
                   values=values, aggfunc=['max', 'count'])
    df.pivot_table(index=index, columns=columns, values=values,
                   aggfunc=['max', 'count']).to_csv('sat_rl_futo1.csv')


def satnet_futo():
    df = main('satfuto_models/rl_select/')
    df.describe()
    values = ['test_acc']
    columns = ['test_file', 's', 'nos', 'cpf']
    index = ['m', 'aux', 'rlrc', 'min', 'wds', 'hds']
    df.pivot_table(index=index, columns=columns,
                   values=values, aggfunc=['max', 'count'])
    df.pivot_table(index=index, columns=columns, values=values,
                   aggfunc=['max', 'count']).to_csv('sat_rl_futo1.csv')
    df = main('satfuto_models/pretrain_chk/')
    index = ['min', 'm', 'wds']
    columns = ['test_file', 's', 'aux']
    df.pivot_table(index=index, columns=columns, values=values, aggfunc=[
                   'max', 'count']).to_csv('sat_baseline_futo1.csv')


def satnet_nqueens():
    numeric = ['m', 'aux', 's', 'min', 'rlrc']
    df = main('satnqueens_models/rl/')
    df.describe()
    values = ['test_acc']
    columns = ['test_file', 's']
    index = ['m', 'aux', 'rlrc', 'min', 'wds', 'hds']
    df.pivot_table(index=index, columns=columns, values=values,
                   aggfunc=['max', 'count']).to_csv('sat_rl_futo1.csv')
    df = main('satfuto_models/pretrain_chk/')
    df = to_numeric(df, numeric)

    index = ['min', 'm', 'wds']
    columns = ['test_file', 's', 'aux']
    df.pivot_table(index=index, columns=columns,
                   values=values, aggfunc=['max', 'count'])
    df.pivot_table(index=index, columns=columns, values=values, aggfunc=[
                   'max', 'count']).to_csv('sat_baseline_futo1.csv')


def main(dire, analyze=False,acc_type='corrected accuracy'):
    df = collate(dire, analyze,acc_type)
    df = df.set_index(["test_file", "config"])
    df.sort_index(inplace=True)
    df = df.reset_index()
    return df

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        help='parent directory containing experiment logs')
    parser.add_argument('--out', default='', type=str,
                        help='output file for collated_results')
    parser.add_argument('--analyze', action='store_true',
                        help='analyze different types of pointwise accuracies')
    parser.add_argument('--ts_acc', action='store_true',
                        help='return target_set accuracies')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = read_args()
    df = main(args.dir,args.analyze)
    if args.out != '':
        df.to_csv(args.out, index=False)
