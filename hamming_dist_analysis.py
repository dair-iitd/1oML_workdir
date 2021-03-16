import torch
import pickle
import numpy as np

task = 'futoshiki'
ifile = '/home/yatin/nlm/futo_models/pretrain_chk/e-0_lrh-0.0005_min-1_d-30_nul-16_nmc-5_nm-14_s-3120_tsb-6_tse-6_ts-10000_tr-5_wds-ambiguous_we-200/train_amb_5_5_14_pred_dump.pkl'

#ifile = '/home/yatin/nlm/models/regime_seed_val/e-0_hds-rs_hmin-0_lrh-0.0005_min-0_d-30_nm-5_s-31_tesb-11_tese-11_trs-10_wds-unique_we-200/train10k_10_5_pred_dump.pkl'
#task = 'nqueens'
#ifile = '/home/yatin/nlm/sudoku_models/pretrain_chk_add_e/arb-0_e-0_clip-5.0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-three-one_we-200_wtd-0.0001/train_e_all_pred_dump.pkl'
#ifile = '/home/yatin/hpcscratch/nlm/sudoku_models/analysis_add/arb-0_e-0_clip-5.0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-three-one_we-200_wtd-0.0001/train_d_all_pred_dump.pkl'


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


train = pickle.load(open(ifile,'rb'))

compare_func_futo = lambda x,query: match_query(query[:,0].float(),x) and is_safe_futoshiki(x,query[:,1:])
compare_func_nqueens = lambda x,query: match_query(query[:,0].float(),x) and is_safe_nqueens(x)
compare_func_tower = lambda x,query: is_safe_towers(query[:,0].float(),x, args.test_number_begin if return_float else args.train_number) 
# query doesn't have to be matched for towers
compare_func_sudoku = lambda x,query: match_query(query,x) and is_safe_sudoku(x, args.test_number_begin if return_float else args.train_number)

if task=='futoshiki':
    compare_func = compare_func_futo
    pred = [(x['output_dict']['pred'] > 0.5).int() for x in train]
elif task=='tower':
    compare_func = compare_func_tower
elif task=='sudoku':
    compare_func = compare_func_sudoku
    pred = [x['output_dict']['pred'].argmax(dim=1) for x in train]
elif task == 'nqueens':
    pred = [(x['output_dict']['pred'] > 0.5).int() for x in train]
    compare_func = compare_func_nqueens 
    #

pred  = torch.cat(pred,dim=0)
count = [x['feed_dict']['count'] for x in train]
count = torch.cat(count)
ts = [x['feed_dict']['target_set'] for x in train]
ts = torch.cat(ts,dim=0)
dif = (pred.int().unsqueeze(1).expand_as(ts) != ts.int())
hd = dif.sum(dim=-1)
mask  = [x['feed_dict']['mask'] for x in train]
mask = torch.cat(mask,dim=0)

if task == 'sudoku':
    k = mask.size(-1) - 1
else:
    k = mask.size(-1)

hd = hd[:,:k]
mask = mask[:,:k]
hda = hd[count > 1]
maska = mask[count > 1]
hda[maska == 0] = 0

hdatk = hda.topk(k=k,dim=1)
hdatk= hdatk[0]
hdatk.sum(dim=0).float()/maska.sum(dim=0).float()
counta = count[count > 1]

for i in range(2,(k+1)):
    print(i,(counta == i).sum(),hdatk[counta == i].sum(dim=0).float()/maska[counta == i].sum(dim=0).float())


query = [x['feed_dict']['query'] for x in train]
query = torch.cat(query,dim=0)
querya = query[count > 1]
preda = pred[count > 1]

correcta = [compare_func(x.float(),y.float()) for x,y in zip(preda,querya)]
correcta = torch.tensor(correcta)

hda[maska == 0] = -1
ints = (hda == 0).any(dim=1)
hda[maska == 0] = 0

for i in range(2,(k+1)):
    ind = ((counta == i) & (correcta != 0))
    print('{},{},{},{}'.format("correct",i,ind.sum(),list((hdatk[ind].sum(dim=0).float()/maska[ind].sum(dim=0).float()).numpy())))
    if i == k:
        ind = ((counta == i) & (correcta != 0) & (ints))
        print('{},{},{},{}'.format("correct In TS",i,ind.sum(),list((hdatk[ind].sum(dim=0).float()/maska[ind].sum(dim=0).float()).numpy())))
        ind = ((counta == i) & (correcta != 0) & (~ints))
        print('{},{},{},{}'.format("correct Not in TS",i,ind.sum(),list((hdatk[ind].sum(dim=0).float()/maska[ind].sum(dim=0).float()).numpy())))

for i in range(2,(k+1)):
    ind = ((counta == i) & (correcta == 0))
    print('{},{},{},{}'.format("INcorrect",i,ind.sum(),list((hdatk[ind].sum(dim=0).float()/maska[ind].sum(dim=0).float()).numpy())))

