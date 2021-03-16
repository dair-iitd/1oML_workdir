import pickle
import torch
import numpy as np
import torch.nn.functional as F
l = pickle.load(open('history_1.pkl','rb'))
minloss = torch.tensor([x['loss_tensor'].mean(dim=1).mean(dim=1).min(dim=1)[0].mean() for x in l])

def getlost(bn):
    a = l[bn]
    itn = a['loss_tensor'].mean(dim=1).mean(dim=1).min(dim=-1)[0].argmax().item()
    worst = torch.tensor([[torch.gather(F.softmax(a['logits'][itn,:,:,i],dim=0),0,a['feed_dict']['target_set'][itn][j].long().unsqueeze(0)).min() for i in range(32)] for j in range(6)]).transpose(0,1)
    itn = a['loss_tensor'].mean(dim=1).mean(dim=1).min(dim=-1)[0].argmin().item()
    best  = torch.tensor([[torch.gather(F.softmax(a['logits'][itn,:,:,i],dim=0),0,a['feed_dict']['target_set'][itn][j].long().unsqueeze(0)).min() for i in range(32)] for j in range(6)]).transpose(0,1)
    return (worst, best)

bn = minloss.argmax().item()
worst = getlost(bn)
bn = minloss.argmin().item()
best = getlost(bn)
bn = 470

