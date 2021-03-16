import numpy as np
from torch.distributions.categorical import Categorical
import torch
from collections import defaultdict
from IPython.core.debugger import Pdb
class SMAnalysis:
    def __init__(self,dataset,args):
        #self.argmax_switch_matrix = torch.zeros(len(dataset.dataset),args.epochs+1).fill_(-1)
        #self.minloss_switch_matrix = torch.zeros(len(dataset.dataset),args.epochs+1).fill_(-1)
        #self.sample_switch_matrix = torch.zeros(len(dataset.dataset),args.epochs+1).fill_(-1)
        #self.prob_matrix = torch.zeros(len(dataset.dataset), args.epochs, dataset.max_count) 
        self.argmax_choice_list = defaultdict(list)
        self.sample_choice_list = defaultdict(list)
        self.minloss_choice_list = defaultdict(list)
        self.prob_dist = defaultdict(list)
        self.loss_dist = defaultdict(list)
    
    def update(self,feed_dict,loss_matrix, epoch,itr_num):
        #Pdb().set_trace()
        with torch.no_grad():
            dist = Categorical(feed_dict['weights'])
            sample = dist.sample()
            argmax = feed_dict['weights'].argmax(dim=1)
            #minloss = loss_matrix.argmin(dim=1) 
            for i in range(feed_dict['ind'].size(0)):
                if feed_dict['is_ambiguous'][i].item() == 1:
                    ind = feed_dict['ind'][i].item()
                    #entire prob distribution
                    #self.prob_matrix[ind,epoch,:] = feed_dict['weights'][i].detach().cpu()
                    #
                    self.prob_dist[ind].append((feed_dict['weights'][i][:feed_dict['count'][i]].detach().cpu(),epoch,itr_num))
                    self.loss_dist[ind].append((loss_matrix[i][:feed_dict['count'][i]].detach().cpu(),epoch,itr_num))
                    self.argmax_choice_list[ind].append((argmax[i].item(),epoch,itr_num))
                    self.sample_choice_list[ind].append((sample[i].item(),epoch,itr_num))
                    self.minloss_choice_list[ind].append((loss_matrix[i][:feed_dict['count'][i]].argmin().item(),epoch,itr_num))
                    
                    """#
                    if len(self.argmax_choice_list[ind]) >= 2:
                        #Pdb().set_trace()
                        self.argmax_switch_matrix[ind,epoch] = max(self.argmax_switch_matrix[ind,epoch],
                                            self.argmax_choice_list[ind][-1] != self.argmax_choice_list[ind][-2])

                        self.sample_switch_matrix[ind,epoch] = max(self.sample_switch_matrix[ind,epoch],
                                            self.sample_choice_list[ind][-1] != self.sample_choice_list[ind][-2])
                        
                        self.minloss_switch_matrix[ind,epoch] = max(self.minloss_switch_matrix[ind,epoch],self.minloss_choice_list[ind][-1] != self.minloss_choice_list[ind][-2])
                    """


