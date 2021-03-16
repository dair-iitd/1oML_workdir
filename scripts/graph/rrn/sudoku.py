"""
SudokuNN module based on RRN for solving sudoku puzzles
"""

#import sys,os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rrn import RRN
from torch import nn
import torch
from IPython.core.debugger import Pdb 

class SudokuNN(nn.Module):
    def __init__(self,
                 num_steps,
                 embed_size=16,
                 hidden_dim=96,
                 edge_drop=0.1,
                 learnable_z = None):
        super(SudokuNN, self).__init__()
        self.num_steps = num_steps
        self.learnable_z = learnable_z
        self.digit_embed = nn.Embedding(10, embed_size)
        self.row_embed = nn.Embedding(9, embed_size)
        self.col_embed = nn.Embedding(9, embed_size)
        learnable_z_len = 0 if learnable_z is None else learnable_z.numel()
        self.input_layer = nn.Sequential(
            nn.Linear(3*embed_size+learnable_z_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lstm = nn.LSTMCell(hidden_dim*2, hidden_dim, bias=False)

        msg_layer = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rrn = RRN(msg_layer, self.node_update_func, num_steps, edge_drop)

        self.output_layer = nn.Linear(hidden_dim, 10)

        self.loss_func = nn.CrossEntropyLoss()
        #self.learnable_z = learnable_z

    def forward(self, g, z_latent = None, is_training=True):
        #labels = g.ndata.pop('a')
        #Pdb().set_trace()
        
        #print('Base: ',self.digit_embed.weight.data[2,:4], self.row_embed.weight.data[2,:4])
        input_digits = self.digit_embed(g.ndata.pop('q'))
        #input_digits = self.digit_embed(g.ndata['q'])
        rows = self.row_embed(g.ndata.pop('row'))
        cols = self.col_embed(g.ndata.pop('col'))
        if self.learnable_z is None:
            x = self.input_layer(torch.cat([input_digits, rows, cols], -1))
        elif z_latent is None:
            x = self.input_layer(torch.cat([input_digits, rows, cols, self.learnable_z.unsqueeze(0).expand(input_digits.size(0),-1)], -1))
        else:
            x = self.input_layer(torch.cat([input_digits, rows, cols, z_latent.unsqueeze(0).expand(input_digits.size(0),-1)], -1))
            
        g.ndata['x'] = x
        g.ndata['h'] = x
        g.ndata['rnn_h'] = torch.zeros_like(x, dtype=torch.float)
        g.ndata['rnn_c'] = torch.zeros_like(x, dtype=torch.float)

        outputs = self.rrn(g, is_training)
        logits = self.output_layer(outputs)
        return logits

        """
        preds = torch.argmax(logits, -1)
        if is_training:
            labels = torch.stack([labels]*self.num_steps, 0)
        logits = logits.view([-1, 10])
        labels = labels.view([-1])
        loss = self.loss_func(logits, labels)
        return preds, loss
        """

    def node_update_func(self, nodes):
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'], nodes.data['rnn_c']
        new_h, new_c = self.lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}


