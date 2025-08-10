import torch
import torch.nn as nn
import numpy as np

class supervised(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.decomposition = None
        self.ReLU = nn.ReLU()
        self.loss_type = self.params['energynet']['supervised']['loss']
        if self.loss_type == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()
        self.softmax1 = nn.Softmax(dim = 1)

    def forward(self, pair, side = None, pair_only = False):
        """
        side = 0(consistent pair), 1(negative pair), or list (each element represents the gold)
        """

        output_vec, _ = self.decomposition(pair, pair_only) # shape: (bat_size, 2)
        
        probs = self.softmax1(output_vec)

        # preds = torch.argmax(probs, dim = -1)

        bat_size = len(output_vec)

        if type(side) == int:
            gold = torch.tensor([side]*bat_size).to(self.params['device']) # shape: (2*bat_size)
        elif type(side) == list and len(side) == bat_size:
            gold = torch.tensor(side).to(self.params['device'])
        else:
            gold = None

        if gold == None:
            loss = None
        else:
            loss = self.loss_fn(output_vec, gold)
            gold = np.array(gold.detach().cpu()).tolist()
        # preds = np.array(preds.detach().cpu()).tolist()
        probs = np.array(probs.detach().cpu()).tolist()
        # preds = [float(p) for p in preds]
        # probs = [float(p) for p in probs]

        return loss, { "gold": gold, "probs": probs}