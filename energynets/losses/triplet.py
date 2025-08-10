import math
import torch
import torch.nn as nn

class triplet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.decomposition = None
        self.full_separation = self.params['energynet']['loss_fully_separate']
        self.full_separation = False
        self.subset_ordering = self.params['energynet']['loss_subset_ordering']
        self.incon_incon_ordering = self.params['energynet']['loss_incon_incon_ordering']
        self.con_incon_ordering = self.params['energynet']['loss_con_incon_ordering']

        self.weight_full_separation = self.params['energynet']['weight_fully_separate']
        self.weight_subset_ordering = self.params['energynet']['weight_subset_ordering']
        self.weight_incon_incon_ordering = self.params['energynet']['weight_incon_incon_ordering']
        self.weight_con_incon_ordering = self.params['energynet']['weight_con_incon_ordering']
        
        self.margin = self.params['energynet']['triplet']['margin']
        self.full_separation_margin = self.params['energynet']['triplet']['fully_separate_margin']
        self.ReLU = nn.ReLU()
        
    def forward(self, pos_pair, neg_pair = None):
        """
        pos_pair, neg_pair : [xypairs, inconsistent_pair_indices]
            xypairs: list of xypairs(=set).
                list length = batch_size
            inconsistent_pair_indiecs: list of list.
                Outer list length = batch_size
                Inner list: indices of inconsistent pairs
        """
        e_pos, hidden_states = self.decomposition(pos_pair) # get the energy value for true pair
        e_neg, hidden_states = self.decomposition(neg_pair) # get the energy value for false pair

        if e_pos.shape != e_neg.shape:
            loss = self.ReLU(e_pos[:1] - e_neg[:1] + self.margin)
        else:
            loss = self.ReLU(e_pos - e_neg + self.margin) # triplet loss for energtNet
        loss = torch.sum(loss) / len(loss)

        if self.full_separation == True:
            loss += self.full_separation_loss(e_pos, e_neg)
        if self.subset_ordering == True:
            loss += self.subset_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)
        if (self.incon_incon_ordering == True) and (self.con_incon_ordering == True):
            loss += self.ci_ii_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)
        else:
            if self.incon_incon_ordering == True:
                loss += self.incon_incon_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)
            elif self.con_incon_ordering == True:
                loss += self.con_incon_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)

        e_pos = torch.sum(e_pos) / len(e_pos)
        e_neg = torch.sum(e_neg) / len(e_neg)

        return loss, {"e_pos": e_pos, "e_neg": e_neg}
    
    # def forward_one_side(self, pair):
    #     """
    #     pos_pair, neg_pair : [xypairs, inconsistent_pair_indices]
    #         xypairs: list of xypairs(=set).
    #             list length = batch_size
    #         inconsistent_pair_indiecs: list of list.
    #             Outer list length = batch_size
    #             Inner list: indices of inconsistent pairs
    #     """
    #     e_pos, hidden_states = self.decomposition(pair) # get the energy value for true pair


    #     if self.full_separation == True:
    #         loss += self.full_separation_loss(e_pos, e_neg)
    #     if self.subset_ordering == True:
    #         loss += self.subset_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)
    #     if (self.incon_incon_ordering == True) and (self.con_incon_ordering == True):
    #         loss += self.ci_ii_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)
    #     else:
    #         if self.incon_incon_ordering == True:
    #             loss += self.incon_incon_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)
    #         elif self.con_incon_ordering == True:
    #             loss += self.con_incon_ordering_loss(e_pos, e_neg, pos_pair, neg_pair)

    #     e_pos = torch.sum(e_pos) / len(e_pos)
    #     e_neg = torch.sum(e_neg) / len(e_neg)

    #     return loss, {"e_pos": e_pos, "e_neg": e_neg}
    
    def full_separation_loss(self, e_pos, e_neg):
        """
        [full separation loss]
        For every consistent set “s” and every inconsistent set “c”,
            Energy(s) - Energy(c) <= 0
        must be satisfied.
        To inject this principle, we add a loss function for every comparable consistent-inconsistent sets "in a batch".
        
        [shape info]
        e_pos = (bat_size, )
        e_neg = (bat_size, )
        """
        # step 1. Expand energy values
        e_pos_extended = e_pos.expand(len(e_pos), len(e_pos)).permute(1, 0)
        e_neg_extended = e_neg.expand(len(e_neg), len(e_neg))

        # step 2. Subtraction
        sep_loss = self.ReLU(e_pos_extended - e_neg_extended + self.full_separation_margin)

        return self.weight_full_separation * torch.sum(sep_loss) / (math.prod(sep_loss.shape))

    def subset_ordering_loss(self, e_pos, e_neg, pos_pair, neg_pair):
        
        """
        [subset ordering loss]
        Let A, B be two non-empty sets.
            If A is_subset B, then E(A) <= E(B)
        must be satisfied.

        At the implementation level, this loss is composed of two parts:
            (1) subset ordering - extension
            (2) subset ordering - division
        
        (1): For two sets "s1", "s2" given as data, we construct a new set "s1 union s2" to compare the value of energy.
        (2): For a set "S" given as data, we divide "S" as "S = A union B" to compare the value of energy.
        """

        loss = 0

        loss_extension = self.subset_ordering_loss_extension(e_pos, e_neg, pos_pair, neg_pair)
        loss_division = 0 # Not implemented yet

        loss = loss + loss_extension + loss_division
        return self.weight_subset_ordering * loss

    def subset_ordering_loss_extension(self, e_pos, e_neg, pos_pair, neg_pair):
        """
        We use in-batch construction.

        Construct a matrix M:
            M_{i,j} = Concat(s_{i}, s_{j})

        """

        loss = 0

        pos_pair = [p[0] for p in pos_pair]
        neg_pair = [p[0] for p in neg_pair]

        # pos pair
        concat_input_matrix = self.concat_pair(pos_pair, pos_pair) # list of list of str, shape: (batch_size, batch_size)
        energy_matrix = self.concat_energy(concat_input_matrix)

        e_row = e_pos.expand(len(e_pos), len(e_pos))
        e_col = e_row.permute(1, 0)
        loss = loss + self.ReLU(e_row + e_col - 2*energy_matrix) * (1- torch.eye(len(e_row)).to(self.params['device']))

        # neg pair
        concat_input_matrix = self.concat_pair(neg_pair, neg_pair) # list of list of str, shape: (batch_size, batch_size)
        energy_matrix = self.concat_energy(concat_input_matrix)

        e_row = e_neg.expand(len(e_neg), len(e_neg))
        e_col = e_row.permute(1, 0)
        loss = loss + self.ReLU(e_row + e_col - 2*energy_matrix) * (1- torch.eye(len(e_row)).to(self.params['device']))

        return 0.25 * torch.sum(loss) / (math.prod(loss.shape) - len(loss))

    def incon_incon_ordering_loss(self, e_pos, e_neg, pos_pair, neg_pair):

        loss = 0

        neg_pair = [p[0] for p in neg_pair]

        # neg pair
        concat_input_matrix = self.concat_pair(neg_pair, neg_pair) # list of list of str, shape: (batch_size, batch_size)
        energy_matrix = self.concat_energy(concat_input_matrix)

        e_row = e_neg.expand(len(e_neg), len(e_neg))
        e_col = e_row.permute(1, 0)
        loss = loss + self.ReLU(e_row + e_col - 2*energy_matrix) * (1- torch.eye(len(e_row)).to(self.params['device']))

        return self.weight_incon_incon_ordering * 0.5 * torch.sum(loss) / (math.prod(loss.shape) - len(loss))
    
    def con_incon_ordering_loss(self, e_pos, e_neg, pos_pair, neg_pair):

        loss = 0

        pos_pair = [p[0] for p in pos_pair]
        neg_pair = [p[0] for p in neg_pair]

        # con  < con_incon 
        concat_input_matrix = self.concat_pair(pos_pair, neg_pair) # list of list of str, shape: (batch_size, batch_size)
        energy_matrix = self.concat_energy(concat_input_matrix)

        e_row = e_pos.expand(len(e_pos), len(e_pos))
        e_col = e_row.permute(1, 0)
        loss = loss + self.ReLU(e_row + e_col - 2*energy_matrix) * (1- torch.eye(len(e_row)).to(self.params['device']))

        # con_incon < incon
        e_row = e_neg.expand(len(e_neg), len(e_neg))
        e_col = e_row.permute(1, 0)
        loss = loss + self.ReLU(2*energy_matrix - e_row - e_col) * (1- torch.eye(len(e_row)).to(self.params['device']))


        return self.weight_con_incon_ordering * 0.25 * torch.sum(loss) / (math.prod(loss.shape) - len(loss))

    def ci_ii_ordering_loss(self, e_pos, e_neg, pos_pair, neg_pair):

        loss = 0

        pos_pair = [p[0] for p in pos_pair]
        neg_pair = [p[0] for p in neg_pair]

        # con  < con_incon 
        concat_input_matrix = self.concat_pair(pos_pair, neg_pair) # list of list of str, shape: (batch_size, batch_size)
        energy_matrix = self.concat_energy(concat_input_matrix)

        e_row = e_pos.expand(len(e_pos), len(e_pos))
        e_col = e_row.permute(1, 0)
        loss = loss + self.weight_con_incon_ordering * self.ReLU(e_row + e_col - 2*energy_matrix) * (1- torch.eye(len(e_row)).to(self.params['device']))

        # con_incon < incon
        e_row = e_neg.expand(len(e_neg), len(e_neg))
        e_col = e_row.permute(1, 0)
        loss = loss + self.weight_con_incon_ordering * self.ReLU(2*energy_matrix - e_row - e_col) * (1- torch.eye(len(e_row)).to(self.params['device']))

        # incon < incon_incon
        concat_input_matrix = self.concat_pair(neg_pair, neg_pair) # list of list of str, shape: (batch_size, batch_size)
        energy_matrix = self.concat_energy(concat_input_matrix)
        loss = loss + self.weight_incon_incon_ordering * self.ReLU(e_row + e_col - 2*energy_matrix) * (1- torch.eye(len(e_row)).to(self.params['device']))

        return (1/6) * torch.sum(loss) / (math.prod(loss.shape) - len(loss))

    def concat_pair(self, pair1, pair2):
        if self.params['energynet']['decomposition'] == 'no':
            return self.concat_pair_no_decomposition(pair1, pair2)
        
        else:
            raise NotImplementedError

    def concat_pair_no_decomposition(self, pair1, pair2):
        """
        pair: list of xypairs(= a set of xy pairs).
            list length = batch_size
        """
        cls_token_len = len(self.decomposition.tokenizer.cls_token)
        # sep_token_len = len(self.decomposition.sep_token)

        concatted_matrix = []
        for idx1, batch1 in enumerate(pair1):
            concatted_matrix.append([
                pair1[idx1] + p[cls_token_len:] for p in pair2
            ])
            
        # print(concatted_matrix[0][1])

        return concatted_matrix
    
    def concat_energy(self, input_matrix):
        """
        input_matrix: list of list of str, shape: (batch_size, batch_size)
        """

        e_mat = [self.decomposition(m)[0] for m in input_matrix]

        return torch.stack(e_mat, dim = 0)