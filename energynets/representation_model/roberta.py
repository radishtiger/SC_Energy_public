import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers import RobertaTokenizerFast


class roberta(nn.Module):

    def __init__(self, params):
        super(roberta, self).__init__()
        self.RoBERTa = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.ReLU = nn.ReLU()
        self.params = params
        self.output_form = self.params['energynet']['output_form']
        self.linear1 = None
        self.sigmoid = nn.Sigmoid()
        self.initialize()
        
        # special tokens
        # single sequence: <s> X </s>
        # pair of sequences: <s> A </s></s> B </s>
        
    def forward(self, inputs):
        """forward function

        Args:
            inputs (tuple): two elements, 'input_ids', 'attention_mask'
            'input_ids': LongTensor. shape=(bat_size, seq_len)
            'mask': LongTensor. shape=(bat_size, seq_len)

        Returns:
            output: prediction of relationship . shape=(bat_size, 1)
        """
        # print("inputs")
        # print(type(inputs), len(inputs))
        if len(inputs)==2:
            input_tensor, mask = inputs['input_ids'], inputs['attention_mask']
            input_tensor, mask = torch.tensor(input_tensor).to(self.params['device']), torch.tensor(mask).to(self.params['device'])
        else:
            input_tensor = inputs
            mask = None
    
        if mask == None:
            mask = torch.ones_like(input_tensor)
            
            
        output_all = self.RoBERTa(input_ids = input_tensor,
                        attention_mask = mask)
        # print(f"output_all len:, {len(output_all)}")
        # print(f"output_all['hidden_states']:, {output_all['hidden_states'][0].shape}")
        output = output_all[0][:,0,:]
        # print("output shape:", output.shape)
        output = torch.reshape(output, (-1, 768))
        norms = torch.norm(output, dim = -1).detach().unsqueeze(-1)
        output = output / norms
        output = self.linear1(output)
        
        if self.output_form == 'real_num':
            return (output)
        elif self.output_form == '2dim_vec':
            return output
        
    def initialize(self, turn_off_LM_grad = False):
        
        if turn_off_LM_grad:
            for param in self.RoBERTa.embeddings.parameters():
                param.requires_grad = False
            for param in self.RoBERTa.encoder.parameters():
                param.requires_grad = False


        # depending on the output dimension, define the corresponding final layer.
        if self.output_form == 'real_num':
            self.linear1 = nn.Linear(768, 1) # Regard output as a compatibility score (a single real value)
        elif self.output_form == '2dim_vec':
            self.linear1 = nn.Linear(768, 2) # Regard output as a classification result = (consistent, in_consistent)
        