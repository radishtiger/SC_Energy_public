import torch.nn as nn
from torch.utils.data import DataLoader

class lm_loader():
    def __init__(self, dataset = None, tokenizer = None, params = None):
        super().__init__()
        self.dataset_text = dataset.dataset
        self.params = params
        self.tokenizer = tokenizer
        self.dataset_decompose = []        
    
    def decompose_transform(self):
        if self.params['dataset'] in {"lconvqa"}:
            for data in self.dataset_text:
                pairs = []
                consistency_list = []
                for i, xy_pair in enumerate(data):
                    one_pair_text = f"question: {xy_pair[0]}, answer: {xy_pair[1]}."
                    if xy_pair[2] == False:
                        consistency_list.append(i)
                    pairs.append(one_pair_text)
                self.dataset_decompose.append([pairs, consistency_list])
            self.dataset_text = None
        
        elif self.params['dataset'] in {"set_nli"}:
            for data in self.dataset_text:
                # print("\ndata:")
                # print(data)
                # print("====")
                pairs = []
                consistency_list = []
                for i, xy_pair in enumerate(data):
                    # print("xy_pair")
                    # print(xy_pair)
                    # print("--")
                    one_pair_text = xy_pair[0]
                    if xy_pair[1] == False:
                        consistency_list.append(i)
                    pairs.append(one_pair_text)
                self.dataset_decompose.append([pairs, consistency_list])
            self.dataset_text = None

    def get_loader(self):
        self.decompose_transform()
        return DataLoader(self.dataset_decompose, 1, shuffle = False, collate_fn = collate_fn)

def collate_fn(batch):
    return batch