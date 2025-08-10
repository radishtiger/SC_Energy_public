import os
import json 
import argparse
from pathlib import Path
from copy import deepcopy
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset


def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def strip_and_but(text):
    if text.startswith('And') or text.startswith('But'):
        text = text[4:]
        text = text[0].upper() + text[1:]
    return text

def gen_and(sentence):
    sentence = sentence.lower().rstrip('.')
    return f"And {sentence}."

def prepare_split_conjunction_data(la_data_path, force_overwrite=False):
    
    save_path = la_data_path.split('.json')[0] + '_split_conj.jsonl'
    
    if os.path.exists(save_path) and not force_overwrite:
        print(f"Loading split conjunction data from {save_path}")
        with open(save_path, 'r') as f:
            la_split_conj_data = [json.loads(line) for line in f.readlines()]
        return la_split_conj_data

    la_data = load_data(la_data_path)['instances']

    la_split_conj_data = []
    for pair in la_data:
        for i, attack in enumerate(pair['attack_samples']):
            if attack['used_rule'] in ['modus_tollens',
                                    'disjunctive_syllogism',
                                    'constructive_dilemma',
                                    'destructive_dilemma',
                                    'bidirectional_dilemma']:
                attack_data = {}
                attack_data['pairID'] = attack['pairID']
                attack_data['used_rule'] = attack['used_rule']
                attack_data['label'] = attack['label']
                attack_data['premise'] = [strip_and_but(x) + '.' if x[-1] != '.' else strip_and_but(x) for x in attack['premise'].split('. ')]
                attack_data['original_premise'] = attack['premise']
                attack_data['hypothesis'] = attack['hypothesis']
                assert sum([x[0].islower() for x in attack_data['premise']]) == 0
                la_split_conj_data.append(attack_data)

    with open(save_path, 'w') as f:
        f.writelines([json.dumps(x) + '\n' for x in la_split_conj_data])
    
    return la_split_conj_data


class CollatorForLogicAttack:
    def __init__(self, tokenizer, nli_model):
        self.tokenizer = tokenizer
        self.nli_model = nli_model
    
    def collate_fn(self, batch):
        premise = [x['original_premise'] for x in batch]
        hypothesis = [x['hypothesis'] for x in batch]
        return premise, hypothesis

# define dataloader for regular NLI data (no explicit coordinating conjunctions like "And" or "But")
class CollatorForRegularNLI:
    def __init__(self, tokenizer, nli_model, dataloader_type):
        self.tokenizer = tokenizer
        self.nli_model = nli_model
        self.dataloader_type = dataloader_type
    
    def collate_fn(self, batch):
        if self.dataloader_type == 'regular':
            premise = [' '.join(x['premise']) for x in batch]
        elif self.dataloader_type == 'regular_with_and':
            premise = [x['premise'][0] + ' ' + ' '.join([gen_and(x_i) for x_i in x['premise'][1:]]) for x in batch]
        hypothesis = [x['hypothesis'] for x in batch]
        return premise, hypothesis
    
# define dataloader for split conjunction data
class CollatorForSplitConjunction:
    def __init__(self, tokenizer, nli_model):
        self.tokenizer = tokenizer
        self.nli_model = nli_model
    
    def collate_fn(self, batch):
        if self.nli_model in ['ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', 
                              'cross-encoder/nli-roberta-base']:
            sep_indicator = self.tokenizer.sep_token * 2
        else:
            sep_indicator = self.tokenizer.sep_token
        premise = [sep_indicator.join(x['premise']) for x in batch]
        hypothesis = [x['hypothesis'] for x in batch]
        return premise, hypothesis



        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--logic_attack_data_path', type=str, default='/data/hyeryung/set_consistency_energy/datasets/logicattack/logic_attacks_snli_test.json')
    parser.add_argument('--nli_model', type=str, default='cross-encoder/nli-deberta-v3-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_root_dir', type=str, default='tasks/nli/split_conjunction_ablation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataloader_type', choices=['logic_attack','regular', 'regular_with_and', 'split_conjunction'], default='regular')
    args = parser.parse_args()
    
    model_nickname = args.nli_model.replace('-', '_').replace('/', '_')
    output_dir = Path(args.output_root_dir) / model_nickname / args.dataloader_type
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model)
    model = model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    label_mapping = model.config.id2label

    
    la_split_conj_data = prepare_split_conjunction_data(args.logic_attack_data_path)
    dataset = Dataset.from_list(la_split_conj_data)
    if args.dataloader_type == 'regular' or args.dataloader_type == 'regular_with_and':
        collator = CollatorForRegularNLI(tokenizer, args.nli_model, args.dataloader_type)
    elif args.dataloader_type == 'split_conjunction':
        collator = CollatorForSplitConjunction(tokenizer, args.nli_model)
    elif args.dataloader_type == 'logic_attack':
        collator = CollatorForLogicAttack(tokenizer, args.nli_model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate_fn)
    
    predictions = []
    predictions_c = []
    sample_inputs = []
    for i, batch in enumerate(dataloader):
        
        features = tokenizer(batch[0], batch[1],  
                            padding=True, 
                            truncation=True, 
                            return_tensors="pt")
        features = features.to(args.device)
        if i == 0:
            sample_inputs = tokenizer.batch_decode(features.input_ids, skip_special_tokens=False)
            print("Example of input ids")
            print(sample_inputs)
            print("-"*50)
        with torch.no_grad():
            scores = model(**features).logits
            probas = scores.softmax(dim=-1).tolist()
            labels = [label_mapping[score_max.item()] for score_max in scores.argmax(dim=1)]
            
            predictions_c.extend(labels)
            predictions.extend(probas)
            
    acc = sum([x == 'entailment' for x in predictions_c])/ len(predictions_c)
    
    with open(os.path.join(output_dir, 'predictions_c.txt'), 'w') as f:
        f.writelines([x+ '\n' for x in predictions_c])
    with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
        f.writelines([','.join([str(y) for y in x])+ '\n' for x in predictions])
    with open(os.path.join(output_dir, 'sample_input.txt'), 'w') as f:
        f.writelines([x + '\n' for x in sample_inputs[:1]])
        
    
    if not os.path.exists(os.path.join(args.output_root_dir, 'performances.csv')):
        with open(os.path.join(args.output_root_dir, 'performances.csv'), 'w') as f:
            f.write("nli_model,data_loader_type,accuracy\n")
            f.write(f"{args.nli_model},{args.dataloader_type},{acc}\n")
    else:
        with open(os.path.join(args.output_root_dir, 'performances.csv'), 'a') as f:
            f.write(f"{args.nli_model},{args.dataloader_type},{acc}\n")
        



    
    
    
    