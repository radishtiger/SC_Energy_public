import yaml, math, os, datetime, json, time
import argparse

import torch

from trainer.modules import evaluate_baseline, locate_baseline

from tasks.dataset_loader import dataset_loader_fine_grained, transform_arbitrary_pairs_to_two_pairs, concat_arbitrary_pairs
from baselines.baseline_model import baseline_model
from baselines.LLM.lm_loader import lm_loader

from utils import merge_dict, parser_add, params_add

parser = argparse.ArgumentParser()
parser = parser_add(parser)

args = parser.parse_args()
params = params_add(args)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
params['device'] = device
params['batch_size'] = 1

runname = params['pairwise']

folder_path = os.path.join('results', params['task'], params['dataset'])
folder_path = os.path.join(folder_path, 
                           params['baseline']['type'], 
                           params['baseline']['model'],
                           str(params['baseline']['shot_num'])+'shot')

record_path = os.path.join(folder_path, f"{runname}.json")

params['folder_path'] = folder_path
params['record_path'] = record_path



def main():

    # test_con_dataset_two_pairs = dataset_loader_fine_grained(params, 'test', 'con', pairwise='two_pairs')
    # test_incon_dataset_two_pairs = dataset_loader_fine_grained(params, 'test', 'incon', pairwise='two_pairs')
    # test_con_and_incon_dataset_two_pairs = dataset_loader_fine_grained(params, 'test', 'con_and_incon', pairwise='two_pairs')
    # test_incon_and_incon_dataset_two_pairs = dataset_loader_fine_grained(params, 'test', 'incon_and_incon', pairwise='two_pairs')
    test_con_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'test', 'con', pairwise='arbitrary_pairs')
    test_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'test', 'incon', pairwise='arbitrary_pairs')
    test_con_dataset_arbitrary_pairs.dataset = [t for t in test_con_dataset_arbitrary_pairs.dataset if len(t) >=4]
    test_incon_dataset_arbitrary_pairs.dataset = [t for t in test_incon_dataset_arbitrary_pairs.dataset if len(t) >=4]
    # test_con_and_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'test', 'con_and_incon', pairwise='arbitrary_pairs')
    # test_incon_and_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'test', 'incon_and_incon', pairwise='arbitrary_pairs')
    two_pairs_dataset, two_pairs_names, two_pairs_set_sizes = transform_arbitrary_pairs_to_two_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs])
    concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=2)
    concat3_dataset, concat3_names, concat3_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=3)
    concat4_dataset, concat4_names, concat4_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=4)

    test_steps_names = ['con', 'incon'] + concat2_names+ concat3_names+ concat4_names
    test_datasets = [test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs
            ] + concat2_dataset + concat3_dataset + concat4_dataset
    test_steps = [
            lm_loader(e, params = params).get_loader()
                for e in test_datasets
        ]

    model = baseline_model(params, 'locate')
    results = {}
    time_seconds = {}
    set_sizes = {}
    accuracy_by_sizes = {}
    for es_idx, data_loader in enumerate(test_steps):
        data_name = test_steps_names[es_idx]
        # if es_idx+1 !=5:
            # continue
        if es_idx+1 <= 11:
            continue

        print(f"=========== {es_idx+1} - {data_name} ===========")
        locate_result = locate_baseline(model, data_loader, device = device, params = params)
        results[f"locate_accuracy-{es_idx+1}-{data_name}"] =  locate_result.get('accuracy', 0)
        results[f"locate_precision-{es_idx+1}-{data_name}"] =  locate_result.get('precision', 0)
        results[f"locate_recall-{es_idx+1}-{data_name}"] =  locate_result.get('recall', 0)
        results[f"locate_f1-{es_idx+1}-{data_name}"] =  locate_result.get('f1', 0)
        print(f"[{es_idx+1}] {data_name}\n\taccuracy:{locate_result['accuracy']:.3f}, precision:{locate_result['precision']:.3f}, recall:{locate_result['recall']:.3f}, f1:{locate_result['f1']:.3f}")

        # results[f"time_seconds-{es_idx+1}-{data_name}"] =  locate_result['time_seconds']
        # results[f"set_sizes-{es_idx+1}-{data_name}"] =  locate_result['set_sizes']
        # results[f"accuracy_by_sizes-{es_idx+1}-{data_name}"] = locate_result['accuracy_by_sizes']

        # for s in locate_result['set_sizes']:
        #     if s not in set_sizes:
        #         set_sizes[s] = 0
        #     set_sizes[s] += locate_result['set_sizes'][s]
        # for s in locate_result['time_seconds']:
        #     if s not in time_seconds:
        #         time_seconds[s] = 0
        #     time_seconds[s] += locate_result['time_seconds'][s] * locate_result['set_sizes'][s]
        # for s in locate_result['accuracy_by_sizes']:
        #     if s not in accuracy_by_sizes:
        #         accuracy_by_sizes[s] = 0
        #     accuracy_by_sizes[s] += locate_result['accuracy_by_sizes'][s] * locate_result['set_sizes'][s]

    # for s in time_seconds.keys():
    #     time_seconds[s] /= set_sizes[s]
    # for s in accuracy_by_sizes.keys():
    #     accuracy_by_sizes[s] /= set_sizes[s]
    # results['set_sizes'] = {s:set_sizes[s] for s in sorted(set_sizes)}
    # results['time_seconds'] = {s:time_seconds[s] for s in sorted(time_seconds)}
    # results['accuracy_by_sizes'] = {s:accuracy_by_sizes[s] for s in sorted(accuracy_by_sizes)}

    results = {r:results[r] for r in sorted(results)}


    total_time_seconds = {}

    os.makedirs(params['folder_path'], exist_ok=True)

    try:
        with open(params['record_path'], 'r') as f:
            record = json.load(f)
    except:
        record = {}
    if "locate" not in record:
        record["locate"] = {}
    record["locate"][params['baseline']['locate_type']] = results
    # with open(params['record_path'], 'w') as f:
        # json.dump(record, f, indent = 4)

    print("================")
    print("Evaluation Ended.")
    print(f"{record_path}")
    for key, val in record.items():
        print(f"[{key}]")
        for k, v in val.items():
            print(f"\t{k}: {v}")
    print("================")

main()