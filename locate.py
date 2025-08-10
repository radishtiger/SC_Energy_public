import yaml, json, math, os, datetime
import argparse

import torch
import torch.nn as nn

from trainer.modules import learn_energy_threshold, locate_energy

from tasks.dataset_loader import dataset_loader_fine_grained, transform_arbitrary_pairs_to_two_pairs, concat_arbitrary_pairs
from energynets.energynet import energynet
from energynets.decomposition.no_decomposition import no_decomposition_loader
from locate_and_edit.locate import locate
from utils import merge_dict, parser_add, params_add, draw_hist
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser = parser_add(parser)

args = parser.parse_args()
params = params_add(args)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
params['device'] = device
params['batch_size'] = 1
runname = f"{params['energynet']['repre_model']}-{params['energynet']['decomposition']}-{params['energynet']['loss_type']}-{params['pairwise']}-{str(params['energynet']['loss_fully_separate'])}"
if params['scratch'] == True:
    runname += '-scratch'
params['eval']['pairwise'] = params['pairwise']

task_dataset_candidates = [
    os.path.join("vqa", "lconvqa"),
    os.path.join("nli", "set_nli"),
    os.path.join("mixed", "lconvqa+set_nli"),
    os.path.join("mixed", "set_nli+lconvqa"),
]

model_path = None
for td in task_dataset_candidates:
    folder_path = os.path.join('results', td)
    if "job_id" in params and params['job_id'] !=0:
        folder_path = os.path.join(folder_path, str(params['job_id']))
        print("folder_path candidate:", folder_path)
    else:
        folder_path = os.path.join(folder_path, params['time_key'])
        print("folder_path candidate:", folder_path)

    try:
        file_lists = os.listdir(folder_path)
    except:
        continue
    for f in file_lists:
        if f.endswith('.pth'):
            model_path = os.path.join(folder_path, f)
            print("found model path:", model_path)
    if model_path != None:
        break

if model_path == None:
    raise f"No model in {folder_path}"
params['folder_path'] = folder_path
params['model_path'] = model_path


def main():
    lossNet = energynet(params).to(device)

    # eval_con_dataset_two_pairs = dataset_loader_fine_grained(params, 'eval', 'con', pairwise='two_pairs')
    # eval_incon_dataset_two_pairs = dataset_loader_fine_grained(params, 'eval', 'incon', pairwise='two_pairs')
    # eval_con_and_incon_dataset_two_pairs = dataset_loader_fine_grained(params, 'eval', 'con_and_incon', pairwise='two_pairs')
    # eval_incon_and_incon_dataset_two_pairs = dataset_loader_fine_grained(params, 'eval', 'incon_and_incon', pairwise='two_pairs')
    eval_con_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'eval', 'con', pairwise='arbitrary_pairs')
    eval_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'eval', 'incon', pairwise='arbitrary_pairs')

    
    # eval_con_and_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'eval', 'con_and_incon', pairwise='arbitrary_pairs')
    # eval_incon_and_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(params, 'eval', 'incon_and_incon', pairwise='arbitrary_pairs')
    # two_pairs_dataset, two_pairs_names, two_pairs_set_sizes = transform_arbitrary_pairs_to_two_pairs([eval_con_dataset_arbitrary_pairs, eval_incon_dataset_arbitrary_pairs])
    concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs([eval_con_dataset_arbitrary_pairs, eval_incon_dataset_arbitrary_pairs], concat_num=2)

    eval_steps_names = ['con', 'incon'] + concat2_names
    eval_datasets = [eval_con_dataset_arbitrary_pairs, eval_incon_dataset_arbitrary_pairs
            ]  + concat2_dataset

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
    # two_pairs_dataset, two_pairs_names, two_pairs_set_sizes = transform_arbitrary_pairs_to_two_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs])
    concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=2)
    concat3_dataset, concat3_names, concat3_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=3)
    concat4_dataset, concat4_names, concat4_set_sizes = concat_arbitrary_pairs([test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs], concat_num=4)

    test_steps_names = ['con', 'incon'] + concat2_names + concat3_names + concat4_names
    test_datasets = [test_con_dataset_arbitrary_pairs, test_incon_dataset_arbitrary_pairs
            ] + concat2_dataset + concat3_dataset + concat4_dataset

    if params['energynet']['decomposition'] == 'no':
        
        eval_steps = [
            no_decomposition_loader(e, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
                for e in eval_datasets
        ]

        test_steps = [
            no_decomposition_loader(e, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
                for e in test_datasets
        ]


    eval_steps_labels = []
    for n in eval_steps_names:
        if 'incon' in n:
            eval_steps_labels.append(1)
        else:
            eval_steps_labels.append(0)

    eval_acc = [0 for _ in range(len(test_steps_names))]
    
    if params['scratch'] == False:
        lossNet.load_state_dict(torch.load(params['model_path'], map_location=device)['state_dict'])
        if 'threshold' in torch.load(params['model_path'], map_location=device):
            lossNet.threshold = torch.load(params['model_path'], map_location=device)['threshold']
    
    if lossNet.threshold == 0.5:
        print("Start threshold learning")
        acc_thre, acc = learn_energy_threshold(lossNet, eval_steps, eval_steps_labels)
        print(f"threshold: {acc_thre}, with accuracy {acc}")
        torch.save({
                    "state_dict":lossNet.state_dict(),
                    "threshold": acc_thre
                    }, params['model_path'])
        assert acc_thre == lossNet.threshold

    locateNet = locate(params, lossNet)
    
    total_info_for_hist = []
    column_names_for_hist = ['e_val', "side", "threshold","datatype"]
    result = {}
    set_sizes = {}
    correct_by_sizes = {}
    time_seconds_by_sizes =  {}

    print("test data names:")
    for i, n in enumerate(test_steps_names):
        print(f"{i+1}: {n}")
    print()


    for es_idx, eval_dataloader in enumerate(test_steps):
        # if es_idx+1 != 2:
        #     continue
        names = test_steps_names[es_idx]

        print(f"[{es_idx+1}] - {names}")

        locate_result = locate_energy(locateNet, eval_dataloader, device, params)
                
        eval_acc[es_idx] = locate_result['accuracy']
        print(f"[{es_idx+1}] {names}\n\taccuracy:{locate_result['accuracy']:.3f}, precision:{locate_result['precision']:.3f}, recall:{locate_result['recall']:.3f}, f1:{locate_result['f1']:.3f}")

        # violin plot prepare
        # eval_this_step_name = test_steps_names[es_idx].split("-")
        # pos_name = eval_this_step_name[0]
        # neg_name = eval_this_step_name[1]
        # pair_num = eval_this_step_name[2]
        
        result = merge_dict(result, {
        f"locate_acc-{es_idx+1}-{test_steps_names[es_idx]}": locate_result.get('accuracy', 0),
        # 'locate_accuracy': locate_result.get('accuracy', 0)
            })
        # locate_result = locate_energy(locateNet, pos_eval_dataloader, neg_eval_dataloader, device, params)
        # for size in locate_result['set_sizes']:
        #     if size not in set_sizes:
        #         set_sizes[size] = 0
        #     if size not in correct_by_sizes:
        #         correct_by_sizes[size] = 0
        #     if size not in time_seconds_by_sizes:
        #         time_seconds_by_sizes[size] = 0
            
        #     set_sizes[size] += locate_result['set_sizes'][size]
        #     correct_by_sizes[size] += locate_result['correct_by_sizes'][size]
        #     time_seconds_by_sizes[size] += locate_result['time_seconds_by_sizes'][size]

    for size in set_sizes.keys():
        result[f"data_proportion_size_{size}"] = set_sizes[size] / sum([v for v in set_sizes.values()])
        result[f"accuracy_size_{size}"] = correct_by_sizes[size] / set_sizes[size]
        result[f"time_seconds_size_{size}"] = time_seconds_by_sizes[size] / set_sizes[size]
        
    # result['eval_two_pairs_acc'] = sum([eval_acc[i] for i in [2, 3]])/2
    result['eval_arbitrary_pairs_acc'] = sum([eval_acc[i] for i in [0, 1]])/2
    result['eval_concat2_acc'] = sum([eval_acc[i] for i in [2,3,4]])/3
    result['eval_concat3_acc'] = sum([eval_acc[i] for i in [5,6,7,8]])/4
    result['eval_concat4_acc'] = sum([eval_acc[i] for i in [9,10,11,12,13]])/4
    result['eval_total_acc'] = sum(eval_acc)/len(test_steps_names)
    
    result = {k:result[k] for k in sorted(result.keys())}
    
    print("====")
    for key, val in result.items():
        print(f"key:{key}")
        print(f"val:{val}")
        print("--")

    if len(total_info_for_hist)!=0:
        paths = ["evalate_hist.png", "evalate_hist.svg"]
        save_paths = [os.path.join(folder_path, p) for p in paths]
        draw_hist(total_info_for_hist, column_names_for_hist, save_paths)

    print("====")
    print(f"Evaluation Ended.\nFolder path: {folder_path}")
    with open(os.path.join(folder_path, f"{params['task']}_{params['dataset']}_locate_result.json"), 'w') as f:
        json.dump(result, f, indent = '\t')

main()