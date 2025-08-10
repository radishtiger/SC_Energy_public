import yaml, json, math, os, datetime
import argparse
import glob
from pathlib import Path
import pickle

import torch
import torch.nn as nn

from trainer.modules import learn_energy_threshold, locate_energy
# dataset_loader_fine_grained / concat_* 사용하지 않고, 사전 저장된 피클을 직접 로드합니다.
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
    if "job_id" in params and params['job_id'] != 0:
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
    if model_path is not None:
        break

if model_path is None:
    raise RuntimeError(f"No model in {folder_path}")
params['folder_path'] = folder_path
params['model_path'] = model_path


# =========================
# set_consistency_dataset 로더 유틸
# =========================
def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    set_consistency_dataset/{dataset_name}/ 경로의 피클 파일 경로를 반환.
    파일명 규칙: {dataset_name}_{split}_{NAME}_dataset.pickle
      예) lconvqa_eval_C_dataset.pickle, lconvqa_test_CI_dataset.pickle
    name 인자는 "eval_C", "test_CI" 등 split 접두사를 포함한 문자열을 기대.
    """
    base = Path("set_consistency_dataset") / dataset_name
    fname = f"{dataset_name}_{name}_dataset.pickle"
    return base / fname


def list_split_pickles(dataset_name: str, split: str):
    """
    split ∈ {"eval","test"} 에 대해
    set_consistency_dataset/{dataset_name}/{dataset_name}_{split}_*_dataset.pickle 수집/정렬.
    반환: (names_sorted, paths_sorted)
      names_sorted: ["C","I","CC","CI","II","CCC", ...]
    정렬: "C","I" 먼저, 이후 길이/사전순.
    """
    base = Path("set_consistency_dataset") / dataset_name
    patt = str(base / f"{dataset_name}_{split}_*_dataset.pickle")
    paths = sorted(glob.glob(patt))

    names = []
    for p in paths:
        fn = Path(p).name  # e.g., lconvqa_eval_C_dataset.pickle
        prefix = f"{dataset_name}_{split}_"
        if not (fn.startswith(prefix) and fn.endswith("_dataset.pickle")):
            raise RuntimeError(f"Unexpected filename: {fn}")
        name = fn[len(prefix):-len("_dataset.pickle")]  # "C","I","CI","CCC",...
        names.append(name)

    def _order_key(x):
        if x == "C": return (0, 0, x)
        if x == "I": return (0, 1, x)
        return (1, len(x), x)

    names_sorted = sorted(names, key=_order_key)
    paths_sorted = [_pkl_path(dataset_name, split, f"{split}_{nm}") for nm in names_sorted]
    return names_sorted, paths_sorted


def load_split_pickles_as_datasets(dataset_name: str, split: str):
    names, paths = list_split_pickles(dataset_name, split)
    datasets = []
    for nm, p in zip(names, paths):
        if not p.exists():
            raise FileNotFoundError(f"Missing pickle: {p}")
        with open(p, "rb") as f:
            ds = pickle.load(f)
        datasets.append(ds)
    return names, datasets


def to_display_name(nm: str) -> str:
    # "C"→"con", "I"→"incon", 나머지는 그대로("CI","CC" 등)
    return "con" if nm == "C" else ("incon" if nm == "I" else nm)


def main():
    lossNet = energynet(params).to(device)
    dataset_name = params['dataset']

    # eval: C, I, 그리고 concat2 (파일명에서 자동)
    eval_raw_names, eval_datasets = load_split_pickles_as_datasets(dataset_name, "eval")
    eval_steps_names = [to_display_name(n) for n in eval_raw_names]

    # test: C, I, 그리고 concat2/3/4 (파일명에서 자동)
    test_raw_names, test_datasets = load_split_pickles_as_datasets(dataset_name, "test")
    test_steps_names = [to_display_name(n) for n in test_raw_names]

    if params['energynet']['decomposition'] == 'no':
        eval_steps = [
            no_decomposition_loader(e, params=params, tokenizer=lossNet.representation_model.tokenizer).get_loader(split='eval')
            for e in eval_datasets
        ]
        test_steps = [
            no_decomposition_loader(e, params=params, tokenizer=lossNet.representation_model.tokenizer).get_loader(split='eval')
            for e in test_datasets
        ]

    # threshold 학습 라벨: 이름에 'I'가 하나라도 포함되면 incon(1), 아니면 con(0)
    eval_steps_labels = [1 if ('I' in n) else 0 for n in eval_raw_names]

    eval_acc = [0 for _ in range(len(test_steps_names))]

    if params['scratch'] == False:
        state = torch.load(params['model_path'], map_location=device)
        lossNet.load_state_dict(state['state_dict'])
        if 'threshold' in state:
            lossNet.threshold = state['threshold']

    if lossNet.threshold == 0.5:
        print("Start threshold learning")
        acc_thre, acc = learn_energy_threshold(lossNet, eval_steps, eval_steps_labels)
        print(f"threshold: {acc_thre}, with accuracy {acc}")
        torch.save({
                    "state_dict": lossNet.state_dict(),
                    "threshold": acc_thre
                    }, params['model_path'])
        assert acc_thre == lossNet.threshold

    locateNet = locate(params, lossNet)

    total_info_for_hist = []
    column_names_for_hist = ['e_val', "side", "threshold", "datatype"]
    result = {}
    set_sizes = {}
    correct_by_sizes = {}
    time_seconds_by_sizes = {}

    print("test data names:")
    for i, n in enumerate(test_steps_names):
        print(f"{i+1}: {n}")
    print()

    for es_idx, eval_dataloader in enumerate(test_steps):
        names = test_steps_names[es_idx]

        print(f"[{es_idx+1}] - {names}")

        locate_result = locate_energy(locateNet, eval_dataloader, device, params)

        eval_acc[es_idx] = locate_result['accuracy']
        print(f"[{es_idx+1}] {names}\n\taccuracy:{locate_result['accuracy']:.3f}, precision:{locate_result['precision']:.3f}, recall:{locate_result['recall']:.3f}, f1:{locate_result['f1']:.3f}")

        result = merge_dict(result, {
            f"locate_acc-{es_idx+1}-{test_steps_names[es_idx]}": locate_result.get('accuracy', 0),
        })

    for size in set_sizes.keys():
        result[f"data_proportion_size_{size}"] = set_sizes[size] / sum([v for v in set_sizes.values()])
        result[f"accuracy_size_{size}"] = correct_by_sizes[size] / set_sizes[size]
        result[f"time_seconds_size_{size}"] = time_seconds_by_sizes[size] / set_sizes[size]

    # 원 코드의 그룹 집계(인덱스 기반) 유지, 개수 다르면 방어
    try:
        result['eval_arbitrary_pairs_acc'] = sum([eval_acc[i] for i in [0, 1]])/2
        result['eval_concat2_acc'] = sum([eval_acc[i] for i in [2,3,4]])/3
        result['eval_concat3_acc'] = sum([eval_acc[i] for i in [5,6,7,8]])/4
        result['eval_concat4_acc'] = sum([eval_acc[i] for i in [9,10,11,12,13]])/4
    except IndexError:
        print("[WARN] Unexpected number of test splits; skipping some grouped averages.")
    result['eval_total_acc'] = sum(eval_acc)/len(test_steps_names)

    result = {k: result[k] for k in sorted(result.keys())}

    print("====")
    for key, val in result.items():
        print(f"key:{key}")
        print(f"val:{val}")
        print("--")

    if len(total_info_for_hist) != 0:
        paths = ["evalate_hist.png", "evalate_hist.svg"]
        save_paths = [os.path.join(folder_path, p) for p in paths]
        draw_hist(total_info_for_hist, column_names_for_hist, save_paths)

    print("====")
    print(f"Evaluation Ended.\nFolder path: {folder_path}")
    with open(os.path.join(folder_path, f"{params['task']}_{params['dataset']}_locate_result.json"), 'w') as f:
        json.dump(result, f, indent='\t')


main()
