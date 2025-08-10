import yaml, math, os, datetime, json, time
import argparse
import glob
from pathlib import Path
import pickle

import torch

from trainer.modules import evaluate_baseline, locate_baseline
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


# =========================
# set_consistency_dataset 로더 유틸
# =========================
def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    set_consistency_dataset/{dataset_name}/ 경로의 피클 파일 경로를 반환.
    파일명 규칙: {dataset_name}_{split}_{NAME}_dataset.pickle
      예) lconvqa_test_C_dataset.pickle, lconvqa_test_CI_dataset.pickle
    name 인자는 "test_C", "test_CI" 등 split 접두사를 포함한 문자열을 기대.
    """
    base = Path("set_consistency_dataset") / dataset_name
    fname = f"{dataset_name}_{name}_dataset.pickle"
    return base / fname


def list_split_pickles(dataset_name: str, split: str):
    """
    split ∈ {"test","eval"} 등에 대해
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
        fn = Path(p).name  # e.g., lconvqa_test_C_dataset.pickle
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
    dataset_name = params['dataset']

    # test: C, I, 그리고 concat2/3/4 (파일명으로부터 자동)
    test_raw_names, test_datasets = load_split_pickles_as_datasets(dataset_name, "test")
    test_steps_names = [to_display_name(n) for n in test_raw_names]

    # 로더 생성
    test_steps = [lm_loader(e, params=params).get_loader() for e in test_datasets]

    model = baseline_model(params, 'locate')
    results = {}
    # time_seconds = {}
    # set_sizes = {}
    # accuracy_by_sizes = {}

    for es_idx, data_loader in enumerate(test_steps):
        data_name = test_steps_names[es_idx]

        print(f"=========== {es_idx+1} - {data_name} ===========")
        locate_result = locate_baseline(model, data_loader, device=device, params=params)

        results[f"locate_accuracy-{es_idx+1}-{data_name}"]  = locate_result.get('accuracy', 0)
        results[f"locate_precision-{es_idx+1}-{data_name}"] = locate_result.get('precision', 0)
        results[f"locate_recall-{es_idx+1}-{data_name}"]    = locate_result.get('recall', 0)
        results[f"locate_f1-{es_idx+1}-{data_name}"]        = locate_result.get('f1', 0)
        print(f"[{es_idx+1}] {data_name}\n\taccuracy:{locate_result['accuracy']:.3f}, precision:{locate_result['precision']:.3f}, recall:{locate_result['recall']:.3f}, f1:{locate_result['f1']:.3f}")



    results = {r: results[r] for r in sorted(results)}

    os.makedirs(params['folder_path'], exist_ok=True)

    try:
        with open(params['record_path'], 'r') as f:
            record = json.load(f)
    except:
        record = {}
    if "locate" not in record:
        record["locate"] = {}
    record["locate"][params['baseline']['locate_type']] = results


    print("================")
    print("Evaluation Ended.")
    print(f"{record_path}")
    for key, val in record.items():
        print(f"[{key}]")
        for k, v in val.items():
            print(f"\t{k}: {v}")
    print("================")

main()
