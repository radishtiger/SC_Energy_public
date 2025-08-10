import yaml, math, os, datetime, json
import argparse, pickle, glob
from pathlib import Path

import torch
import torch.nn as nn

from trainer.modules import (
    evaluate_energy,
    learn_energy_threshold,
    evaluate_energy_fine_grained,
    evaluate_energy_one_semantic,
    evaluate_supervised_one_semantic,
    evaluate_supervised_one_to_one,
    evaluate_energy_one_to_one,
    evaluate_energy_many_to_one,
    evaluate_supervised_many_to_one,
)

from energynets.energynet import energynet
from energynets.decomposition.no_decomposition import no_decomposition_loader
from utils import (
    merge_dict, parser_add, params_add, draw_hist,
    draw_violin_plot_supervised, draw_violin_plot, draw_violin_plot_noneface,
    draw_violin_plot_noneface_noarrow, draw_violin_plot_whiteface_noarrow
)

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
# prepared_dataset 로더 유틸
# =========================
# =========================
# set_consistency_dataset 경로용 유틸 (교체본)
# =========================

def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    set_consistency_dataset/{dataset_name}/ 하위의 피클 경로를 돌려줍니다.
    (중간 split 디렉터리 없음)

    기대 파일명:
      {dataset_name}_{split}_*_dataset.pickle
      예: lconvqa_eval_C_dataset.pickle, lconvqa_test_CI_dataset.pickle
    """
    base = Path("set_consistency_dataset") / dataset_name
    # name은 "eval_C", "test_CI" 등 split 접두사를 포함한 문자열로 전달됨
    fname = f"{dataset_name}_{name}_dataset.pickle"
    return base / fname


def list_split_pickles(dataset_name: str, split: str):
    """
    split ∈ {"eval","test"} 에 대해 set_consistency_dataset/{dataset_name}/에서
    {dataset_name}_{split}_*_dataset.pickle 파일을 수집/정렬합니다.

    반환: (names_sorted, paths_sorted)
      names_sorted: ["C","I","CC","CI","II","CCC", ...]
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

    # "C","I" 먼저, 이후 길이/사전순
    def _order_key(x):
        if x == "C":
            return (0, 0, x)
        if x == "I":
            return (0, 1, x)
        return (1, len(x), x)

    names_sorted = sorted(names, key=_order_key)
    paths_sorted = [ _pkl_path(dataset_name, split, f"{split}_{nm}") for nm in names_sorted ]
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
    # 로그/프린트 용: "C"→"con", "I"→"incon", 나머지는 그대로("CI","CC" 등)
    return "con" if nm == "C" else ("incon" if nm == "I" else nm)


# =========
# 메인 루틴
# =========
def main():
    lossNet = energynet(params).to(device)

    dataset_name = params['dataset']

    # eval: C, I, 그리고 concat2/3 (파일명으로부터 자동)
    eval_raw_names, eval_datasets = load_split_pickles_as_datasets(dataset_name, "eval")
    eval_steps_names = [to_display_name(n) for n in eval_raw_names]

    # test: C, I, 그리고 concat2/3/4 (파일명으로부터 자동)
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

    # threshold 학습 라벨: "I"가 하나라도 들어가면 incon(1), 순수 "C"만이면 con(0)
    eval_steps_labels = [ 1 if ('I' in n) else 0 for n in eval_raw_names ]

    eval_acc = [0 for _ in range(len(test_steps_names))]

    if params['scratch'] == False:
        state = torch.load(params['model_path'], map_location=device)
        lossNet.load_state_dict(state['state_dict'])
        if 'threshold' in state:
            lossNet.threshold = state['threshold']

    if False:
        print("Start threshold learning")
        acc_thre, acc = learn_energy_threshold(lossNet, eval_steps, eval_steps_labels)
        print(f"threshold: {acc_thre}, with accuracy {acc}")
        torch.save({
                    "state_dict": lossNet.state_dict(),
                    "threshold": acc_thre
                    }, params['model_path'])
        assert acc_thre == lossNet.threshold

    total_info_for_hist = []
    column_names_for_hist = ['energy value', "side", "threshold", "datatype"]
    result = {}
    set_sizes = {}
    correct_by_sizes = {}
    time_seconds_by_sizes = {}

    for i, n in enumerate(test_steps_names):
        print(n, "length", len(test_steps[i]))

    for es_idx, eval_dataloader in enumerate(test_steps):
        raw_name = test_raw_names[es_idx]      # ex) "C","I","CI","CCC",...
        names = test_steps_names[es_idx]       # display name

        print(f"[{es_idx+1}] - {names}")

        # "I"가 들어가면 Inconsistent 취급 (원래 'incon' substring 로직을 보완)
        if ('I' in raw_name):
            side = -1
            sidename = "Inconsistent"
        else:
            side = 1
            sidename = "Consistent"

        if 'set' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == 'real_num':
            eval_result = evaluate_energy_one_semantic(lossNet, eval_dataloader, side, device, params)
            total_info_for_hist += [[eval_result['e_vals'][i], sidename, lossNet.threshold, names] for i in range(len(eval_result['e_vals']))]
        elif 'set' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == '2dim_vec':
            gold = 0 if side == 1 else 1
            eval_result = evaluate_supervised_one_semantic(lossNet, eval_dataloader, gold, device, params)
            total_info_for_hist += [[eval_result['neg_probs'][i], side, lossNet.threshold, names] for i in range(len(eval_result['neg_probs']))]
        elif 'oto' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == '2dim_vec':
            eval_result = evaluate_supervised_one_to_one(lossNet, eval_dataloader, side, device, params)
        elif 'oto' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == 'real_num':
            eval_result = evaluate_energy_one_to_one(lossNet, eval_dataloader, side, device, params)
        elif 'mto' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == 'real_num':
            eval_result = evaluate_energy_many_to_one(lossNet, eval_dataloader, side, device, params)
        elif 'mto' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == '2dim_vec':
            eval_result = evaluate_supervised_many_to_one(lossNet, eval_dataloader, side, device, params)
        else:
            print('extra for eval:', params['extra_for_eval'])
            print(f"output form: {params['energynet']['output_form']}")
            continue

        eval_acc[es_idx] = eval_result['eval_acc']

        result = merge_dict(result, {
            f"eval_loss-{es_idx+1}-{names}": eval_result.get('eval_loss', 0),
            f"eval_auroc-{es_idx+1}-{names}": eval_result.get('auroc', 0),
            f"eval_precision-{es_idx+1}-{names}": eval_result.get('eval_precision', 0),
            f"eval_recall-{es_idx+1}-{names}": eval_result.get('eval_recall', 0),
            f"eval_f1-{es_idx+1}-{names}": eval_result.get('eval_f1', 0),
            f"eval_acc-{es_idx+1}-{names}": eval_result.get('eval_acc', 0),
            f"eval_oracle_acc-{es_idx+1}-{names}": eval_result.get('eval_oracle_acc', 0),
        })

        for size in eval_result['set_sizes']:
            if size not in set_sizes:
                set_sizes[size] = 0
            if size not in correct_by_sizes:
                correct_by_sizes[size] = 0
            if size not in time_seconds_by_sizes:
                time_seconds_by_sizes[size] = 0

            set_sizes[size] += eval_result['set_sizes'][size]
            correct_by_sizes[size] += eval_result['correct_by_sizes'][size]
            time_seconds_by_sizes[size] += eval_result['time_seconds_by_sizes'][size]

    for size in set_sizes.keys():
        result[f"data_proportion_size_{size}"] = set_sizes[size] / sum([v for v in set_sizes.values()])
        result[f"accuracy_size_{size}"] = correct_by_sizes[size] / set_sizes[size]
        result[f"time_seconds_size_{size}"] = time_seconds_by_sizes[size] / set_sizes[size]

    # 인덱스 집계는 원래 코드의 가정(3/4/5개씩)과 동일하게 유지
    # test_steps_names: ["con","incon"] + (concat2: 3개) + (concat3: 4개) + (concat4: 5개)
    try:
        result['eval_arbitrary_pairs_acc'] = sum([eval_acc[i] for i in [0, 1]]) / 2
        result['eval_concat2_acc'] = sum([eval_acc[i] for i in [2,3,4]]) / 3
        result['eval_concat3_acc'] = sum([eval_acc[i] for i in [5,6,7,8]]) / 4
        result['eval_concat4_acc'] = sum([eval_acc[i] for i in [9,10,11,12,13]]) / 4
    except IndexError:
        # 준비된 피클 개수가 다를 수 있으니 방어 코드
        print("[WARN] Unexpected number of test splits; skipping grouped averages.")
    result['eval_total_acc'] = sum(eval_acc)/len(test_steps_names)

    result = {k: result[k] for k in sorted(result.keys())}

    print("====")
    for key, val in result.items():
        print(f"key:{key}")
        print(f"val:{val}")
        print("--")

    if len(total_info_for_hist) != 0:
        paths = [f"{params['task']}_{params['dataset']}_evaluate_box.png",
                 f"{params['task']}_{params['dataset']}_evaluate_box.svg"]
        save_paths = [os.path.join(folder_path, p) for p in paths]
        draw_violin_plot(total_info_for_hist, column_names_for_hist, test_steps_names, save_paths)
        draw_violin_plot_whiteface_noarrow(total_info_for_hist, column_names_for_hist, test_steps_names, save_paths)
        draw_violin_plot_noneface_noarrow(total_info_for_hist, column_names_for_hist, test_steps_names, save_paths)
        draw_violin_plot_noneface(total_info_for_hist, column_names_for_hist, test_steps_names, save_paths)

    print("====")
    print(f"Evaluation Ended.\nFolder path: {folder_path}")
    with open(os.path.join(folder_path, f"{params['task']}_{params['dataset']}_evaluate_result.json"), 'w') as f:
        json.dump(result, f, indent='\t')


main()
