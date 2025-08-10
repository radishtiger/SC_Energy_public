import yaml, math, os, json, time, datetime, wandb, pickle
import argparse
import pandas as pd
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from trainer.modules import (
    train_energy,
    evaluate_energy_fine_grained,
    locate_energy,
    learn_energy_threshold,
)
from trainer.modules import (
    train_supervised,
    evaluate_supervised_one_semantic,
    locate_energy,
    learn_energy_threshold,
    evaluate_energy_one_semantic,
    evaluate_energy_one_to_one,
    evaluate_supervised_one_to_one,
)

from tasks.dataset_loader import (
    dataset_loader,
    dataset_loader_fine_grained,
    concat_arbitrary_pairs,
)
from energynets.energynet import energynet
from energynets.decomposition.no_decomposition import no_decomposition_loader

from locate_and_edit.locate import locate

from utils import merge_dict, parser_add, params_add, draw_violin_plot


# -----------------------------
# Pre-saved dataset direct loader (set_consistency_dataset/{dataset}/...)
# -----------------------------
def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    Return path to a pickle under set_consistency_dataset/{dataset_name}/.

    Expected file naming (based on your original saving rules):
      - train:
          {dataset}_train_con_dataset.pickle
          {dataset}_train_incon_dataset.pickle
      - eval:
          {dataset}_eval_C_dataset.pickle
          {dataset}_eval_I_dataset.pickle
          {dataset}_eval_{NAME}_dataset.pickle    (e.g., CI, II, ...)
      - eval2:
          {dataset}_eval2_C_dataset.pickle
          {dataset}_eval2_I_dataset.pickle
          {dataset}_eval2_{NAME}_dataset.pickle   (e.g., CCI, CII, III, CCCC, ...)

    NOTE: There is NO split subdirectory. Files live directly under:
          set_consistency_dataset/{dataset_name}/
    """
    base = Path("set_consistency_dataset") / dataset_name
    # name is like "train_con", "eval_C", "eval2_CI", etc.
    fname = f"{dataset_name}_{name}_dataset.pickle"
    return base / fname


def load_train_pickles(dataset_name: str):
    con_p = _pkl_path(dataset_name, "train", "train_con")
    incon_p = _pkl_path(dataset_name, "train", "train_incon")
    assert con_p.exists() and incon_p.exists(), f"[train] pickle not found: {con_p} / {incon_p}"
    with open(con_p, "rb") as f:
        train_con = pickle.load(f)
    with open(incon_p, "rb") as f:
        train_incon = pickle.load(f)
    return train_con, train_incon


def list_eval_pickles(dataset_name: str, split: str):
    """
    List NAME pieces for {dataset}_{split}_{NAME}_dataset.pickle under:
        set_consistency_dataset/{dataset_name}/

    Returns (names_sorted, paths_sorted_by_name).
    """
    base = Path("set_consistency_dataset") / dataset_name
    patt = str(base / f"{dataset_name}_{split}_*_dataset.pickle")
    paths = sorted(glob.glob(patt))
    names = []
    for p in paths:
        fn = Path(p).name
        prefix = f"{dataset_name}_{split}_"
        assert fn.startswith(prefix) and fn.endswith("_dataset.pickle"), f"unexpected filename: {fn}"
        name = fn[len(prefix):-len("_dataset.pickle")]  # e.g., "C", "I", "CI", "CCI", ...
        names.append(name)

    # Ensure canonical ordering that matches original expected indices:
    # First "C", then "I", then by length and lex for the rest (e.g., "CC","CI","II", then "CCC","CCI","CII","III", ...)
    def _order_key(x):
        if x == "C":
            return (0, 0, x)
        if x == "I":
            return (0, 1, x)
        return (1, len(x), x)

    names_sorted = sorted(names, key=_order_key)
    paths_sorted = []
    for nm in names_sorted:
        paths_sorted.append(str(_pkl_path(dataset_name, split, f"{split}_{nm}")))
    return names_sorted, [Path(p) for p in paths_sorted]


def load_eval_pickles_as_datasets(dataset_name: str, split: str):
    names, _ = list_eval_pickles(dataset_name, split)
    datasets = []
    for nm in names:
        p = _pkl_path(dataset_name, split, f"{split}_{nm}")
        with open(p, "rb") as f:
            ds = pickle.load(f)
        datasets.append(ds)
    return names, datasets


# -----------------------------


parser = argparse.ArgumentParser()
parser = parser_add(parser)

args = parser.parse_args()
params = params_add(args)

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
params["device"] = device

#
runname = f"SetCon-{params['energynet']['repre_model']}-{params['energynet']['decomposition']}-{params['energynet']['loss_type']}-{params['energynet']['loss_fully_separate']}-fg_tot"

sweep_configuration = {
    "method": "random",
    "name": runname,
    "metric": {"goal": "minimize", "name": "eval_loss"},
    "parameters": {
        "lossNet_lr": {"values": [pow(10, -1 * i) for i in range(6, 7)]},
        # roberta-base trains well when the exponent falls within range(5, 8)
        "margin": {"values": [pow(10, -1 * i) for i in range(2, 3)]},
        # roberta-base performs well when the exponent is within range(1, 3)
    },
}

coarse_folder_path = os.path.join("results", params["task"], params["dataset"])
projectName = f"{coarse_folder_path.replace('/', '-')}_v7"
sweep_id = wandb.sweep(sweep=sweep_configuration, project=projectName)

sw_count = math.prod(
    [len(val["values"]) for val in sweep_configuration["parameters"].values()]
)
sw_count = 1


def main():

    folder_path = os.path.join(
        "results", params["task"], params["dataset"], str(params["job_id"])
    )
    model_path = os.path.join(folder_path, f"{runname}.pth")
    threshold_path = os.path.join(folder_path, "thresholds.json")

    params["folder_path"] = folder_path
    params["model_path"] = model_path
    params["threshold_path"] = threshold_path
    wandb.init(config=params)
    wandb.run.name = runname
    # params['energynet']['lr'] = 0.1
    # params['energynet']['margin']['margin'] = 0.001
    params["energynet"]["lr"] = wandb.config.lossNet_lr
    params["energynet"]["margin"]["margin"] = wandb.config.margin

    wandb.run.save()
    # print(f"runname: {wandb.run.name}")
    print(f"lossNet_lr: {params['energynet']['lr']}")
    print(f"margin: {params['energynet']['margin']['margin']}")
    print()

    # Define and initialize the energy network
    lossNet = energynet(params)
    lossNet.to(device)
    energy_opt = torch.optim.Adam(lossNet.parameters(), lr=params["energynet"]["lr"])
    locateNet = locate(params, lossNet)
    dataset_name = params['dataset']

    # ===========================================
    # Load dataset DIRECTLY from pre-saved pickles
    # ===========================================
    # (1) train
    train_con_dataset_load_all, train_incon_dataset_load_all = load_train_pickles(dataset_name)

    # (2) eval (contains C, I, and concat2 variants such as CC/CI/II)
    eval_names, eval_datasets = load_eval_pickles_as_datasets(dataset_name, "eval")

    # display names mapping for logs/labels: "C"->"con", "I"->"incon"
    def _map_eval_display_name(nm: str):
        return "con" if nm == "C" else ("incon" if nm == "I" else nm)

    eval_steps_names = [_map_eval_display_name(n) for n in eval_names]

    # (3) eval2 (contains C, I, and concat2/3/4 variants)
    eval2_names, eval2_datasets = load_eval_pickles_as_datasets(dataset_name, "eval2")
    eval2_steps_names = [_map_eval_display_name(n) for n in eval2_names]

    # Build DataLoaders
    train_con_dataloader_load_all = no_decomposition_loader(
        train_con_dataset_load_all,
        params=params,
        tokenizer=lossNet.representation_model.tokenizer,
    ).get_loader()

    train_incon_dataloader_load_all = no_decomposition_loader(
        train_incon_dataset_load_all,
        params=params,
        tokenizer=lossNet.representation_model.tokenizer,
    ).get_loader()

    eval_steps = [
        no_decomposition_loader(
            ds, params=params, tokenizer=lossNet.representation_model.tokenizer
        ).get_loader(split="eval")
        for ds in eval_datasets
    ]

    eval2_steps = [
        no_decomposition_loader(
            ds, params=params, tokenizer=lossNet.representation_model.tokenizer
        ).get_loader(split="eval")
        for ds in eval2_datasets
    ]

    # ===========================================

    # Define the appropriate dataloader considering the 'decomposition'
    if params["energynet"]["decomposition"] == "no":
        # (already built above)
        pass

    training_steps = [
        (train_con_dataloader_load_all, train_incon_dataloader_load_all),
    ]

    best_train_loss = 100000

    train_loss = 100000

    eval_acc = [0 for _ in range(len(eval2_steps_names))]

    best_eval_total_acc = 0

    # Train and evaluate model.
    # The trained model and the evaluation result figures will be stored at 'folder_path' (defined above)

    os.makedirs(params["folder_path"], exist_ok=True)

    # params + config file save
    with open(os.path.join(folder_path, "params.pickle"), "wb") as f:
        pickle.dump(params, f)

    # Test with initialized model
    result = {}
    set_sizes = {}
    correct_by_sizes = {}
    time_seconds_by_sizes = {}

    for e_total in range(params["energynet"]["epoch"]):

        train_loss = 1000
        for ts_idx, (pos_train_dataloader, neg_train_dataloader) in enumerate(
            training_steps
        ):
            e = e_total * len(training_steps) + ts_idx

            result = {}

            print(f"[Epoch {e+1}]")

            if ts_idx == 0 or ts_idx == 4:
                lt = True
            else:
                lt = False
            train_loss_step, threshold_result = train_energy(
                lossNet,
                energy_opt,
                pos_train_dataloader,
                neg_train_dataloader,
                lt,
                device,
                params,
            )
            train_loss += float(train_loss_step)

        # threshold labels: 1 if "incon" or exactly "I", else 0
        eval_steps_labels = [1 if (("incon" in n) or (n == "I")) else 0 for n in eval_steps_names]

        acc_thre, acc = learn_energy_threshold(lossNet, eval_steps, eval_steps_labels)
        print(f"threshold: {acc_thre}, with accuracy {acc}")
        os.makedirs(params["folder_path"], exist_ok=True)

        if best_train_loss >= train_loss_step:
            best_train_loss = train_loss_step

        print(f"train_loss = {train_loss_step:.4f}")
        print(f"best_train_loss = {best_train_loss:.4f}")

        # evaluation for eval2 dataset
        set_sizes = {}
        correct_by_sizes = {}
        time_seconds_by_sizes = {}
        for es_idx, eval_dataloader in enumerate(eval2_steps):
            names = eval2_steps_names[es_idx]

            print(f"[{es_idx+1}] - {names}")

            # side: -1 for incon / I, +1 otherwise
            if ("incon" in names) or (names == "I"):
                side = -1
            else:
                side = 1

            eval_result = evaluate_energy_one_semantic(
                lossNet, eval_dataloader, side, device, params
            )

            eval_acc[es_idx] = eval_result["eval_acc"]

            result = merge_dict(
                result,
                {
                    f"eval_loss-{es_idx+1}-{eval2_steps_names[es_idx]}": eval_result.get(
                        "eval_loss", 0
                    ),
                    f"eval_auroc-{es_idx+1}-{eval2_steps_names[es_idx]}": eval_result.get(
                        "auroc", 0
                    ),
                    f"eval_precision-{es_idx+1}-{eval2_steps_names[es_idx]}": eval_result.get(
                        "eval_precision", 0
                    ),
                    f"eval_recall-{es_idx+1}-{eval2_steps_names[es_idx]}": eval_result.get(
                        "eval_recall", 0
                    ),
                    f"eval_f1-{es_idx+1}-{eval2_steps_names[es_idx]}": eval_result.get(
                        "eval_f1", 0
                    ),
                    f"eval_acc-{es_idx+1}-{eval2_steps_names[es_idx]}": eval_result.get(
                        "eval_acc", 0
                    ),
                    f"eval_oracle_acc-{es_idx+1}-{eval2_steps_names[es_idx]}": eval_result.get(
                        "eval_oracle_acc", 0
                    ),
                },
            )

            for size in eval_result["set_sizes"]:
                if size not in set_sizes:
                    set_sizes[size] = 0
                if size not in correct_by_sizes:
                    correct_by_sizes[size] = 0
                if size not in time_seconds_by_sizes:
                    time_seconds_by_sizes[size] = 0

                set_sizes[size] += eval_result["set_sizes"][size]
                correct_by_sizes[size] += eval_result["correct_by_sizes"][size]
                time_seconds_by_sizes[size] += eval_result["time_seconds_by_sizes"][size]

        for size in set_sizes.keys():
            result[f"data_proportion_size_{size}"] = set_sizes[size] / sum(
                [v for v in set_sizes.values()]
            )
            result[f"accuracy_size_{size}"] = correct_by_sizes[size] / set_sizes[size]
            result[f"time_seconds_size_{size}"] = (
                time_seconds_by_sizes[size] / set_sizes[size]
            )

        # NOTE: These aggregates follow the original index-based averaging.
        # Assumes eval2_steps_names ordered as: ["con","incon", (3x concat2), (4x concat3), (5x concat4)]
        result["eval_arbitrary_pairs_acc"] = sum([eval_acc[i] for i in [0, 1]]) / 2
        result["eval_concat2_acc"] = sum([eval_acc[i] for i in [2, 3, 4]]) / 3
        result["eval_concat3_acc"] = sum([eval_acc[i] for i in [5, 6, 7, 8]]) / 4
        result["eval_concat4_acc"] = sum([eval_acc[i] for i in [9, 10, 11, 12, 13]]) / 4  # original code

        result["eval_total_acc"] = sum(eval_acc) / len(eval2_steps_names)

        if best_eval_total_acc < result["eval_total_acc"]:
            best_eval_total_acc = result["eval_total_acc"]
            torch.save(
                {"state_dict": lossNet.state_dict(), "threshold": lossNet.threshold},
                params["model_path"],
            )
            print("model saved.")
        print(
            f"best_eval_total_acc: {best_eval_total_acc:.4f}\ncurrent eval total acc: {result['eval_total_acc']:.4f}"
        )

        result = {k: result[k] for k in sorted(result.keys())}

        print("====")
        for key, val in result.items():
            print(f"key:{key}")
            print(f"val:{val}")
            print("--")

        print("====")
        print(f"Evaluation Ended.\nFolder path: {folder_path}")
        with open(
            os.path.join(
                folder_path,
                f"{params['task']}_{params['dataset']}_evaluate_result.json",
            ),
            "w",
        ) as f:
            json.dump(result, f, indent="\t")

        wandb.log(result, step=e + 1)
        print("======================================")


wandb.agent(sweep_id, function=main, count=sw_count)
