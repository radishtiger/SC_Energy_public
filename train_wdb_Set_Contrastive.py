import yaml, math, os, json, time, datetime, wandb, pickle
import argparse
import pandas as pd

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
        "triplet_margin": {"values": [pow(10, -1 * i) for i in range(2, 3)]},
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
    # params['energynet']['triplet']['margin'] = 0.001
    params["energynet"]["lr"] = wandb.config.lossNet_lr
    params["energynet"]["triplet"]["margin"] = wandb.config.triplet_margin

    wandb.run.save()
    # print(f"runname: {wandb.run.name}")
    print(f"lossNet_lr: {params['energynet']['lr']}")
    print(f"triplet_margin: {params['energynet']['triplet']['margin']}")
    print()

    # Define and initialize the energy network
    lossNet = energynet(params)
    lossNet.to(device)
    energy_opt = torch.optim.Adam(lossNet.parameters(), lr=params["energynet"]["lr"])
    locateNet = locate(params, lossNet)

    # Load dataset
    train_con_dataset_load_all = dataset_loader_fine_grained(
        params, "train", "con", load_in_once_everything=True
    )
    train_incon_dataset_load_all = dataset_loader_fine_grained(
        params, "train", "incon", load_in_once_everything=True
    )

    eval_con_dataset_arbitrary_pairs = dataset_loader_fine_grained(
        params, "eval", "con", pairwise="arbitrary_pairs"
    )
    eval_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(
        params, "eval", "incon", pairwise="arbitrary_pairs"
    )
    concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs(
        [eval_con_dataset_arbitrary_pairs, eval_incon_dataset_arbitrary_pairs],
        concat_num=2,
    )
    eval_steps_names = ["con", "incon"] + concat2_names
    eval_datasets = [
        eval_con_dataset_arbitrary_pairs,
        eval_incon_dataset_arbitrary_pairs,
    ] + concat2_dataset

    eval_steps = [
        no_decomposition_loader(
            e, params=params, tokenizer=lossNet.representation_model.tokenizer
        ).get_loader(split="eval")
        for e in eval_datasets
    ]
    del eval_datasets

    eval2_con_dataset_arbitrary_pairs = dataset_loader_fine_grained(
        params, "eval2", "con", pairwise="arbitrary_pairs"
    )
    eval2_incon_dataset_arbitrary_pairs = dataset_loader_fine_grained(
        params, "eval2", "incon", pairwise="arbitrary_pairs"
    )
    concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs(
        [eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs],
        concat_num=2,
    )
    concat3_dataset, concat3_names, concat3_set_sizes = concat_arbitrary_pairs(
        [eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs],
        concat_num=3,
    )
    concat4_dataset, concat4_names, concat4_set_sizes = concat_arbitrary_pairs(
        [eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs],
        concat_num=4,
    )

    eval2_steps_names = ["con", "incon"] + concat2_names + concat3_names + concat4_names
    eval2_datasets = (
        [eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs]
        + concat2_dataset
        + concat3_dataset
        + concat4_dataset
    )

    eval2_steps = [
        no_decomposition_loader(
            e, params=params, tokenizer=lossNet.representation_model.tokenizer
        ).get_loader(split="eval")
        for e in eval2_datasets
    ]

    del eval2_datasets

    # Define the appropriate dataloader considering the 'decomposition'
    if params["energynet"]["decomposition"] == "no":
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

        # train_con_dataloader_two_pairs = no_decomposition_loader(train_con_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
        # train_incon_dataloader_two_pairs = no_decomposition_loader(train_incon_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
        # train_con_and_incon_dataloader_two_pairs = no_decomposition_loader(train_con_and_incon_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
        # train_incon_and_incon_dataloader_two_pairs = no_decomposition_loader(train_incon_and_incon_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')

        # eval_con_dataloader_two_pairs = no_decomposition_loader(eval_con_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
        # eval_incon_dataloader_two_pairs = no_decomposition_loader(eval_incon_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
        # eval_con_and_incon_dataloader_two_pairs = no_decomposition_loader(eval_con_and_incon_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')
        # eval_incon_and_incon_dataloader_two_pairs = no_decomposition_loader(eval_incon_and_incon_dataset_two_pairs, params = params, tokenizer = lossNet.representation_model.tokenizer).get_loader(split = 'eval')

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

        eval_steps_labels = []
        for n in eval_steps_names:
            if "incon" in n:
                eval_steps_labels.append(1)
            else:
                eval_steps_labels.append(0)

        acc_thre, acc = learn_energy_threshold(lossNet, eval_steps, eval_steps_labels)
        print(f"threshold: {acc_thre}, with accuracy {acc}")
        os.makedirs(params["folder_path"], exist_ok=True)

        if best_train_loss >= train_loss_step:
            best_train_loss = train_loss_step

        print(f"train_loss = {train_loss_step:.4f}")
        print(f"best_train_loss = {best_train_loss:.4f}")

        # evaluation for eval dataset
        set_sizes = {}
        correct_by_sizes = {}
        time_seconds_by_sizes = {}
        for es_idx, eval_dataloader in enumerate(eval2_steps):
            names = eval2_steps_names[es_idx]

            print(f"[{es_idx+1}] - {names}")

            if "incon" in names:
                side = -1
            else:
                side = 1

            # if 'set' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == 'real_num':
            #     eval_result = evaluate_energy_one_semantic(lossNet, eval_dataloader,side, device, params)
            # elif 'set' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == '2dim_vec':
            #     if side == 1:
            #         gold = 0
            #     elif side == -1:
            #         gold = 1
            #     eval_result = evaluate_supervised_one_semantic(lossNet, eval_dataloader,gold, device, params)
            # elif 'oto' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == '2dim_vec':
            #     eval_result = evaluate_supervised_one_to_one(lossNet, eval_dataloader,side, device, params)
            # elif 'oto' in params['extra_for_eval'].lower() and params['energynet']['output_form'] == 'real_num':
            #     eval_result = evaluate_energy_one_to_one(lossNet, eval_dataloader,side, device, params)
            # else:
            #     print('extra for eval:', params['extra_for_eval'])
            #     print(f"output form: {params['energynet']['output_form']}")
            eval_result = evaluate_energy_one_semantic(
                lossNet, eval_dataloader, side, device, params
            )

            eval_acc[es_idx] = eval_result["eval_acc"]

            # violin plot prepare
            # eval_this_step_name = eval2_steps_names[es_idx].split("-")
            # pos_name = eval_this_step_name[0]
            # neg_name = eval_this_step_name[1]
            # pair_num = eval_this_step_name[2]

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
                    # 'locate_accuracy': locate_result.get('accuracy', 0)
                },
            )
            # locate_result = locate_energy(locateNet, pos_eval_dataloader, neg_eval_dataloader, device, params)
            for size in eval_result["set_sizes"]:
                if size not in set_sizes:
                    set_sizes[size] = 0
                if size not in correct_by_sizes:
                    correct_by_sizes[size] = 0
                if size not in time_seconds_by_sizes:
                    time_seconds_by_sizes[size] = 0

                set_sizes[size] += eval_result["set_sizes"][size]
                correct_by_sizes[size] += eval_result["correct_by_sizes"][size]
                time_seconds_by_sizes[size] += eval_result["time_seconds_by_sizes"][
                    size
                ]

        for size in set_sizes.keys():
            result[f"data_proportion_size_{size}"] = set_sizes[size] / sum(
                [v for v in set_sizes.values()]
            )
            result[f"accuracy_size_{size}"] = correct_by_sizes[size] / set_sizes[size]
            result[f"time_seconds_size_{size}"] = (
                time_seconds_by_sizes[size] / set_sizes[size]
            )

        # result['eval_two_pairs_acc'] = sum([eval_acc[i] for i in [2, 3]])/2
        result["eval_arbitrary_pairs_acc"] = sum([eval_acc[i] for i in [0, 1]]) / 2
        result["eval_concat2_acc"] = sum([eval_acc[i] for i in [2, 3, 4]]) / 3
        result["eval_concat3_acc"] = sum([eval_acc[i] for i in [5, 6, 7, 8]]) / 4
        result["eval_concat4_acc"] = sum([eval_acc[i] for i in [9, 10, 11, 12, 13]]) / 4
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
