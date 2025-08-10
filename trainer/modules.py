import os, itertools, time, random, torch, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from tqdm import tqdm
from torch.nn.functional import softmax


def train_energy(
    energynet,
    opt,
    pos_dataloader,
    neg_dataloader,
    learn_threshold=True,
    device="cpu",
    params=None,
):

    energynet.train()
    total_loss = 0
    e_pos = []
    e_neg = []
    threshold_result = {}

    for pairs in tqdm(zip(pos_dataloader, neg_dataloader)):
        # print("\n\npos_pairs:")
        # print(pairs[0])
        # print("neg_pairs:")
        # print(pairs[1])

        loss, additional_info = energynet(pairs)
        # print(f"loss: {loss}")
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
        e_pos.append(float(additional_info.get("e_pos", 0)))
        e_neg.append(float(additional_info.get("e_neg", 0)))

    threshold_result = determine_best_split(e_pos, e_neg)
    # for key,val in threshold_result.items():
    #     print(key)
    #     print(val)
    #     print()
    if learn_threshold:
        energynet.threshold = threshold_result.get("acc_threshold", 0)
    return total_loss / len(pos_dataloader), threshold_result


# def evaluate_energy_one_loader(energynet, dataloader, device= 'cpu', params = None):


def train_supervised(energynet, opt, dataloader, device="cpu", params=None):
    energynet.train()
    total_loss = 0
    preds = []
    probs = []

    for pairs in tqdm(dataloader):
        error_locations = [p[1] for p in pairs]
        labels = [int(len(e) > 0) for e in error_locations]
        # print("pairs[0] input:")
        # print(pairs[0][0])
        # print("error_locations")
        # print(pairs[0][1])
        # print("labels")
        # print(labels[0])

        loss, additional_info = energynet((pairs, labels))
        # print(f"loss: {loss}")
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
        preds.extend(additional_info.get("pred", 0))
        probs.extend(additional_info.get("probs", 0))

    return total_loss / len(dataloader), {"preds": preds, "probs": probs}


def evaluate_energy(
    energynet, pos_dataloader, neg_dataloader, device="cpu", params=None
):
    if params["eval"]["pairwise"] == "arbitrary_pairs":
        return evaluate_energy_arbitrary(
            energynet, pos_dataloader, neg_dataloader, device, params
        )
    elif params["eval"]["pairwise"] == "two_pairs":
        return evaluate_energy_two_pairs(
            energynet, pos_dataloader, neg_dataloader, device, params
        )


def evaluate_energy_fine_grained(
    energynet,
    pos_dataloader,
    neg_dataloader,
    side_pos=1,
    side_neg=-1,
    device="cpu",
    params=None,
):
    print("[eval_energy_fine_Grained]")
    energynet.eval()
    total_loss = 0
    result = {
        "eval_loss": 0,
        "e_pos": [],
        "e_neg": [],
        "pred": [],
        "gold": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    # inference
    for pairs in tqdm(zip(pos_dataloader, neg_dataloader)):
        start = time.time()
        loss, additional_info = energynet(pairs)
        consumed_time = time.time() - start
        if params["energynet"]["output_form"] == "real_num":
            e_pos = additional_info.get("e_pos", 0)
            e_neg = additional_info.get("e_neg", 0)
            result["e_pos"].append(float(e_pos))
            result["e_neg"].append(float(e_neg))
        elif params["energynet"]["output_form"] == "2dim_vec":
            pred, gold = additional_info["pred"], additional_info["gold"]
            result["pred"].extend(pred)
            result["gold"].extend(gold)
        total_loss += loss.detach().item()

        pos_set_size = len(pairs[0][0][2])  # 0-0-1 : positive-first_batch-set&label
        if pos_set_size not in result["set_sizes"]:
            result["set_sizes"][pos_set_size] = 0
        if pos_set_size not in result["correct_by_sizes"]:
            result["correct_by_sizes"][pos_set_size] = 0
        if pos_set_size not in result["time_seconds_by_sizes"]:
            result["time_seconds_by_sizes"][pos_set_size] = 0

        neg_set_size = len(pairs[1][0][2])  # 1-0-2 : negative-first_batch-set&label
        if neg_set_size not in result["set_sizes"]:
            result["set_sizes"][neg_set_size] = 0
        if neg_set_size not in result["correct_by_sizes"]:
            result["correct_by_sizes"][neg_set_size] = 0
        if neg_set_size not in result["time_seconds_by_sizes"]:
            result["time_seconds_by_sizes"][neg_set_size] = 0

        result["set_sizes"][pos_set_size] += 1
        result["set_sizes"][neg_set_size] += 1
        if side_pos * (float(e_pos) - energynet.threshold) < 0:
            result["correct_by_sizes"][pos_set_size] += 1
        if side_neg * (float(e_neg) - energynet.threshold) < 0:
            result["correct_by_sizes"][neg_set_size] += 1
        result["time_seconds_by_sizes"][pos_set_size] += consumed_time / 2
        result["time_seconds_by_sizes"][neg_set_size] += consumed_time / 2

    # loss
    result["eval_loss"] = total_loss / len(pos_dataloader)

    if params["energynet"]["output_form"] == "real_num":
        # hist plot
        plot_hist_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_hist.svg"
        )
        plot_hist_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_hist.png"
        )

        # ROC plot
        plot_roc_curve_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_ROC.svg"
        )
        auroc = plot_roc_curve_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_ROC.png"
        )
        thresholds = determine_best_split(result["e_pos"], result["e_neg"])
        precision = thresholds["precision"]
        recall = thresholds["recall"]
        f1 = thresholds["f1"]
        acc_by_eval = thresholds["acc"]

        pos_result = [
            e for e in result["e_pos"] if side_pos * (e - energynet.threshold) < 0
        ]
        neg_result = [
            e for e in result["e_neg"] if side_neg * (e - energynet.threshold) < 0
        ]
        correct_by_trained_threshold = len(pos_result) + len(neg_result)
        acc = correct_by_trained_threshold / (
            len(result["e_pos"]) + len(result["e_neg"])
        )

    elif params["energynet"]["output_form"] == "2dim_vec":
        auroc = 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(
            result["gold"], result["pred"], average="macro"
        )
        acc = accuracy_score(result["gold"], result["pred"])

    result["auroc"] = auroc
    result["eval_precision"] = precision
    result["eval_recall"] = recall
    result["eval_f1"] = f1
    result["eval_acc"] = acc
    result["eval_oracle_acc"] = acc_by_eval

    return result


def evaluate_energy_one_semantic(
    energynet, dataloader, side, device="cpu", params=None
):
    print("[eval_energy_one_semantic]")
    energynet.eval()
    result = {
        "eval_loss": 0,
        "e_vals": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    # inference
    for pairs in tqdm(dataloader):
        set_size = len(pairs[0][2])
        start = time.time()

        e_vals, _ = energynet.energy_model(pairs)
        consumed_time = time.time() - start

        result["e_vals"].append(float(e_vals))

        if set_size not in result["set_sizes"]:
            result["set_sizes"][set_size] = 0
        result["set_sizes"][set_size] += 1

        if set_size not in result["correct_by_sizes"]:
            result["correct_by_sizes"][set_size] = 0
        if side * (float(e_vals) - energynet.threshold) < 0:
            result["correct_by_sizes"][set_size] += 1

        if set_size not in result["time_seconds_by_sizes"]:
            result["time_seconds_by_sizes"][set_size] = 0
        result["time_seconds_by_sizes"][set_size] += consumed_time

    # loss

    e_val_result = [e for e in result["e_vals"] if side * (e - energynet.threshold) < 0]
    correct_by_trained_threshold = len(e_val_result)
    acc = correct_by_trained_threshold / (len(result["e_vals"]))

    result["eval_acc"] = acc

    return result


def evaluate_supervised_one_semantic(
    energynet, dataloader, side=None, device="cpu", params=None
):
    print("[eval_energy_one_side]")
    energynet.eval()
    total_loss = 0
    result = {
        "eval_loss": 0,
        #   'e_pos': [],
        #   'e_neg': [],
        "pred": [],
        "gold": [],
        "neg_probs": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    # inference
    for pairs in tqdm(dataloader):
        assert len(pairs) == 1
        set_size = len(pairs[0][2])
        start = time.time()
        loss, additional_info = energynet((pairs, side))
        consumed_time = time.time() - start
        pred, gold = additional_info["pred"], additional_info["gold"]
        probs = additional_info["probs"]

        result["pred"].extend(pred)
        result["gold"].extend(gold)
        result["neg_probs"].extend(
            [p[1] for p in probs]
        )  # we only store dim_1 values. (which means the probability for inconsistent state)
        total_loss += loss.detach().item()

        if set_size not in result["set_sizes"]:
            result["set_sizes"][set_size] = 0
        result["set_sizes"][set_size] += 1

        if set_size not in result["correct_by_sizes"]:
            result["correct_by_sizes"][set_size] = 0
        if pred == gold:
            result["correct_by_sizes"][set_size] += 1

        if set_size not in result["time_seconds_by_sizes"]:
            result["time_seconds_by_sizes"][set_size] = 0
        result["time_seconds_by_sizes"][set_size] += consumed_time

    # loss
    result["eval_loss"] = total_loss / len(dataloader)

    # if params['energynet']['output_form'] == 'real_num':
    #     # hist plot
    #     plot_hist_real_num(result['e_pos'], result['e_neg'], params['model_path'][:-1] + '_hist.svg')
    #     plot_hist_real_num(result['e_pos'], result['e_neg'], params['model_path'][:-1] + '_hist.png')

    #     # ROC plot
    #     plot_roc_curve_real_num(result['e_pos'], result['e_neg'], params['model_path'][:-1] + '_ROC.svg')
    #     auroc = plot_roc_curve_real_num(result['e_pos'], result['e_neg'], params['model_path'][:-1] + '_ROC.png')
    #     thresholds = determine_best_split(result['e_pos'], result['e_neg'])
    #     precision = thresholds['precision']
    #     recall = thresholds['recall']
    #     f1 = thresholds['f1']
    #     acc_by_eval = thresholds['acc']

    #     pos_result = [e for e in result['e_pos'] if side*(e - energynet.threshold) < 0]
    #     neg_result = [e for e in result['e_neg'] if side_neg*(e - energynet.threshold) < 0]
    #     correct_by_trained_threshold = len(pos_result) + len(neg_result)
    #     acc = correct_by_trained_threshold / (len(result['e_pos']) + len(result['e_neg']))

    if params["energynet"]["output_form"] == "2dim_vec":
        acc = accuracy_score(result["gold"], result["pred"])

    result["eval_acc"] = acc

    return result


def evaluate_supervised_one_to_one(
    energynet, dataloader, side=None, device="cpu", params=None
):

    energynet.eval()
    cls_token = energynet.representation_model.tokenizer.cls_token
    sep_token = "."
    total_loss = 0
    result = {
        "eval_loss": 0,
        "pred": [],
        "gold": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    for pairs in tqdm(dataloader):

        for pair in pairs:

            pred_one_example = []

            inputs = pair[0]
            set_size = len(pair[2])

            inputs_comb = comb_input(inputs, cls_token=cls_token, sep_token=sep_token)

            start = time.time()
            for idx in range(0, len(inputs_comb), 4):
                temp_input = inputs_comb[idx : idx + 4]
                # print("temp_input:")
                # print(temp_input)
                loss, additional_info = energynet((temp_input, None), pair_only=True)
                pred_one_example.extend(additional_info["pred"])
            consumed_time = time.time() - start

            if len(pair[1]) > 0:
                label = 1
            else:
                label = 0
            # print("inputs_comb[0]:", inputs_comb[0])
            # loss, additional_info = energynet((inputs_comb, None))
            preds = pred_one_example
            pred = 0 if sum(preds) <= 0 else 1
            result["pred"].extend([pred])
            result["gold"].extend([label])
            # result['neg_probs'].extend([p[1] for p in probs]) # we only store dim_1 values. (which means the probability for inconsistent state)

            if set_size not in result["set_sizes"]:
                result["set_sizes"][set_size] = 0
            result["set_sizes"][set_size] += 1

            if set_size not in result["correct_by_sizes"]:
                result["correct_by_sizes"][set_size] = 0
            if pred == label:
                result["correct_by_sizes"][set_size] += 1

            if set_size not in result["time_seconds_by_sizes"]:
                result["time_seconds_by_sizes"][set_size] = 0
            result["time_seconds_by_sizes"][set_size] += consumed_time

            # print("\n\ninputs:")
            # print(pair[0])
            # print("incon_sets:", pair[1])
            # print("set_total:", pair[2])
            # print("-----")
            # print("combinations:")
            # for comb in inputs_comb:
            #     print(comb)
            # print("-----")
            # print("preds:", preds)
            # print("prediction_final:", pred)
            # print("-----")
            # print("probs")
            # print(additional_info['probs'])

    # loss
    result["eval_loss"] = total_loss / len(dataloader)

    acc = len(
        [
            i
            for i in range(len(result["pred"]))
            if result["pred"][i] == result["gold"][i]
        ]
    ) / len(result["pred"])

    result["eval_acc"] = acc

    return result


def evaluate_energy_one_to_one(
    energynet, dataloader, side=None, device="cpu", params=None
):

    energynet.eval()
    cls_token = energynet.representation_model.tokenizer.cls_token
    sep_token = "."
    total_loss = 0
    result = {
        "eval_loss": 0,
        "pred": [],
        "gold": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    for pairs in tqdm(dataloader):

        for pair in pairs:

            pred_one_example = []
            # print("pair:", pair)
            inputs = pair[0]
            set_size = len(pair[2])

            inputs_comb = comb_input(inputs, cls_token=cls_token, sep_token=sep_token)
            start = time.time()
            for idx in range(0, len(inputs_comb), 4):
                temp_input = inputs_comb[idx : idx + 4]
                # print("temp_input:")
                # print(temp_input)
                e_vals, _ = energynet.energy_model(temp_input, pair_only=True)
                pred_one_example.extend([float(e_v) for e_v in e_vals])
            consumed_time = time.time() - start

            incon_preds = [e for e in pred_one_example if (e - energynet.threshold) > 0]
            preds = [int(e - energynet.threshold > 0) for e in pred_one_example]

            if sum(pair[1]) > 0:
                label = 1
            else:
                label = 0
            # print("inputs_comb[0]:", inputs_comb[0])
            # loss, additional_info = energynet((inputs_comb, None))
            pred = 0 if len(incon_preds) <= 0 else 1
            result["pred"].extend([pred])
            result["gold"].extend([label])

            if set_size not in result["set_sizes"]:
                result["set_sizes"][set_size] = 0
            result["set_sizes"][set_size] += 1

            if set_size not in result["correct_by_sizes"]:
                result["correct_by_sizes"][set_size] = 0
            if pred == label:
                result["correct_by_sizes"][set_size] += 1

            if set_size not in result["time_seconds_by_sizes"]:
                result["time_seconds_by_sizes"][set_size] = 0
            result["time_seconds_by_sizes"][set_size] += consumed_time
            # print("\n\ninputs:")
            # print(pair[0])
            # print("incon_sets:", pair[1])
            # print("set_total:", pair[2])
            # print("-----")
            # print("combinations:")
            # for comb in inputs_comb:
            #     print(comb)
            # print("-----")
            # print("preds:", preds)
            # print("prediction_final:", pred)
            # print("-----")
            # print("e_val")
            # print(e_vals)

    # loss
    result["eval_loss"] = total_loss / len(dataloader)

    acc = len(
        [
            i
            for i in range(len(result["pred"]))
            if result["pred"][i] == result["gold"][i]
        ]
    ) / len(result["pred"])

    result["eval_acc"] = acc

    return result


def evaluate_energy_many_to_one(
    energynet, dataloader, side=None, device="cpu", params=None
):

    energynet.eval()
    cls_token = energynet.representation_model.tokenizer.cls_token
    sep_token = "."
    total_loss = 0
    result = {
        "eval_loss": 0,
        "pred": [],
        "gold": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    for pairs in tqdm(dataloader):

        for pair in pairs:

            pred_one_example = []
            # print("pair:", pair)
            inputs = pair[0]
            set_size = len(pair[2])

            inputs_comb = mto_comb_input(inputs)
            start = time.time()
            for idx in range(0, len(inputs_comb), 4):
                temp_input = inputs_comb[idx : idx + 4]
                # print("temp_input:")
                # print(temp_input)
                e_vals, _ = energynet.energy_model(temp_input, pair_only=True)
                pred_one_example.extend([float(e_v) for e_v in e_vals])
            consumed_time = time.time() - start

            incon_preds = [e for e in pred_one_example if (e - energynet.threshold) > 0]
            preds = [int(e - energynet.threshold > 0) for e in pred_one_example]

            if sum(pair[1]) > 0:
                label = 1
            else:
                label = 0
            # print("inputs_comb[0]:", inputs_comb[0])
            # loss, additional_info = energynet((inputs_comb, None))
            pred = 0 if len(incon_preds) <= 0 else 1
            result["pred"].extend([pred])
            result["gold"].extend([label])

            if set_size not in result["set_sizes"]:
                result["set_sizes"][set_size] = 0
            result["set_sizes"][set_size] += 1

            if set_size not in result["correct_by_sizes"]:
                result["correct_by_sizes"][set_size] = 0
            if pred == label:
                result["correct_by_sizes"][set_size] += 1

            if set_size not in result["time_seconds_by_sizes"]:
                result["time_seconds_by_sizes"][set_size] = 0
            result["time_seconds_by_sizes"][set_size] += consumed_time
            # print("\n\ninputs:")
            # print(pair[0])
            # print("incon_sets:", pair[1])
            # print("set_total:", pair[2])
            # print("-----")
            # print("combinations:")
            # for comb in inputs_comb:
            #     print(comb)
            # print("-----")
            # print("preds:", preds)
            # print("prediction_final:", pred)
            # print("-----")
            # print("e_val")
            # print(e_vals)

    # loss
    result["eval_loss"] = total_loss / len(dataloader)

    acc = len(
        [
            i
            for i in range(len(result["pred"]))
            if result["pred"][i] == result["gold"][i]
        ]
    ) / len(result["pred"])

    result["eval_acc"] = acc

    return result


def evaluate_supervised_many_to_one(
    energynet, dataloader, side=None, device="cpu", params=None
):

    energynet.eval()
    cls_token = energynet.representation_model.tokenizer.cls_token
    sep_token = "."
    total_loss = 0
    result = {
        "eval_loss": 0,
        "pred": [],
        "gold": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    for pairs in tqdm(dataloader):

        for pair in pairs:

            pred_one_example = []

            inputs = pair[0]
            set_size = len(pair[2])

            inputs_comb = mto_comb_input(inputs, cls_token=cls_token)

            start = time.time()
            for idx in range(0, len(inputs_comb), 4):
                temp_input = inputs_comb[idx : idx + 4]
                # print("temp_input:")
                # print(temp_input)
                loss, additional_info = energynet((temp_input, None), pair_only=True)
                pred_one_example.extend(additional_info["pred"])
            consumed_time = time.time() - start

            if len(pair[1]) > 0:
                label = 1
            else:
                label = 0
            # print("inputs_comb[0]:", inputs_comb[0])
            # loss, additional_info = energynet((inputs_comb, None))
            preds = pred_one_example
            pred = 0 if sum(preds) <= 0 else 1
            result["pred"].extend([pred])
            result["gold"].extend([label])
            # result['neg_probs'].extend([p[1] for p in probs]) # we only store dim_1 values. (which means the probability for inconsistent state)

            if set_size not in result["set_sizes"]:
                result["set_sizes"][set_size] = 0
            result["set_sizes"][set_size] += 1

            if set_size not in result["correct_by_sizes"]:
                result["correct_by_sizes"][set_size] = 0
            if pred == label:
                result["correct_by_sizes"][set_size] += 1

            if set_size not in result["time_seconds_by_sizes"]:
                result["time_seconds_by_sizes"][set_size] = 0
            result["time_seconds_by_sizes"][set_size] += consumed_time

            # print("\n\ninputs:")
            # print(pair[0])
            # print("incon_sets:", pair[1])
            # print("set_total:", pair[2])
            # print("-----")
            # print("combinations:")
            # for comb in inputs_comb:
            #     print(comb)
            # print("-----")
            # print("preds:", preds)
            # print("prediction_final:", pred)
            # print("-----")
            # print("probs")
            # print(additional_info['probs'])

    # loss
    result["eval_loss"] = total_loss / len(dataloader)

    acc = len(
        [
            i
            for i in range(len(result["pred"]))
            if result["pred"][i] == result["gold"][i]
        ]
    ) / len(result["pred"])

    result["eval_acc"] = acc

    return result


def one_to_one_maximum_tolerance_ratio(
    energynet, seq_of_dataloaders, labels, device="cpu", params=None
):

    energynet.eval()
    cls_token = energynet.representation_model.tokenizer.cls_token
    sep_token = "."
    total_loss = 0
    result = {
        "eval_loss": 0,
        "pred": [],
        "gold": [],
        "set_sizes": {},
        "correct_by_sizes": {},
        "time_seconds_by_sizes": {},
    }

    set_size_incon_ratios = {}

    assert len(seq_of_dataloaders) == len(labels)

    for i, dataloader in enumerate(seq_of_dataloaders):

        for pairs in tqdm(dataloader):
            assert len(pairs) == 1

            for pair in pairs:

                pred_one_example = []
                # print("pair:", pair)
                inputs = pair[0]
                set_size = len(pair[2])

                inputs_comb = comb_input(
                    inputs, cls_token=cls_token, sep_token=sep_token
                )
                start = time.time()
                for idx in range(0, len(inputs_comb), 4):
                    temp_input = inputs_comb[idx : idx + 4]
                    # print("temp_input:")
                    # print(temp_input)
                    e_vals, _ = energynet.energy_model(temp_input, pair_only=True)
                    pred_one_example.extend([float(e_v[-1]) for e_v in e_vals])
                consumed_time = time.time() - start

                incon_preds = [
                    e for e in pred_one_example if (e - energynet.threshold) > 0
                ]

                if sum(pair[1]) > 0:
                    label = 1
                else:
                    label = 0
                # print("inputs_comb[0]:", inputs_comb[0])
                # loss, additional_info = energynet((inputs_comb, None))
                pred = 0 if len(incon_preds) <= 0 else 1
                incon_ratio = len(incon_preds) / len(inputs_comb)
                result["pred"].extend([pred])
                result["gold"].extend([label])

                if set_size not in result["set_sizes"]:
                    result["set_sizes"][set_size] = 0
                    set_size_incon_ratios[set_size] = {}
                    set_size_incon_ratios[set_size]["consistent"] = []
                    set_size_incon_ratios[set_size]["inconsistent"] = []
                result["set_sizes"][set_size] += 1
                if label == 0:
                    set_size_incon_ratios[set_size]["consistent"].append(incon_ratio)
                if label == 1:
                    set_size_incon_ratios[set_size]["inconsistent"].append(incon_ratio)

                if set_size not in result["correct_by_sizes"]:
                    result["correct_by_sizes"][set_size] = 0
                if pred == label:
                    result["correct_by_sizes"][set_size] += 1

                if set_size not in result["time_seconds_by_sizes"]:
                    result["time_seconds_by_sizes"][set_size] = 0
                result["time_seconds_by_sizes"][set_size] += consumed_time

    tol_acc = {}
    tol_f1 = {}
    for set_size in result["set_sizes"].keys():
        tol_acc[set_size] = []
        tol_f1[set_size] = []
    best_tolerance_acc = []
    best_tolerance_f1 = []
    for set_size in result["set_sizes"].keys():
        tolerance_ratio_result = tolerance_ratio_analysis(
            set_size_incon_ratios[set_size]["consistent"],
            set_size_incon_ratios[set_size]["inconsistent"],
        )
        tol_acc[set_size].extend(tolerance_ratio_result["overal_acc"])
        tol_f1[set_size].extend(tolerance_ratio_result["overal_f1"])
        best_tolerance_acc.append(tolerance_ratio_result["acc"])
        best_tolerance_f1.append(tolerance_ratio_result["f1"])

    consistent_set_total = []
    inconsistent_set_total = []
    for set_size in result["set_sizes"].keys():
        consistent_set_total.extend(set_size_incon_ratios[set_size]["consistent"])
        inconsistent_set_total.extend(set_size_incon_ratios[set_size]["inconsistent"])
    tolerance_ratio_result = tolerance_ratio_analysis(
        consistent_set_total, inconsistent_set_total
    )
    tol_acc["total"] = tolerance_ratio_result["overal_acc"]
    tol_f1["total"] = tolerance_ratio_result["overal_f1"]
    best_tolerance_acc.append(tolerance_ratio_result["acc"])
    best_tolerance_f1.append(tolerance_ratio_result["f1"])

    result["tol_acc"] = tol_acc
    result["tol_f1"] = tol_f1
    result["best_tolerance_acc"] = best_tolerance_acc
    result["best_tolerance_f1"] = best_tolerance_f1

    # loss
    result["eval_loss"] = total_loss / len(dataloader)

    acc = len(
        [
            i
            for i in range(len(result["pred"]))
            if result["pred"][i] == result["gold"][i]
        ]
    ) / len(result["pred"])

    result["eval_acc"] = acc

    return result


def evaluate_energy_arbitrary(
    energynet, pos_dataloader, neg_dataloader, device="cpu", params=None
):
    print("[eval_arbitrary]")
    energynet.eval()
    total_loss = 0
    result = {"eval_loss": 0, "e_pos": [], "e_neg": [], "pred": [], "gold": []}

    # inference
    for pairs in tqdm(zip(pos_dataloader, neg_dataloader)):
        loss, additional_info = energynet(pairs)
        if params["energynet"]["output_form"] == "real_num":
            e_pos = additional_info.get("e_pos", 0)
            e_neg = additional_info.get("e_neg", 0)
            result["e_pos"].append(float(e_pos[-1]))
            result["e_neg"].append(float(e_neg[-1]))
        elif params["energynet"]["output_form"] == "2dim_vec":
            pred, gold = additional_info["pred"], additional_info["gold"]
            result["pred"].extend(pred)
            result["gold"].extend(gold)
        total_loss += loss.detach().item()

    # loss
    result["eval_loss"] = total_loss / len(pos_dataloader)

    if params["energynet"]["output_form"] == "real_num":
        # hist plot
        plot_hist_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_hist.svg"
        )
        plot_hist_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_hist.png"
        )

        # ROC plot
        plot_roc_curve_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_ROC.svg"
        )
        auroc = plot_roc_curve_real_num(
            result["e_pos"], result["e_neg"], params["model_path"][:-1] + "_ROC.png"
        )
        thresholds = determine_best_split(result["e_pos"], result["e_neg"])
        precision = thresholds["precision"]
        recall = thresholds["recall"]
        f1 = thresholds["f1"]
        acc_by_eval = thresholds["acc"]

        correct_by_trained_threshold = len(
            [e for e in result["e_pos"] if e <= energynet.threshold]
            + [e for e in result["e_neg"] if e > energynet.threshold]
        )
        acc = correct_by_trained_threshold / (
            len(result["e_pos"]) + len(result["e_neg"])
        )

    elif params["energynet"]["output_form"] == "2dim_vec":
        auroc = 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(
            result["gold"], result["pred"], average="macro"
        )
        acc = accuracy_score(result["gold"], result["pred"])

    result["auroc"] = auroc
    result["eval_precision"] = precision
    result["eval_recall"] = recall
    result["eval_f1"] = f1
    result = {
        "auroc": result["auroc"],
        "eval_precision": result["eval_precision"],
        "eval_recall": result["eval_recall"],
        "eval_f1": result["eval_f1"],
        "eval_loss": result["eval_loss"],
        "eval_acc": acc,
    }

    return result


def evaluate_energy_two_pairs(
    energynet, pos_dataloader, neg_dataloader, device="cpu", params=None
):

    energynet.eval()
    cls_token = energynet.representation_model.tokenizer.cls_token
    sep_token = "."
    total_loss = 0
    result = {"eval_loss": 0, "e_pos": [], "e_neg": [], "pred": [], "gold": []}
    if params["energynet"]["output_form"] == "real_num":
        # inference
        for pairs in tqdm(zip(pos_dataloader, neg_dataloader)):
            pos = pairs[0]
            neg = pairs[1]
            # print("pos:", pos)

            for pair in pos:
                # print("pair:", pair)
                pos_inputs = pair[0]
                pos_inputs_comb = comb_input(
                    pos_inputs, cls_token=cls_token, sep_token=sep_token
                )
                # print("pos_inputs_comb[0]:", pos_inputs_comb[0])
                e_pos_out, _ = energynet.energy_model(pos_inputs_comb, pair_only=True)
                e_pos_incon_pred = [
                    float(e) for e in e_pos_out if float(e) > energynet.threshold
                ]
                e_pos_pred = (
                    0 if len(e_pos_incon_pred) <= (len(pos_inputs) - 1) / 2 else 1
                )
                result["pred"].extend([e_pos_pred])
                result["gold"].extend([0])

            for pair in neg:
                neg_inputs = pair[0]
                neg_inputs_comb = comb_input(
                    neg_inputs, cls_token=cls_token, sep_token=sep_token
                )
                e_neg_out, _ = energynet.energy_model(neg_inputs_comb, pair_only=True)
                e_neg_incon_pred = [
                    float(e) for e in e_neg_out if float(e) > energynet.threshold
                ]
                e_neg_pred = (
                    0 if len(e_neg_incon_pred) <= (len(neg_inputs) - 1) / 2 else 1
                )
                result["pred"].extend([e_neg_pred])
                result["gold"].extend([1])

    elif params["energynet"]["output_form"] == "2dim_vec":
        for pairs in tqdm(zip(pos_dataloader, neg_dataloader)):

            pos = pairs[0]
            neg = pairs[1]
            # print("pos:", pos)

            for pair in pos:
                # print("pair:", pair)
                pos_inputs = pair[0]
                pos_inputs_comb = comb_input(
                    pos_inputs, cls_token=cls_token, sep_token=sep_token
                )
                # print("pos_inputs_comb[0]:", pos_inputs_comb[0])
                loss, additional_info = energynet((pos_inputs_comb, None))
                pred, gold = additional_info["pred"], additional_info["gold"]
                pos_pred = 0 if len(pred) < 2 else 1
                result["pred"].extend([pos_pred])
                result["gold"].extend([0])

            for pair in neg:
                neg_inputs = pair[0]
                neg_inputs_comb = comb_input(
                    neg_inputs, cls_token=cls_token, sep_token=sep_token
                )
                loss, additional_info = energynet((None, neg_inputs_comb))
                pred, gold = additional_info["pred"], additional_info["gold"]
                neg_pred = 0 if len(pred) < 2 else 1
                result["pred"].extend([neg_pred])
                result["gold"].extend([1])

    # loss
    result["eval_loss"] = total_loss / len(pos_dataloader)

    acc = len(
        [
            i
            for i in range(len(result["pred"]))
            if result["pred"][i] == result["gold"][i]
        ]
    ) / len(result["pred"])

    result = {"eval_loss": result["eval_loss"], "eval_acc": acc}

    return result


def evaluate_baseline(model, dataloader, label=None, device="cpu", params=None):

    result = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }
    pred = []
    gold = []  # 0: consistent, 1: inconsistent
    time_seconds = {}
    set_sizes = {}
    accuracy_by_sizes = {}
    error_count = 0

    if label != None:
        if type(label) == int:
            pre_defined_gold = [label for _ in range(len(dataloader))]
        else:
            assert len(label) == len(dataloader)
            pre_defined_gold = label

    for i, pairs in enumerate(dataloader):
        # assert len(pairs) == 1

        # print("pairs:", pairs)
        start = time.time()
        evaluate_result = model.predict(pairs)

        time_second = time.time() - start
        set_size = len(pairs[0][0])
        if set_size not in set_sizes:
            set_sizes[set_size] = 0
        set_sizes[set_size] += 1
        if set_size not in time_seconds:
            time_seconds[set_size] = 0
        time_seconds[set_size] += time_second
        pred.extend(evaluate_result["pred"])
        gold.extend(evaluate_result["gold"])

        if set_size not in accuracy_by_sizes:
            accuracy_by_sizes[set_size] = 0
        if evaluate_result["pred"] == evaluate_result["gold"]:
            accuracy_by_sizes[set_size] += 1

        if i % 100 == 0 or i == len(dataloader) - 1:
            if label == None:
                accuracy = accuracy_score(gold, pred, normalize=True)
            else:
                accuracy = accuracy_score(
                    pre_defined_gold[: len(pred)], pred, normalize=True
                )

            print(f"accurcy for i={i}:", accuracy)

        ######################################################
        # if evaluate_result['pred'] != evaluate_result['gold']:
        #     error_count +=1
        #     print("====start =======error===========")
        #     print(pairs)
        #     print("===========error===== end ======")
        #     print()

        # if error_count > 5:
        #     break
        ######################################################

    # precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, average = 'macro')
    if label == None:
        accuracy = accuracy_score(gold, pred, normalize=True)
    else:
        accuracy = accuracy_score(pre_defined_gold, pred, normalize=True)
    # result['precision'] = precision
    # result['recall'] = recall
    # result['f1'] = f1
    result["accuracy"] = accuracy

    for size in time_seconds.keys():
        time_seconds[size] = time_seconds[size] / set_sizes[size]
    for size in time_seconds.keys():
        accuracy_by_sizes[size] = accuracy_by_sizes[size] / set_sizes[size]

    result["time_seconds"] = time_seconds
    result["set_sizes"] = set_sizes
    result["accuracy_by_sizes"] = accuracy_by_sizes

    return result


def locate_energy(locate_net, dataloader, device="cpu", params=None):

    locate_net.eval()
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    data_len = 0
    result = {"eval_loss": 0, "e_pos": [], "e_neg": []}
    original_text = []
    pred_list = []
    gold_list = []
    result_text = []

    if params["locate"]["type"] == "subtraction":

        # inference
        for pairs in tqdm(dataloader):
            accuracy, additional_info = locate_net(pairs)
            if additional_info["pair_num"] >= 4:
                data_len += 1
            else:
                continue

            total_accuracy += accuracy
            total_precision += additional_info["precision"]
            total_recall += additional_info["recall"]
            total_f1 += additional_info["f1"]
            inputs = [p[0] for p in pairs]
            original_text.extend(inputs)
            pred_list.extend(additional_info["pred"])
            gold_list.extend(additional_info["gold"])

    # loss
    if data_len == 0:
        acc = 1
        precision = 1
        recall = 1
        f1 = 1
    else:
        acc = total_accuracy / data_len
        precision = total_precision / data_len
        recall = total_recall / data_len
        f1 = total_f1 / data_len
    result["accuracy"] = acc
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["gold"] = gold_list
    result["original_text"] = original_text
    result["masked_text"] = result_text
    result["pred"] = pred_list

    return result


def locate_baseline(model, dataloader, device="cpu", params=None):
    """Evaluate the locate performance of baseline models such as GPT.

    Args:
        model (_type_): _description_
        pos_dataloader (_type_): _description_
        neg_dataloader (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.
        params (_type_, optional): _description_. Defaults to None.

    """
    result = {}

    time_seconds = {}
    set_sizes = {}
    accuracy_by_sizes = {}

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    data_len = 0

    for i, pairs in enumerate(dataloader):
        # assert len(pairs) == 1

        # print("pairs:", pairs)
        start = time.time()
        locate_result = model.locate(pairs)
        if locate_result["pair_num"] != 2:
            data_len += 1
        else:
            continue
        total_accuracy += locate_result["accuracy"]
        total_precision += locate_result["precision"]
        total_recall += locate_result["recall"]
        total_f1 += locate_result["f1"]

        ######################################################
        # if locate_result['pred'] != locate_result['gold']:
        #     error_count +=1
        #     print("====start =======error===========")
        #     print(pairs)
        #     print("===========error===== end ======")
        #     print()

        # if error_count > 5:
        #     break
        ######################################################

    if data_len == 0:
        acc = 1
        precision = 1
        recall = 1
        f1 = 1
    else:
        acc = total_accuracy / data_len
        precision = total_precision / data_len
        recall = total_recall / data_len
        f1 = total_f1 / data_len
    result["accuracy"] = acc
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1

    return result


def fine_tune_energy(
    energynet_finetune,
    energynet_pre_trained,
    opt,
    pos_dataloader,
    neg_dataloader,
    learn_threshold=True,
    device="cpu",
    params=None,
):
    """
    Assumes the dataloader mixes original and fine-tune data simultaneously (mixed_dataset).

    """
    energynet_finetune.train()
    total_loss = 0
    e_pos = []
    e_neg = []
    threshold_result = {}

    for pairs in tqdm(zip(pos_dataloader, neg_dataloader)):
        # print("pairs:")
        # print(pairs)
        loss, additional_info = energynet_finetune(pairs)

        # print(f"loss: {loss}")
        opt.zero_grad()
        (loss).backward()
        opt.step()

        total_loss += loss.detach().item()
        e_pos.append(float(additional_info.get("e_pos", 0)))
        e_neg.append(float(additional_info.get("e_neg", 0)))

    threshold_result = determine_best_split(e_pos, e_neg)
    # for key,val in threshold_result.items():
    #     print(key)
    #     print(val)
    #     print()
    if learn_threshold:
        energynet_finetune.threshold = threshold_result.get("acc_threshold", 0)
    return total_loss / len(pos_dataloader), threshold_result


def fine_tune_energy_with_l2(
    energynet_finetune,
    energynet_pre_trained,
    opt,
    pos_dataloader,
    neg_dataloader,
    learn_threshold=True,
    device="cpu",
    params=None,
):

    energynet_finetune.train()
    total_loss = 0
    e_pos = []
    e_neg = []
    threshold_result = {}

    for pairs in tqdm(zip(pos_dataloader, neg_dataloader)):
        # print("pairs:")
        # print(pairs)
        loss, additional_info = energynet_finetune(pairs)

        # L2 Regularization Term
        l2_loss = 0.0
        for (name, param), (_, pretrained_param) in zip(
            energynet_finetune.named_parameters(), energynet_pre_trained.items()
        ):
            if param.requires_grad:
                l2_loss += torch.norm(param - pretrained_param, p=2) ** 2

        # print(f"loss: {loss}")
        opt.zero_grad()
        (loss + 0.00001 * l2_loss).backward()
        opt.step()

        total_loss += loss.detach().item()
        e_pos.append(float(additional_info.get("e_pos", 0)))
        e_neg.append(float(additional_info.get("e_neg", 0)))

    threshold_result = determine_best_split(e_pos, e_neg)
    # for key,val in threshold_result.items():
    #     print(key)
    #     print(val)
    #     print()
    if learn_threshold:
        energynet_finetune.threshold = threshold_result.get("acc_threshold", 0)
    return total_loss / len(pos_dataloader), threshold_result


def plot_hist_real_num(array_positive, array_negative, path_to_save_figure):
    plt.hist([array_positive, array_negative], label=["gold", "negative"])
    plt.legend(loc="upper left")
    plt.title("hist")
    plt.savefig(path_to_save_figure)
    plt.close()


def plot_roc_curve_real_num(array_positive, array_negative, path_to_save_figure):
    # Combine the positive and negative arrays
    y_true = np.array([1] * len(array_positive) + [0] * len(array_negative))
    pos = [-1 * a for a in array_positive]
    neg = [-1 * a for a in array_negative]
    scores = np.array(pos + neg)

    # Calculate FPR and TPR for various thresholds
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auroc = roc_auc_score(y_true, scores)

    # Plotting the ROC curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve with AUROC={auroc:.4f}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save the plot to the specified path
    plt.savefig(path_to_save_figure)
    plt.close()
    return auroc


def determine_best_split(array_positive, array_negative):
    y_true = np.array([1] * len(array_positive) + [0] * len(array_negative))
    pos = [a for a in array_positive]
    neg = [a for a in array_negative]

    (
        best_precision_threshold_list,
        best_recall_threshold_list,
        best_f1_threshold_list,
        best_acc_threshold_list,
    ) = ([0], [0], [0], [0])
    best_precision, best_recall, best_f1, best_acc = 0, 0, 0, 0

    if (len(pos) == 0) and (len(neg) == 0):
        return {
            "precision": best_precision,
            "precision_threshold": min(best_precision_threshold_list),
            "recall": best_recall,
            "recall_threshold": max(best_recall_threshold_list),
            "f1": best_f1,
            "f1_threshold": sum(best_f1_threshold_list) / len(best_f1_threshold_list),
            "acc": best_acc,
            "acc_threshold": sum(best_acc_threshold_list)
            / len(best_acc_threshold_list),
        }
    if all([p == 0 for p in pos]) and all([n == 0 for n in neg]):
        return {
            "precision": best_precision,
            "precision_threshold": min(best_precision_threshold_list),
            "recall": best_recall,
            "recall_threshold": max(best_recall_threshold_list),
            "f1": best_f1,
            "f1_threshold": sum(best_f1_threshold_list) / len(best_f1_threshold_list),
            "acc": best_acc,
            "acc_threshold": sum(best_acc_threshold_list)
            / len(best_acc_threshold_list),
        }

    scores = np.array(pos + neg)

    threshold_min = float(min(scores))
    threshold_max = float(max(scores))
    difference = threshold_max - threshold_min

    threshold = threshold_min
    while threshold < threshold_max:
        threshold += difference / 100
        y_pred = np.where(scores > threshold, 0, 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        accuracy = accuracy_score(y_true, y_pred)

        # estimating precision
        if best_precision < precision:
            best_precision = precision
            best_precision_threshold_list = [threshold]
        elif best_precision == precision:
            best_precision_threshold_list.append(threshold)

        # estimating recall
        if best_recall < recall:
            best_recall = recall
            best_recall_threshold_list = [threshold]
        elif best_recall == recall:
            best_recall_threshold_list.append(threshold)

        # estimating f1
        if best_f1 < f1:
            best_f1 = f1
            best_f1_threshold_list = [threshold]
        elif best_f1 == f1:
            best_f1_threshold_list.append(threshold)

        # estimating acc
        if best_acc < accuracy:
            best_acc = accuracy
            best_acc_threshold_list = [threshold]
        elif best_acc == accuracy:
            best_acc_threshold_list.append(threshold)

    return {
        "precision": best_precision,
        "precision_threshold": min(best_precision_threshold_list),
        "recall": best_recall,
        "recall_threshold": max(best_recall_threshold_list),
        "f1": best_f1,
        "f1_threshold": sum(best_f1_threshold_list) / len(best_f1_threshold_list),
        "acc": best_acc,
        "acc_threshold": sum(best_acc_threshold_list) / len(best_acc_threshold_list),
    }


def learn_energy_threshold(energynet, seq_of_dataloaders, labels):

    energynet.eval()

    e_vals_pos = []
    e_vals_neg = []
    assert len(seq_of_dataloaders) == len(labels)

    for i, dataloader in enumerate(seq_of_dataloaders):
        for pairs in tqdm(dataloader):
            e_val, _ = energynet.energy_model(pairs)
            if energynet.output_form == "2dim_vec":
                e_val = softmax(e_val, dim=-1)[0][-1]
            if labels[i] == 0:
                e_vals_pos.append(float(e_val))
            elif labels[i] == 1:
                e_vals_neg.append(float(e_val))

    split = determine_best_split(e_vals_pos, e_vals_neg)
    energynet.threshold = split["acc_threshold"]
    return split["acc_threshold"], split["acc"]


def learn_2dimvec_threshold(energynet, dataloader):
    energynet.eval()

    e_vals_pos = []
    e_vals_neg = []

    for i, pairs in enumerate(dataloader):
        e_val, _ = energynet.energy_model(pairs)
        e_val = softmax(e_val, dim=-1)[0][-1]
        label = int(len(pairs[0][1]) > 0)
        if label == 0:
            e_vals_pos.append(float(e_val))
        elif label == 1:
            e_vals_neg.append(float(e_val))

    split = determine_best_split(e_vals_pos, e_vals_neg)
    energynet.threshold = split["acc_threshold"]
    return split["acc_threshold"], split["acc"]


def comb_input(long_text, cls_token="<s>", sep_token="."):
    out = long_text[len(cls_token) :].split(sep_token)[:-1]
    out = list(itertools.combinations(out, 2))
    out = [f"{cls_token}{o[0]}{sep_token}{o[1]}{sep_token}" for o in out]
    return out


def mto_comb_input(long_text, cls_token="<s>", sep_token="</s>", period="."):
    out = long_text[len(cls_token) :].split(period)[:-1]
    comb_result = []
    for i in range(len(out)):
        tmp = cls_token
        for j in range(len(out)):
            if j != i:
                tmp += f"{out[j]}{period}"
        tmp += f" {sep_token}{out[i]}{period}"
        comb_result.append(tmp)
    return comb_result


def cal_length(long_text, cls_token="<s>", sep_token="."):
    out = long_text[len(cls_token) :].split(sep_token)[:-1]
    return len(out)


def tolerance_ratio_analysis(consistent_list, inconsistent_list):
    y_true = np.array([0] * len(consistent_list) + [1] * len(inconsistent_list))
    pos = [a for a in consistent_list]
    neg = [a for a in inconsistent_list]

    (
        best_precision_threshold_list,
        best_recall_threshold_list,
        best_f1_threshold_list,
        best_acc_threshold_list,
    ) = ([0], [0], [0], [0])
    best_precision, best_recall, best_f1, best_acc = 0, 0, 0, 0

    overal_acc = []
    overal_f1 = []

    scores = np.array(pos + neg)

    for threshold in [0.05 * n for n in range(21)]:
        y_pred = np.where(scores <= threshold, 0, 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        accuracy = accuracy_score(y_true, y_pred)
        overal_acc.append(accuracy)
        overal_f1.append(f1)

        # estimating precision
        if best_precision < precision:
            best_precision = precision
            best_precision_threshold_list = [threshold]
        elif best_precision == precision:
            best_precision_threshold_list.append(threshold)

        # estimating recall
        if best_recall < recall:
            best_recall = recall
            best_recall_threshold_list = [threshold]
        elif best_recall == recall:
            best_recall_threshold_list.append(threshold)

        # estimating f1
        if best_f1 < f1:
            best_f1 = f1
            best_f1_threshold_list = [threshold]
        elif best_f1 == f1:
            best_f1_threshold_list.append(threshold)

        # estimating acc
        if best_acc < accuracy:
            best_acc = accuracy
            best_acc_threshold_list = [threshold]
        elif best_acc == accuracy:
            best_acc_threshold_list.append(threshold)

    return {
        "f1": best_f1,
        "f1_threshold": sum(best_f1_threshold_list) / len(best_f1_threshold_list),
        "acc": best_acc,
        "acc_threshold": sum(best_acc_threshold_list) / len(best_acc_threshold_list),
        "overal_acc": overal_acc,
        "overal_f1": overal_f1,
    }
