import os, json, ast
import pandas as pd
from tqdm import tqdm

import torch
from energynets.energynet import energynet

from baselines.baseline_model import baseline_model

## 0. prepare
job_id = 1225068
total_result = {
    "file_id": [],
    "set_size": [],
    "separate_answer_accuracy": [],
    "gold_consistency": [],
    "gold_consistency_by_correct": [],
    "gold_consistency_by_wrong": [],
    "gpt_consistency_check_accuracy": [],
    "energy_consistency_check_accuracy": [],
    "predictions": [],
}  # will be converted into csv later.


set_sizes = {}  # set size : example num
separate_inference_consistency_by_sizes = {}  # set size : correct example num

set_sizes_list = []


separate_inference_consistency_list = []

consistency_result_for_gpt = []  # True if considered consistent, otherwise False
consistency_result_for_energy = []  # True if considered consistent, otherwise False

separate_inference_correct = 0
separate_inference_total = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset folder
data_folder_path = os.path.join("datasets", "wiqa", "datasets_clustered_generated")

files_tmp = os.listdir(data_folder_path)
files_tmp = [f for f in files_tmp if f.endswith("json")]
sizes = {}
files = []

for idx, filename in tqdm(enumerate(files_tmp)):

    valid_file = False

    file_path = os.path.join(data_folder_path, filename)
    with open(file_path, "r") as f:
        one_json_file = json.load(f)
    for qa in one_json_file["qa_pairs"]:
        if "no_effect" != qa["answer_label"]:
            valid_file = True
            break

    if len(one_json_file["qa_pairs"]) > 20:
        valid_file = False

    if valid_file:
        files.append(filename)


print(f"{len(files)}-number of json files")


params = {
    "job_id": job_id,
    "baseline": {
        "type": "llm",
        "model": "deepseek-r1",
        "shot_num": 0,
        "prediction_type": None,
        "do_not_initialize": True,
    },
    "dataset": None,
    "energynet": {
        "repre_model": "roberta",
        "decomposition": "no",
        "loss_type": "margin",
        "loss_fully_separate": False,
        "weight_fully_separate": 0.1,
        "loss_subset_ordering": False,
        "weight_subset_ordering": 0.01,
        "loss_con_incon_ordering": False,
        "weight_con_incon_ordering": 0.1,
        "loss_incon_incon_ordering": False,
        "weight_incon_incon_ordering": 0.1,
        "margin": {"margin": 0.1, "fully_separate_margin": 0.001},
        "output_form": "real_num",
    },
    "locate": {"type": None},
    "device": device,
}

## llm load
model = baseline_model(params, "prediction")

## energynet load
task_dataset_candidates = [
    os.path.join("vqa", "lconvqa"),
    os.path.join("nli", "set_nli"),
    os.path.join("mixed", "lconvqa+set_nli"),
    os.path.join("mixed", "set_nli+lconvqa"),
]

model_path = None
for td in task_dataset_candidates:
    folder_path = os.path.join("results", td)
    if "job_id" in params and params["job_id"] != 0:
        folder_path = os.path.join(folder_path, str(params["job_id"]))
        print("folder_path candidate:", folder_path)
    else:
        folder_path = os.path.join(folder_path, params["time_key"])
        print("folder_path candidate:", folder_path)

    try:
        file_lists = os.listdir(folder_path)
    except:
        continue
    for f in file_lists:
        if f.endswith(".pth"):
            model_path = os.path.join(folder_path, f)
            print("found model path:", model_path)
    if model_path != None:
        break

if model_path == None:
    raise f"No model in {folder_path}"
params["folder_path"] = folder_path
params["model_path"] = model_path
lossNet = energynet(params).to(device)
lossNet.load_state_dict(
    torch.load(params["model_path"], map_location=device)["state_dict"]
)
if "threshold" in torch.load(params["model_path"], map_location=device):
    lossNet.threshold = torch.load(params["model_path"], map_location=device)[
        "threshold"
    ]


prev_result = pd.read_csv(os.path.join(folder_path, "wiqa_result.csv"))


##################################################
for idx, filename in tqdm(enumerate(files)):
    # if idx > 5:
    #     break
    # file_path = os.path.join(data_folder_path, filename)
    # with open(file_path, 'r') as f:
    #     one_json_file = json.load(f)
    filename = str(prev_result["file_id"].iloc[idx])

    print(f"====index {idx}, filename: {filename}====")
    # if idx > 5:
    #     break
    file_path = os.path.join(data_folder_path, filename)
    with open(file_path, "r") as f:
        one_json_file = json.load(f)

    new_qa_pair = []
    for qa in one_json_file["qa_pairs"]:
        if "no_effect" != qa["answer_label"]:
            new_qa_pair.append(qa)

    one_json_file["qa_pairs"] = new_qa_pair

    # inference_result = model.wiqa_separate_inference(one_json_file)
    # size = inference_result['set_size']
    # consistency = inference_result['consistency']
    # predictions = inference_result['pred']

    size = len(one_json_file["qa_pairs"])
    consistency = prev_result["gold_consistency"].iloc[idx]
    predictions = ast.literal_eval(prev_result["predictions"].iloc[idx])
    separate_inference_consistency_list.append(consistency)

    if size not in set_sizes:
        set_sizes[size] = 0
    set_sizes[size] += 1
    set_sizes_list.append(size)

    if size not in separate_inference_consistency_by_sizes:
        separate_inference_consistency_by_sizes[size] = 0
    if consistency:
        separate_inference_consistency_by_sizes[size] += 1

    # separate_inference_total += len(inference_result['gold'])
    # separate_inference_correct += len([
    # i for i in range(len(predictions)) if predictions[i] == inference_result['gold'][i]
    # ])

    ##### incorporate prediction before sending to gpt
    consistency_check_for_llm = model.wiqa_consistency_check_for_pred(
        one_json_file, predictions
    )
    # type(consistency_check_for_llm) == bool
    consistency_result_for_gpt.append(consistency_check_for_llm)

    ##### incorporate prediction before sending to energy net
    energy_input = "<s>"

    qa_pairs = one_json_file["qa_pairs"]
    for idx, qa_pair in enumerate(qa_pairs):
        energy_input += (
            f" {qa_pair['question'][:-1]}? The answer is {predictions[idx]}.\n"
        )

    e_val, _ = lossNet.energy_model([energy_input], pair_only=True)
    if float(e_val) <= lossNet.threshold:
        consistency_check_for_energy = True
    else:
        consistency_check_for_energy = False
    consistency_result_for_energy.append(consistency_check_for_energy)

    total_result["set_size"].append(size)
    total_result["file_id"].append(filename)
    total_result["gold_consistency"].append(consistency)
    # total_result['gold_consistency_by_correct'].append(inference_result['gold_consistency_by_correct'])
    # total_result['gold_consistency_by_wrong'].append(inference_result['gold_consistency_by_wrong'])
    total_result["gpt_consistency_check_accuracy"].append(consistency_check_for_llm)
    total_result["energy_consistency_check_accuracy"].append(
        consistency_check_for_energy
    )
    # total_result['separate_answer_accuracy'].append(len([
    # i for i in range(len(predictions)) if predictions[i] == inference_result['gold'][i]
    # ])/len(inference_result['gold']))
    total_result["predictions"].append(predictions)

#####################################

print("set_sizes")
for size, count in set_sizes.items():
    print(f"size {size}: count {count}")

print("\n--------------\n")

print("separate_inference_consistency_by_sizes")
for size, count in separate_inference_consistency_by_sizes.items():
    print(f"size {size}: count {count}")

print("\n--------------\n")

print("separate_inference_consistency_list")
print(
    f"consistent rate = {sum(separate_inference_consistency_list)} / {len(separate_inference_consistency_list)} = {sum(separate_inference_consistency_list)/len(separate_inference_consistency_list):.4f}"
)

# print(f"separate_inference_correct rate: {separate_inference_correct}/{separate_inference_total} = {separate_inference_correct/separate_inference_total:.4f}")

print("\n--------------\n")

print("gpt separate inference & gpt consistency check")
infodict = {
    "TT": 0,
    "TF": 0,
    "FT": 0,
    "FF": 0,
}
for i in range(len(separate_inference_consistency_list)):
    if separate_inference_consistency_list[i]:
        if consistency_result_for_gpt[i]:
            infodict["TT"] += 1
        else:
            infodict["TF"] += 1
    else:
        if consistency_result_for_gpt[i]:
            infodict["FT"] += 1
        else:
            infodict["FF"] += 1

for key, val in infodict.items():
    print(
        f"key: {key}, val: {val} ({val/len(separate_inference_consistency_list):.4f})"
    )

print("\n--------------\n")

print("gpt separate inference & energy consistency check")
infodict = {
    "TT": 0,
    "TF": 0,
    "FT": 0,
    "FF": 0,
}
for i in range(len(separate_inference_consistency_list)):
    if separate_inference_consistency_list[i]:
        if consistency_result_for_energy[i]:
            infodict["TT"] += 1
        else:
            infodict["TF"] += 1
    else:
        if consistency_result_for_energy[i]:
            infodict["FT"] += 1
        else:
            infodict["FF"] += 1

for key, val in infodict.items():
    print(
        f"key: {key}, val: {val} ({val/len(separate_inference_consistency_list):.4f})"
    )


total_result = pd.DataFrame.from_dict(total_result)
total_result.to_csv(os.path.join(folder_path, "wiqa_result.csv"))
