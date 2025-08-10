import itertools, random
import numpy as np
from torch.utils.data import Dataset
from .multi_hop.musique import musique_dataset
from .vqa.lconvqa import l_convqa_dataset, l_convqa_fine_grained_dataset
from .vqa.lconvqa import l_convqa_dataset
from .nli.set_nli import set_nli_dataset


def dataset_loader(
    params,
    split="train",
    consistency="con",
    additional_info={"pairwise": "arbitrary_pairs", "load_full": False},
):
    if params["task"] == "vqa":
        if params["dataset"] == "lconvqa":
            dataset = l_convqa_dataset(
                params, split, consistency, additional_info["pairwise"]
            )

    elif params["task"] == "multi-hop":
        if params["dataset"] == "musique":
            dataset = musique_dataset(params, split)

    elif params["task"] == "nli":
        if params["dataset"] == "set_nli":
            dataset = set_nli_dataset(
                params,
                split,
                consistency,
                additional_info["pairwise"],
                additional_info["load_full"],
            )

    return dataset


def dataset_loader_fine_grained(
    params,
    split="train",
    semantic="con",
    pairwise="two_pairs",
    load_in_once_everything=False,
    output_form="real_num",
    input_type="set",
):
    if params["finetune_job_id"] == 0:
        if params["task"] == "mixed":
            seq_of_dataset = []
            seq_of_data_name = []
            if "lconvqa" in params["dataset"]:
                seq_of_dataset.append(
                    l_convqa_fine_grained_dataset(
                        params,
                        split,
                        semantic,
                        pairwise,
                        load_in_once_everything,
                        output_form,
                        input_type,
                    )
                )
                seq_of_data_name.append("lconvqa")
            if "set_nli" in params["dataset"]:
                seq_of_dataset.append(
                    set_nli_dataset(
                        params,
                        split,
                        semantic,
                        pairwise,
                        load_in_once_everything,
                        output_form,
                        input_type,
                    )
                )
                seq_of_data_name.append("set_nli")

            dataset = mixed_dataset(params, seq_of_dataset, seq_of_data_name)

        elif params["task"] == "vqa":
            if params["dataset"] == "lconvqa":
                dataset = l_convqa_fine_grained_dataset(
                    params,
                    split,
                    semantic,
                    pairwise,
                    load_in_once_everything,
                    output_form,
                    input_type,
                )

        elif params["task"] == "multi-hop":
            if params["dataset"] == "musique":
                dataset = musique_dataset(params, split)

        elif params["task"] == "nli":
            if params["dataset"] == "set_nli":
                dataset = set_nli_dataset(
                    params,
                    split,
                    semantic,
                    pairwise,
                    load_in_once_everything,
                    output_form,
                    input_type,
                )

    elif params["finetune_job_id"] != 0:
        # We only consider two cases: from nli to lconvqa, from lconvqa to nli
        seq_of_dataset = []
        seq_of_data_name = []
        seq_of_dataset.append(
            l_convqa_fine_grained_dataset(
                params,
                split,
                semantic,
                pairwise,
                load_in_once_everything,
                output_form,
                input_type,
            )
        )
        seq_of_data_name.append("lconvqa")
        seq_of_dataset.append(
            set_nli_dataset(
                params,
                split,
                semantic,
                pairwise,
                load_in_once_everything,
                output_form,
                input_type,
            )
        )
        seq_of_data_name.append("set_nli")

        dataset = mixed_dataset(params, seq_of_dataset, seq_of_data_name)

    # dataset.dataset = [
    #     [
    #       (data_point1, consistency1), (data_point2, consistency2), ...
    #     ],
    #     ....
    #   ]
    return dataset


def transform_arbitrary_pairs_to_diverse_pairs(seq_of_dataset):
    """

    seq_of_dataset collects datasets; any iterable (set, tuple, list) works in Python.
    This function uses only arbitrary con_dataset and arbitrary incon_dataset.
    This produces datasets such as

        # size = 2
        two_pairs con
        two_pairs incon

        # size = arbitrary
        con
        incon

        # concat 2 dataset
        con + con
        con + incon
        incon + incon

        # concat 3 dataset
        con + con + con
        ...
        incon + incon + incon

    Construct datasets of these types


    dataset: l_convqa_fine_grained_dataset.dataset

    for d in dataset: # type(dataset) == list
        for data_point in d: # type(d) = list
            question, answer, consistency = data_point # type(data_point) = tuple
            # type(question): usually string
            # type(answer): usually string
            # type(consistency): bool. Consistency indicates whether this qa pair is consistent or inconsistent with respect to the rest.

    """


def transform_arbitrary_pairs_to_two_pairs(seq_of_dataset):

    transformed_dataset_con = []
    transformed_dataset_incon = []
    set_sizes = {}

    for dataset in seq_of_dataset:
        for data in dataset:
            if len(data) == 2:
                if all([d[2] for d in data]):
                    transformed_dataset_con.append(data)
                else:
                    transformed_dataset_incon.append(data)

            else:
                combinations = list(
                    itertools.combinations(data, 2)
                )  # "len(d)-choose-2"-number of combinations
                for two_pair in combinations:
                    if all([p[2] for p in two_pair]):
                        transformed_dataset_con.append(two_pair)
                    else:
                        transformed_dataset_incon.append(two_pair)

            if len(data) not in set_sizes:
                set_sizes[len(data)] = 0
            set_sizes[len(data)] += 1

    random.Random(1014).shuffle(transformed_dataset_con)
    random.Random(1014).shuffle(transformed_dataset_incon)

    transformed_dataset_con = transformed_dataset_con[
        : min([len(s) for s in seq_of_dataset])
    ]
    transformed_dataset_incon = transformed_dataset_incon[
        : min([len(s) for s in seq_of_dataset])
    ]

    return (
        [
            general_dataset(transformed_dataset_con),
            general_dataset(transformed_dataset_incon),
        ],
        ["con_two_pairs", "incon_two_pairs"],
        set_sizes,
    )


def concat_arbitrary_pairs(seq_of_dataset, concat_num=2):
    """
    seq_of_dataset = [con_dataset, incon_dataset] exactly two
    """

    set_sizes = {}
    data_num = min([len(s) for s in seq_of_dataset])

    index_combinations = list(itertools.combinations(range(data_num), concat_num))

    total_dataset = []
    dataset_names = []

    semantic_combinations = [
        (concat_num - i, i) for i in range(concat_num + 1)
    ]  # enumerate all con/incon mixing ratios
    for s_comb in semantic_combinations:
        one_dataset = []
        random.Random(1014).shuffle(index_combinations)
        idx_combs = index_combinations[
            :data_num
        ]  # e.g., when concat_num=3, idx_combs = [(1,2,3), (3,6,2), ...]

        con_num, incon_num = s_comb  # e.g., con_num, incon_num = (2, 1)
        dataset_names.append("_".join(["con"] * con_num + ["incon"] * incon_num))

        data_name = []
        for indices in idx_combs:
            tmp = []  # one data. Treated as a single data point

            for semantic_idx, data_idx in enumerate(indices):  # e.g., indices = (2,7,6)
                if semantic_idx + 1 <= con_num:
                    tmp += seq_of_dataset[0][data_idx]
                else:
                    tmp += seq_of_dataset[1][data_idx]
            random.shuffle(tmp)

            if len(tmp) not in set_sizes:
                set_sizes[len(tmp)] = 0
            set_sizes[len(tmp)] += 1
            one_dataset.append(tmp)
            if hasattr(seq_of_dataset[0], "data_name"):
                data_name.append(seq_of_dataset[0].data_name[data_idx])

        gen_dataset = general_dataset(one_dataset)
        if hasattr(seq_of_dataset[0], "data_name"):
            gen_dataset.data_name = data_name
        else:
            gen_dataset.data_name = None

        total_dataset.append(gen_dataset)

    return total_dataset, dataset_names, set_sizes


class general_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_name = []

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class mixed_dataset(Dataset):
    def __init__(self, params, seq_of_dataset, seq_of_data_name):
        self.params = params
        self.seed = 1014
        self.seq_of_dataset = seq_of_dataset
        self.seq_of_data_name = seq_of_data_name
        self.dataset = []
        self.data_name = []
        self.initialize()

    def initialize(self):
        data = []
        for data_name, dataset in zip(self.seq_of_data_name, self.seq_of_dataset):
            data += [(d, data_name) for d in dataset.dataset]

        random.Random(self.seed).shuffle(self.dataset)
        self.data_name = [d[1] for d in data]
        self.dataset = [d[0] for d in data]

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
