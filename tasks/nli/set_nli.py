import os, json, pickle, itertools
import numpy as np
import random
from tqdm import tqdm

from torch.utils.data import Dataset

random.seed(1014)


class set_nli_dataset(Dataset):
    def __init__(
        self,
        params,
        mode="train",
        semantic="con",
        pairwise="arbitrary_pairs",
        load_in_once_everything=False,
        output_form="real_num",
        input_type="set",
    ):
        super().__init__()
        self.params = params
        self.pairwise = pairwise
        self.seed = params["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.mode = mode  # one of 'train', 'dev', 'test.
        self.semantic = semantic
        self.output_form = output_form
        self.input_type = input_type
        self.load_everything = load_in_once_everything
        self.dataset_num = {
            "train": self.params["set_nli"]["stepwise_dataset_train_num"],
            "eval": self.params["set_nli"]["stepwise_dataset_eval_num"],
            "eval2": self.params["set_nli"]["stepwise_dataset_eval2_num"],
            "test": self.params["set_nli"]["stepwise_dataset_test_num"],
        }[self.mode]

        self.folder_path = os.path.join("datasets", "set_nli", "final")
        self.filename = "set_nli_data_snli_test_full.json"
        self.processed_dataset_path = os.path.join(
            "datasets", "set_nli", "processed_data"
        )
        self.dataset = None
        self.load_dataset()
        random.Random(self.seed).shuffle(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self):
        required_raw_dataset = [
            f"raw_{mode}_file.pickle" for mode in ["train", "eval", "eval2", "test"]
        ]
        if not all(
            [r in os.listdir(self.processed_dataset_path) for r in required_raw_dataset]
        ):
            self.raw_data_split()

        self.dataset = []
        if self.output_form == "2dim_vec":
            if self.input_type == "set":
                if self.load_everything == True and type(self.load_everything) == bool:
                    # self.dataset += self.load_dataset_two_pairs_con()
                    # self.dataset += self.load_dataset_two_pairs_incon()
                    # self.dataset += self.load_dataset_two_pairs_con_and_incon()
                    # self.dataset += self.load_dataset_two_pairs_incon_and_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_con()
                    self.dataset += self.load_dataset_arbitrary_pairs_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()
                    self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_incon_and_incon()
                elif self.load_everything == "only_con_and_incon":
                    self.dataset = []
                    # self.dataset += self.load_dataset_two_pairs_con()
                    # self.dataset += self.load_dataset_two_pairs_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_con()
                    self.dataset += self.load_dataset_arbitrary_pairs_incon()
            elif self.input_type.lower() in {"oto", "one_to_one"}:
                transformed_dataset = transform_arbitrary_pairs_to_two_pairs(
                    [
                        self.load_dataset_two_pairs_con(),
                        self.load_dataset_two_pairs_incon(),
                    ]
                )
                self.dataset += transformed_dataset[0]
                self.dataset += transformed_dataset[1]

        if self.output_form == "real_num":
            if self.input_type == "set":
                if self.load_everything == True and type(self.load_everything) == bool:
                    self.dataset = []
                    if self.semantic == "con":
                        self.dataset += self.load_dataset_arbitrary_pairs_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()

                        self.dataset += (
                            self.load_dataset_arbitrary_pairs_con_and_incon()
                        )
                        self.dataset += self.load_dataset_arbitrary_pairs_incon()

                    elif self.semantic == "incon":
                        self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        self.dataset += (
                            self.load_dataset_arbitrary_pairs_con_and_incon()
                        )
                        self.dataset += (
                            self.load_dataset_arbitrary_pairs_incon_and_incon()
                        )
                        self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        self.dataset += (
                            self.load_dataset_arbitrary_pairs_con_and_incon()
                        )
                        self.dataset += (
                            self.load_dataset_arbitrary_pairs_incon_and_incon()
                        )

                        self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        self.dataset += (
                            self.load_dataset_arbitrary_pairs_incon_and_incon()
                        )

                else:
                    if self.load_everything == "divide_by_semantic":
                        self.dataset = []
                        if self.semantic == "con":
                            # self.dataset += self.load_dataset_two_pairs_con()
                            self.dataset += self.load_dataset_arbitrary_pairs_con()
                        elif self.semantic == "incon":
                            # self.dataset += self.load_dataset_two_pairs_incon()
                            self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        elif self.semantic == "con_and_incon":
                            # self.dataset += self.load_dataset_two_pairs_con_and_incon()
                            self.dataset += (
                                self.load_dataset_arbitrary_pairs_con_and_incon()
                            )
                        elif self.semantic == "incon_and_incon":
                            # self.dataset += self.load_dataset_two_pairs_incon_and_incon()
                            self.dataset += (
                                self.load_dataset_arbitrary_pairs_incon_and_incon()
                            )
                    elif self.load_everything == "divide_by_length":
                        self.dataset = []
                        if self.pairwise == "two_pairs":
                            self.dataset += self.load_dataset_two_pairs_con()
                            self.dataset += self.load_dataset_two_pairs_incon()
                            self.dataset += self.load_dataset_two_pairs_con_and_incon()
                            self.dataset += (
                                self.load_dataset_two_pairs_incon_and_incon()
                            )
                        elif self.pairwise == "arbitrary_pairs":
                            self.dataset += self.load_dataset_arbitrary_pairs_con()
                            self.dataset += self.load_dataset_arbitrary_pairs_incon()
                            self.dataset += (
                                self.load_dataset_arbitrary_pairs_con_and_incon()
                            )
                            self.dataset += (
                                self.load_dataset_arbitrary_pairs_incon_and_incon()
                            )
                    else:
                        if self.pairwise == "two_pairs":
                            if self.semantic == "con":
                                self.dataset = self.load_dataset_two_pairs_con()
                            elif self.semantic == "incon":
                                self.dataset = self.load_dataset_two_pairs_incon()
                            elif self.semantic == "con_and_incon":
                                self.dataset = (
                                    self.load_dataset_two_pairs_con_and_incon()
                                )
                            elif self.semantic == "incon_and_incon":
                                self.dataset = (
                                    self.load_dataset_two_pairs_incon_and_incon()
                                )
                            else:
                                raise f"invalid dataset mode: {self.pairwise} - {self.semantic}"
                        elif self.pairwise == "arbitrary_pairs":
                            if self.semantic == "con":
                                self.dataset = self.load_dataset_arbitrary_pairs_con()
                            elif self.semantic == "incon":
                                self.dataset = self.load_dataset_arbitrary_pairs_incon()
                            elif self.semantic == "con_and_incon":
                                self.dataset = (
                                    self.load_dataset_arbitrary_pairs_con_and_incon()
                                )
                            elif self.semantic == "incon_and_incon":
                                self.dataset = (
                                    self.load_dataset_arbitrary_pairs_incon_and_incon()
                                )
                            else:
                                raise f"invalid dataset mode: {self.pairwise} - {self.semantic}"

            elif self.input_type.lower() in {"oto", "one_to_one"}:
                if self.semantic == "con":
                    self.dataset += self.load_dataset_two_pairs_con()
                    self.dataset += self.load_dataset_two_pairs_con()
                    self.dataset += self.load_dataset_two_pairs_con()

                    # self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                    # self.dataset += self.load_dataset_arbitrary_pairs_incon()

                elif self.semantic == "incon":
                    self.dataset += self.load_dataset_two_pairs_incon()
                    self.dataset += self.load_dataset_two_pairs_incon()
                    self.dataset += self.load_dataset_two_pairs_incon()

        print(
            f"dataset loaded: {self.mode} - {self.pairwise} - {self.semantic}. With length {len(self.dataset)}"
        )

    def load_dataset_two_pairs_con(self):
        dataset_name = f"two_pairs_con_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(
                os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"),
                "rb",
            ) as f:
                dataset = pickle.load(f)
            return dataset

        data = []
        # Load raw data for dataset generation; required file indices differ for train/eval/test.
        with open(
            os.path.join(self.processed_dataset_path, f"raw_{self.mode}_file.pickle"),
            "rb",
        ) as f:
            raw_source_file = pickle.load(f)

        for one_seed_pair in raw_source_file:
            if "pairID" not in one_seed_pair:
                continue
            seed_pair_original_label = one_seed_pair[
                "label"
            ]  # one of 'entailment', 'neutral', 'contradiction'
            if seed_pair_original_label == "entailment":
                sets = one_seed_pair["sets"]
                consistent_indices = [
                    i for i in range(len(sets)) if sets[i]["label"] == "consistent"
                ]
                used_rule_index = random.choice(consistent_indices)
                one_set = sets[used_rule_index]["set"]
                random.shuffle(one_set)

                data.append([(s, True) for s in one_set[:2]])

        dataset = data[: self.params["set_nli"][f"stepwise_dataset_{self.mode}_num"]]
        with open(
            os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), "wb"
        ) as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_two_pairs_incon(self):
        dataset_name = f"two_pairs_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(
                os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"),
                "rb",
            ) as f:
                dataset = pickle.load(f)
            return dataset

        data = []
        # Load raw data for dataset generation; required file indices differ for train/eval/test.
        with open(
            os.path.join(self.processed_dataset_path, f"raw_{self.mode}_file.pickle"),
            "rb",
        ) as f:
            raw_source_file = pickle.load(f)

        entailment_two_pair_incon_gen_functions = {
            "p_and_notp": lambda x: [x["premise"], x["premise_negated"]],
            "h_and_noth": lambda x: [x["hypothesis"], x["hypothesis_negated"]],
            "p_and_noth": lambda x: [x["premise"], x["hypothesis_negated"]],
            "p_or_h_and_noth": lambda x: [
                f"Either {x['premise']} or {x['hypothesis']}.",
                x["hypothesis_negated"],
            ],
        }
        neutral_two_pair_incon_gen_functions = {
            "p_and_notp": lambda x: [x["premise"], x["premise_negated"]],
            "h_and_noth": lambda x: [x["hypothesis"], x["hypothesis_negated"]],
        }
        contradiction_two_pair_incon_gen_functions = {
            "p_and_h": lambda x: [x["premise"], x["hypothesis"]],
        }

        entailment_gen_rules = list(entailment_two_pair_incon_gen_functions.keys())
        neutral_gen_rules = list(neutral_two_pair_incon_gen_functions.keys())
        contradiction_gen_rules = list(
            contradiction_two_pair_incon_gen_functions.keys()
        )

        for one_seed_pair in raw_source_file:
            if "pairID" not in one_seed_pair:
                continue
            seed_pair_original_label = one_seed_pair[
                "label"
            ]  # one of 'entailment', 'neutral', 'contradiction'
            if seed_pair_original_label == "entailment":
                generation_rule = random.choice(entailment_gen_rules)
                one_set = entailment_two_pair_incon_gen_functions[generation_rule](
                    one_seed_pair
                )
            if seed_pair_original_label == "neutral":
                generation_rule = random.choice(neutral_gen_rules)
                one_set = neutral_two_pair_incon_gen_functions[generation_rule](
                    one_seed_pair
                )
            if seed_pair_original_label == "contradiction":
                generation_rule = random.choice(contradiction_gen_rules)
                one_set = contradiction_two_pair_incon_gen_functions[generation_rule](
                    one_seed_pair
                )

            random.shuffle(one_set)
            data.append([(s, False) for s in one_set])

        dataset = data[: self.params["set_nli"][f"stepwise_dataset_{self.mode}_num"]]
        with open(
            os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), "wb"
        ) as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_arbitrary_pairs_con(self):
        dataset_name = f"arbitrary_pairs_con_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(
                os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"),
                "rb",
            ) as f:
                dataset = pickle.load(f)
            return dataset

        data = []
        # Load raw data for dataset generation; required file indices differ for train/eval/test.
        with open(
            os.path.join(self.processed_dataset_path, f"raw_{self.mode}_file.pickle"),
            "rb",
        ) as f:
            raw_source_file = pickle.load(f)

        for one_seed_pair in raw_source_file:
            sets = one_seed_pair["sets"]
            consistent_indices = [
                i for i in range(len(sets)) if sets[i]["label"] == "consistent"
            ]
            used_rule_index = random.choice(consistent_indices)
            one_set = sets[used_rule_index]["set"]
            random.shuffle(one_set)
            data.append([(s, True) for s in one_set])

        dataset = data[: self.params["set_nli"][f"stepwise_dataset_{self.mode}_num"]]
        with open(
            os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), "wb"
        ) as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_arbitrary_pairs_incon(self):
        dataset_name = f"arbitrary_pairs_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(
                os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"),
                "rb",
            ) as f:
                dataset = pickle.load(f)
            return dataset

        data = []
        # Load raw data for dataset generation; required file indices differ for train/eval/test.
        with open(
            os.path.join(self.processed_dataset_path, f"raw_{self.mode}_file.pickle"),
            "rb",
        ) as f:
            raw_source_file = pickle.load(f)

        for one_seed_pair in raw_source_file:
            sets = one_seed_pair["sets"]
            inconsistent_indices = [
                i for i in range(len(sets)) if sets[i]["label"] == "inconsistent"
            ]
            used_rule_index = random.choice(inconsistent_indices)
            one_set = sets[used_rule_index]["set"]
            random.shuffle(one_set)
            data.append([(s, False) for s in one_set])

        dataset = data[: self.params["set_nli"][f"stepwise_dataset_{self.mode}_num"]]
        with open(
            os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), "wb"
        ) as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_arbitrary_pairs_con_and_con(self):
        dataset_name = (
            f"arbitrary_pairs_con_and_con_{self.mode}_{str(self.dataset_num)}"
        )
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(
                os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"),
                "rb",
            ) as f:
                dataset = pickle.load(f)
            return dataset

        data = []
        # Load raw data for dataset generation; required file indices differ for train/eval/test.
        with open(
            os.path.join(self.processed_dataset_path, f"raw_{self.mode}_file.pickle"),
            "rb",
        ) as f:
            raw_source_file = pickle.load(f)

        source_file_another_indices = list(range(len(raw_source_file)))
        random.Random(self.seed).shuffle(source_file_another_indices)

        for i, one_seed_pair in enumerate(raw_source_file):
            sets1 = one_seed_pair["sets"]
            sets2 = raw_source_file[source_file_another_indices[i]]["sets"]

            consistent_indices1 = [
                i for i in range(len(sets1)) if sets1[i]["label"] == "consistent"
            ]
            used_rule_index1 = random.choice(consistent_indices1)
            one_set1 = sets1[used_rule_index1]["set"]

            consistent_indices2 = [
                i for i in range(len(sets2)) if sets2[i]["label"] == "consistent"
            ]
            used_rule_index2 = random.choice(consistent_indices2)
            one_set2 = sets2[used_rule_index2]["set"]

            one_set = one_set1 + one_set2

            random.shuffle(one_set)
            data.append([(s, True) for s in one_set])

        dataset = data[: self.params["set_nli"][f"stepwise_dataset_{self.mode}_num"]]
        with open(
            os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), "wb"
        ) as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_arbitrary_pairs_con_and_incon(self):
        dataset_name = (
            f"arbitrary_pairs_con_and_incon_{self.mode}_{str(self.dataset_num)}"
        )
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(
                os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"),
                "rb",
            ) as f:
                dataset = pickle.load(f)
            return dataset

        data = []
        # Load raw data for dataset generation; required file indices differ for train/eval/test.
        with open(
            os.path.join(self.processed_dataset_path, f"raw_{self.mode}_file.pickle"),
            "rb",
        ) as f:
            raw_source_file = pickle.load(f)

        source_file_another_indices = list(range(len(raw_source_file)))
        random.Random(self.seed).shuffle(source_file_another_indices)

        for i, one_seed_pair in enumerate(raw_source_file):
            sets1 = one_seed_pair["sets"]
            sets2 = raw_source_file[source_file_another_indices[i]]["sets"]

            consistent_indices1 = [
                i for i in range(len(sets1)) if sets1[i]["label"] == "consistent"
            ]
            used_rule_index1 = random.choice(consistent_indices1)
            one_set1 = sets1[used_rule_index1]["set"]

            inconsistent_indices2 = [
                i for i in range(len(sets2)) if sets2[i]["label"] == "inconsistent"
            ]
            used_rule_index2 = random.choice(inconsistent_indices2)
            one_set2 = sets2[used_rule_index2]["set"]

            one_set = one_set1 + one_set2

            random.shuffle(one_set)
            data.append([(s, False) for s in one_set])

        dataset = data[: self.params["set_nli"][f"stepwise_dataset_{self.mode}_num"]]
        with open(
            os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), "wb"
        ) as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_arbitrary_pairs_incon_and_incon(self):
        dataset_name = (
            f"arbitrary_pairs_incon_and_incon_{self.mode}_{str(self.dataset_num)}"
        )
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(
                os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"),
                "rb",
            ) as f:
                dataset = pickle.load(f)
            return dataset

        data = []
        # Load raw data for dataset generation; required file indices differ for train/eval/test.
        with open(
            os.path.join(self.processed_dataset_path, f"raw_{self.mode}_file.pickle"),
            "rb",
        ) as f:
            raw_source_file = pickle.load(f)

        source_file_another_indices = list(range(len(raw_source_file)))
        random.Random(self.seed).shuffle(source_file_another_indices)

        for i, one_seed_pair in enumerate(raw_source_file):
            sets1 = one_seed_pair["sets"]
            sets2 = raw_source_file[source_file_another_indices[i]]["sets"]

            inconsistent_indices1 = [
                i for i in range(len(sets1)) if sets1[i]["label"] == "inconsistent"
            ]
            used_rule_index1 = random.choice(inconsistent_indices1)
            one_set1 = sets1[used_rule_index1]["set"]

            inconsistent_indices2 = [
                i for i in range(len(sets2)) if sets2[i]["label"] == "inconsistent"
            ]
            used_rule_index2 = random.choice(inconsistent_indices2)
            one_set2 = sets2[used_rule_index2]["set"]

            one_set = one_set1 + one_set2

            random.shuffle(one_set)
            data.append([(s, False) for s in one_set])

        dataset = data[: self.params["set_nli"][f"stepwise_dataset_{self.mode}_num"]]
        with open(
            os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), "wb"
        ) as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def raw_data_split(self):
        """
        Split the downloaded dataset into train/eval/test subsets.
        Save the file lists so future runs can simply read them.
        """
        print("=================")
        print("raw data split start.")
        # filter out empty JSON files
        source_file_path = os.path.join(self.folder_path, self.filename)
        with open(source_file_path, "r") as f:
            source_json_data = json.load(f)
        source_json_data = source_json_data[
            "instances"
        ]  # type(source_json_data) = list

        pairIDs = set()
        for idx, one_instance in enumerate(source_json_data):
            if "pairID" in one_instance:
                pairIDs.add(one_instance["pairID"])
        print("source_json_data len, pairID len:", len(source_json_data), len(pairIDs))
        pairIDs = list(pairIDs)
        random.shuffle(pairIDs)
        train_pairIDs = set(pairIDs[: int(len(pairIDs) * 0.6)])
        eval_pairIDs = set(pairIDs[int(len(pairIDs) * 0.6) : int(len(pairIDs) * 0.8)])
        eval2_pairIDs = set(pairIDs[int(len(pairIDs) * 0.8) : int(len(pairIDs) * 0.9)])
        test_pairIDs = set(pairIDs[int(len(pairIDs) * 0.9) :])

        train_file, eval_file, eval2_file, test_file = [], [], [], []  # simple data
        train_file_full, eval_file_full, eval2_file_full, test_file_full = (
            [],
            [],
            [],
            [],
        )  # full data

        (
            train_two_pair_seed_index,
            eval_two_pair_seed_index,
            eval2_two_pair_seed_index,
            test_two_pair_seed_index,
        ) = (set(), set(), set(), set())
        for train_pairid in train_pairIDs:
            selected_pair = select_another(train_pairIDs, train_pairid)
            train_two_pair_seed_index.add((train_pairid, selected_pair))
        for eval_pairid in eval_pairIDs:
            selected_pair = select_another(eval_pairIDs, eval_pairid)
            eval_two_pair_seed_index.add((eval_pairid, selected_pair))
        for eval2_pairid in eval2_pairIDs:
            selected_pair = select_another(eval2_pairIDs, eval2_pairid)
            eval2_two_pair_seed_index.add((eval2_pairid, selected_pair))
        for test_pairid in test_pairIDs:
            selected_pair = select_another(test_pairIDs, test_pairid)
            test_two_pair_seed_index.add((test_pairid, selected_pair))

        for one_instance in tqdm(source_json_data):

            if "pairID" in one_instance:  # Sets are generated by a single seed pair
                if one_instance["pairID"] in train_pairIDs:
                    train_file.append(one_instance)
                    train_file_full.append(one_instance)
                elif one_instance["pairID"] in eval_pairIDs:
                    eval_file.append(one_instance)
                    eval_file_full.append(one_instance)
                elif one_instance["pairID"] in eval2_pairIDs:
                    eval2_file.append(one_instance)
                    eval2_file_full.append(one_instance)
                elif one_instance["pairID"] in test_pairIDs:
                    test_file.append(one_instance)
                    test_file_full.append(one_instance)

            elif ("pairID_1" in one_instance) and (
                "pairID_2" in one_instance
            ):  # Sets are generated by two seed pairs
                if (one_instance["pairID_1"] in train_pairIDs) and (
                    one_instance["pairID_2"] in train_pairIDs
                ):
                    train_file_full.append(one_instance)
                elif (one_instance["pairID_1"] in eval_pairIDs) and (
                    one_instance["pairID_2"] in eval_pairIDs
                ):
                    eval_file_full.append(one_instance)
                elif (one_instance["pairID_1"] in eval2_pairIDs) and (
                    one_instance["pairID_2"] in eval2_pairIDs
                ):
                    eval2_file_full.append(one_instance)
                elif (one_instance["pairID_1"] in test_pairIDs) and (
                    one_instance["pairID_2"] in test_pairIDs
                ):
                    test_file_full.append(one_instance)

                if (
                    one_instance["pairID_1"],
                    one_instance["pairID_2"],
                ) in train_two_pair_seed_index:
                    train_file.append(one_instance)
                if (
                    one_instance["pairID_1"],
                    one_instance["pairID_2"],
                ) in eval_two_pair_seed_index:
                    eval_file.append(one_instance)
                if (
                    one_instance["pairID_1"],
                    one_instance["pairID_2"],
                ) in eval2_two_pair_seed_index:
                    eval2_file.append(one_instance)
                if (
                    one_instance["pairID_1"],
                    one_instance["pairID_2"],
                ) in test_two_pair_seed_index:
                    test_file.append(one_instance)

        for file in (
            train_file,
            eval_file,
            eval2_file,
            test_file,
            train_file_full,
            eval_file_full,
            eval2_file_full,
            test_file_full,
        ):
            random.shuffle(file)

        # save results
        with open(
            os.path.join(self.processed_dataset_path, "raw_train_file.pickle"), "wb"
        ) as f:
            pickle.dump(train_file, f)
        with open(
            os.path.join(self.processed_dataset_path, "raw_eval_file.pickle"), "wb"
        ) as f:
            pickle.dump(eval_file, f)
        with open(
            os.path.join(self.processed_dataset_path, "raw_eval2_file.pickle"), "wb"
        ) as f:
            pickle.dump(eval2_file, f)
        with open(
            os.path.join(self.processed_dataset_path, "raw_test_file.pickle"), "wb"
        ) as f:
            pickle.dump(test_file, f)

        with open(
            os.path.join(self.processed_dataset_path, "raw_train_file_full.pickle"),
            "wb",
        ) as f:
            pickle.dump(train_file_full, f)
        with open(
            os.path.join(self.processed_dataset_path, "raw_eval_file_full.pickle"), "wb"
        ) as f:
            pickle.dump(eval_file_full, f)
        with open(
            os.path.join(self.processed_dataset_path, "raw_eval2_file_full.pickle"),
            "wb",
        ) as f:
            pickle.dump(eval2_file_full, f)
        with open(
            os.path.join(self.processed_dataset_path, "raw_test_file_full.pickle"), "wb"
        ) as f:
            pickle.dump(test_file_full, f)

        print("raw data split ends.")
        print(
            f"raw_train_filename: {os.path.join(self.processed_dataset_path, 'raw_train_file.pickle')}"
        )
        print(f"raw_train dataset length: {len(train_file)}")
        print(
            f"raw_eval_filename: {os.path.join(self.processed_dataset_path, 'raw_eval_file.pickle')}"
        )
        print(f"raw_eval dataset length: {len(eval_file)}")
        print(
            f"raw_eval2_filename: {os.path.join(self.processed_dataset_path, 'raw_eval2_file.pickle')}"
        )
        print(f"raw_eval2 dataset length: {len(eval2_file)}")
        print(
            f"raw_test_filename: {os.path.join(self.processed_dataset_path, 'raw_test_file.pickle')}"
        )
        print(f"raw_test dataset length: {len(test_file)}")
        print("=================")


def transform_arbitrary_pairs_to_two_pairs(seq_of_dataset):
    """

    seq_of_dataset collects datasets; any iterable (set, tuple, list) works in Python.

    dataset: l_set_nli_fine_grained_dataset.dataset


    for d in dataset: # type(dataset) == list
        for data_point in d: # type(d) = list
            question_answer-pair, consistency = data_point # type(data_point) = tuple
            # type(question): usually string
            # type(answer): usually string
            # type(consistency): bool. Consistency indicates whether this qa pair is consistent or inconsistent with respect to the rest.

    """

    transformed_dataset_con = []
    transformed_dataset_incon = []
    set_sizes = {}

    for dataset in seq_of_dataset:
        for data in dataset:
            if len(data) == 2:
                if all([d[1] for d in data]):
                    transformed_dataset_con.append(data)
                else:
                    transformed_dataset_incon.append(data)

            else:
                combinations = itertools.combinations(
                    data, 2
                )  # "len(d)-choose-2"-number of combinations
                for two_pair in combinations:
                    if all([p[1] for p in two_pair]):
                        transformed_dataset_con.append(two_pair)
                    else:
                        transformed_dataset_incon.append(two_pair)

            if len(data) not in set_sizes:
                set_sizes[len(data)] = 0
            set_sizes[len(data)] += 1

    additional_info = {}
    for i, s in enumerate(seq_of_dataset):
        additional_info[f"original_dataset_number_{i+1}_length"] = len(s)
    additional_info["con_transformed"] = len(transformed_dataset_con)
    additional_info["incon_transformed"] = len(transformed_dataset_incon)
    for i in set_sizes.keys():
        additional_info[f"original_data_set_size_with_|set|={i}"] = set_sizes[i]

    print("===========================")
    print("transformed arbitrary-pairs dataset to two-pairs dataset.")
    for key, val in additional_info.items():
        print("key:", key)
        print("key:", val)
        print("\n")
    print("===========================")

    return transformed_dataset_con, transformed_dataset_incon, additional_info


def select_another(candidate, curr):
    selected = random.choice(tuple(candidate))
    if selected == curr:
        return select_another
    else:
        return selected
