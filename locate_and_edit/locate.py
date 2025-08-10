import torch.nn as nn
from .locate_modules.locate_by_gradnorm import locate_by_gradnorm
from .locate_modules.locate_by_subtraction import locate_by_subtraction


class locate(nn.Module):
    def __init__(self, params, energynet):
        super().__init__()
        self.params = params

        self.locate_method = None
        self.locate_type = self.params["locate"]["type"]

        self.initialize(energynet)

    def initialize(self, energynet):
        if self.locate_type == "gradnorm":
            self.locate_method = locate_by_gradnorm(self.params, energynet)
            print("locate by gradnorm")
        elif "subtraction" in self.locate_type:
            self.locate_method = locate_by_subtraction(self.params, energynet)
            print("locate by subtraction")
        else:
            print(f"invalid location method. Current type is {self.locate_type}")

    def forward(self, inputs):

        # assert len(inputs) == 1

        # aim: detect examples that appear inconsistent for a given pair

        additional_info = {}
        locate_result = self.locate_method.locate(inputs)

        pred = locate_result["prediction_list"]
        gold = locate_result["errors_gold"]
        pair_num = locate_result["pair_num"]

        # print("pred:", pred)
        # print("gold:", gold)
        # print("pair_num:", pair_num)

        ########## statistics ##########

        # exact match
        if set(pred[0]) == set(gold[0]):
            correct = 1
        else:
            correct = 0

        # precision
        if len(pred[0]) == 0:
            precision = 1
        else:
            precision = len(set(pred[0]) & set(gold[0])) / len(set(pred[0]))

        # recall
        if len(gold[0]) == 0:
            recall = 1
        else:
            recall = len(set(pred[0]) & set(gold[0])) / len(set(gold[0]))

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = correct

        additional_info["pred"] = pred
        additional_info["gold"] = gold
        additional_info["pair_num"] = pair_num
        additional_info["precision"] = precision
        additional_info["recall"] = recall
        additional_info["f1"] = f1

        return accuracy, additional_info
