import os, sys
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    LongformerConfig,
    LongformerModel,
    LongformerTokenizer,
    AutoTokenizer,
)


from .losses.triplet import triplet
from .losses.NCE import NCE
from .losses.supervised import supervised


from .decomposition.no_decomposition import no_decomposition

from .representation_model.gte_qwen import gte_qwen
from .representation_model.roberta import roberta
from .representation_model.longformer import longformer


class energynet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.output_form = None  # determined by self.initialize()

        ## hierarchy: loss_type -> decomposition -> representation_model

        # Method to compute the Energy Net loss. Default = triplet_loss
        self.loss_function = None  # determined by self.initialize()
        self.loss_type = params["energynet"][
            "loss_type"
        ]  # e.g., triplet_loss, regression-based loss

        # How to split x and y pairs for feature extraction. Default = no decomposition
        self.energy_model = None  # determined by self.initialize()
        self.decomposition_type = params["energynet"][
            "decomposition"
        ]  # e.g., (no / x-y / pairwise) decomposition

        # representation model. Default = RoBERTa
        self.representation_model = None  # determined by self.initialize()
        self.representation_model_type = params["energynet"][
            "repre_model"
        ]  # e.x., Longformer, RoBERTa

        self.threshold = 0.5
        self.initialize()

    def forward(self, input, pair_only=False):
        additional_info = {}
        if self.loss_type in {"triplet", "NCE"}:
            pos_pair, neg_pair = input
            loss, additionals = self.loss_function(pos_pair, neg_pair)
            additional_info["e_pos"] = additionals["e_pos"]
            additional_info["e_neg"] = additionals["e_neg"]

        if self.loss_type == "supervised":
            pairs, label = input
            # print("pairs:", pairs)
            # print("label:", label)
            loss, additionals = self.loss_function(pairs, label, pair_only=pair_only)
            additional_info["pred"] = [
                int(e[-1] >= self.threshold) for e in additionals["probs"]
            ]
            additional_info["gold"] = additionals["gold"]
            additional_info["probs"] = additionals["probs"]

        return loss, additional_info

    def initialize(self):

        # loss_type
        if self.loss_type == "triplet":
            self.loss_function = triplet(self.params)
        if self.loss_type == "NCE":
            self.loss_function = NCE(self.params)
        elif self.loss_type == "supervised":
            self.loss_function = supervised(self.params)

        # decomposition_type
        if self.decomposition_type == "no":
            self.energy_model = no_decomposition(self.params)

        # representation model.
        if self.representation_model_type == "longformer":
            self.representation_model = longformer(self.params)
        elif self.representation_model_type == "roberta":
            self.representation_model = roberta(self.params)
        elif "gte" in self.representation_model_type.lower():
            self.representation_model = gte_qwen(self.params)

        self.energy_model.set_representation_model(self.representation_model)
        self.loss_function.decomposition = self.energy_model
        self.output_form = self.representation_model.output_form
