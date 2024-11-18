"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

from sympy import false

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.gpt4ts_custom import gpt4ts
from models.loss import get_loss_module
from optimizers import get_optimizer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AdamW


def save_classification_report(predicted_targets, actual_targets):
    sensor_names = [
        "appliance temperature",
        "humidity",
        "pressure",
        "indoor air temperature",
        "outdoor air temperature",
        "wind speed",
        "light",
        "air quality",
        "wind direction",
        "wind temperature",
        "voltage",
        "current intensity",
        "wireless RSSI",
        "heat index",
        "dewpoint",
        "rain index",
        "UV",
        "PM 1",
        "PM 2.5",
        "PM 10",
        "CO2",
    ]

    clf_report = classification_report(
        actual_targets,
        predicted_targets,
        digits=5,
        zero_division=0,
        target_names=sensor_names,
        output_dict=True,
    )
    # Convert report to DataFrame (Test)
    clf_df = pd.DataFrame(clf_report).transpose()
    clf_df.to_csv(f"OFA_classification_report_only_TS.csv")


def main(dataframe, config):
    X = dataframe.drop(
        labels=[
            "metadata",
            "label",
        ],
        axis=1,
    )

    mask_value = 0.0000000000000001
    X = X.fillna(mask_value)

    X_meta = pd.DataFrame({"metadata": []})
    X_meta["metadata"] = dataframe["metadata"].astype(str)

    y = dataframe["label"]
    X_train, X_test, y_train, y_test, X_meta_train, X_meta_test = train_test_split(
        X, y, X_meta, test_size=0.3, stratify=y, random_state=100
    )

    X_train = np.reshape(np.array(X_train), (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(np.array(X_test), (X_test.shape[0], X_test.shape[1], 1))

    sequence_length = X_train.shape[1]

    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)

    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)

    num_targets = len(np.unique(y_train))

    config["sequence_length"] = sequence_length
    config["num_targets"] = num_targets

    model = gpt4ts(config=config)
    print(model)
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config["gpu"] != "-1") else "cpu"
    )
    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    model.to(device)

    # Below Code For Freezing Model
    if config["freeze"]:
        for name, param in model.named_parameters():
            if name.startswith("out_layer"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Check which layers are frozen
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    # logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(utils.count_parameters(model, trainable=True))
    )

    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 200
    best_val_epoch = 0
    best_val_accuracy = 0
    best_val_predictions = None
    val_accuracy_list = []
    macro_f1_list = []
    # model.train() # By default is set to train mode
    for epoch in range(num_epochs):
        # model.train() # By default is set to train mode
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_train)
        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(y_train, num_targets)
        # Calculate loss
        loss = criterion(outputs.float(), one_hot_labels.float())

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        _, predicted_classes = torch.max(outputs, 1)
        accuracy = (predicted_classes == y_train).float().mean().item()

        # Forward pass on validation data
        # model.eval() # To set model on evaluation mode, no training
        with torch.no_grad():
            validation_outputs = model(X_test)

        # Calculate accuracy or other evaluation metrics
        _, val_predicted_classes = torch.max(validation_outputs, 1)
        val_accuracy = (val_predicted_classes == y_test).float().mean().item()
        val_accuracy_list.append(val_accuracy * 100)
        from sklearn.metrics import f1_score

        # Calculate macro F1 score
        macro_f1 = f1_score(y_test, val_predicted_classes, average="macro")
        macro_f1_list.append(macro_f1 * 100)
        # Print loss and accuracy for monitoring training progress
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}, Validation Accuracy: {val_accuracy}"
        )
    #     if val_accuracy > best_val_accuracy:
    #         best_val_accuracy = val_accuracy
    #         best_val_predictions = val_predicted_classes
    #         best_val_epoch = epoch + 1

    # print("Results Summary \n")
    # print(f"Best Validation Accuracy: {best_val_accuracy} at Epoch {best_val_epoch}")
    # # Save classification report on test data (best epoch while training)
    # save_classification_report(best_val_predictions, y_test)
    top_epochs = 5
    # Sort the validation accuracies to get the best 5 epochs
    best_val_accuracies = sorted(val_accuracy_list, reverse=True)[:top_epochs]
    best_f1_macro = sorted(macro_f1_list, reverse=True)[:top_epochs]

    results = []
    for accuracy, f1_macro in zip(best_val_accuracies, best_f1_macro):
        results.append({"accuracy": accuracy, "f1_macro": f1_macro})
    print(
        "Best Accuracy: ",
        max(best_val_accuracies),
        "\n",
        "Best F1-Macro: ",
        max(best_f1_macro),
    )
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate mean and standard deviation of the top 5 epochs
    mean_accuracy = results_df["accuracy"].mean()
    std_accuracy = results_df["accuracy"].std()
    mean_f1_macro = results_df["f1_macro"].mean()
    std_f1_macro = results_df["f1_macro"].std()
    results_list = [
        "OFA only TS",
        mean_accuracy,
        std_accuracy,
        mean_f1_macro,
        std_f1_macro,
    ]
    import csv

    with open(
        f"/home/muhammadinan/Documents/GitHub/Multi-modal-Journal-Paper/results/senseurcity/combine.csv",
        "a",
        newline="",
    ) as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(results_list)


if __name__ == "__main__":
    config = {
        "d_model": 768,
        "patch_size": 64,
        "stride": 64,
        "scale": 1,
        "adapter_dim": 32,
        "dropout": 0.1,
        "gpu": "-1",
        "freeze": False,
    }
    dataframe = pd.read_excel(
        "/home/muhammadinan/Documents/GitHub/Multi-modal-Journal-Paper/datasets/senseurcity/senseurcity_combine_cities.xlsx"
    )
    label_dict = {
        "Atmospheric pressure": 0,
        "Ambient Temperature": 1,
        "Ambient Relative humidity": 2,
        "Internal Temperature": 3,
        "Internal Relative humidity": 4,
        "CO": 5,
        "CO2": 6,
        "NO": 7,
        "NO2": 8,
        "PM10": 9,
        "PM2.5": 10,
        "PM1": 11,
    }

    dataframe["label"] = dataframe["label"].replace(label_dict)

    main(dataframe, config=config)
