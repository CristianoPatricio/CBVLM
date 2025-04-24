import pandas as pd
import numpy as np
import ast
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, f1_score, multilabel_confusion_matrix, cohen_kappa_score
from torchmetrics.classification import MultilabelExactMatch
import torch
import argparse
import os
import json

from src.utils import utils

def compute_metrics(csv_path):
    # Read CSV to a Pandas DF
    df = pd.read_excel(csv_path)

    info = {}

    # disease classification
    if "pred_label" in df.columns:
        perc_unk = 0
        # Check if y_pred contains -1
        if -1 in df["pred_label"].values.tolist():
            perc_unk = ((df["pred_label"] == -1).sum()) / len(df)
            df = df[df["pred_label"] != -1] 

        # Extract y_true and y_pred from df
        y_true = df["gt_label"].values.tolist()
        y_pred = df["pred_label"].values.tolist()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        ck = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        # Save results to a txt file
        info["perc_unk"] = perc_unk
        info["accuracy"] = accuracy
        info["cm"] = cm
        info["balanced_acc"] = balanced_acc
        info["f1"] = f1
        info["ck"] = ck
    
    # concepts prediction
    if "pred_concepts" in list(df.columns):
        concepts_true = df["gt_concepts"].values.tolist()
        concepts_pred = df["pred_concepts"].values.tolist()

        # Clean lists   
        clean_concepts_true = [ast.literal_eval(item) for item in concepts_true]
        clean_concepts_pred = [ast.literal_eval(item) for item in concepts_pred]

        # Prevent 1 dimension in array
        clean_concepts_true = np.array(clean_concepts_true)
        clean_concepts_pred = np.array(clean_concepts_pred)
        count_unk = np.count_nonzero(clean_concepts_pred == -1)
        perc_unk = count_unk / (clean_concepts_pred.shape[0] * clean_concepts_pred.shape[1])
        clean_concepts_pred = np.where(clean_concepts_pred == -1, 0, clean_concepts_pred)
        
        if "derm7pt" in csv_path:
            clean_concepts_pred_multi = clean_concepts_pred[:, :3].copy()
            clean_concepts_pred_binary = clean_concepts_pred[:, 3:].copy()
            clean_concepts_pred_multi = np.eye(3, dtype=int)[clean_concepts_pred_multi].reshape(clean_concepts_pred_multi.shape[0], -1)
            clean_concepts_pred = np.hstack((clean_concepts_pred_multi, clean_concepts_pred_binary))

            clean_concepts_true_multi = clean_concepts_true[:, :3].copy()
            clean_concepts_true_binary = clean_concepts_true[:, 3:].copy()
            clean_concepts_true_multi = np.eye(3, dtype=int)[clean_concepts_true_multi].reshape(clean_concepts_true_multi.shape[0], -1)
            clean_concepts_true = np.hstack((clean_concepts_true_multi, clean_concepts_true_binary))


        concepts_f1 = f1_score(y_true=clean_concepts_true.flatten(), y_pred=clean_concepts_pred.flatten())
        ml_cm = multilabel_confusion_matrix(y_true=clean_concepts_true, y_pred=clean_concepts_pred)
        concepts_f1_per_concept = f1_score(y_true=clean_concepts_true, y_pred=clean_concepts_pred, average=None)

        concepts_bacc = balanced_accuracy_score(y_true=clean_concepts_true.flatten(), y_pred=clean_concepts_pred.flatten())
 
        metric = MultilabelExactMatch(num_labels=clean_concepts_true.shape[1])
        exact_match = metric(torch.tensor(clean_concepts_pred), torch.tensor(clean_concepts_true)).item()

        info["perc_unk"] = perc_unk
        info["concepts_bacc"] = concepts_bacc
        info["concepts_f1"] = concepts_f1
        info["concepts_f1_per_concept"] = concepts_f1_per_concept.tolist()
        info["exact_match"] = exact_match
        info["ml_cm"] = ml_cm
        
    utils.save_results_to_txt(os.path.dirname(csv_path), **info)
    if "cm" in info:
        del info["cm"]
    if "ml_cm" in info:
        del info["ml_cm"]
    with open(os.path.join(os.path.dirname(csv_path), "results.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run evaluation metrics.')
    parser.add_argument('csv_path', help='Path to csv with predictions')
    args = parser.parse_args()
    compute_metrics(args.csv_path)
