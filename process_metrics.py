import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import json
import yaml

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument("path", nargs="+", default=[], help="Path to folder where json files are stored.")
    args = parser.parse_args()

    all_jsons = []
    for p in args.path:
        all_jsons.extend(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
   
    all_data_concepts = []
    all_data_clf = []
    for json_file in tqdm(all_jsons):
        with open(os.path.join(os.path.dirname(json_file), ".hydra", "config.yaml")) as f:
            cfg = yaml.safe_load(f)

        if "classification" in json_file:
            assert "use_concepts" in cfg

        with open(json_file, "r") as f:
            json_data = json.load(f)

        if "bs" not in cfg:
            cfg["bs"] = 1

        indv_data = [json_file, cfg["name"], cfg["data"]["name"], cfg["bs"], cfg["n_demos"], cfg["demo_selection"], cfg["feature_extractor"], cfg["filter"]*100 if "filter" in cfg else 100, cfg["intervention"] if "intervention" in cfg else None,  cfg["intervention_perc"] if "intervention_perc" in cfg else None, cfg["seed"]]
        if "classification" in json_file:
            indv_data.extend([cfg["use_concepts"], json_data["perc_unk"]*100, json_data["accuracy"]*100, json_data["balanced_acc"]*100, json_data["f1"]*100])
            all_data_clf.append(indv_data)
        else:
            indv_data.extend([json_data["perc_unk"]*100, json_data["concepts_bacc"]*100, json_data["concepts_f1"]*100, json_data["concepts_f1_per_concept"], json_data["exact_match"]*100])
            all_data_concepts.append(indv_data)

    columns_concepts = ["path", "model", "dataset", "bs", "n_demos", "demo_selection", "feature_extractor", "filter", "intervention", "intervention_perc", "seed", "perc_unk", "attr_bacc", "attr_f1", "attr_f1_per_concept", "exact_match"]
    columns_clf = ["path", "model", "dataset", "bs", "n_demos", "demo_selection", "feature_extractor", "filter", "intervention", "intervention_perc", "seed", "use_concepts", "perc_unk", "acc", "bacc", "f1"]

    df_concepts = pd.DataFrame.from_records(all_data_concepts, columns=columns_concepts)
    df_clf = pd.DataFrame.from_records(all_data_clf, columns=columns_clf)

    subset_concepts = ["model", "dataset", "n_demos", "demo_selection", "filter", "intervention", "intervention_perc", "seed"]
    subset_clf = ["model", "dataset", "n_demos", "demo_selection", "filter", "intervention", "intervention_perc", "seed"]

    # group demo_selection + feature_extractor
    # group n_demos + use_concepts    
    for c, df, s in zip(["concepts", "classification"], [df_concepts, df_clf], [subset_concepts, subset_clf]):
        df.loc[(df["demo_selection"] == "rices") & (df["feature_extractor"] == "clip"), "demo_selection"] = "rices_clip"
        df.loc[(df["demo_selection"] == "rices") & (df["feature_extractor"] == "biomedclip"), "demo_selection"] = "rices_biomedclip"
        df.loc[(df["demo_selection"] == "rices") & (df["feature_extractor"] == "medimageinsight"), "demo_selection"] = "rices_medii"
        df.loc[(df["demo_selection"] == "rices") & (~pd.isna(df["feature_extractor"])) & (~df["feature_extractor"].isin(["clip", "biomedclip", "medii"])), "demo_selection"] = "rices_model"
        df["n_demos"] = df["n_demos"].astype(str)
        cond = ((df["demo_selection"] == "rices_per_class_global") | (df["demo_selection"] == "rices_per_class_max")) & (df["n_demos"] == "1")
        df.loc[cond, "demo_selection"] = df.loc[cond].apply(lambda x: x.demo_selection.replace("_global", "_mean").replace("_max", "_mean") + f"_{x.feature_extractor.replace('medimageinsight', 'medii')}", axis=1)
        cond = (df["demo_selection"].isin(["rices_per_class_global", "rices_per_class_max", "rices_per_class_mean"]) & (df["n_demos"] != "1"))
        df.loc[cond, "demo_selection"] = df.loc[cond].apply(lambda x: x.demo_selection + f"_{x.feature_extractor.replace('medimageinsight', 'medii')}", axis=1)
        df = df.drop("feature_extractor", axis=1)
        if "use_concepts" in df.columns:
            df.loc[(df["n_demos"] == "0") & (pd.isna(df["use_concepts"])), "n_demos"] = "0 w/o"
            df = df.drop("use_concepts", axis=1)
        # find duplicates
        duplicates = df[df.duplicated(subset=s, keep=False)]
        if len(duplicates) > 0:
            duplicates.to_excel(f"logs/duplicates_{c}.xlsx")
            print("Found these duplicates")
            print(duplicates)
            print("No results generated. Delete the duplicate folders.")
            quit()

        df.to_excel(f"logs/results_{c}.xlsx")
        print(len(df))

if __name__ == "__main__":
    main()