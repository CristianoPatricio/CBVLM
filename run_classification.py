import hydra
from omegaconf import DictConfig
import os
import dotenv
from tqdm import tqdm
import gc
import torch
from PIL import Image
import re
import pandas as pd
import yaml
import ast
import random
import numpy as np

from torch.utils.data import DataLoader

from src.utils import utils
from src.utils.dataset_loaders import get_dataset, CustomIndexSampler
from src.models.mistral.demo import Mistral
from src.utils.random_selection import RANDOM
from src.utils.rices import RICES
from src.utils.mmices import MMICES
from run_metrics import compute_metrics

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dir_path = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(dir_path + '/var_environment.env', override=True)

log = utils.get_logger(__name__) # init logger

def process_multiple_choice(model_name, answer, option2gt, class_mapping, mistral, mistral_max_new_tokens, debug=False):
    mistral_output = None
    mistral_option2gt = None

    n_options = len(option2gt)
    final_letter = chr(ord('A') + n_options)
    pattern = f'[A-{final_letter}a-{final_letter.lower()}]\)'

    og_answer = answer.lower()
    if model_name == "med-flamingo" or model_name == "skingpt4":
        # get first sentence
        answer = answer.split("\n")[0]
    if len(answer) == 1:               
        # exception for instruct-blip that likes to answer a single lowercase letter
        answer = f"{answer.upper()})"
    elif len(answer) == 2:
        # exception for open-flamingo that likes to answer a single uppercase letter followed by a full stop
        answer = f"{answer[0].upper()})"

    if model_name == "skingpt4":
        # ignore option because sometimes it answers e.g. "A) present" when the options are "A) absent B) present"
        answer = answer.split(")")[-1]
    
    answer = re.findall(pattern, answer)
    exists = False
    if len(answer) == 0:
        # check if model answered in the first word with the option text and not the option letter
        gttext2option = {class_mapping[v].lower(): k for k,v in option2gt.items()}
        if og_answer in list(gttext2option.keys()):
            answer = [gttext2option[og_answer]]
            exists = True
        else:
            for k_,v_ in gttext2option.items():
                if (og_answer.startswith(k_ + " ")) or (og_answer.endswith(" " + k_)) or (" " + k_ + " " in og_answer) or (" " + k_ + "." in og_answer):
                    answer = [v_]
                    exists = True
                    break
    else:
        exists = True
    
    if exists == False:
        # only load mistral if needed
        if mistral is None:
            mistral = Mistral()

        # might still have answered but without choosing an option
        mistral_output, mistral_option2gt = mistral.predict(
                og_answer, None, class_mapping, "multiple_choice", max_new_tokens=mistral_max_new_tokens
        )

        if debug:
            print(f">>> LVLM: {og_answer} // Mistral: {mistral_output}\n")

        answer = mistral_output["Answer"]
        if answer in mistral_option2gt:
            y_pred = mistral_option2gt[answer]
        else:
            y_pred = -1
    elif answer[0] in option2gt:
        y_pred = option2gt[answer[0]]
    else:
        y_pred = -1 

    return y_pred, mistral_output, mistral_option2gt, mistral

def process_answers(csv_file, class_mapping, mistral_max_new_tokens=100, debug=False):
    log.info(f"Processing predictions of file {csv_file}")

    output = pd.read_excel(csv_file, index_col=0)

    with open(os.path.join(os.path.dirname(csv_file), ".hydra", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    mistral = None
    
    # process extracted options
    processed_df = {}
    for idx, row in tqdm(output.iterrows()):
        processed_df[idx] = {}
        answer = str(row["clf_answer"])
        option2gt = row[f"option2gt"]
        option2gt = ast.literal_eval(option2gt)
        y_pred, mistral_output, mistral_option2gt, mistral = process_multiple_choice(cfg["name"], answer, option2gt, class_mapping, mistral, mistral_max_new_tokens=mistral_max_new_tokens, debug=debug)
        processed_df[idx]["option2gt_mistral"] = mistral_option2gt
        processed_df[idx]["clf_mistral"] = mistral_output
        processed_df[idx]["pred_label"] = y_pred
        
    processed_df = pd.DataFrame.from_dict(processed_df, orient="index")
    final_df = pd.concat([output, processed_df], axis=1)
    csv_name = csv_file.replace("_raw_classification.xlsx", "_processed_classification.xlsx")
    final_df.to_excel(csv_name)
    log.info(f"Classification predictions processed and saved to {csv_name}")

    log.info("Computing metrics")
    compute_metrics(csv_name)

@hydra.main(version_base=None, config_path="configs", config_name="classification.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # torch.set_num_threads(8)

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        utils.print_config(cfg, resolve=True, save_to_file=True)

    # set seed
    seed_everything(cfg.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.precomputed_file is None:
        if cfg.demo_selection is not None:
            assert cfg.demo_selection in ["random", "random_per_class", "rices", "rices_per_class_global", "rices_per_class_mean", "rices_per_class_max", "mmices"]
            assert cfg.n_demos > 0
        if cfg.n_demos > 0:
            assert cfg.demo_selection is not None
            assert cfg.demo_selection in ["random", "random_per_class", "rices", "rices_per_class_global", "rices_per_class_mean", "rices_per_class_max", "mmices"]
    
        if (cfg.demo_selection is not None) and ("rices" in cfg.demo_selection or cfg.demo_selection == "mmices"):
            assert cfg.feature_extractor is not None
        
        if cfg.demo_selection == "mmices":
            assert cfg.mmices_text_features is not None
            assert cfg.mmices_text_features in ["descriptions", "concepts"]

        if cfg.use_concepts is not None:
            assert (".xlsx" in cfg.use_concepts) or ("automatic" in cfg.use_concepts)
            pred_concepts_df = None
            if "automatic" in cfg.use_concepts:
                all_results_df = pd.read_excel("logs/results_concepts.xlsx", index_col=0)
                if cfg.n_demos == 0:
                    pred_concepts_file = all_results_df.loc[(all_results_df["model"] == cfg.name) & (all_results_df["dataset"] == cfg.data.name) & (all_results_df["n_demos"] == cfg.n_demos)]
                else:
                    ds = cfg.demo_selection
                    if "rices" in ds or "mmices" in ds:
                        if cfg.feature_extractor not in ["clip", "biomedclip", "medimageinsight"]:
                            ds += "_model"
                        else:
                            ds += f"_{cfg.feature_extractor}"
                        ds = ds.replace("medimageinsight", "medii")
                    
                    pred_concepts_file = all_results_df.loc[(all_results_df["model"] == cfg.name) & (all_results_df["dataset"] == cfg.data.name) & (all_results_df["n_demos"] == cfg.n_demos) & (all_results_df["demo_selection"] == ds)]
                if len(pred_concepts_file) == 0:
                    raise Exception("Could not find the concepts file for these specs.")
                pred_concepts_file = pred_concepts_file["path"].values[0].replace("results.json", "model_output_processed_concepts.xlsx")
                cfg.use_concepts = pred_concepts_file

            if ".xlsx" in cfg.use_concepts:
                print(f"Using concepts from: {cfg.use_concepts}")
                pred_concepts_df = pd.read_excel(cfg.use_concepts, index_col=0)
                assert "pred_concepts" in pred_concepts_df.columns
                pred_concepts_df.index = pred_concepts_df.index.map(str)
                pred_concepts_df["pred_concepts"] = pred_concepts_df["pred_concepts"].apply(eval)

        assert cfg.intervention_perc >= 0.
        assert cfg.intervention_perc <= 1.

    # Get dataloaders
    train_dataset, test_dataset = get_dataset(cfg)
    batch_size = cfg.get("bs") if hasattr(cfg, "bs") else 1
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=cfg.num_workers, batch_size=batch_size)
    
    if cfg.precomputed_file:
        process_answers(cfg.precomputed_file, test_dataloader.dataset.class_mapping, mistral_max_new_tokens=cfg.mistral_max_new_tokens, debug=cfg.debug)
    else:
        if cfg.get("name") == 'med-flamingo':
            """ Run Med-Flamingo """
            from src.models.med_flamingo.demo import MedFlamingo

            log.info("Starting evaluation on Med-Flamingo ...")
            model = MedFlamingo(os.environ["LLAMA_PATH"])
        
        elif cfg.get("name") == 'open-flamingo':
            """ Run OpenFlamingo """
            from src.models.open_flamingo.demo import OpenFlamingo

            log.info("Starting evaluation on Open-Flamingo ...")
            model = OpenFlamingo()

        elif cfg.get("name") == 'chexagent':
            """ Run CheXagent """
            from src.models.chexagent.demo import CheXagent

            log.info("Starting evaluation on CheXagent ...")
            model = CheXagent(cfg.max_memory)

        elif cfg.get("name") == 'llava-med':
            """ Run LLaVA-Med """
            from src.models.llava_med.demo import LlavaMed

            log.info("Starting evaluation on LLaVA-Med ...")
            model = LlavaMed()

        elif cfg.get("name") == 'vila8B':
            """ Run VILA """
            from src.models.vila.demo import Vila

            log.info("Starting evaluation on VILA 8B ...")
            model = Vila(version="8B")

        elif cfg.get("name") == 'vila40B':
            """ Run VILA """
            from src.models.vila.demo import Vila

            log.info("Starting evaluation on VILA 40B ...")
            model = Vila(version="40B")
        
        elif cfg.get("name") == 'skingpt4':
            """ Run SkinGPT-4 """
            from src.models.skingpt4.demo import SkinGPT4

            log.info("Starting evaluation on SkinGPT-4 ...")
            model = SkinGPT4(os.environ["LLAMA2_PATH"], os.environ["SKINGPT4_PATH"])
        
        elif cfg.get("name") == 'llava-ov':
            """ Run LLaVA-OneVision """
            from src.models.llava_ov.demo import LlavaOV

            log.info("Starting evaluation on LLaVA-OneVision ...")
            model = LlavaOV()
        
        elif cfg.get("name") == 'qwen2-vl':
            """ Run Qwen2-VL """
            from src.models.qwen2_vl.demo import Qwen2VL

            log.info("Starting evaluation on Qwen2-VL ...")
            model = Qwen2VL()
        
        elif cfg.get("name") == 'minicpm':
            """ Run MiniCPM """
            from src.models.mini_cpm.demo import miniCPM

            log.info("Starting evaluation on MiniCPM ...")
            model = miniCPM()

        elif cfg.get("name") == 'internvl2':
            """ Run InternVL2 """
            from src.models.internvl.demo import InternVL2

            log.info("Starting evaluation on InternVL2-8B ...")
            model = InternVL2()

        elif cfg.get("name") == 'idefics3':
            """ Run Idefics3 """
            from src.models.idefics.demo import Idefics3

            log.info("Starting evaluation on Idefics3 ...")
            model = Idefics3()

        elif cfg.get("name") == 'mplug':
            """ Run mPLUG-Owl3 """
            from src.models.mplug_owl3.demo import mPLUGOwl3

            log.info("Starting evaluation on mPLUG-Owl3 ...")
            model = mPLUGOwl3()

        else:
            raise ValueError(f"The experiment {cfg.get('name')} has not a valid implementation.")
        
        model.model.eval()
        log.info("Starting evaluation...")
        output = {}

        if cfg.demo_selection is not None:
            if "random" in cfg.demo_selection:
                id_selector = RANDOM(
                    cfg,
                    train_dataset.valid_ids,
                    annotations=train_dataset.prepare_data_for_rices() if "per_class" in cfg.demo_selection else None,
                )
            elif "rices" in cfg.demo_selection:
                id_selector = RICES(
                    cfg,
                    train_dataset.valid_ids,
                    annotations=train_dataset.prepare_data_for_rices() if "per_class" in cfg.demo_selection else None,
                    mode=cfg.demo_selection.split("_")[-1] if "per_class" in cfg.demo_selection else None,
                )
            elif cfg.demo_selection == "mmices":
                if cfg.mmices_text_features == "concepts":
                    concepts_dict = {}
                    for b in train_dataset:
                        ids = b["img_id"]
                        cpts = b["clinical_concepts"]
                        for i in range(len(ids)):
                            concepts_dict[ids[i]] = cpts[i].cpu().numpy()
                id_selector = MMICES(cfg, train_dataset.valid_ids, K=cfg.mmices_K, train_concepts=concepts_dict)
        
        for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            demos_images = []
            demos_labels = None
            demos_concepts = None

            if cfg.debug:
                if not((idx < 1) or (idx > len(test_dataloader) - 2)): continue
            
            if cfg.use_concepts is None:
                query_concepts = None
            elif ".xlsx" in cfg.use_concepts:
                query_concepts = [pred_concepts_df.loc[imgid]["pred_concepts"].copy() for imgid in batch["img_id"]]
            else:
                raise ValueError
            
            # ICL
            if cfg.demo_selection is not None:
                if "rices" in cfg.demo_selection:
                    demo_ids = np.array([id_selector.get_context_keys(key=x, n=cfg.n_demos, data_column="labels" if "_per_class" in cfg.demo_selection else None) for x in batch["img_id"]])
                else:
                    demo_ids = np.array([id_selector.get_context_keys(key=batch["img_id"][x], n=cfg.n_demos, query_concepts=query_concepts[x] if cfg.mmices_text_features == "concepts" else None) for x in range(len(batch["img_id"]))])
                n_demos_per_sample = len(demo_ids[0])
                demo_ids = np.array(demo_ids).flatten()
                sampler = CustomIndexSampler(train_dataset, demo_ids)
                train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=len(demo_ids), num_workers=cfg.num_workers, drop_last=False)
                train_batch = next(iter(train_dataloader))
                demos_images = [Image.open(x).convert("RGB") for x in train_batch["img_path"]]
                demos_images = [demos_images[i:i+n_demos_per_sample] for i in range(0, len(demos_images), n_demos_per_sample)]
                demos_labels = train_batch["class_label"].tolist()
                demos_labels = [demos_labels[i:i+n_demos_per_sample] for i in range(0, len(demos_labels), n_demos_per_sample)]
                if cfg.use_concepts is not None:
                    demos_concepts = train_batch["clinical_concepts"].tolist()
                    demos_concepts = [demos_concepts[i:i+n_demos_per_sample] for i in range(0, len(demos_concepts), n_demos_per_sample)]

            # finally predict on query image
            query_images = [Image.open(x).convert("RGB") for x in batch["img_path"]]

            for x in range(len(batch["img_id"])):
                output[batch["img_id"][x]] = {
                    "gt_label": batch["class_label"][x].item(),
                    "gt_concepts": batch["clinical_concepts"].tolist()[x]
                }
            
            if cfg.intervention_perc > 0.:
                query_concepts = np.array(query_concepts)
                gt_concepts = np.array(batch["clinical_concepts"])
                n_concepts = len(gt_concepts[0])
                scores = np.abs(query_concepts - gt_concepts)
                intervention_order = np.argsort(scores)[::-1]
                concepts_to_intervene = intervention_order[:, :int(cfg.intervention_perc*n_concepts)]
                if int(cfg.intervention_perc*n_concepts) > 0:
                    rows = np.arange(concepts_to_intervene.shape[0])
                    query_concepts[rows[:, None], concepts_to_intervene] = gt_concepts[rows[:, None], concepts_to_intervene]
            
            if cfg.get('debug', False):
                print(f'>>> ID: {batch["img_id"][0]}')
            
            # predict final classification (based on concepts)
            clf_prompts = []
            for i in range(len(query_images)):
                clf_instruction, clf_query_prompt, clf_demos_prompts, option2gt = test_dataloader.dataset.get_classification_prompt(
                    query_concepts[i] if query_concepts is not None else None,
                    demos_labels[i] if demos_labels is not None else None,
                    demos_concepts[i] if demos_concepts is not None else None
                )
                clf_prompt_model = model.get_prompt(clf_instruction, clf_query_prompt, clf_demos_prompts)
                output[batch["img_id"][i]]["clf_question"] = clf_query_prompt
                output[batch["img_id"][i]]["option2gt"] = option2gt
                clf_prompts.append(clf_prompt_model)
            
            clf_answers = model.predict(query_images, clf_prompts, cfg.max_new_tokens, demo_images=demos_images)
            for x in range(len(batch["img_id"])): 
                output[batch["img_id"][x]]["clf_answer"] = clf_answers[x].strip()

            if cfg.get('debug', False):
                print(f'>>> GT: {batch["class_label"][-1].item()} // Option2GT: {option2gt} // Prompt: {clf_prompt_model} // LVLM: {clf_answers[-1]}')

        csv_file = os.path.join(cfg.paths.output_dir, 'model_output_raw_classification.xlsx')
        df = pd.DataFrame.from_dict(output, orient="index")
        df = df.sort_index(axis=1)
        df.index.name = "img_id"
        df.to_excel(csv_file)
                    
        log.info(f"Results saved to {cfg.paths.output_dir}")

        # free GPU memory
        del model
        del query_images
        del demos_images
        gc.collect()
        torch.cuda.empty_cache()

        process_answers(csv_file, test_dataset.class_mapping, mistral_max_new_tokens=cfg.mistral_max_new_tokens, debug=cfg.debug)

if __name__ == "__main__":
    main()