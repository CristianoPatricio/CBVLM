import hydra
from omegaconf import DictConfig
import os
import dotenv
from tqdm import tqdm
import gc
import torch
from PIL import Image
import re
import random
import pandas as pd
import yaml
import ast
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

def process_answers(csv_file, concepts, mistral_max_new_tokens=100, debug=False, mapping=None):
    log.info(f"Processing predictions of file {csv_file}")

    output = pd.read_excel(csv_file, index_col=0)

    with open(os.path.join(os.path.dirname(csv_file), ".hydra", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    mistral = None
    
    # process extracted options
    processed_df = {}
    for idx, row in tqdm(output.iterrows(), total=len(output)):
        processed_df[idx] = {}

        pred_concepts = [None for _ in range(len(concepts))]
        for cpt_idx, cpt in enumerate(concepts):
            answer = str(row[f"{cpt}_answer"])
            option2gt = row[f"{cpt}_option2gt"]
            option2gt = ast.literal_eval(option2gt)
            y_pred, mistral_output, mistral_option2gt, mistral = process_multiple_choice(cfg["name"], answer, option2gt, mapping[cpt] if mapping is not None else None, mistral, mistral_max_new_tokens=mistral_max_new_tokens, debug=debug)
            processed_df[idx][f"{cpt}_option2gt_mistral"] = mistral_option2gt
            processed_df[idx][f"{cpt}_mistral"] = None
            processed_df[idx][f"{cpt}_mistral"] = mistral_output
            pred_concepts[cpt_idx] = y_pred

        processed_df[idx]["pred_concepts"] = pred_concepts
        
    processed_df = pd.DataFrame.from_dict(processed_df, orient="index")
    final_df = pd.concat([output, processed_df], axis=1)
    final_df.index = final_df.index.map(str)
    csv_name = csv_file.replace("_raw_concepts.xlsx", "_processed_concepts.xlsx")
    final_df.to_excel(csv_name)
    log.info(f"Concepts processed and saved to {csv_name}")

    log.info("Computing metrics")
    compute_metrics(csv_name)

@hydra.main(version_base=None, config_path="configs", config_name="concepts.yaml")
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

    # Get dataloaders
    train_dataset, test_dataset = get_dataset(cfg)
    batch_size = cfg.get("bs") if hasattr(cfg, "bs") else 1
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=cfg.num_workers, batch_size=batch_size)
    
    if cfg.precomputed_file:
        process_answers(cfg.precomputed_file, test_dataloader.dataset.clinical_concepts, mistral_max_new_tokens=cfg.mistral_max_new_tokens, debug=cfg.debug, mapping=test_dataloader.dataset.clinical_concepts_mapping)
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
                id_selector = MMICES(
                    cfg,
                    train_dataset.valid_ids,
                    K=cfg.mmices_K
                )
        
        for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            demos_images = []
            demos_concepts = None

            if cfg.debug:
                if not((idx < 1) or (idx > len(test_dataset) - 2)): continue
            
            # ICL (when it's the same for all concepts of the same image)
            if cfg.demo_selection is not None and "_per_class" not in cfg.demo_selection:
                demo_ids = np.array([id_selector.get_context_keys(key=x, n=cfg.n_demos) for x in batch["img_id"]]).flatten()
                sampler = CustomIndexSampler(train_dataset, demo_ids)
                train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=len(demo_ids), num_workers=cfg.num_workers, drop_last=False)
                train_batch = next(iter(train_dataloader))
                demos_images = [Image.open(x).convert("RGB") for x in train_batch["img_path"]]
                demos_images = [demos_images[i:i+cfg.n_demos] for i in range(0, len(demos_images), cfg.n_demos)]
                demos_concepts = train_batch["clinical_concepts"].tolist()
                demos_concepts = [train_dataloader.dataset.convert_numbers_to_concepts(dc) for dc in demos_concepts]
                demos_concepts = [demos_concepts[i:i+cfg.n_demos] for i in range(0, len(demos_concepts), cfg.n_demos)]
            
            # finally predict on query image
            query_images = [Image.open(x).convert("RGB") for x in batch["img_path"]]

            concepts = test_dataloader.dataset.clinical_concepts.copy()
            random.shuffle(concepts)

            for x in range(len(batch["img_id"])):
                output[batch["img_id"][x]] = {
                    "gt_label": batch["class_label"][x].item(),
                    "gt_concepts": batch["clinical_concepts"].tolist()[x]
                }
            query_concepts = [test_dataloader.dataset.convert_numbers_to_concepts(x.tolist() for x in batch["clinical_concepts"])] 
            if cfg.get('debug', False):
                print(f'>>> ID: {batch["img_id"][0]}')
            
            for cpt in concepts:
                # in random/rices_per_class the demo_images depend on the concept
                if cfg.demo_selection is not None and "per_class" in cfg.demo_selection and cfg.n_demos > 0:
                    demo_ids = [id_selector.get_context_keys(key=x, n=cfg.n_demos, data_column=cpt) for x in batch["img_id"]]
                    n_demos_per_sample = len(demo_ids[0])
                    demo_ids = np.array(demo_ids).flatten()
                    sampler = CustomIndexSampler(train_dataset, demo_ids)
                    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=len(demo_ids), num_workers=cfg.num_workers, drop_last=False)
                    train_batch = next(iter(train_dataloader))
                    demos_images = [Image.open(x).convert("RGB") for x in train_batch["img_path"]]
                    demos_images = [demos_images[i:i+n_demos_per_sample] for i in range(0, len(demos_images), n_demos_per_sample)]
                    demos_concepts = train_batch["clinical_concepts"].tolist()
                    demos_concepts = [train_dataloader.dataset.convert_numbers_to_concepts(dc) for dc in demos_concepts]
                    demos_concepts = [demos_concepts[i:i+n_demos_per_sample] for i in range(0, len(demos_concepts), n_demos_per_sample)]

                cpt_prompts = []
                for i in range(len(query_images)):
                    cpt_instruction, cpt_query_prompt, cpt_demos_prompts, option2gt = test_dataloader.dataset.get_concept_prompt(cpt, demos_concepts[i] if demos_concepts is not None else None)
                    cpt_prompt_model = model.get_prompt(cpt_instruction, cpt_query_prompt, cpt_demos_prompts)
                    output[batch["img_id"][i]][f"{cpt}_question"] = cpt_query_prompt
                    output[batch["img_id"][i]][f"{cpt}_option2gt"] = option2gt
                    cpt_prompts.append(cpt_prompt_model)
                
                cpt_answers = model.predict(query_images, cpt_prompts, cfg.max_new_tokens, demo_images=demos_images)
                for x in range(len(batch["img_id"])):
                    output[batch["img_id"][x]][f"{cpt}_answer"] = cpt_answers[x].strip()

                if cfg.get('debug', False):
                    print(f'>>> Question for {cpt}: {cpt_prompts[-1]} // Option2GT: {option2gt}\n')    
                    print(f">>> {query_concepts}")                
                    print(f'\t LVLM: {cpt_answers[-1]}\n')

        csv_file = os.path.join(cfg.paths.output_dir, 'model_output_raw_concepts.xlsx')
        df = pd.DataFrame.from_dict(output, orient="index")
        df = df.sort_index(axis=1)
        df.index.name = "img_id"
        df.index = df.index.map(str)
        df.to_excel(csv_file)
                    
        log.info(f"Results saved to {cfg.paths.output_dir}")

        # free GPU memory
        del model
        del query_images
        del demos_images
        gc.collect()
        torch.cuda.empty_cache()

        process_answers(csv_file, test_dataset.clinical_concepts, mistral_max_new_tokens=cfg.mistral_max_new_tokens, debug=cfg.debug, mapping=test_dataloader.dataset.clinical_concepts_mapping)

if __name__ == "__main__":
    main()