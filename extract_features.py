import hydra
from omegaconf import DictConfig
import os
import dotenv
from tqdm import tqdm
from PIL import Image
import pickle
import numpy as np
import json
import base64

from src.utils.dataset_loaders import get_dataset
from torch.utils.data import DataLoader

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dir_path = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(dir_path + '/var_environment.env', override=True)

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

@hydra.main(version_base=None, config_path="configs", config_name="extract_features.yaml")
def run(cfg: DictConfig):
    """Extracts features from a dataset and save it into numpy files.

    Args:
        cfg (DictConfig): DictConfig configuration composed by Hydra.
    """
    assert cfg.features in ["image", "descriptions", "concepts"]

    train_dataset, test_dataset = get_dataset(cfg)
    batch_size = cfg.get("bs") if hasattr(cfg, "bs") else 1
    train_dataloader = DataLoader(train_dataset, shuffle=False, num_workers=cfg.num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=cfg.num_workers, batch_size=batch_size)

    if cfg.get("name") == 'med-flamingo':
        """ Extract features from Med-Flamingo """
        from src.models.med_flamingo.demo import MedFlamingo
        model = MedFlamingo(os.environ["LLAMA_PATH"])
         
    elif cfg.get("name") == 'open-flamingo':
        """ Extract features from OpenFlamingo """
        from src.models.open_flamingo.demo import OpenFlamingo
        model = OpenFlamingo()

    elif cfg.get("name") == 'llava-next':
        """ Extract features from LLaVA-NeXT """
        from src.models.llava_next.demo import LlavaNext
        model = LlavaNext(cfg.max_memory)

    elif cfg.get("name") == 'chexagent':
        """ Extract features from CheXagent """
        from src.models.chexagent.demo import CheXagent
        model = CheXagent(cfg.max_memory)

    elif cfg.get("name") == 'llava-med':
        """ Extract features from LLaVA-Med """
        from src.models.llava_med.demo import LlavaMed
        model = LlavaMed()
    
    elif cfg.get("name") == 'idefics':
        """ Extract features from IDEFICS """
        from src.models.idefics.demo import Idefics
        model = Idefics()

    elif cfg.get("name") == 'vila8B':
        """ Extract features from VILA """
        from src.models.vila.demo import Vila
        model = Vila(version="8B")

    elif cfg.get("name") == 'vila40B':
        """ Extract features from VILA """
        from src.models.vila.demo import Vila
        model = Vila(version="40B")

    elif cfg.get("name") == 'skingpt4':
        """ Extract features from SkinGPT-4 """
        from src.models.skingpt4.demo import SkinGPT4
        model = SkinGPT4(os.environ["LLAMA2_PATH"], os.environ["SKINGPT4_PATH"])
    
    elif cfg.get("name") == 'llava-ov':
        """ Extract features from LLaVA-OneVision """        
        assert cfg.bs == 1, "llava-ov only supports batch size 1"
        from src.models.llava_ov.demo import LlavaOV
        model = LlavaOV()

    elif cfg.get("name") == 'qwen2-vl':
        """ Extract features from Qwen2-VL """
        assert cfg.bs == 1, "qwen2-vl only supports batch size 1"
        from src.models.qwen2_vl.demo import Qwen2VL
        model = Qwen2VL()
    
    elif cfg.get("name") == 'minicpm':
        """ Extract features from MiniCPM """
        from src.models.mini_cpm.demo import miniCPM
        model = miniCPM()

    elif cfg.get("name") == 'internvl2':
        """ Extract features from InternVL2 """
        from src.models.internvl.demo import InternVL2
        model = InternVL2()

    elif cfg.get("name") == 'idefics3':
        """ Extract features from Idefics3 """
        from src.models.idefics3.demo import Idefics3
        model = Idefics3()

    elif cfg.get("name") == 'mplug':
        """ Extract features from mPLUG-Owl3 """
        from src.models.mplug_owl3.demo import mPLUGOwl3
        model = mPLUGOwl3()
    
    elif cfg.get("name") == 'clip':
        """ Extract features from BiomedCLIP """
        from src.models.clip_vitb16 import CLIPViTB16
        model = CLIPViTB16()

    elif cfg.get("name") == 'biomedclip':
        """ Extract features from CLIP """
        from src.models.biomedclip import BiomedCLIP
        model = BiomedCLIP()
    
    elif cfg.get("name") == 'medimageinsight':
        """ Extract features from MedImageInsight """
        from src.models.MedImageInsights.medimageinsightmodel import MedImageInsight
        model = MedImageInsight(
            model_dir="src/models/MedImageInsights/2024.09.27",
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth"
        )
        model.load_model()

    else:
        raise ValueError(f"The experiment {cfg.get('name')} has not a valid implementation.")

    model.model.eval()
    
    # Create dir if not exists
    custom_dir = os.path.join("data", f"{cfg.features}_features", cfg.data.name)
    if not os.path.exists(custom_dir):
        os.makedirs(custom_dir)

    features = {}
    for loader, split in zip([train_dataloader, test_dataloader], ["train", "test"]):
        if cfg.features == "descriptions":
            with open(os.path.join("data", "descriptions", f"{cfg.data.name}", f"{cfg.data.name}_{cfg.description_model}_descriptions_{split}.json"), "r") as f:
                desc = json.load(f)
        for batch in tqdm(loader):
            img_ids = batch["img_id"]
            if cfg.features == "image":
                if cfg.get("name") == 'medimageinsight':
                    imgs = [base64.encodebytes(read_image(x)).decode("utf-8") for x in batch["img_path"]]
                    feats = model.encode(images=imgs)["image_embeddings"]
                else:
                    imgs = [Image.open(x).convert("RGB") for x in batch["img_path"]]
                    feats = model.extract_image_features(imgs)
            elif cfg.features == "descriptions":
                inputs = [desc[id] for id in img_ids]
                feats = model.extract_text_features(inputs)

            for id, ft in zip(img_ids, feats):
                features[id] = ft / (np.linalg.norm(ft)) # L2 norm
        
        if cfg.features == "descriptions":
            fname = os.path.join(custom_dir, f"{cfg.data.name}_{cfg.name}_{cfg.description_model}_{cfg.features}_features_{split}.pkl")
        else:
            fname = os.path.join(custom_dir, f"{cfg.data.name}_{cfg.name}_{cfg.features}_features_{split}.pkl")

        with open(fname, "wb") as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    run()