import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


class Fitzpatrick_Dataset(Dataset):
    """Fitzpatrick17k Dataset.

    Args:
        root         (str): Root directory of dataset.
        csv_path     (str): Path to the metadata CSV file. Defaults to `{root}/ddi_metadata.csv`
        transform         : Function to transform and collate image input. (can use test_transform from this file) 
    """
    def __init__(self, root, csv_path=None, filter=0):    
        assert filter >= 0.
        assert filter <= 1.0    
        if csv_path is None:
            csv_path = os.path.join(root, "fitzpatrick17k_original_annotations.csv")
    
        self.data = pd.read_csv(csv_path)
        self.root = root
        self.class_mapping = {0: "benign", 1: "malignant"}
        self.clinical_concepts = ["papule", "plaque", "pustule", "bulla", "patch", "nodule", 
                                  "ulcer", "crust", "erosion", "atrophy", "exudate", "telangiectasia", 
                                  "scale", "scar", "friable", "dome-shaped", "brown(hyperpigmentation)", 
                                  "white(hypopigmentation)", "purple", "yellow", "black", "erythema"]
        
        self.clinical_concepts_mapping = {x: {0: "absent", 1: "present"} for x in self.clinical_concepts}
        
        self.valid_ids = self.filter_data(filter)

    def filter_data(self, keep_perc):
        def sample_by_label(group, frac):
            return group.sample(frac=frac)
        
        df = self.data[["img_path", "benign_malignant"]].copy()
        df["img_path"] = df["img_path"].apply(lambda x: x[x.rfind("/")+1 : -4])
        df = df.groupby("benign_malignant").apply(sample_by_label, frac=keep_perc, include_groups=False)
        
        return list(df["img_path"].values)

    def prepare_data_for_rices(self):
        new_df = self.data.copy(deep=True)
        split = new_df["attribute_label"].apply(eval).apply(pd.Series)
        split = split.rename(columns={idx: cpt for idx,cpt in enumerate(self.clinical_concepts)})
        new_df = pd.concat([new_df, split], axis=1)
        new_df["image_id"] = new_df["img_path"].apply(lambda x: x[x.rfind("/")+1 : -4])
        new_df = new_df.rename(columns={"benign_malignant": "labels"})
        return new_df

    def convert_numbers_to_concepts(self, concepts: list):
        return {name: concept for name, concept in zip(self.clinical_concepts, concepts)}

    def convert_concepts_to_numbers(self, concepts: list):
        return [1 if c in concepts else 0 for c in self.clinical_concepts] 
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        # Get sample
        sample = self.data.iloc[idx]

        # Get image ID
        img_id = sample["img_path"][sample["img_path"].rfind('/')+1 : -4]

        # Get image path
        img_path = os.path.join(self.root, img_id + ".jpg")

        # Get malignancy
        malignant = sample["benign_malignant"]

        # Get skin concepts
        skinconcepts = np.array(eval(sample["attribute_label"]))

        return {"img_id": img_id, "img_path": img_path, "class_label": malignant, "clinical_concepts": skinconcepts}

    def get_options(self, mapping):
        options_keys = list(mapping.copy().keys())

        option2gt = {f"{chr(ord('A') + i)})": options_keys[i] for i in range(len(options_keys))}
        gt2option = {options_keys[i]: f"{chr(ord('A') + i)})" for i in range(len(options_keys))}

        options_prompt = "\n"
        for k, v in option2gt.items():
            options_prompt += f"{k} {mapping[v]}\n"

        return options_prompt, option2gt, gt2option
    
    def get_concept_prompt(self, concept, demos_concepts=None):
        if concept == "papule":
            instruction = "A papule is a small, solid, raised lesion with a diameter less than 1 cm, typically palpable and may have a distinct border.\n"
        elif concept == "plaque":
            instruction = "Plaque is a broad, elevated lesion with a diameter greater than 1 cm, often with a rough or scaly surface.\n"
        elif concept == "pustule":
            instruction = "A pustule is a small, elevated lesion containing pus, appearing white or yellowish, often surrounded by erythema.\n"
        elif concept == "bulla":
            instruction = "A bulla is a large, fluid-filled blister with a diameter greater than 1 cm, which can be tense and easily ruptured.\n"
        elif concept == "patch":
            instruction = "A patch is a flat, discolored area of skin with a diameter greater than 1 cm, often lighter or darker than surrounding skin.\n"
        elif concept == "nodule":
            instruction = "A nodule is a solid, raised lesion deeper than a papule, typically with a diameter greater than 1 cm, and may be palpable.\n"
        elif concept == "ulcer":
            instruction = "An ulcer is a break in the skin or mucous membrane with a loss of epidermis and dermis, often presenting as a depressed lesion with irregular borders.\n"
        elif concept == "crust":
            instruction = "A crust is a dry, rough surface resulting from the drying of exudate or serum on the skin, often forming over an ulcer or wound.\n"
        elif concept == "erosion":
            instruction = "An erosion is a superficial loss of skin that does not extend into the dermis, usually appearing as a moist, depressed area.\n"
        elif concept == "atrophy":
            instruction = "Atrophy is the thinning or loss of skin tissue, leading to a depressed appearance, often resulting in a fragile and wrinkled surface.\n"
        elif concept == "exudate":
            instruction = "An exudate is a fluid that oozes out of a lesion or wound, which can be serous, purulent, or hemorrhagic, depending on its composition.\n"
        elif concept == "telangiectasia":
            instruction = "Telangiectasia appears as dilated, small blood vessels near the surface of the skin, often appearing as fine, red or purple lines.\n"
        elif concept == "scale":
            instruction = "Scale appears as a flake or layer of dead skin cells that may shed from the surface, often seen in conditions like psoriasis or eczema.\n"
        elif concept == "scar":
            instruction = "A scar is a mark left on the skin after the healing of a wound or injury, which may be flat, raised, or depressed compared to surrounding skin.\n"
        elif concept == "friable":
            instruction = "Friable skin corresponds to skin that easily breaks or bleeds with minimal trauma, often seen in conditions like malignancies or chronic irritation.\n"
        elif concept == "dome-shaped":
            instruction = "A dome-shaped lesion is a lesion with a rounded, elevated appearance, resembling the shape of a dome, which can be smooth or irregular.\n"
        elif concept == "brown(hyperpigmentation)":
            instruction = "Hyperpigmentation, or an area of the skin that appears brown, corresponds to darkened skin area due to excess melanin production, often seen in conditions like age spots or melasma.\n"
        elif concept == "white(hypopigmentation)":
            instruction = "Hypopigmentation, or an area of the skin that appears white, corresponds to lightened skin area due to reduced melanin production, which may result from conditions like vitiligo or post-inflammatory hypopigmentation.\n"
        elif concept == "purple":
            instruction = "A purple area on the skin might indicate bruising or bleeding beneath the skin, often seen in conditions like purpura or ecchymosis.“\n"
        elif concept == "yellow":
            instruction = "A yellow area on the skin corresponds to a color change indicating the presence of bilirubin or lipids, often seen in conditions like jaundice or xanthomas.\n"
        elif concept == "black":
            instruction = "Dark pigmentation often indicates the presence of melanin or necrosis, frequently observed in melanoma or necrotic tissue.\n"
        elif concept == "erythema":
            instruction = "Erythema corresponds to redness of the skin caused by increased blood flow to the capillaries, commonly seen in inflammation or irritation.\n"
        else:
            raise ValueError
        
        options, option2gt, gt2option = self.get_options(self.clinical_concepts_mapping[concept])
        question = f"In the image, {concept} is:{options}Choose one option. Do not provide additional information.".format(options=options)
        
        demos_prompts = []
        if demos_concepts is not None:
            instruction += "Consider the following examples:\n"
            for dcpts in demos_concepts:
                dprompt = f"{question} Answer: {gt2option[dcpts[concept]]} {self.clinical_concepts_mapping[concept][dcpts[concept]]}"
                demos_prompts.append(dprompt)

        query_prompt = f"{question} Answer:"

        return instruction, query_prompt, demos_prompts, option2gt

    def get_classification_prompt(self, query_concepts=None, demos_labels=None, demos_concepts=None):
        if query_concepts is None:
            question = "What is the diagnosis shown in this image?\nOptions:{options}Choose one option. Do not provide additional information. Answer:"
        else:
            question = "What is the diagnosis that is associated with the following concepts: {concepts} \nOptions:{options}Choose one option. Do not provide additional information. Answer:"

        options, option2gt, gt2option = self.get_options(self.class_mapping)

        instruction = ""
        if query_concepts is not None:
            instruction += "Consider the following useful concepts to diagnose skin lesions.\n" \
            "A papule is a small, solid, raised lesion with a diameter less than 1 cm, typically palpable and may have a distinct border.\n" \
            "Plaque is a broad, elevated lesion with a diameter greater than 1 cm, often with a rough or scaly surface.\n" \
            "A pustule is a small, elevated lesion containing pus, appearing white or yellowish, often surrounded by erythema.\n" \
            "A bulla is a large, fluid-filled blister with a diameter greater than 1 cm, which can be tense and easily ruptured.\n" \
            "A patch is a flat, discolored area of skin with a diameter greater than 1 cm, often lighter or darker than surrounding skin.\n" \
            "A nodule is a solid, raised lesion deeper than a papule, typically with a diameter greater than 1 cm, and may be palpable.\n" \
            "An ulcer is a break in the skin or mucous membrane with a loss of epidermis and dermis, often presenting as a depressed lesion with irregular borders.\n" \
            "A crust is a dry, rough surface resulting from the drying of exudate or serum on the skin, often forming over an ulcer or wound.\n" \
            "An erosion is a superficial loss of skin that does not extend into the dermis, usually appearing as a moist, depressed area.\n" \
            "Atrophy is the thinning or loss of skin tissue, leading to a depressed appearance, often resulting in a fragile and wrinkled surface.\n" \
            "An exudate is a fluid that oozes out of a lesion or wound, which can be serous, purulent, or hemorrhagic, depending on its composition.\n" \
            "Telangiectasia appears as dilated, small blood vessels near the surface of the skin, often appearing as fine, red or purple lines.\n" \
            "Scale appears as a flake or layer of dead skin cells that may shed from the surface, often seen in conditions like psoriasis or eczema.\n" \
            "A scar is a mark left on the skin after the healing of a wound or injury, which may be flat, raised, or depressed compared to surrounding skin.\n" \
            "Friable skin corresponds to skin that easily breaks or bleeds with minimal trauma, often seen in conditions like malignancies or chronic irritation.\n" \
            "A dome-shaped lesion is a lesion with a rounded, elevated appearance, resembling the shape of a dome, which can be smooth or irregular.\n" \
            "Hyperpigmentation, or an area of the skin that appears brown, corresponds to darkened skin area due to excess melanin production, often seen in conditions like age spots or melasma.\n" \
            "Hypopigmentation, or an area of the skin that appears white, corresponds to lightened skin area due to reduced melanin production, which may result from conditions like vitiligo or post-inflammatory hypopigmentation.\n" \
            "A purple area on the skin might indicate bruising or bleeding beneath the skin, often seen in conditions like purpura or ecchymosis.“\n" \
            "A yellow area on the skin corresponds to a color change indicating the presence of bilirubin or lipids, often seen in conditions like jaundice or xanthomas.\n" \
            "Dark pigmentation often indicates the presence of melanin or necrosis, frequently observed in melanoma or necrotic tissue.\n" \
            "Erythema corresponds to redness of the skin caused by increased blood flow to the capillaries, commonly seen in inflammation or irritation.\n"

       # demonstration prompts
        demos_prompts = []
        if demos_labels is not None:
            instruction += "Consider the following examples:\n"
            
            for didx, dlabel in enumerate(demos_labels):
                cptp = ""
                if demos_concepts is not None:                
                    # in case we are using concepts to predict the classification label
                    for cidx, dcpts in enumerate(demos_concepts[didx]):
                        if cidx == len(demos_concepts[didx]) - 1:
                            cptp += f"and {'no ' + self.clinical_concepts[cidx] if dcpts == 0 else self.clinical_concepts[cidx]}."
                        else:
                            cptp += f"{'no ' + self.clinical_concepts[cidx] if dcpts == 0 else self.clinical_concepts[cidx]}, "
                                    
                    dprompt = f"{question.format(concepts=cptp, options=options)} {gt2option[dlabel]} {self.class_mapping[dlabel]}"                        
                
                else:
                    # when no concepts are being used to predict the classification label
                    dprompt = f"{question.format(options=options)} {gt2option[dlabel]} {self.class_mapping[dlabel]}"                        

                demos_prompts.append(dprompt)

        # query prompt
        query_prompt = ""
        qcpt = ""
        if query_concepts is not None:           
            for cidx, qcpts in enumerate(query_concepts):
                if cidx == len(query_concepts) - 1:
                    qcpt += f"and {'no ' + self.clinical_concepts[cidx] if qcpts <= 0 else self.clinical_concepts[cidx]}."
                else:
                    qcpt += f"{'no ' + self.clinical_concepts[cidx] if qcpts <= 0 else self.clinical_concepts[cidx]}, "  
            query_prompt = f"{question.format(concepts=qcpt, options=options)}"
        else:
            query_prompt = f"{question.format(options=options)}"
        
        return instruction, query_prompt, demos_prompts, option2gt

    def get_description_prompt(self):
        return "Describe the image. Answer: "
    
if __name__ == '__main__':
    # Debug dataset

    dataset = Fitzpatrick_Dataset(
        root="/home/icrto/Dermatology/SKINCON/data/fitzpatrick17k", 
        csv_path="data/splits/SkinCon_train.csv",
        filter=0.25
    )
    print(len(dataset))
    print(len(dataset.valid_ids))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 3:
            break
