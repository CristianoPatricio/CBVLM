import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class Derm7ptDataset(Dataset):
    """
    A custom Pytorch dataset for reading data from a CSV file regarding the Derm7pt dataset.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        img_extension (str): The file extension of the images.
        path_to_images (str): The path to the images folder.

    Attributes:
        data (DataFrame): Pandas DataFrame containing the data from the CSV file.

    """

    def __init__(self, csv_file, clinical_concepts_path, img_extension, path_to_images, filter=0):
        assert filter >= 0.
        assert filter <= 1.0   
        self.data = pd.read_csv(csv_file)

        self.clinical_concepts_path = clinical_concepts_path
        self.clinical_concepts_df = pd.read_csv(self.clinical_concepts_path)

        self.img_extension = img_extension
        self.path_to_images = path_to_images
        self.class_mapping = {0: "nevus", 1: "melanoma"}
        self.clinical_concepts = [
            "pigment network",
            "streaks",
            "dots and globules",
            "blue-whitish veil",
            "regression structures"
        ]
        self.clinical_concepts_mapping = {
            "pigment network": {0: "absent", 1: "typical", 2: "atypical"},
            "streaks": {0: "absent", 1: "regular", 2: "irregular"},
            "dots and globules": {0: "absent", 1: "regular", 2: "irregular"},
            "blue-whitish veil": {0: "absent", 1: "present"},
            "regression structures": {0: "absent", 1: "present"},
        }

        self.valid_ids = self.filter_data(filter)

    def filter_data(self, keep_perc):
        def sample_by_label(group, frac):
            return group.sample(frac=frac)
        
        df = self.data[["images", "labels"]]
        df = df.groupby("labels").apply(sample_by_label, frac=keep_perc, include_groups=False)
        
        return list(df["images"].values)
    
    def convert_numbers_to_concepts(self, concepts: list):
        return {name: concept for name, concept in zip(self.clinical_concepts, concepts)}
    
    def prepare_data_for_rices(self):
        new_df = self.data.copy(deep=True)
        new_df = new_df.rename(columns={"images": "image_id"})
        new_df["image_id"] = new_df["image_id"].astype(str)
        new_df = new_df.set_index("image_id")

        ccdf = self.clinical_concepts_df.copy(deep=True)
        ccdf["image_id"] = ccdf["image_id"].astype(str)
        ccdf = ccdf.rename(columns={"PN": "pigment network", "STR": "streaks", "DG": "dots and globules", "BWV": "blue-whitish veil" , "RS": "regression structures"})
        ccdf = ccdf.set_index("image_id")
        
        new_df = new_df.join(ccdf)
        new_df = new_df.reset_index()
        return new_df
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary containing information of a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: A dictionary containing information about the image[idx]
                img_id (str): The ID of the image.
                img_path (str): The full path of the image. 
                class_label (str): The diagnostic category of the dermoscopic image.
                clinical_concepts (list): A list containing binary values indicating the presence/absence of dermoscopic attributes in the image.
        """
        
        # Get sample
        sample = self.data.iloc[idx]
        
        # Get image ID
        img_id = sample["images"]

        # Create image_path
        img_path = os.path.join(self.path_to_images, img_id + '.' + self.img_extension)

        # Get class label
        class_label = sample["labels"]
        
        # Get clinical_concepts
        clinical_concepts_sample = self.clinical_concepts_df.loc[self.clinical_concepts_df.image_id == img_id]
        clinical_concepts = np.array([
            clinical_concepts_sample["PN"],
            clinical_concepts_sample["STR"],
            clinical_concepts_sample["DG"],
            clinical_concepts_sample["BWV"],
            clinical_concepts_sample["RS"]
        ])

        return {"img_id": img_id, "img_path": img_path, "class_label": class_label, "clinical_concepts": clinical_concepts.squeeze()}

    def get_options(self, mapping):
        options_keys = list(mapping.copy().keys())

        option2gt = {f"{chr(ord('A') + i)})": options_keys[i] for i in range(len(options_keys))}
        gt2option = {options_keys[i]: f"{chr(ord('A') + i)})" for i in range(len(options_keys))}

        options_prompt = "\n"
        for k, v in option2gt.items():
            options_prompt += f"{k} {mapping[v]}\n"

        return options_prompt, option2gt, gt2option
    
    def get_concept_prompt(self, concept, demos_concepts=None):
        options, option2gt, gt2option = self.get_options(self.clinical_concepts_mapping[concept])
        if concept == "pigment network":
            instruction = "The pigment network consists of intersecting brown lines forming a grid-like reticular pattern. " \
                "It can be absent, typical, or atypical. A typical pigment network appears as a regular grid-like pattern " \
                "on dermoscopy, consisting of thin lines that form an even mesh. The spaces (holes) between these lines are " \
                "relatively uniform in size and shape. In an atypical pigment network the lines forming the network are uneven in thickness, " \
                "and the holes or spaces between the lines vary in size and shape.\n"
            question = "In the image, the pigment network is:{options}Choose one option. Do not provide additional information.".format(options=options)
        elif concept == 'streaks':
            instruction = "Streaks are lineal pigmented projections at the periphery of a melanocytic lesion and include radial streaming (lineal streaks) and pseudopods (bulbous projections). " \
                "They can be absent, regular, or irregular. Regular streaks are symmetrically arranged around the periphery of the lesion, appearing consistently in both length and spacing. " \
                "Irregular streaks appear as projections at the periphery of a lesion and are irregular in length, thickness, and distribution.\n"
            question = "In the image, streaks are:{options}Choose one option. Do not provide additional information.".format(options=options)
        elif concept == 'dots and globules':
            instruction = "Dots and globules can be absent, regular, or irregular. Regular dots and globules are consistent in size and shape throughout the lesion, appearing either as small, " \
                "round dots or larger, round globules. Irregular dots and globules vary widely in size and shape, with some appearing small and round, while others are larger and more irregular.\n"
            question = "In the image, dots and globules are:{options}Choose one option. Do not provide additional information.".format(options=options)
        elif concept == 'blue-whitish veil':
            instruction = "The blue-whitish veil appears as an opaque, bluish-white area on the surface of the lesion, giving it a hazy or clouded appearance.\n"
            question = "In the image, the blue-whitish veil is:{options}Choose one option. Do not provide additional information.".format(options=options)
        elif concept == 'regression structures':
            instruction = "Regression structures appear as whitish, scar-like depigmented areas within the lesion, indicating areas where the pigment cells have been destroyed or the lesion has undergone partial regression.\n"
            question = "In the image, regression structures are:{options}Choose one option. Do not provide additional information.".format(options=options)
        else:
            raise ValueError
        
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
            question = "What is the type of skin lesion shown in this image?\nOptions:{options}Choose one option. Do not provide additional information. Answer:"
        else:
            question = "What is the type of skin lesion that is associated with the following dermoscopic concepts: {concepts} \nOptions:{options}Choose one option. Do not provide additional information. Answer:"
        
        options, option2gt, gt2option = self.get_options(self.class_mapping)
        
        instruction = ""
        if query_concepts is not None:
            instruction += "Consider the following useful concepts to diagnose melanoma.\n" \
                "The pigment network consists of intersecting brown lines forming a grid-like reticular pattern. " \
                "It can be absent, typical, or atypical. A typical pigment network appears as a regular grid-like pattern " \
                "on dermoscopy, consisting of thin lines that form an even mesh. The spaces (holes) between these lines are " \
                "relatively uniform in size and shape. In an atypical pigment network the lines forming the network are uneven in thickness, " \
                "and the holes or spaces between the lines vary in size and shape.\n" \
                "Streaks are lineal pigmented projections at the periphery of a melanocytic lesion and include radial streaming (lineal streaks) and pseudopods (bulbous projections). " \
                "They can be absent, regular, or irregular. Regular streaks are symmetrically arranged around the periphery of the lesion, appearing consistently in both length and spacing. " \
                "Irregular streaks appear as projections at the periphery of a lesion and are irregular in length, thickness, and distribution.\n" \
                "Dots and globules can be absent, regular, or irregular. Regular dots and globules are consistent in size and shape throughout the lesion, appearing either as small, " \
                "round dots or larger, round globules. Irregular dots and globules vary widely in size and shape, with some appearing small and round, while others are larger and more irregular.\n" \
                "The blue-whitish veil appears as an opaque, bluish-white area on the surface of the lesion, giving it a hazy or clouded appearance.\n" \
                "Regression structures appear as whitish, scar-like depigmented areas within the lesion, indicating areas where the pigment cells have been destroyed or the lesion has undergone partial regression.\n"
        
        # demonstration prompts
        demos_prompts = []
        if demos_labels is not None:
            instruction += "Consider the following examples:\n"
            
            for didx, dlabel in enumerate(demos_labels):
                cptp = ""
                if demos_concepts is not None:                
                    # in case we are using concepts to predict the classification label
                    for cidx, dcpts in enumerate(demos_concepts[didx]): 
                        cpt = self.clinical_concepts[cidx]
                        pred = self.clinical_concepts_mapping[cpt][dcpts]
                        if cidx == len(demos_concepts[didx]) - 1:
                            cptp += f"and {pred} {cpt}."
                        else:
                            cptp += f"{pred} {cpt}, "
                    
                    dprompt = f"{question.format(concepts=cptp, options=options)} {gt2option[dlabel]} {self.class_mapping[dlabel]}".replace("absent", "no").replace("present", "").replace("  ", " ")
                else:
                    # when no concepts are being used to predict the classification label
                    dprompt = f"{question.format(options=options)} {gt2option[dlabel]} {self.class_mapping[dlabel]}"                        
                
                demos_prompts.append(dprompt)

        # query prompt
        query_prompt = ""
        qcpt = ""
        if query_concepts is not None:
            for cidx, qcpts in enumerate(query_concepts):
                cpt = self.clinical_concepts[cidx]
                if qcpts == -1:
                    qcpts = 0
                pred = self.clinical_concepts_mapping[cpt][qcpts]
                if cidx == len(query_concepts) - 1:
                    qcpt += f"and {pred} {cpt}."
                else:
                    qcpt += f"{pred} {cpt}, "
            query_prompt = f"{question.format(concepts=qcpt, options=options)}".replace("absent", "no").replace("present", "").replace("  ", " ")
        else:
            query_prompt = f"{question.format(options=options)}"

        return instruction, query_prompt, demos_prompts, option2gt

    def get_description_prompt(self):
        return "Describe the dermoscopic image. Answer: "

if __name__ == '__main__':
    # Debug dataset

    dataset = Derm7ptDataset(
        csv_file="data/splits/derm7pt_train.csv", 
        clinical_concepts_path="data/splits/dermoscopic_concepts_Derm7pt_all.csv",
        img_extension='jpg', 
        path_to_images='/home/icrto/datasets/Dermatology/Derm7pt/Derm7pt_All_Images',
        filter=0.25
    )
    print(len(dataset))
    print(len(dataset.valid_ids))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for batch in dataloader:
        print(batch)
        break
