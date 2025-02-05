import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class CORDADataset(Dataset):
    """
    A custom Pytorch dataset for reading data from a CSV file regarding the CORDA dataset.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        img_extension (str): The file extension of the images.
        path_to_images (str): The path to the images folder.

    Attributes:
        data (DataFrame): Pandas DataFrame containing the data from the CSV file.

    """

    def __init__(self, csv_file, img_extension, path_to_images, filter=0):    
        assert filter >= 0.
        assert filter <= 1.0    
        self.data = pd.read_csv(csv_file)
        self.data = self.data.fillna(-1)
        self.img_extension = img_extension
        self.path_to_images = path_to_images
        self.class_mapping = {0: "no covid-19", 1: "covid-19"}
        self.clinical_concepts = [
            "enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "edema",
            "consolidation",
            "pneumonia",
            "pneumothorax",
            "pleural effusion"
        ]
        self.clinical_concepts_mapping = {x: {0: "absent", 1: "present"} for x in self.clinical_concepts}

        self.valid_ids = self.filter_data(filter)

    def filter_data(self, keep_perc):
        def sample_by_label(group, frac):
            return group.sample(frac=frac)
        
        df = self.data[["path", "covid"]]
        df = df.groupby("covid").apply(sample_by_label, frac=keep_perc, include_groups=False)
        
        return list(df["path"].values)
    
    def prepare_data_for_rices(self):
        new_df = self.data.copy(deep=True)
        split = new_df["attribute_labels"].apply(eval).apply(pd.Series)
        split = split.rename(columns={
            0: "ignore", 
            1: "enlarged cardiomediastinum",
            2: "cardiomegaly",
            3: "lung opacity",
            4: "edema",
            5: "consolidation",
            6: "pneumonia",
            7: "pneumothorax",
            8: "pleural effusion"
        })
        new_df = pd.concat([new_df, split], axis=1)
        new_df = new_df.rename(columns={"path": "image_id", "covid": "labels"})
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
        img_id = sample["path"]

        # Create image_path
        img_path = os.path.join(self.path_to_images, img_id + '.' + self.img_extension)

        # Get class label
        class_label = sample["covid"]
        
        # Get clinical_concepts
        clinical_concepts = np.array(eval(sample["attribute_labels"]))[1:]

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
        if concept == "enlarged cardiomediastinum":
            instruction = "An enlarged cardiomediastinum is an increase in the width of the mediastinum, which may suggest conditions like aortic aneurysm, lymphadenopathy, or heart failure.\n"
        elif concept == "cardiomegaly":
            instruction = "Cardiomegaly is an abnormal enlargement of the heart, often indicating underlying conditions such as heart failure, hypertension, or cardiomyopathy.\n"
        elif concept == "lung opacity":
            instruction = "Lung opacity corresponds to an area on the radiograph where the lung appears more opaque than normal, potentially due to fluid, consolidation, or masses within the lung tissue.\n"
        elif concept == "edema":
            instruction = "Edema is the accumulation of fluid in the lung interstitium or alveoli, causing increased opacity on imaging, often indicative of conditions such as heart failure or acute respiratory distress syndrome (ARDS).\n"
        elif concept == "consolidation":
            instruction = "Consolidation corresponds to an area of the lung where normal air-filled spaces have been replaced with fluid, cells, or other material, often seen in pneumonia or other infections.\n"
        elif concept == "pneumonia":
            instruction = "Pneumonia is an infection of the lung parenchyma causing inflammation and consolidation, which appears as areas of opacity on radiographs.\n"
        elif concept == "pneumothorax":
            instruction = "A pneumothorax corresponds to the presence of air in the pleural space, leading to lung collapse, often visible as a dark area on the radiograph where the lung is not present.\n"
        elif concept == "pleural effusion":
            instruction = "A pleural effusion corresponds to the accumulation of fluid in the pleural space between the lung and chest wall, appearing as a blunting of the costophrenic angles or a meniscus sign on radiographs.\n"
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
            instruction += "Consider the following useful concepts to diagnose covid-19.\n" \
            "An enlarged cardiomediastinum is an increase in the width of the mediastinum, which may suggest conditions like aortic aneurysm, lymphadenopathy, or heart failure.\n" \
            "Cardiomegaly is an abnormal enlargement of the heart, often indicating underlying conditions such as heart failure, hypertension, or cardiomyopathy.\n" \
            "Lung opacity corresponds to an area on the radiograph where the lung appears more opaque than normal, potentially due to fluid, consolidation, or masses within the lung tissue.\n" \
            "Edema is the accumulation of fluid in the lung interstitium or alveoli, causing increased opacity on imaging, often indicative of conditions such as heart failure or acute respiratory distress syndrome (ARDS).\n" \
            "Consolidation corresponds to an area of the lung where normal air-filled spaces have been replaced with fluid, cells, or other material, often seen in pneumonia or other infections.\n" \
            "Pneumonia is an infection of the lung parenchyma causing inflammation and consolidation, which appears as areas of opacity on radiographs.\n" \
            "A pneumothorax corresponds to the presence of air in the pleural space, leading to lung collapse, often visible as a dark area on the radiograph where the lung is not present.\n" \
            "A pleural effusion corresponds to the accumulation of fluid in the pleural space between the lung and chest wall, appearing as a blunting of the costophrenic angles or a meniscus sign on radiographs.\n"
        
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

    dataset = CORDADataset(
        csv_file="data/splits/corda_train.csv", 
        img_extension="png", 
        path_to_images='/home/icrto/CORDA/224x224',
        filter=0.25
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print(len(dataset.valid_ids))
    for batch in loader:
        print(batch)
        break
