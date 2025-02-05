import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class DDRDataset(Dataset):
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
        self.class_mapping = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy", 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
        self.clinical_concepts = [
            "hard exudates", 
            "haemorrhages", 
            "microaneurysms", 
            "soft exudates"
        ]
        self.clinical_concepts_mapping = {x: {0: "absent", 1: "present"} for x in self.clinical_concepts}

        self.ALL_CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        self.ALL_CONCEPTS = ['EX', 'HE', 'MA', 'SE']
        self.valid_ids = self.filter_data(filter)

    def filter_data(self, keep_perc):
        def sample_by_label(group, frac):
            return group.sample(frac=frac)
        
        columns = ["ID"]
        columns.extend(self.ALL_CLASSES)
        df = self.data[columns].copy()
        df["label"] = df.apply(lambda row: row[self.ALL_CLASSES].values.tolist().index(1), axis=1)

        df = df.groupby("label").apply(sample_by_label, frac=keep_perc, include_groups=False)
        
        return list(df["ID"].values)
    
    def prepare_data_for_rices(self):
        new_df = self.data.copy(deep=True)
        new_df["labels"] = new_df.apply(lambda row: row[self.ALL_CLASSES].values.tolist().index(1), axis=1)
        new_df = new_df.rename(columns={"ID": "image_id", "EX": "hard exudates", "HE": "haemorrhages", "MA": "microaneurysms", "SE": "soft exudates"})
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
        img_id = sample["ID"]

        # Create image_path
        img_path = os.path.join(self.path_to_images, img_id + '.' + self.img_extension)

        # Get class label
        class_label = sample[self.ALL_CLASSES].values.tolist().index(1)
        
        # Get clinical_concepts
        clinical_concepts = np.array(sample[self.ALL_CONCEPTS].values.tolist())

        return {"img_id": img_id, "img_path": img_path, "class_label": class_label, "clinical_concepts": clinical_concepts}
    
    def get_options(self, mapping):
        options_keys = list(mapping.copy().keys())

        option2gt = {f"{chr(ord('A') + i)})": options_keys[i] for i in range(len(options_keys))}
        gt2option = {options_keys[i]: f"{chr(ord('A') + i)})" for i in range(len(options_keys))}

        options_prompt = "\n"
        for k, v in option2gt.items():
            options_prompt += f"{k} {mapping[v]}\n"

        return options_prompt, option2gt, gt2option
    
    def get_concept_prompt(self, concept, demos_concepts=None):
        if concept == "hard exudates":
            instruction = "Hard exudates typically appear as yellowish or whitish, well-defined spots or patches on the retina. "\
                "They can vary in size and shape and often form clusters or rings around areas of retinal edema (swelling).\n"
        elif concept == "soft exudates":
            instruction = "Soft exudates, also known as cotton-wool spots appear as pale, fluffy, or cloud-like spots on the retina. "\
                "Unlike hard exudates, they are not well-defined and have a more feathery edge.\n"
        elif concept == "haemorrhages":
            instruction = "Haemorrhages can appear as red or dark spots, streaks, or blotches on the retina.\n"
        elif concept == "microaneurysms":
            instruction = "Microaneurysms appear as tiny, red dots on the retina. They are typically round and uniform in shape. "\
                "They can be distinguished from other retinal features by their small size and bright red color.\n"
        else:
            raise ValueError
        
        options, option2gt, gt2option = self.get_options(self.clinical_concepts_mapping[concept])
        question = f"In the image, {concept} are:{options}Choose one option. Do not provide additional information.".format(options=options)
        
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
            question = "What is the type of diabetic retinopathy shown in this image?\nOptions:{options}Choose one option. Do not provide additional information. Answer:"
        else:
            question = "What is the type of diabetic retinopathy associated with the following concepts: {concepts}\nOptions:{options}Choose one option. Do not provide additional information. Answer:"
        
        options, option2gt, gt2option = self.get_options(self.class_mapping)

        instruction = ""
        if query_concepts is not None:
            instruction += "Consider the following useful concepts to diagnose diabetic retinopathy.\n" \
                "Hard exudates typically appear as yellowish or whitish, well-defined spots or patches on the retina. "\
                "They can vary in size and shape and often form clusters or rings around areas of retinal edema (swelling).\n"\
                "Soft exudates, also known as cotton-wool spots appear as pale, fluffy, or cloud-like spots on the retina. "\
                "Unlike hard exudates, they are not well-defined and have a more feathery edge.\n"\
                "Haemorrhages can appear as red or dark spots, streaks, or blotches on the retina.\n"\
                "Microaneurysms appear as tiny, red dots on the retina. They are typically round and uniform in shape. "\
                "They can be distinguished from other retinal features by their small size and bright red color.\n"
        
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
    dataset = DDRDataset(
        csv_file="data/splits/DDR_train.csv", 
        img_extension='jpg', 
        path_to_images='/home/icrto/Fundus/DDR/DDR-dataset/DR_grading/all_images',
        filter=0.25
    )

    print(len(dataset))
    print(len(dataset.valid_ids))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        print(batch)
        break
