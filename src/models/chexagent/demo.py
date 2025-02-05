from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
from PIL import Image
import numpy as np
from collections import defaultdict

from src.utils import utils

log = utils.get_logger(__name__) # init logger

class CheXagent:
    
    def __init__(self, max_memory=None):

        self.processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
        self.model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True, device_map="auto", max_memory=max_memory, torch_dtype=torch.float16, low_cpu_mem_usage=True)    

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        Template 1:
        USER: <s>[instruction][question] ASSISTANT: <s>
        
        Template 2:
        
        USER: <s>[instruction]
        repeat cfg.n_demonstrations:
            USER: <s>[question] ASSISTANT: <s>[gt_answer]
        USER: <s>[question] ASSISTANT: <s>
        """

        if instruction != "":
            prompt = f"USER: <s>{instruction}"
        else:
            prompt = ""
        
        for d in demos_prompts:
            x = d.split("Answer:")
            if d[-1] == ".":
                prompt += f"USER: <s>{x[0].strip()}\nAnswer: ASSISTANT: <s>{x[1].strip()}\n"
            else:
                prompt += f"USER: <s>{x[0].strip()}\nAnswer: ASSISTANT: <s>{x[1].strip()}.\n"

        prompt += f"USER: <s>{query_prompt} ASSISTANT: <s>"

        return prompt

    def predict(self, query_images, prompts, max_new_tokens, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0
        
        if len(demo_images) > 0:
            images = [demo_images[i] + [query_images[i]] for i in range(len(query_images))]
        else:
            images = query_images
        
        proc_inputs = [self.processor(images=images[i], text=prompts[i], return_tensors="pt").to(0, dtype=torch.float16) for i in range(len(query_images))]
        inputs = {}
        # pad to max_length
        max_length = max(item["input_ids"].size(1) for item in proc_inputs)
        for k in proc_inputs[0].keys():
            if k == "input_ids" or k == "attention_mask":
                inputs[k] = torch.stack([torch.nn.functional.pad(item[k], (max_length - item[k].size(1), 0), mode='constant', value=self.processor.tokenizer.unk_token_id) for item in proc_inputs]).squeeze(dim=1)
            else:
                inputs[k] = torch.stack([x[k] for x in proc_inputs], dim=0).squeeze(dim=1)
                
        output = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True
        )
        
        response = self.processor.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # decoded_prompt = self.processor.tokenizer.batch_decode(inputs['input_ids'][0])
        # attentions = output.attentions

        return response #, attentions, decoded_prompt

    def get_concept_attention_weights(self, attentions, decoded_prompt, plot=False):
        """Calculates the averaged attention per concept and returns the sorted list of concepts with higher attention in descending order.

        Args:
            attentions (tuple): Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor of shape (batch_size, num_heads, generated_length, sequence_length).
            decoded_prompt (list): List of decoded tokens of the prompt.
            plot (bool, optional): If true generates the plot. Defaults to False.

        Returns:
            list: A sorted list in descending order of the concepts with higher attention.
        """
        # Select attention for the first generated token from the last layer averaged over heads
        attention = attentions[0][-1][0].mean(dim=0)[-1].detach().cpu().numpy()

        ######################################
        # Calculate the mean per concept
        ######################################
        # 1. Find the index of the word 'shows' (The image shows...) from the inverse decoded input prompt
        decoded_prompt = [d.strip() for d in decoded_prompt]
        index_shows = len(decoded_prompt) - decoded_prompt[::-1].index("shows")

        # 2. Iterate over the decoded_response starting from the index_shows until the end and append the concepts until encouter a point '.'
        concepts = dict()
        for index, substr in enumerate(decoded_prompt[index_shows:]):
            if '.' in substr:
                break
            
            concepts[index + index_shows] = substr

        encountered_concepts = "".join(concepts.values()).split(",")

        # 3. Get index values corresponding to the tokens of each concepts and save it into a dict where keys are the name of the concept and the values are the indexes
        i = 0
        indexes_concepts = defaultdict(list)
        for idx, (k, v) in enumerate(concepts.items()):
            if v != ',':
                if (v == 'and' and list(concepts.values())[idx-1] == ',') or v == '-':  # to avoid putting the last 'and' token in the list or the '-' of e.g. blue-whitish veils 
                    continue
                else:
                    indexes_concepts[encountered_concepts[i]].append(k)
            else:
                i += 1
        
        # 4. Get attention values averaged per concept and return a list sorted descendently
        averaged_weights = []
        for k, v in indexes_concepts.items():
            averaged_weights.append(attention[v[0]:v[-1]].mean())

        # 5. Sort the list in descending order
        sorted_indices = np.argsort(averaged_weights)[::-1]

        # Generate a heatmap (Optional)
        if plot:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(20, 20))
            ax = sns.heatmap(np.expand_dims(attention, axis=0),
                            xticklabels=decoded_prompt,
                            cmap="YlGnBu")
            plt.savefig(f"attention_heatmap.png")

        return sorted_indices
    
    @torch.no_grad()
    def extract_image_features(self, img_batch):
        encoding_image_processor = self.processor.image_processor(img_batch, return_tensors="pt").to(0, dtype=torch.float16)
        pixel_values = encoding_image_processor["pixel_values"]
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
        )
        
        return vision_outputs["pooler_output"].cpu().numpy()
        
        