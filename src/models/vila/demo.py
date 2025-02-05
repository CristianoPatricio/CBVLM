"""
+info: https://huggingface.co/Efficient-Large-Model/Llama-3-VILA1.5-8B
https://github.com/NVlabs/VILA/tree/main?tab=readme-ov-file#inference
"""

import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from src.utils import utils


log = utils.get_logger(__name__) # init logger

class Vila:

    def __init__(self, version="8B"):
        assert version in ["8B", "40B"]
        disable_torch_init()
        if version == "8B":
            model_path = os.path.expanduser("Efficient-Large-Model/Llama-3-VILA1.5-8B")
        else:
            model_path = os.path.expanduser("Efficient-Large-Model/VILA1.5-40b")
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, model_name, device_map="auto")

        self.version = version

    def get_prompt(self, instruction, query_prompt, demos_prompts=[]):
        if self.version == "8B":
            return self.get_prompt8B(instruction, query_prompt, demos_prompts=demos_prompts)
        else:
            return self.get_prompt40B(instruction, query_prompt, demos_prompts=demos_prompts)

    def get_prompt8B(self, instruction, query_prompt, demos_prompts=[]):
        """
        Template 1:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n[instruction]<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{DEFAULT_IMAGE_TOKEN}\n[question]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        
        Template 2:
        
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n[instruction]<|eot_id|>
        repeat cfg.n_demonstrations:
            <|start_header_id|>user<|end_header_id|>\n\nDEFAULT_IMAGE_TOKEN}\n[question]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[gt_answer]}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>\n\n{DEFAULT_IMAGE_TOKEN}\n[question]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        """

        if instruction != "":
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
        else:
            prompt = "<|begin_of_text|>"

        for d in demos_prompts:
            x = d.split("Answer:")
            if d[-1] == ".":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{x[1].strip()}<|eot_id|>"
            else:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{x[1].strip()}.<|eot_id|>"

        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{DEFAULT_IMAGE_TOKEN}\n{query_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt
    
    def get_prompt40B(self, instruction, query_prompt, demos_prompts=[]):
        """
        Template 1:
        <|im_start|>system\n[instruction]<|im_end|><|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n[question]<|im_end|><|im_start|>assistant\n
        
        Template 2:
        
        <|im_start|>system\n[instruction]<|im_end|>
        repeat cfg.n_demonstrations:
            <|im_start|>user\nDEFAULT_IMAGE_TOKEN}\n[question]<|im_end|><|im_start|>assistant\n[gt_answer]}<|im_end|>
        <|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n[question]<|im_end|><|im_start|>assistant\n
        """

        if instruction != "":
            prompt = f"<|im_start|>system<\n{instruction}<|im_end|>"
        else:
            prompt = ""
        
        for d in demos_prompts:
            x = d.split("Answer:")
            if d[-1] == ".":
                prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}\nAnswer:<im_end><|im_start|>assistant\n{x[1].strip()}<|im_end|>"
            else:
                prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}\nAnswer:<im_end><|im_start|>assistant\n{x[1].strip()}.<|im_end|>"

        prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{query_prompt}<|im_end|><|im_start|>assistant\n"

        return prompt
    
    def predict(self, query_images, prompts, max_new_tokens, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0
        
        input_ids = torch.stack([tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for p in prompts]).cuda()
        if len(demo_images) > 0:
            images = [demo_images[i] + [query_images[i]] for i in range(len(query_images))]
            image_tensor = torch.stack([process_images(x, self.image_processor, self.model.config) for x in images], dim=0)
        else:
            image_tensor = process_images(query_images, self.image_processor, self.model.config)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.2
        )

        response = self.tokenizer.batch_decode(output_ids.to("cpu"), skip_special_tokens=True)

        return response

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        image_tensor = process_images(img_batch, self.image_processor, self.model.config)

        visual_features = self.model.vision_tower(
            image_tensor
        )
        return visual_features.mean(dim=1).cpu().numpy()