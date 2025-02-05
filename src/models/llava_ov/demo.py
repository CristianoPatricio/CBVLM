from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

import torch
import warnings

class LlavaOV:

    def __init__(self):   
        warnings.filterwarnings("ignore")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2", low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
        self.processor.tokenizer.padding_side = "left"

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):   

        """
            <|im_start|>system\n
            You are a helpful assistant.<|im_end|>\n
            <|im_start|>user\n
            <image>\nWhat is shown in this image?<|im_end|>\n
            <|im_start|>assistant\n
        """
        
        if instruction != "":
            prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n"
        else:
            prompt = ""
        
        for d in demos_prompts:
            x = d.split("Answer:")
            if d[-1] == ".":
                prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}<|im_end|>\n<|im_start|>assistant\n{x[1].strip()}<|im_end|>\n"
            else:
                prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}<|im_end|>\n<|im_start|>assistant\n{x[1].strip()}.<|im_end|>\n"

        prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{query_prompt}<|im_end|>\n<|im_start|>assistant\n"

        return prompt

    def predict(self, query_images, prompts, max_new_tokens=None, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0

        if len(demo_images) > 0:
            images = []
            for i in range(len(query_images)):
                images.extend(demo_images[i] + [query_images[i]])
        else:
            images = query_images

        inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.model.device, torch.float16)

        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=max_new_tokens
        )

        response = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response = [x.split("assistant\n")[-1] for x in response]

        return response

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        image_tensor = process_images(img_batch, self.image_processor, self.model.config).to(dtype=torch.float16, device=self.device) # (bs, 2, c, h, w)
        bs, grid = image_tensor.shape[:2]
        image_tensor = image_tensor.view(bs * grid, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1])

        visual_features = self.model.model.vision_tower(
            image_tensor
        )
        visual_features = visual_features.view(bs, grid, visual_features.shape[-2], -1).mean(dim=1) # (bs, 729, 1152)
        
        return visual_features.mean(dim=1).cpu().numpy()