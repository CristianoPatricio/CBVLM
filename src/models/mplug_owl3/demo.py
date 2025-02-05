"""
+info: https://huggingface.co/mPLUG/mPLUG-Owl3-7B-241101
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip uninstall flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
"""

import torch
from transformers import AutoModel, AutoTokenizer

class mPLUGOwl3:

    def __init__(self) -> None:
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained("mPLUG/mPLUG-Owl3-7B-241101", device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained("mPLUG/mPLUG-Owl3-7B-241101", trust_remote_code=True)
        self.processor = self.model.init_processor(self.tokenizer)
        self.generation_config = {
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
        }

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        <|im_start|>system\n
        You are a helpful assistant.<|im_end|>\n
        <|im_start|>user\n
        (<image>./</image>)What is in the image?<|im_end|>\n
        <|im_start|>assistant\n
        """

        DEFAULT_IMAGE_TOKEN = "<|image|>"
        
        prompt = []
        if instruction != "":
            prompt.append({"role": "user", "content": instruction})
        else:
            prompt = []
        
        if demos_prompts is not None:
            for d in demos_prompts:
                x = d.split("Answer:")
                s = "<|image|>" + x[0].strip()
                prompt.append({"role": "user", "content": s})
                gt = x[1].strip()
                prompt.append({"role": "assistant", "content": gt})

        s = "<|image|>" + query_prompt
        prompt.append({"role": "user", "content": s})
        prompt.append({"role": "assistant", "content": ""})

        return prompt

    def predict(self, query_images, prompts, max_new_tokens, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0
        
        if len(demo_images) > 0:
            images = [demo_images[i] + [query_images[i]] for i in range(len(query_images))]
            inputs = [self.processor(prompts[i], images=images[i], videos=None, return_tensors="pt").to(0, dtype=torch.float16) for i in range(len(images))]
        else:
            images = query_images
            inputs = [self.processor(prompts[i], images=[images[i]], videos=None, return_tensors="pt") for i in range(len(images))]
        
        new_inputs = {"media_offset": []}
        new_inputs["pixel_values"] = torch.cat([x["pixel_values"] for x in inputs], dim=0).to(0, dtype=torch.float16)
        new_inputs["input_ids"], _ = self.processor.pad([inp["input_ids"].squeeze(0) for inp in inputs], padding_value=self.tokenizer.pad_token_id, padding_side="left")
        new_inputs["input_ids"] = new_inputs["input_ids"].to(0)
        for x in inputs:
            for z in x["media_offset"]:
                new_inputs["media_offset"].append(z)

        response = self.model.generate(
            **new_inputs,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            decode_text=True,
            **self.generation_config
        )

        return response 

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        encoded_images = self.processor.image_processor(images=img_batch, return_tensors="pt")
        img_features = self.model.forward_image(encoded_images['pixel_values'].to(0))

        return img_features.view(len(img_batch), -1, 3584).mean(dim=1).cpu().float().numpy()
    