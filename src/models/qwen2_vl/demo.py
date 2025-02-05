"""
+info: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
"""

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

class Qwen2VL:

    def __init__(self) -> None:
    
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):   

        """
            <|im_start|>system\n
            You are a helpful assistant.<|im_end|>\n
            <|im_start|>user\n
            <|vision_start|><|image_pad|><|vision_end|>\nWhat is shown in this image?<|im_end|>\n
            <|im_start|>assistant\n
        """
        DEFAULT_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        
        if instruction != "":
            prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n"
        else:
            prompt = ""
        
        if demos_prompts is not None:
            for d in demos_prompts:
                x = d.split("Answer:")
                if d[-1] == ".":
                    prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}{x[0].strip()}<|im_end|>\n<|im_start|>assistant\n{x[1].strip()}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}{x[0].strip()}<|im_end|>\n<|im_start|>assistant\n{x[1].strip()}.<|im_end|>\n"

        prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}{query_prompt}<|im_end|>\n<|im_start|>assistant\n"

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

        inputs = self.processor(
            text=prompts,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(0)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return response

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        inputs = self.processor.image_processor(img_batch, return_tensors="pt").to("cuda")
        vision_outputs = self.model.visual(inputs['pixel_values'], inputs['image_grid_thw'])
        return vision_outputs.view(len(img_batch), -1, 3584).mean(dim=1).cpu().float().numpy()