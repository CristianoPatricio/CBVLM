"""
+info: https://huggingface.co/OpenGVLab/InternVL2-8B
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN='<img>' 
IMG_END_TOKEN='</img>' 
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

class InternVL2:

    def __init__(self) -> None:
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True, use_flash_attn=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True)
        self.generation_config = {
            "do_sample": True,
            "eos_token_id": self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        }
        self.tokenizer.padding_side = "left"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.transform = self.build_transform(input_size=448)

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        <|im_start|>system\n
        You are a helpful assistant.<|im_end|>\n
        <|im_start|>user\n
        <image>\nWhat is in the image?<|im_end|>\n
        <|im_start|>assistant\n
        """

        DEFAULT_IMAGE_TOKEN = "<image>"
        
        if instruction != "":
            prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n"
        else:
            prompt = ""
        
        if demos_prompts is not None:
            for d in demos_prompts:
                x = d.split("Answer:")
                if d[-1] == ".":
                    prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}<|im_end|>\n<|im_start|>assistant\n{x[1].strip()}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()}<|im_end|>\n<|im_start|>assistant\n{x[1].strip()}.<|im_end|>\n"

        prompt += f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{query_prompt}<|im_end|>\n<|im_start|>assistant\n"

        return prompt

    def predict(self, query_images, prompts, max_new_tokens, demo_images=[]):
        # Adapted from modeling_internvl_chat.py
        # and from https://github.com/OpenGVLab/InternVL/blob/869d3be88d40d79162ca23b1ff5380d657883b55/internvl_chat/internvl/train/internvl_chat_pretrain.py#L518
        # https://github.com/OpenGVLab/InternVL/blob/869d3be88d40d79162ca23b1ff5380d657883b55/internvl_chat/internvl/train/dataset.py#L711
        assert query_images is not None
        assert len(query_images) > 0

        if len(demo_images) > 0:
            images = []
            for i in range(len(query_images)):
                images.extend(demo_images[i] + [query_images[i]])
        else:
            images = query_images
        
        num_image = len(images)
        proc_images, num_tiles = [], []
        for img in images:
            image = self.dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12)
            proc_images += image
            num_tiles.append(len(image))

        pixel_values = [self.transform(image) for image in proc_images]
        pixel_values = torch.stack(pixel_values)
        num_image_tokens = [self.model.num_image_token * num_tile for num_tile in num_tiles]

        new_prompts = []
        current_image_idx = 0
        for p in prompts:
            image_cnt = p.count("<image>")
            for _ in range(image_cnt):
                if current_image_idx == num_image: break
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_tokens[current_image_idx]}{IMG_END_TOKEN}'
                p = p.replace('<image>', image_tokens, 1)
                current_image_idx += 1
            new_prompts.append(p)
        
        model_inputs = self.tokenizer(
            new_prompts,
            return_tensors='pt',
            padding=True,
            max_length=self.tokenizer.model_max_length,
        )

        generation_output = self.model.generate(
            pixel_values=pixel_values.to(0, dtype=torch.bfloat16),
            input_ids=model_inputs["input_ids"].to(0),
            attention_mask=model_inputs["attention_mask"].to(0),
            max_new_tokens=max_new_tokens,
            **self.generation_config,
        )

        responses = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)

        return responses

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        list_pixel_values = []
        for img in img_batch:
            list_pixel_values.append(self.load_image(img).to(torch.bfloat16).to(0))

        pixel_values = torch.cat(list_pixel_values, dim=0)

        vision_features = self.model.extract_feature(pixel_values)
        
        return vision_features.mean(dim=1).cpu().float().numpy()