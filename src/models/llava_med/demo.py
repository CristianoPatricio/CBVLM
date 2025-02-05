import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

from src.utils import utils

log = utils.get_logger(__name__) # init logger

class LlavaMed:

    def __init__(self):

        disable_torch_init()
        model_path = os.path.expanduser("microsoft/llava-med-v1.5-mistral-7b")
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, None, model_name, device_map="auto")
        self.tokenizer.padding_side = "left"

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        Template 1:

        [INST] [instruction] [/INST] DEFAULT_IMAGE_TOKEN\n[question] </s>

        Template 2:

        [INST] [instruction] [/INST]
        repeat n_demons:
            DEFAULT_IMAGE_TOKEN\n[question]</s> [INST] [gt_answer] [/INST]
        DEFAULT_IMAGE_TOKEN [question] </s>
        """

        if instruction != "":   
            prompt = "[INST] " + instruction
        else:
            prompt = "[INST] "
        
        for d in demos_prompts:
            x = d.split("Answer:")
            if d[-1] == ".":
                prompt += f"{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()} [/INST] {x[1].strip()} </s>[INST] "
            else:
                prompt += f"{DEFAULT_IMAGE_TOKEN}\n{x[0].strip()} </s> [INST] {x[1].strip()}. [/INST] "

        prompt += f"{DEFAULT_IMAGE_TOKEN}\n{query_prompt} [/INST]"

        return prompt
    
    def predict(self, query_images, prompts, max_new_tokens, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0

        input_ids = [tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for p in prompts]
        # pad to max_length
        max_length = max(tensor.size(0) for tensor in input_ids)
        input_ids = torch.stack([torch.nn.functional.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=self.tokenizer.pad_token_id) for tensor in input_ids]).cuda()
        
        if len(demo_images) > 0:
            images = []
            for i in range(len(query_images)):
                images.extend(demo_images[i] + [query_images[i]])
            image_tensor = process_images(images, self.image_processor, self.model.config)
        else:
            image_tensor = process_images(query_images, self.image_processor, self.model.config)
        
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            output_attentions=True,
            temperature=0.2,
            do_sample=True
        )

        response = self.tokenizer.batch_decode(output_ids.to("cpu"), skip_special_tokens=True)

        return response

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        image_tensor = process_images(img_batch, self.image_processor, self.model.config).half().cuda()
        output = self.model.model.vision_tower(image_tensor) # [bs, 576, 1024]
        output_gap = torch.mean(output, dim=1) # [16, 1024]
        return output_gap.cpu().numpy()