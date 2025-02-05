from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
import os
from einops import repeat

from src.utils import utils

log = utils.get_logger(__name__) # init logger

class MedFlamingo:
    def __init__(self, llama_path):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log.info('Loading model ...')

        if not os.path.exists(llama_path):
            raise ValueError('Llama model not yet set up, please check README for instructions!')

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            cross_attn_every_n_layers=4
        )
        del self.model.lang_encoder.old_decoder_blocks
        del self.model.lang_encoder.gated_cross_attn_layers
        self.model.to(0)

        # load med-flamingo checkpoint:
        checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
        log.info(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        self.model.to(torch.float16)

        log.info("Model successfully loaded")
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left

    def get_length_of_prompt(self, original_prompt: str) -> str:
        return len(original_prompt.replace("<image>", " ").replace("<|endofchunk|>", " "))

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        Template 1:

        [instruction] <image>[question]

        Template 2:

        [instruction] 
        repeat n_demons:
            <image>[question]<|endofchunk|>
        <image>[question]
        """
           
        prompt = instruction
        
        for d in demos_prompts:
            if d[-1] == ".":
                prompt += f"<image>{d}<|endofchunk|>\n"
            else:
                prompt += f"<image>{d}.<|endofchunk|>\n"

        prompt += f"<image>{query_prompt}"

        return prompt
    
    def predict(self, query_images, prompts, max_new_tokens, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0

        if len(demo_images) > 0:
            images = [demo_images[i] + [query_images[i]] for i in range(len(query_images))]   
            n_imgs_per_sample = len(images[0])
            images = [item for sublist in images for item in sublist]
        else:
            images = query_images
            n_imgs_per_sample = 1

        pixels = [self.image_processor(im).unsqueeze(0) for im in images]
        pixels = torch.cat(pixels, dim=0)
        pixels = pixels.view(len(query_images), n_imgs_per_sample, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])
        pixels = repeat(pixels, 'b n c h w -> b n T c h w', T=1)
        
        tokenized_data = self.tokenizer(
            prompts,
            return_tensors="pt",
        )

        generated_text = self.model.generate(
            vision_x=pixels.to(0, dtype=torch.float16),
            lang_x=tokenized_data["input_ids"].to(0),
            attention_mask=tokenized_data["attention_mask"].to(0),
            max_new_tokens=max_new_tokens,
            top_k=None,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        response = self.tokenizer.batch_decode(generated_text.to("cpu"), skip_special_tokens=True)
        response = [response[i][self.get_length_of_prompt(prompts[i]):] for i in range(len(response))]

        return response
    
    @torch.no_grad()
    def extract_image_features(self, img_batch):
        vision_x = [self.image_processor(img).unsqueeze(0) for img in img_batch]
        vision_x = torch.cat(vision_x, dim=0).to(0, dtype=torch.float16)
        vision_x = self.model.vision_encoder(vision_x)[0]
        return vision_x.cpu().numpy()
