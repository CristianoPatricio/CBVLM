from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from einops import repeat

from src.utils import utils

log = utils.get_logger(__name__) # init logger

class OpenFlamingo:

    def __init__(self):

        log.info('Loading model ...')

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            cross_attn_every_n_layers=2,
        )
        del self.model.lang_encoder.old_decoder_blocks
        del self.model.lang_encoder.gated_cross_attn_layers

        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)

        log.info("Model successfully loaded")
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        self.model.to(0, dtype=torch.float16)
        
    def get_length_of_prompt(self, original_prompt: str) -> str:
        return len(original_prompt.replace("<image>", "").replace("<|endofchunk|>", ""))

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
        
        """
        Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
        batch_size x num_media x num_frames x channels x height x width. 
        In this case batch_size = 1, num_media = 3, num_frames = 1,
        channels = 3, height = 224, width = 224.
        """
        vision_x = [self.image_processor(img).unsqueeze(0) for img in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.view(len(query_images), n_imgs_per_sample, vision_x.shape[-3], vision_x.shape[-2], vision_x.shape[-1])
        vision_x = repeat(vision_x, 'b n c h w -> b n T c h w', T=1)

        """
        Details: In the text we expect an <image> special token to indicate where an image is.
        We also expect an <|endofchunk|> special token to indicate the end of the text 
        portion associated with an image.
        """
        lang_x = self.tokenizer(
            prompts,
            return_tensors="pt",
        )

        generated_text = self.model.generate(
            vision_x=vision_x.to(0, dtype=torch.float16),
            lang_x=lang_x["input_ids"].to(0),
            attention_mask=lang_x["attention_mask"].to(0),
            max_new_tokens=max_new_tokens,
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