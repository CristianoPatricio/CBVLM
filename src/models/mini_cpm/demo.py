"""
+info: https://huggingface.co/openbmb/MiniCPM-V-2_6
Pillow==10.1.0
torch==2.1.2
torchvision==0.16.2
transformers==4.40.0
sentencepiece==0.1.99
decord
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor

class miniCPM:

    def __init__(self) -> None:
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16).to(0)
        self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
        self.generation_config = {
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.05
        }

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        <|im_start|>system\n
        You are a helpful assistant.<|im_end|>\n
        <|im_start|>user\n
        (<image>./</image>)What is in the image?<|im_end|>\n
        <|im_start|>assistant\n
        """

        DEFAULT_IMAGE_TOKEN = "(<image>./</image>)"
        
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

    def predict(self, query_images, prompts, max_new_tokens, demo_images=[]):
        # Adapted from modeling_minicpmv.py
        assert query_images is not None
        assert len(query_images) > 0

        if len(demo_images) > 0:
            images = [demo_images[i] + [query_images[i]] for i in range(len(query_images))]
        else:
            images = [[x] for x in query_images]
        
        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(0)
        inputs.pop("image_sizes")

        response = self.model.generate(
            **inputs, 
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            decode_text=True,
            **self.generation_config
        )
        return response   

    def get_vision_embeddings(self, data):
        # adapted from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/modeling_minicpmv.py - get_vllm_embedding
        dtype = data['pixel_values'][0][0].dtype
        device = data['pixel_values'][0][0].device
        tgt_sizes = data['tgt_sizes']
        pixel_values_list = data['pixel_values']
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
        tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                            padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
        for i in range(B):
            patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        vision_batch_size = self.model.config.vision_batch_size
        all_pixel_values = all_pixel_values.type(dtype)
        if B > vision_batch_size:
            hs = []
            for i in range(0, B, vision_batch_size):
                start_idx = i
                end_idx = i + vision_batch_size
                tmp_hs = self.model.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                hs.append(tmp_hs)
            vision_embedding = torch.cat(hs, dim=0)
        else:
            vision_embedding = self.model.vpm(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes).last_hidden_state
        vision_embedding = self.model.resampler(vision_embedding, tgt_sizes)

        start = 0
        for pixel_values in pixel_values_list:
            img_cnt = len(pixel_values)
            if img_cnt > 0:
                vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                start += img_cnt
            else:
                vision_hidden_states.append([])

        return vision_hidden_states

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        inputs = self.processor.image_processor(img_batch, return_tensors="pt").to(dtype=torch.bfloat16, device="cuda")
        vision_features = self.get_vision_embeddings(inputs)
        vision_outputs = vision_features[0].view(len(img_batch), -1, 3584).mean(dim=[1])
        return vision_outputs.cpu().float().numpy()

    