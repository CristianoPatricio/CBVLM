import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class Idefics3:

    def __init__(self) -> None:
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
        self.model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", device_map="auto", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        
    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        '<|begin_of_text|>User:<image>What do we see in this image?<end_of_utterance>\n
        Assistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty.<end_of_utterance>\n
        User:<image>And how about this image?<end_of_utterance>\n
        Assistant:'
        """

        DEFAULT_IMAGE_TOKEN = "<image>"
        
        if instruction != "":
            prompt = f"<|begin_of_text|>User:\n{instruction}<end_of_utterance>\n"
        else:
            prompt = ""
        
        if demos_prompts is not None:
            for d in demos_prompts:
                x = d.split("Answer:")
                if d[-1] == ".":
                    prompt += f"<|begin_of_text|>User:{DEFAULT_IMAGE_TOKEN}{x[0].strip()}<end_of_utterance>\nAssistant:{x[1].strip()}<end_of_utterance>\n"
                else:
                    prompt += f"<|begin_of_text|>User:{DEFAULT_IMAGE_TOKEN}{x[0].strip()}<end_of_utterance>\nAssistant:{x[1].strip()}.<end_of_utterance>\n"

        prompt += f"<|begin_of_text|>User:{DEFAULT_IMAGE_TOKEN}{query_prompt}<end_of_utterance>\nAssistant:"

        return prompt

    def predict(self, query_images, prompts, max_new_tokens=None, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0

        if len(demo_images) > 0:
            images = [demo_images[i] + [query_images[i]] for i in range(len(query_images))]
        else:
            images = query_images
        
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        response = self.processor.batch_decode(output, skip_special_tokens=True)
        response = [x.split("Assistant: ")[-1] for x in response]

        return response    

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        encoding_image_processor = self.processor.image_processor(img_batch, return_tensors="pt").to(0, dtype=torch.bfloat16)
        pixel_values = encoding_image_processor["pixel_values"].squeeze(0)

        vision_outputs = self.model.model.vision_model(
            pixel_values=pixel_values,
        )

        img_features = vision_outputs.last_hidden_state

        return img_features.view(len(img_batch), -1, 1152).mean(dim=1).cpu().float().numpy()