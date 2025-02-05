"""
Adapted from: https://github.com/JoshuaChou2018/SkinGPT-4/demo.py
"""

from PIL import Image
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from src.models.skingpt4.model import skingpt4
from src.models.skingpt4.processors import Blip2ImageEvalProcessor


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class SkinGPT4:

    def __init__(self, llm_path, weights):
        self.model = skingpt4(max_txt_len=160, llm_model=llm_path)
        self.model.to(0)

        ckpt = torch.load(weights, map_location="cuda")
        self.model.load_state_dict(ckpt['model'], strict=False)
        
        self.vis_processor = Blip2ImageEvalProcessor()

        stop_words_ids = [torch.tensor([835]).to(0),
                          torch.tensor([2277, 29937]).to(0)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def get_prompt(self, instruction, query_prompt, demos_prompts=None):
        """
        Example template: 

        Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.
        ###Human: <Img><ImageHere></Img> What's wrong with my skin?
        ###Assistant: The image shows a close up view of the person's ear, with the outer ear visible on the left side and the inner ear visible on the right side. The skin appears to be healthy and free of any infections or wounds. The ear canal is also visible, with a small amount of earwax visible inside.###
        """
        
        if instruction != "":
            prompt = f"{instruction}###Human: "
        else:
            prompt = "###Human: "
        
        if demos_prompts != None:
            for d in demos_prompts:
                x = d.split("Answer:")
                if d[-1] == ".":
                    prompt += f"<Img><ImageHere></Img>{x[0].strip()}###Assistant: {x[1].strip()}###"
                else:
                    prompt += f"<Img><ImageHere></Img>{x[0].strip()}###Assistant: {x[1].strip()}###"

        prompt += f"<Img><ImageHere></Img>{query_prompt}###Assistant: "

        return prompt
    
    def processor(self, images, prompt):
        img_list = []
        for im in images:
            image = self.vis_processor(im.convert("RGB")).unsqueeze(0).to(0)
            image_emb, _ = self.model.encode_img(image.half())
            img_list.append(image_emb)
        
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llm_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(0).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llm_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)

        return mixed_embs

    def predict(self, query_images, prompts, max_new_tokens, max_length=2000, demo_images=[]):
        assert query_images is not None
        assert len(query_images) > 0

        if len(demo_images) > 0:
            images = [demo_images[i] + [query_images[i]] for i in range(len(query_images))] 
            embs = torch.stack([self.processor(images[i], prompts[i]) for i in range(len(query_images))]).squeeze(dim=1)
        else:
            images = query_images 
            embs = torch.stack([self.processor([images[i]], prompts[i]) for i in range(len(query_images))]).squeeze(dim=1)

        # Model Answer
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llm_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            do_sample=False,
        )

        output_texts = []
        for output_token in outputs:
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llm_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_texts.append(output_text)

        return output_texts
    
    @torch.no_grad()
    def extract_image_features(self, img_batch):
        vision_x = [self.vis_processor(img.convert("RGB")).unsqueeze(0) for img in img_batch]
        vision_x = torch.cat(vision_x, dim=0).to(0)
        with self.model.maybe_autocast():
            output = self.model.ln_vision(self.model.visual_encoder(vision_x)) # [bs, 257, 1408]
        output_gap = torch.mean(output, dim=1)  # [bs, 1408]
        
        return output_gap.cpu().numpy()
