"""
https://huggingface.co/Qwen/Qwen2-7B-Instruct
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"

class Qwen2:

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    def predict(self):

        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        breakpoint()

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
if __name__ == "__main__":
    model = Qwen2()
    response = model.predict()
    breakpoint()
