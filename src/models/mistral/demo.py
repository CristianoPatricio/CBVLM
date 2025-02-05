"""
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
"""
from transformers import AutoModelForCausalLM, LlamaTokenizer
import random
import ast
import torch

class Mistral:

    def __init__(self, max_memory=None):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto", max_memory=max_memory, torch_dtype=torch.float16)
        self.tokenizer = LlamaTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_fast=False)

    def get_prompt_yes_no(self, model_response, original_question):
        prompt = "You are tasked with evaluating responses to questions. Consider the following question and response given in between <<< >>>.\n\
<<<Question: {original_question} Response: {model_response}>>>\n\
Does the provided response answer the question affirmatively?\
Take your time and answer in json format by saying only yes or no, following the template: {{'Answer': 'yes or no'}}. \
If the response does not provide enough information to choose between yes or no, provide the following answer: {{'Answer': 'UNK'}}.\n\
Do not provide additional responses, context or explanations.\n\
Answer: "

        prompt = prompt.format(original_question=original_question, model_response=model_response)
        return prompt

    def get_prompt_multiple_choice(self, model_response, options):
        prompt = "Sentence: <<<{sentence}>>>\n\
Consider the sentence given in between <<< >>> and the following options:{options}\
Choose the option that best fits the information conveyed by the sentence. \
Take your time and answer in json format by providing only the letter corresponding to the chosen option, following the template: {{'Answer': 'option letter'}}. \
If the sentence does not provide enough information to choose an option, provide the following answer: {{'Answer': 'UNK'}}.\n\
Do not provide additional responses, context or explanations.\n\
Answer: "

        options_keys = list(options.keys())
        #random.shuffle(options_keys)

        option2gt = {chr(ord("A") + i): options_keys[i] for i in range(len(options_keys))}

        options_prompt = "\n"
        for opt, k in option2gt.items():
            options_prompt += f"{opt}. {options[k]}\n"

        prompt = prompt.format(options=options_prompt, sentence=model_response)
        return prompt, option2gt

    def predict(self, model_response, original_question, dataset_options, answer_type, max_new_tokens):

        option2gt = None

        # Depending on the task, the LLM is instructed to provide different responses
        if answer_type == "yes_no":
            prompt = self.get_prompt_yes_no(model_response, original_question)
        else:
            prompt, option2gt = self.get_prompt_multiple_choice(model_response, dataset_options)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(0)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        mistral_response = decoded[len(prompt.replace("</s>", ""))-1:].strip()
        
        try:
            final_response = ast.literal_eval(mistral_response)
            final_response["Answer"] = final_response["Answer"].upper()
        except:
            final_response = {"Answer": "UNK", "other": mistral_response}

        return final_response, option2gt