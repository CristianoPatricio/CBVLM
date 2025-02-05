from omegaconf import OmegaConf
import random

def join_words(words):
    if not words:
        return ""
    elif len(words) == 1:
        return words[0]
    else:
        return ", ".join(words[:-1]) + ", and " + words[-1]
    
""" def get_generic_prompt(answer_type, question, class_mapping: dict=None, demos_labels: list=None, demos_concepts: list=None, rationale: str=None, dataset_concepts: list=None):
    # Format question with options when answer type is closed_set
    option2gt = None
    if answer_type == "closed_set":
        assert class_mapping is not None
        options_keys = list(class_mapping.keys())
        random.shuffle(options_keys)

        option2gt = {f"{chr(ord('A') + i)})": options_keys[i] for i in range(len(options_keys))}
        gt2option = {options_keys[i]: f"{chr(ord('A') + i)})" for i in range(len(options_keys))}

        options_prompt = "\n"
        for k, v in option2gt.items():
            options_prompt += f"{k} {class_mapping[v]}\n"
        question = question.format(options=options_prompt)
    
    if dataset_concepts is not None:
        random.shuffle(dataset_concepts.copy())

        option2gt = {f"{chr(ord('A') + i)})": dataset_concepts[i] for i in range(len(dataset_concepts))}

        concepts_prompt = "\n"
        for k, v in option2gt.items():
            concepts_prompt += f"{k} {v}\n"
        question = question.format(concepts=concepts_prompt)
    
    
    # In an ICL scenario, we need to build the prompt for each demonstration
    demo_prompts = []
    if demos_labels is not None:
        # TODO concepts in the question, not on the rationale
        for i, label in enumerate(demos_labels):
            if answer_type == "yes_no":
                if label == 1:
                    text_label = "yes"
                else:
                    text_label = "no"
            elif answer_type == "closed_set":
                text_label = f"{gt2option[label]} {class_mapping[label]}"
            else:
                text_label = class_mapping[label]
            
            if demos_concepts is not None:
                assert rationale is not None
                demo_prompt = question + " " + rationale.format(concepts=join_words(demos_concepts[i]), answer=text_label)
            else:
                demo_prompt = question + " " + text_label
           
            demo_prompts.append(demo_prompt)
    
    return question, demo_prompts, option2gt """

def get_generic_prompt(answer_type, question, class_mapping: dict=None, demos_labels: list=None, demos_concepts: list=None, rationale: str=None, dataset_concepts: list=None):
    question = "Answer the following questions about the image.\n" \
        "Question 1: Blue-white veil is confluent blue pigmentation with an overlying white “ground-glass” haze. Does this image present blue-whitish veil?\n"\
        "Question 2: The pigment network consists of intersecting brown lines forming a grid-like reticular pattern. If the pigment network exists and is typical it is characterized by uniform lines (in width and color). If it exists and is atypical it is irregularly meshed with lines varying in size, color, thickness, or distribution." \
        "Is there a pigment network present in the image and, if so, is it typical or atypical?\n"
    
    question = "Blue-whitish veil is confluent blue pigmentation with an overlying white “ground-glass” haze. Does this dermoscopic image present blue-whitish veil? Answer:"
    
    return question, [], None
