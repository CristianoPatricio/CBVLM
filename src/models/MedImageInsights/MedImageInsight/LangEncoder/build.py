import os
import logging

from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .registry import lang_encoders
from .registry import is_lang_encoder

logger = logging.getLogger(__name__)


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if model_name.endswith('pretrain'):
        model_name = 'pretrain'

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unknown model: {model_name}')

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)


def post_process_clip(text):
    text['input_ids'].squeeze_() # torch.Size([1, 77])
    text['attention_mask'].squeeze_() # torch.Size([1, 77])
    return text


def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' # 'true', avoid hanging

    if config_encoder['TOKENIZER'] == 'clip':
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        # print(pretrained_tokenizer)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
        tokenizer.post_process = post_process_clip
    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True)
        tokenizer.post_process = post_process_clip
    elif config_encoder['TOKENIZER'] == 'zcodepp':
        from .zcodepp import ZCodeppTokenizer
        tokenizer = ZCodeppTokenizer(config_encoder)
        tokenizer.post_process = lambda x: x
    elif config_encoder['TOKENIZER'] == 'zcode':
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained(config_encoder['PRETRAINED_TOKENIZER'])
    elif config_encoder['TOKENIZER'] == 'tulrv6':
        from .modeling_tulrv6 import TULRv6Tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'tulrv6-base'
        )
        tokenizer = TULRv6Tokenizer.from_pretrained(pretrained_tokenizer)
        # tokenizer.post_process = post_process_clip
    else:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        pretrained_tokenizer = config_encoder.get('PRETRAINED_TOKENIZER', '')
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_tokenizer
            if pretrained_tokenizer else config_encoder['TOKENIZER']
        )
        tokenizer.post_process = post_process_clip

        # Extra configurations.
        if 'TOKENIZER_CONF' in config_encoder:
            tokenizer_conf = config_encoder['TOKENIZER_CONF']

            num_pretrained_tokens = len(tokenizer)

            addition_special_tokens_config = tokenizer_conf.get('ADDITIONAL_SPECIAL_TOKENS', None)
            if addition_special_tokens_config == 'od+cap':
                # Note: We still keep the additional special tokens from original tokenizer when we add new special tokens.
                #       This is to make sure tokenizer.additional_special_tokens afterwards includes original additional special tokens.
                special_tokens_dict = {
                    'additional_special_tokens': \
                        tokenizer.additional_special_tokens + \
                        ['<od>','</od>','<cap>','</cap>'] + \
                        [f'<loc_{x}>' for x in range(tokenizer_conf.get('NUM_LOCATION_TOKENS', 0))]
                }
                tokenizer.add_special_tokens(special_tokens_dict)
            elif isinstance(addition_special_tokens_config, list):
                special_tokens_dict = {
                    'additional_special_tokens': \
                        tokenizer.additional_special_tokens + \
                        addition_special_tokens_config + \
                        [f'<loc_{x}>' for x in range(tokenizer_conf.get('NUM_LOCATION_TOKENS', 0))]+
                    [f'<time_{x}>' for x in range(
                        tokenizer_conf.get('NUM_TIME_TOKENS', 0))]
                }
                tokenizer.add_special_tokens(special_tokens_dict)
            elif addition_special_tokens_config is not None:
                raise ValueError('ADDITIONAL_SPECIAL_TOKENS type error')

            num_current_tokens = len(tokenizer)
            logger.info(f'{num_pretrained_tokens} tokens in pretrained tokenizer => {num_current_tokens} in current tokenizer')
            logger.info(f'All special tokens in tokenizer: {tokenizer.additional_special_tokens}')

    return tokenizer
