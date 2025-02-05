import pathlib
import tempfile
import logging
import os
import copy

import torch
from torch import nn

from timm.models.layers import trunc_normal_

from .ImageEncoder import build_image_encoder
from .LangEncoder import build_lang_encoder
from .LangEncoder import build_tokenizer

import mup.init
from mup import set_base_shapes

from safetensors.torch import load_file


logger = logging.getLogger(__name__)


class UniCLModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.conf_lang_encoder = config['LANG_ENCODER']
        self.tokenizer = build_tokenizer(self.conf_lang_encoder)

        self.lang_encoder = build_lang_encoder(self.conf_lang_encoder, self.tokenizer, config['VERBOSE'])

        dim_projection = config['UNICL_MODEL']['DIM_PROJECTION']
        if hasattr(self.lang_encoder, 'dim_out'):
            dim_out = self.lang_encoder.dim_out
        else:
            with torch.no_grad():
                dim_out = self.lang_encoder(
                    torch.zeros(1,1).type(torch.LongTensor)
                )['last_hidden_state'].size(2)

        self.lang_projection = nn.Parameter(torch.empty(dim_out, dim_projection))

        self.conf_image_encoder = config['IMAGE_ENCODER']
        self.image_encoder = build_image_encoder(self.conf_image_encoder, config['VERBOSE'])

        self.image_projection = nn.Parameter(
            torch.empty(self.image_encoder.dim_out, dim_projection)
        )

        self.logit_scale = nn.Parameter(torch.ones([]))

        if torch.cuda.is_available():
            self.device = torch.device(type="cuda", index=0)
        else:
            self.device = torch.device(type="cpu")

    def custom_init_weights(self, use_original_init=True):
        self.use_original_init = use_original_init
        logger.info('Custom init: {}'.format('original init' if self.use_original_init else 'muP init'))

        if self.use_original_init:
            # Original initialization.
            # Note: This is not SP init. We do not implement SP init here.
            custom_trunc_normal_ = trunc_normal_  # Note: This should be the same as torch.nn.init.trunc_normal_
        else:
            # muP.
            custom_trunc_normal_ = mup.init.trunc_normal_

        custom_trunc_normal_(self.lang_projection, std=.02)
        custom_trunc_normal_(self.image_projection, std=.02)

    def _convert_old_weights(self, model_dict):
        model_dict_updated = {}
        for k, v in model_dict.items():
            if k.startswith('visual.'):
                model_dict_updated['image_encoder.'+k[7:]] = v
            elif k.startswith('text.'):
                model_dict_updated['lang_encoder.'+k[5:]] = v
            elif k == 'vision_projection':
                model_dict_updated['image_projection'] = v
            elif k == 'text_projection':
                model_dict_updated['lang_projection'] = v
            else:
                model_dict_updated[k] = v

        return model_dict_updated

    def from_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
        if not os.path.isfile(pretrained):
            logger.warning(f'=> Pretrained model ({pretrained}) is not a file, skip init weight')
            return

        ## Load SafeTensors Version of Pretrained Model
        pretrained_dict = load_file(pretrained)
        logger.info(f'=> Loading pretrained model {pretrained}')
        model_dict = self.state_dict()
        pretrained_dict = self._convert_old_weights(pretrained_dict)
        ## To ensure cuda is mapped to all weights in the SafeTensors version model
        pretrained_dict = {
            k: v.to(self.device) for k, v in pretrained_dict.items()
        }
        need_init_state_dict = {}
        image_encoder_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                k.split('.')[0] in pretrained_layers
                or pretrained_layers[0] == '*'
            )

            if need_init:
                if k.startswith('image_encoder.'):
                    image_encoder_state_dict[k] = v.to(self.device)
                else:
                    if verbose:
                        logger.info(f'=> init {k} from {pretrained}')

                    if 'positional_embedding' in k and v.size() != model_dict[k].size():
                        positional_embedding_pretrained = v
                        positional_embedding_current = model_dict[k]
                        L1, nH1 = positional_embedding_pretrained.size()
                        L2, nH2 = positional_embedding_current.size()
                        if nH1 != nH2:
                            logger.info(f"Error in loading {k}, passing")
                        else:
                            if L1 != L2:
                                logger.info(
                                    '=> load_pretrained: resized variant: {} to {}'
                                        .format((L1, nH1), (L2, nH2))
                                )

                                posemb = positional_embedding_pretrained.float()
                                posemb_grid = posemb.unsqueeze(dim=0).permute(0, 2, 1)
                                posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=L2, mode='linear')
                                posemb_grid = posemb_grid.permute(0, 2, 1).squeeze(dim=0)
                                v = posemb_grid

                    need_init_state_dict[k] = v.to(self.device)
        self.image_encoder.from_state_dict(image_encoder_state_dict, ['*'], verbose)
        self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {'logit_scale'}
        if hasattr(self.lang_encoder, 'no_weight_decay'):
            for k in self.lang_encoder.no_weight_decay():
                no_weight_decay.add('lang_encoder.'+k)

        if hasattr(self.image_encoder, 'no_weight_decay'):
            for k in self.visual.no_weight_decay():
                no_weight_decay.add('image_encoder.'+k)

        return no_weight_decay

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def encode_image(self, image, norm=True):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=True):
        x = self.lang_encoder(**text)
        x = x['last_hidden_state']

        if self.conf_lang_encoder['TOKENIZER'] == 'clip':
            x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T


def create_model(config):
    model = UniCLModel(config)
    return model


def create_mup_model(config):
    def gen_config(config, wm):
        # TODO: Currently only support the case that all UniCL, lang encoder, and image encoder use
        #       mu parameterization. This requirement can be relaxed.
        assert (not config['UNICL_MODEL']['STANDPARAM']) and \
               (not config['LANG_ENCODER']['STANDPARAM']) and \
               (not config['IMAGE_ENCODER']['SPEC']['STANDPARAM'])
        new_config = copy.deepcopy(config)
        logger.info(f'Generate config with width mult = {wm}:')

        # Generate config for UniCL head.
        new_config_section = new_config['UNICL_MODEL']
        new_config_section['STANDPARAM'] = True  # Use standard parameterization when determining base shapes.
        for name in ['DIM_PROJECTION']:
            base_name = 'BASE_' + name
            new_values = round(new_config_section[base_name] * wm)  # New value = base value * width multiplier.
            logger.info(f'config["UNICL_MODEL"]["{name}"]: {new_config_section[name]} -> {new_values}')
            new_config_section[name] = new_values

        # Generate config for lang encoder.
        new_config_section = new_config['LANG_ENCODER']
        new_config_section['STANDPARAM'] = True
        for name in ['WIDTH', 'HEADS']:
            base_name = 'BASE_' + name
            new_values = round(new_config_section[base_name] * wm)  # New value = base value * width multiplier.
            logger.info(f'config["LANG_ENCODER"]["{name}"]: {new_config_section[name]} -> {new_values}')
            new_config_section[name] = new_values

        # Generate config for image encoder.
        new_config_section = new_config['IMAGE_ENCODER']['SPEC']
        new_config_section['STANDPARAM'] = True
        for name in ['DIM_EMBED', 'NUM_HEADS', 'NUM_GROUPS']:
            base_name = 'BASE_' + name
            new_values = [round(base_value * wm) for base_value in new_config_section[base_name]]  # New value = base value * width multiplier.
            logger.info(f'config["IMAGE_ENCODER"]["SPEC"]["{name}"]: {new_config_section[name]} -> {new_values}')
            new_config_section[name] = new_values

        return new_config

    logger.info('muP: Create models and set base shapes')
    logger.info('=> Create model')
    model = create_model(config)
    # Temporarily remove the lang and image encoders from model to prevent from
    # setting the base shape for these encoders again.
    lang_encoder, image_encoder = model.lang_encoder, model.image_encoder
    model.lang_encoder, model.image_encoder = None, None

    logger.info('=> Create base model')
    base_config = gen_config(config, wm=1.0)
    base_model = create_model(base_config)
    del base_model.lang_encoder, base_model.image_encoder

    logger.info('=> Create delta model')
    delta_config = gen_config(config, wm=2.0)
    delta_model = create_model(delta_config)
    del delta_model.lang_encoder, delta_model.image_encoder

    logger.info('=> Set base shapes in model for training')
    set_base_shapes(model, base=base_model, delta=delta_model)

    # Restore the lang and image encoders in the model.
    model.lang_encoder, model.image_encoder = lang_encoder, image_encoder

    return model


def build_unicl_model(config, **kwargs):
    standparam = config['UNICL_MODEL'].get('STANDPARAM', True)

    if standparam:
        logger.info('Create model with standard parameterization')
        model = create_model(config)

        use_original_init = True
    else:
        logger.info('Create model with mu parameterization')
        model = create_mup_model(config)
        use_original_init = False

    # Initialize other parameters.
    model.custom_init_weights(use_original_init=use_original_init)

    if config['UNICL_MODEL']['LOAD_PRETRAINED']:
        pretrained_path = config['UNICL_MODEL']['PRETRAINED']
        from .Distributed.Utils import is_valid_url, download_file
        if is_valid_url(pretrained_path):
            with tempfile.TemporaryDirectory() as tmp_path:
                file_local_path = pathlib.Path(tmp_path) / 'base_model.pt'
                download_file(pretrained_path, file_local_path)
                model.from_pretrained(str(file_local_path), config['UNICL_MODEL']['PRETRAINED_LAYERS'], config['VERBOSE'])
        else:
            model.from_pretrained(pretrained_path, config['UNICL_MODEL']['PRETRAINED_LAYERS'], config['VERBOSE'])

    return model
