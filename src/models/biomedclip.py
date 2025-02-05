from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

import torch

class BiomedCLIP:
    """
    Paper: https://arxiv.org/abs/2303.00915
    Model: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    Requirements: pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib (Better to use a separate conda env due to the transformers version)
    """

    def __init__(self) -> None:
        """
        Initialize the attributes of the class.
        """

        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device='cuda')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


    @torch.no_grad()
    def extract_image_features(self, img_batch):
        images = torch.stack([self.preprocess(img) for img in img_batch]).to(0)
        image_features = self.model.encode_image(images, normalize=True)
        return image_features.cpu().numpy()
    
    @torch.no_grad()
    def extract_text_features(self, text):
        inputs = self.tokenizer(text).cuda()
        text_features = self.model.encode_text(inputs, normalize=True)
        return text_features.cpu().numpy()
                                      