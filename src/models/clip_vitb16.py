from transformers import CLIPProcessor, CLIPModel
import torch

class CLIPViTB16:
    """
    Paper: https://arxiv.org/abs/2103.00020
    Model: https://huggingface.co/openai/clip-vit-base-patch16
    """

    def __init__(self) -> None:
        """
        Initialize the attributes of the class.
        """
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(0)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    @torch.no_grad()
    def extract_image_features(self, img_batch):
        images = self.processor.image_processor(img_batch, return_tensors="pt")["pixel_values"].to(0)
        image_features = self.model.get_image_features(images)
        return image_features.cpu().numpy()