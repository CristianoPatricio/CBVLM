"""Medical Image Classification model wrapper class that loads the model, preprocesses inputs and performs inference."""

import torch
from PIL import Image
import os
import tempfile
import base64
import io
from typing import List

from src.models.MedImageInsights.MedImageInsight.UniCLModel import build_unicl_model
from src.models.MedImageInsights.MedImageInsight.Utils.Arguments import load_opt_from_config_files
from src.models.MedImageInsights.MedImageInsight.ImageDataLoader import build_transforms
from src.models.MedImageInsights.MedImageInsight.LangEncoder import build_tokenizer


class MedImageInsight:
    """Wrapper class for medical image classification model."""

    def __init__(
            self,
            model_dir: str,
            vision_model_name: str,
            language_model_name: str
    ) -> None:
        """Initialize the medical image classifier.

        Args:
            model_dir: Directory containing model files and config
            vision_model_name: Name of the vision model
            language_model_name: Name of the language model
        """
        self.model_dir = model_dir
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.model = None
        self.device = None
        self.tokenize = None
        self.preprocess = None
        self.opt = None

    def load_model(self) -> None:
        """Load the model and necessary components."""
        try:
            # Load configuration
            config_path = os.path.join(self.model_dir, 'config.yaml')
            self.opt = load_opt_from_config_files([config_path])

            # Set paths
            self.opt['LANG_ENCODER']['PRETRAINED_TOKENIZER'] = os.path.join(
                self.model_dir,
                'language_model',
                'clip_tokenizer_4.16.2'
            )
            self.opt['UNICL_MODEL']['PRETRAINED'] = os.path.join(
                self.model_dir,
                'vision_model',
                self.vision_model_name
            )

            # Initialize components
            self.preprocess = build_transforms(self.opt, False)
            self.model = build_unicl_model(self.opt)

            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Load tokenizer
            self.tokenize = build_tokenizer(self.opt['LANG_ENCODER'])
            self.max_length = self.opt['LANG_ENCODER']['CONTEXT_LENGTH']

            print(f"Model loaded successfully on device: {self.device}")

        except Exception as e:
            print("Failed to load the model:")
            raise e

    @staticmethod
    def decode_base64_image(base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image and ensure RGB format.

        Args:
            base64_str: Base64 encoded image string

        Returns:
            PIL Image object in RGB format
        """
        try:
            # Remove header if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]

            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert grayscale (L) or grayscale with alpha (LA) to RGB
            if image.mode in ('L', 'LA'):
                image = image.convert('RGB')

            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

    def predict(self, images: List[str], labels: List[str], multilabel: bool = False) -> List[dict]:
        """Perform zero shot classification on the input images.

        Args:
            images: List of base64 encoded image strings
            labels: List of candidate labels for classification

        Returns:
            DataFrame with columns ["probabilities", "labels"]
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not labels:
            raise ValueError("No labels provided")

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Process images
            image_list = []
            for img_base64 in images:
                try:
                    img = self.decode_base64_image(img_base64)
                    image_list.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to process image: {str(e)}")

            # Run inference
            probs = self.run_inference_batch(image_list, labels, multilabel)
            probs_np = probs.cpu().numpy()
            results = []
            for prob_row in probs_np:
                # Create label-prob pairs and sort by probability
                label_probs = [(label, float(prob)) for label, prob in zip(labels, prob_row)]
                label_probs.sort(key=lambda x: x[1], reverse=True)

                # Create ordered dictionary from sorted pairs
                results.append({
                    label: prob
                    for label, prob in label_probs
                })

            return results

    def encode(self, images: List[str] = None, texts: List[str] = None):

        output = {
            "image_embeddings"  : None,
            "text_embeddings" : None,
        }

        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not images and not texts:
            raise  ValueError("You must provide either images or texts")

        if images is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Process images
                image_list = []
                for img_base64 in images:
                    try:
                        img = self.decode_base64_image(img_base64)
                        image_list.append(img)
                    except Exception as e:
                        raise ValueError(f"Failed to process image: {str(e)}")
            images = torch.stack([self.preprocess(img) for img in image_list]).to(self.device)
            with torch.no_grad():
                output["image_embeddings"] = self.model.encode_image(images).cpu().numpy()

        if texts is not None:
            text_tokens = self.tokenize(
                texts,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )

            # Move text tensors to the correct device
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
            output["text_embeddings"] = self.model.encode_text(text_tokens).cpu().numpy()


        return output

    def run_inference_batch(
            self,
            images: List[Image.Image],
            texts: List[str],
            multilabel: bool = False
    ) -> torch.Tensor:
        """Perform inference on batch of input images.

        Args:
            images: List of PIL Image objects
            texts: List of text labels
            multilabel: If True, use sigmoid for multilabel classification.
                       If False, use softmax for single-label classification.

        Returns:
            Tensor of prediction probabilities
        """
        # Prepare inputs
        images = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        # Process text
        text_tokens = self.tokenize(
            texts,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        # Move text tensors to the correct device
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(image=images, text=text_tokens)
            logits_per_image = outputs[0] @ outputs[1].t() * outputs[2]

            if multilabel:
                # Use sigmoid for independent probabilities per label
                probs = torch.sigmoid(logits_per_image)
            else:
                # Use softmax for single-label classification
                probs = logits_per_image.softmax(dim=1)

        return probs


