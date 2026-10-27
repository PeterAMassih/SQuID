"""DINOv3 + Mask2Former local segmentation backend (QASR-C).

Wraps the DINOv3 ViT-L/16 (SAT-493M) + Mask2Former model fine-tuned on
EarthVQA. ``eval_satellite_processing`` routes the topics in
``DINOV3_SUPPORTED_TOPICS`` here once ``init_dinov3_model`` has been called.
The other topics keep going to the HTTP server. The model itself is defined
in ``dinov3_model``.
"""

import logging
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dinov3_model import load_segmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dinov3_eval")

# Must match the training configuration of the checkpoint.
DINOV3_CONFIG = {
    'model': {
        'backbone_name': 'facebook/dinov3-vitl16-pretrain-sat493m',
        'num_classes': 8,
        'hidden_dim': 256,  # Token dimension used during training
    },
    'data': {
        'image_size': 1024,
        # SAT-493M normalization statistics used during training.
        'normalize_mean': [0.430, 0.411, 0.296],
        'normalize_std': [0.213, 0.156, 0.143],
    }
}

# Class ids of the prediction mask: the model's 0-7 argmax shifted by one.
DINOV3_CLASSES = {
    1: 'Background',
    2: 'Building',
    3: 'Road',
    4: 'Water',
    5: 'Barren',
    6: 'Forest',
    7: 'Agriculture',
    8: 'Playground'
}

# QASR topic -> DINOv3 class ids. Other topics fall back to the HTTP server.
TOPIC_TO_DINOV3_CLASS = {
    'forest': [6],
    'agric': [7],
    'barren': [5],
    'water': [4],
    'roof': [2],
}

DINOV3_SUPPORTED_TOPICS = list(TOPIC_TO_DINOV3_CLASS.keys())


class DINOv3SegmentationModel:
    """DINOv3 + Mask2Former model with tiling and padding for inference.

    A single instance is kept in ``_DINOV3_MODEL_INSTANCE`` by
    ``init_dinov3_model``.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """Build the model and load its weights.

        Args:
            checkpoint_path: Hugging Face repo id or local checkpoint, see
                ``dinov3_model.load_segmenter``.
            device: Torch device. Falls back to CPU when CUDA is unavailable.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = DINOV3_CONFIG

        logger.info(f"Loading DINOv3 model from: {checkpoint_path}")
        self.model = load_segmenter(checkpoint_path, self.device)

    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Predict a dense class map for one image.

        Images larger than the training size in either dimension are processed
        with overlapping sliding windows. Smaller ones are reflect-padded.

        Args:
            image_input: Image path, numpy array, or PIL image.

        Returns:
            np.ndarray: Class map (H, W) with values 1-8 (see ``DINOV3_CLASSES``).
        """
        image, original_size = self._load_image(image_input)

        h, w = original_size
        train_size = self.config['data']['image_size']

        use_sliding_window = (h > train_size or w > train_size)

        if use_sliding_window:
            logger.debug(f"Using sliding window inference for {h}x{w} image")
            prediction = self._sliding_window_inference(image, train_size)
        else:
            logger.debug(f"Using direct inference for {h}x{w} image")
            prediction = self._direct_inference(image, train_size, original_size)

        return prediction.cpu().numpy()

    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load an image as a normalized float tensor (3, H, W) and its (H, W) size."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        original_size = img.size[::-1]  # (H, W)

        img_np = np.array(img).astype(np.float32) / 255.0

        mean = np.array(self.config['data']['normalize_mean']).reshape(1, 1, 3)
        std = np.array(self.config['data']['normalize_std']).reshape(1, 1, 3)
        img_np = (img_np - mean) / std

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        return img_tensor, original_size

    def _direct_inference(self, image: torch.Tensor, train_size: int, original_size: Tuple[int, int]) -> torch.Tensor:
        """Run the model on the whole image, padding it up to ``train_size`` if smaller."""
        h, w = original_size

        if h < train_size or w < train_size:
            image, padding = self._pad_image(image, train_size)
        else:
            padding = None

        image = image.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        with torch.no_grad():
            outputs = self.model(image)
            semantic_logits = self._convert_mask2former_to_semantic(
                outputs,
                target_size=(image.shape[2], image.shape[3])
            )

        prediction = semantic_logits.argmax(dim=1)[0] + 1  # [H, W], classes 1-8

        if padding is not None:
            prediction = self._unpad_prediction(prediction, padding)

        return prediction

    def _sliding_window_inference(self, image: torch.Tensor, window_size: int = 1024,
                                  overlap: int = 256) -> torch.Tensor:
        """Average semantic logits over overlapping windows and take the argmax.

        The last window in each dimension is shifted back to end at the image
        border. Assumes both image dimensions are at least ``window_size``.
        """
        _, h, w = image.shape
        stride = window_size - overlap

        n_rows = (h - overlap) // stride + (1 if (h - overlap) % stride else 0)
        n_cols = (w - overlap) // stride + (1 if (w - overlap) % stride else 0)

        logits_sum = torch.zeros(8, h, w, dtype=torch.float32, device=self.device)
        count_map = torch.zeros(h, w, dtype=torch.float32, device=self.device)

        logger.debug(f"  Windows: {n_rows}x{n_cols} = {n_rows * n_cols} total")

        for i in range(n_rows):
            for j in range(n_cols):
                y_start = min(i * stride, h - window_size)
                x_start = min(j * stride, w - window_size)
                y_end = y_start + window_size
                x_end = x_start + window_size

                window = image[:, y_start:y_end, x_start:x_end]
                window_batch = window.unsqueeze(0).to(self.device)  # [1, 3, window_size, window_size]

                with torch.no_grad():
                    outputs = self.model(window_batch)
                    semantic_logits = self._convert_mask2former_to_semantic(
                        outputs,
                        target_size=(window_size, window_size)
                    )  # [1, 8, window_size, window_size]

                logits_sum[:, y_start:y_end, x_start:x_end] += semantic_logits[0]
                count_map[y_start:y_end, x_start:x_end] += 1

        logits_avg = logits_sum / count_map.unsqueeze(0).clamp(min=1)
        prediction = logits_avg.argmax(dim=0) + 1  # classes 1-8

        return prediction

    @staticmethod
    def _convert_mask2former_to_semantic(outputs: dict, target_size: Tuple[int, int]) -> torch.Tensor:
        """Combine Mask2Former query predictions into dense per-class scores.

        Each pixel's score for class c is the sum over queries of
        P(class c | query) * sigmoid(mask logit), with the "no object" class
        dropped.
        """
        pred_logits = outputs['pred_logits']  # [B, 100, 8+1]
        pred_masks = outputs['pred_masks']    # [B, 100, H/4, W/4]

        mask_cls = pred_logits.softmax(dim=-1)[..., :-1]  # [B, 100, 8]

        mask_pred = F.interpolate(
            pred_masks,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )  # [B, 100, H, W]

        mask_pred = mask_pred.sigmoid()

        semantic_logits = torch.einsum('bqc,bqhw->bchw', mask_cls, mask_pred)  # [B, 8, H, W]

        return semantic_logits

    @staticmethod
    def _pad_image(image: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Reflect-pad a (3, H, W) tensor symmetrically up to ``target_size``."""
        _, h, w = image.shape

        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)

        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        padded = F.pad(image, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode='reflect')

        return padded, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)

    @staticmethod
    def _unpad_prediction(prediction: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
        """Crop a prediction back to the unpadded image area."""
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
        h, w = prediction.shape

        h_start = pad_h_top
        h_end = h - pad_h_bottom if pad_h_bottom > 0 else h
        w_start = pad_w_left
        w_end = w - pad_w_right if pad_w_right > 0 else w

        return prediction[h_start:h_end, w_start:w_end]


_DINOV3_MODEL_INSTANCE = None


def init_dinov3_model(checkpoint_path: str, device: str = 'cuda'):
    """Create the module-level model instance. Later calls are no-ops.

    Args:
        checkpoint_path: Path to the model checkpoint.
        device: Torch device ('cuda' or 'cpu').
    """
    global _DINOV3_MODEL_INSTANCE

    if _DINOV3_MODEL_INSTANCE is not None:
        logger.warning("DINOv3 model already initialized, skipping")
        return

    _DINOV3_MODEL_INSTANCE = DINOv3SegmentationModel(checkpoint_path, device)
    logger.info("DINOv3 model initialized and ready")


def is_dinov3_initialized() -> bool:
    """Return whether ``init_dinov3_model`` has loaded the model."""
    return _DINOV3_MODEL_INSTANCE is not None


def segment_image_with_dinov3(
    image_input: Union[str, np.ndarray, Image.Image],
    topics: list,
    gsd: float = 1.0
) -> dict:
    """Segment an image with the DINOv3 model.

    Returns the same format as ``segment_image_from_path``. Topics outside
    ``DINOV3_SUPPORTED_TOPICS`` are ignored, and no minimum-area filter is
    applied.

    Args:
        image_input: Image path, numpy array, or PIL image.
        topics: Topics to segment, e.g. ``['forest', 'water']``.
        gsd: Ground sample distance in meters per pixel.

    Returns:
        dict: ``shapes``, ``image_width``, ``image_height`` and
        ``total_pixels``. If no requested topic is supported, ``shapes`` is
        empty and all sizes are 0.

    Raises:
        RuntimeError: If ``init_dinov3_model`` has not been called.
    """
    if _DINOV3_MODEL_INSTANCE is None:
        raise RuntimeError("DINOv3 model not initialized. Call init_dinov3_model() first.")

    requested_topics = [t for t in topics if t in DINOV3_SUPPORTED_TOPICS]

    if not requested_topics:
        logger.warning(f"No DINOv3-supported topics in {topics}")
        return {'shapes': [], 'image_width': 0, 'image_height': 0, 'total_pixels': 0}

    logger.info(f"Segmenting with DINOv3: {requested_topics}")

    prediction_mask = _DINOV3_MODEL_INSTANCE.predict(image_input)  # [H, W] with values 1-8
    h, w = prediction_mask.shape

    from .eval_shapes import mask_to_shapes

    all_shapes = []
    shape_id = 0

    for topic in requested_topics:
        class_indices = TOPIC_TO_DINOV3_CLASS[topic]

        binary_mask = np.zeros((h, w), dtype=np.uint8)
        for class_idx in class_indices:
            binary_mask |= (prediction_mask == class_idx)

        binary_mask = (binary_mask * 255).astype(np.uint8)

        shapes = mask_to_shapes(
            mask=binary_mask,
            class_type=topic,
            min_area_pixels=0,
            gsd=gsd,
            start_id=shape_id
        )

        all_shapes.extend(shapes)
        shape_id += len(shapes)

        logger.info(f"  {topic}: {len(shapes)} shapes")

    return {
        'shapes': all_shapes,
        'image_width': w,
        'image_height': h,
        'total_pixels': h * w
    }

