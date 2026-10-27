"""DINOv3 + Mask2Former semantic segmenter used by QASR-C.

A frozen DINOv3 ViT-L/16 pretrained on SAT-493M feeds the DINOv3 ViT-Adapter,
which builds four feature maps (strides 4 to 32) for a Mask2Former head. The
adapter and the head were trained on EarthVQA. Their weights are released at
https://huggingface.co/PeterAM4/SQuID-DINOv3-Mask2Former. The backbone weights
are loaded from ``facebook/dinov3-vitl16-pretrain-sat493m``.

The adapter and head come from Meta's ``dinov3`` package, pinned in
requirements.txt to the commit the model was trained with.
"""

from pathlib import Path

import torch
import torch.nn as nn
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead
from transformers import AutoModel

BACKBONE = 'facebook/dinov3-vitl16-pretrain-sat493m'
WEIGHTS_REPO = 'PeterAM4/SQuID-DINOv3-Mask2Former'
FROZEN_PREFIX = 'backbone.backbone.model.'


class HFDINOv3Backbone(nn.Module):
    """Exposes a Hugging Face DINOv3 model through the interface of the adapter.

    Args:
        model: A ``transformers`` DINOv3 ViT.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_dim = model.config.hidden_size
        self.num_features = model.config.hidden_size
        self.patch_size = model.config.patch_size

    def forward(self, x):
        return self.model(x)

    def get_intermediate_layers(self, x, n=None, return_class_token=False):
        """Returns the patch tokens of the requested transformer blocks.

        Args:
            x: Images, ``(B, 3, H, W)``.
            n: Block indices. All blocks if None.
            return_class_token: Also return each block's [CLS] token.

        Returns:
            One ``(B, N, D)`` tensor per block, or ``(patches, cls)`` pairs.
        """
        out = self.model(x, output_hidden_states=True)
        # The adapter was trained with transformers 4.x, whose last hidden state
        # is the final layer-normed output. Transformers 5 returns it un-normed.
        hidden = (*out.hidden_states[:-1], out.last_hidden_state)
        n_registers = getattr(self.model.config, 'num_register_tokens', 4)
        blocks = range(len(hidden) - 1) if n is None else n
        out = []
        for i in blocks:
            tokens = hidden[i + 1]  # hidden[0] is the embedding output
            patches = tokens[:, 1 + n_registers:, :]
            out.append((patches, tokens[:, :1, :]) if return_class_token else patches)
        return out

    def requires_grad_(self, requires_grad=True):
        return self.model.requires_grad_(requires_grad)

    def parameters(self):
        return self.model.parameters()


class DINOv3Mask2FormerSegmenter(nn.Module):
    """DINOv3 ViT-L backbone, ViT-Adapter and Mask2Former head.

    Args:
        backbone_name: Hugging Face id of the DINOv3 backbone.
        num_classes: Number of segmentation classes (8 for EarthVQA).
        hidden_dim: Mask2Former hidden dimension.
    """

    def __init__(self, backbone_name=BACKBONE, num_classes=8, hidden_dim=256):
        super().__init__()
        vit = HFDINOv3Backbone(AutoModel.from_pretrained(backbone_name, trust_remote_code=True))
        layers = {12: [2, 5, 8, 11], 24: [4, 11, 17, 23], 48: [9, 19, 29, 39]}
        self.backbone = DINOv3_Adapter(backbone=vit, interaction_indexes=layers[vit.model.config.num_hidden_layers],
                                       pretrain_size=1024)
        dim = vit.embed_dim
        self.decoder = Mask2FormerHead(input_shape={'1': (dim, 4), '2': (dim, 8), '3': (dim, 16), '4': (dim, 32)},
                                       hidden_dim=hidden_dim, num_classes=num_classes, ignore_value=0)

    def forward(self, x):
        """Returns Mask2Former's per-query ``pred_logits`` and ``pred_masks``."""
        return self.decoder(self.backbone(x))

    def predict(self, x, rescale_to=None):
        """Like :meth:`forward`, with masks resized to ``rescale_to`` (default: input size)."""
        return self.decoder.predict(self.backbone(x), rescale_to=rescale_to or tuple(x.shape[2:]))


def load_segmenter(checkpoint=WEIGHTS_REPO, device='cpu'):
    """Builds the segmenter and loads its trained weights.

    Args:
        checkpoint: A Hugging Face repo id, a ``.safetensors`` file with the
            adapter and head weights, or a full training ``.pth`` checkpoint.
        device: Torch device.

    Returns:
        The model in eval mode.

    Raises:
        RuntimeError: If weights other than the frozen backbone are missing or
            unexpected.
    """
    if Path(checkpoint).suffix == '.pth':
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
    else:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = checkpoint if Path(checkpoint).is_file() else hf_hub_download(checkpoint, 'model.safetensors')
        state = load_file(path)

    model = DINOv3Mask2FormerSegmenter()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
    if missing or unexpected:
        raise RuntimeError(f"bad checkpoint: missing {missing[:3]}, unexpected {unexpected[:3]}")
    return model.to(device).eval()
