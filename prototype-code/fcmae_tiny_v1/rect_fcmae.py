"""Rectangular/padding adaptation of pinned Meta FCMAE, retaining sparse blocks."""
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent/'vendor'))
from models.fcmae import FCMAE
from MinkowskiOps import to_sparse
from data_utils import patchify, uniform_mask, reconstruction_loss


def masked_stem_features(encoder, images, mask, valid):
    """Return the dense stem grid after removing masked/invalid locations.

    This is the deterministic isolation boundary: masked image content must be
    gone before MinkowskiEngine receives features and builds a coordinate map.
    """
    n, _, h, w = images.shape
    gh, gw = h//32, w//32
    x = encoder.downsample_layers[0](images * valid)
    active = (1-mask.reshape(n, 1, gh, gw)).repeat_interleave(8, 2).repeat_interleave(8, 3)
    active = active * (F.max_pool2d(valid, 4, 4) > 0)
    return x * active


def rectangular_sparse_forward(self, images, mask, valid):
    n, _, h, w = images.shape
    gh, gw = h//32, w//32
    # Fill invalid pixels with zero in normalized input space. No pixel values
    # from masked patches can enter the non-overlapping stride-4 stem.
    x = to_sparse(masked_stem_features(self, images, mask, valid))
    for i in range(4):
        x = self.downsample_layers[i](x) if i else x
        x = self.stages[i](x)
    # Explicit origin/shape preserve masked outer rows/columns and batch items.
    return x.dense(shape=torch.Size([n, x.F.shape[1], gh, gw]),
                   min_coordinate=torch.zeros(2, dtype=torch.int32))[0]


class RectFCMAE(FCMAE):
    def __init__(self, **kwargs):
        super().__init__(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                         patch_size=32, mask_ratio=.6, norm_pix_loss=True, **kwargs)
        self.encoder.forward = types.MethodType(rectangular_sparse_forward, self.encoder)

    def patchify(self, images):
        return patchify(images, self.patch_size)

    def forward(self, images, valid, mask=None, generator=None):
        if mask is None:
            mask = uniform_mask(valid, self.mask_ratio, generator)
        features = self.encoder(images, mask, valid)
        pred = self.forward_decoder(features, mask)
        loss = reconstruction_loss(images, pred, valid, mask, self.norm_pix_loss)
        return loss, pred, mask


from checkpoint_utils import export_encoder, load_initial
