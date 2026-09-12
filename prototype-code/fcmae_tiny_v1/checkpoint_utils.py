"""Strict official FCMAE encoder initialization and sparse/dense tensor conversion."""
import torch

def sparse_to_dense_key(key, value):
    """Meta/Minkowski layout to the official dense ConvNeXt-V2 layout.

    The spatial transpose follows MinkowskiEngine's kernel-offset ordering.
    GPU preflight checks all-visible sparse/dense feature parity with batch=1.
    """
    key = key.removeprefix('encoder.')
    if key.endswith('.kernel'):
        key = key[:-len('kernel')]+'weight'
        if value.ndim == 3:  # kernel-volume, in, out
            p = int(value.shape[0]**.5)
            value = value.permute(2, 1, 0).reshape(value.shape[2], value.shape[1], p, p).transpose(-1, -2)
        elif value.ndim == 2:  # depthwise: kernel-volume, channels
            p = int(value.shape[0]**.5)
            value = value.T.reshape(value.shape[1], 1, p, p).transpose(-1, -2)
        else:
            raise ValueError(f'Unexpected kernel shape: {key} {value.shape}')
    key = key.replace('.ln.', '.').replace('.linear.', '.')
    if '.grn.' in key:
        value = value.reshape(1, 1, 1, -1)
    if key.endswith('.bias'):
        value = value.reshape(-1)
    return key, value.contiguous()


def export_encoder(model):
    return dict(sparse_to_dense_key(k, v.detach().cpu()) for k, v in model.encoder.state_dict().items())


def load_initial(model, path):
    # Restricted unpickler: the official file is a tensor model dictionary.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model', checkpoint)
    # Support released dense/JAX-converted tensors or native sparse training keys.
    # The released pt_only checkpoint contains 196 ENCODER tensors, no decoder.
    # The FCMAE reconstruction head remains explicitly randomly initialized.
    expected = {'encoder.'+k: v for k, v in model.encoder.state_dict().items()}
    loaded = {}
    used = set()
    for key, target in expected.items():
        if key in state and state[key].shape == target.shape:
            loaded[key] = state[key]; used.add(key); continue
        dense_key, _ = sparse_to_dense_key(key, target) if key.startswith('encoder.') else (key, target)
        candidates = ['encoder.'+dense_key, dense_key] if key.startswith('encoder.') else [dense_key]
        match = next((k for k in candidates if k in state), None)
        if match is None:
            raise ValueError(f'Checkpoint is missing {key}; refusing partial initialization')
        value = state[match]
        if key.endswith('.kernel'):
            value = value.transpose(-1, -2)
            if target.ndim == 3:
                value = value.reshape(value.shape[0], value.shape[1], -1).permute(2, 1, 0)
            else:
                value = value.reshape(value.shape[0], -1).T
        if value.numel() != target.numel():
            raise ValueError(f'Checkpoint shape mismatch: {key}')
        loaded[key] = value.reshape_as(target).contiguous(); used.add(match)
    unexpected = set(state)-used
    if unexpected:
        raise ValueError(f'Unexpected checkpoint keys: {sorted(unexpected)[:10]}')
    model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in loaded.items()}, strict=True)
    return {'loaded_encoder_tensors': len(loaded), 'missing_encoder': [], 'unexpected': [],
            'decoder_initialization': 'random; official pt_only release contains encoder only'}
