"""Checkpoint-layout tests only; actual sparse execution is a GPU preflight."""
import argparse
from pathlib import Path
import tempfile
import types
import unittest
import torch
from checkpoint_utils import load_initial, export_encoder


class ShapeOnlyEncoder:
    """Describe sparse tensor shapes without pretending to execute Minkowski."""
    def __init__(self, dense):
        self.state = {}
        for key, value in dense.items():
            shape = value.shape
            if key.startswith('downsample_layers.') and key.split('.')[1] != '0':
                if key.split('.')[2] == '0':
                    key = key.replace('.0.', '.0.ln.')
                elif key.endswith('weight'):
                    key = key.removesuffix('weight')+'kernel'
                    shape = (4, value.shape[1], value.shape[0])
                else:
                    shape = (1, value.numel())
            elif key.startswith('stages.'):
                if '.dwconv.' in key and key.endswith('weight'):
                    key = key.removesuffix('weight')+'kernel'; shape = (49, value.shape[0])
                elif '.dwconv.' in key:
                    shape = (1, value.numel())
                key = key.replace('.norm.', '.norm.ln.').replace('.pwconv1.', '.pwconv1.linear.').replace('.pwconv2.', '.pwconv2.linear.')
                if '.grn.' in key:
                    shape = (1, value.numel())
            self.state[key] = torch.empty(shape, dtype=value.dtype)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, strict):
        assert strict and set(state) == set(self.state)
        assert all(state[k].shape == v.shape for k, v in self.state.items())
        self.state = state


def verify(path, expected_count=None):
    dense = torch.load(path, map_location='cpu', weights_only=True)['model']
    if expected_count:
        assert len(dense) == expected_count
    model = types.SimpleNamespace(encoder=ShapeOnlyEncoder(dense))
    report = load_initial(model, path)
    converted = export_encoder(model)
    assert set(converted) == set(dense)
    for key, value in dense.items():
        expected = value.reshape(1,1,1,-1) if '.grn.' in key else value
        torch.testing.assert_close(converted[key], expected, rtol=0, atol=0)
    print('Exact encoder tensor roundtrip:', len(dense), report)


class CheckpointTests(unittest.TestCase):
    def test_conversion_and_strict_missing(self):
        state = {'downsample_layers.1.0.weight': torch.randn(3),
                 'downsample_layers.1.1.weight': torch.arange(24.).reshape(2,3,2,2),
                 'downsample_layers.1.1.bias': torch.randn(2),
                 'stages.0.0.dwconv.weight': torch.arange(147.).reshape(3,1,7,7),
                 'stages.0.0.norm.weight': torch.randn(3),
                 'stages.0.0.pwconv1.weight': torch.randn(12,3),
                 'stages.0.0.grn.gamma': torch.randn(1,12)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pt'; torch.save({'model':state}, path)
            verify(path)
            model = types.SimpleNamespace(encoder=ShapeOnlyEncoder(state))
            del state['stages.0.0.norm.weight']; torch.save({'model':state}, path)
            with self.assertRaises(ValueError):
                load_initial(model, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.checkpoint:
        verify(args.checkpoint, 196)
    else:
        unittest.main(argv=['test_checkpoint_utils'])
