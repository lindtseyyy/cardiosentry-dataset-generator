"""CPU regression checks for the notebook's data/masking/loss adaptations."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from data_utils import (letterbox, patchify, unpatchify, uniform_mask,
                        targets, reconstruction_loss, prepare_splits)


class DataTests(unittest.TestCase):
    def test_rectangular_roundtrip(self):
        for h, w in [(64, 96), (768, 1024)]:
            x = torch.randn(1, 3, h, w)
            torch.testing.assert_close(unpatchify(patchify(x), (h//32, w//32)), x)

    def test_padding_invariance_loss_and_stats(self):
        x = torch.randn(1, 3, 64, 96)
        valid = torch.zeros(1, 1, 64, 96); valid[:, :, 7:57, 10:80] = 1
        changed = x + (1-valid)*100
        mask = uniform_mask(valid)
        pred = torch.randn(1, 3072, 2, 3, requires_grad=True)
        a = reconstruction_loss(x, pred, valid, mask)
        b = reconstruction_loss(changed, pred, valid, mask)
        torch.testing.assert_close(a, b)
        for aa, bb in zip(targets(x, valid)[2:], targets(changed, valid)[2:]):
            torch.testing.assert_close(aa, bb)
        a.backward(); self.assertTrue(torch.isfinite(pred.grad).all())

    def test_full_valid_matches_official_loss(self):
        x = torch.randn(2, 3, 64, 96)
        valid = torch.ones(2, 1, 64, 96); mask = uniform_mask(valid)
        pred = torch.randn(2, 6, 3072)
        t = patchify(x); t = (t-t.mean(-1, keepdim=True))/(t.var(-1, keepdim=True)+1e-6).sqrt()
        expected = (((pred-t).square().mean(-1))*mask).sum()/mask.sum()
        torch.testing.assert_close(reconstruction_loss(x, pred, valid, mask), expected)

    def test_uniform_ignores_empty_patches(self):
        valid = torch.zeros(2, 1, 96, 160); valid[:, :, :64, :128] = 1
        a = uniform_mask(valid, generator=torch.Generator().manual_seed(3))
        b = uniform_mask(valid, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(a, b)
        self.assertTrue((a.sum(-1) == 5).all())
        self.assertTrue((a[patchify(valid).sum(-1) == 0] == 0).all())

    def test_letterbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.png'; Image.new('RGB', (100, 50)).save(path)
            image, valid = letterbox(path, (64, 96))
            self.assertEqual(tuple(image.shape), (3,64,96))
            self.assertEqual(valid.sum(), 48*96)

    def test_cache_preserves_pixels_and_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'original.png'
            Image.fromarray(np.random.default_rng(0).integers(0,256,(91,137,3),dtype=np.uint8)).save(path)
            cache = Path(tmp)/'cache.png'
            a, av = letterbox(path, (64,96), cache)
            b, bv = letterbox(path, (64,96), cache)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(av, bv, rtol=0, atol=0)

    def test_patient_holdout_and_stable_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(100):
                path = f'{i}.png'; Image.new('RGB', (8,8)).save(root/path)
                rows.append(dict(image_path=path, fcmae_record_id=str(i), source_dataset='s',
                                 canonical_record_id=str(i), patient_id=str(i//2), patient_id_available='1'))
            frame = pd.DataFrame(rows); frame.to_csv(root/'index.csv', index=False)
            frame.iloc[[0]].to_csv(root/'holdout.csv', index=False)
            a, b, audit = prepare_splits(root/'index.csv', root, root/'out', 42, root/'holdout.csv', .2)
            self.assertEqual(audit['excluded_records'], 2)
            self.assertFalse(set(a.group_key)&set(b.group_key))
            self.assertNotIn('0', set(a.patient_id)|set(b.patient_id))
            frame.loc[2, 'canonical_record_id'] = frame.loc[3, 'canonical_record_id']
            frame.to_csv(root/'bad.csv', index=False)
            with self.assertRaises(ValueError):
                prepare_splits(root/'bad.csv', root, root/'badout', 42)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
