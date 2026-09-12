"""CPU interrupted-vs-uninterrupted tests of the production epoch loop."""
from pathlib import Path
import os
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from training_state import (ResumeBatchSampler, epoch_order, snapshot, save_atomic,
                            load_resume, restore_rng, train_epoch)


class Records(Dataset):
    def __len__(self):
        return 23

    def __getitem__(self, index):
        return torch.tensor([index/23., .3, -.2, .7]), torch.ones(4), str(index)


class MaskedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 9), torch.nn.Dropout(.2), torch.nn.Linear(9, 4))
        self.visited = []

    def forward(self, images, valid):
        self.visited.extend(round(float(x)*23) for x in images[:, 0])
        mask = (torch.rand_like(images) > .4).float()
        # Exercise all saved RNG streams, as well as dropout and random masks.
        target = images * (random.random()+float(np.random.random()))
        loss = (self.net(images*(1-mask))-target).square().mean()
        return loss, None, mask


def seed_all():
    random.seed(81); np.random.seed(81); torch.manual_seed(81)


def make_objects():
    model = MaskedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    return model, optimizer, scaler


def loader(order, start, workers):
    return DataLoader(Records(), batch_sampler=ResumeBatchSampler(order, 2, start, 3),
                      num_workers=workers, generator=torch.Generator().manual_seed(500))


CFG = dict(batch_size=2, accumulation=3, learning_rate=.001, amp=False, clip_grad=1.,
           checkpoint_every_updates=1, checkpoint_every_minutes=0)


def initial_progress():
    return dict(next_batch=0, step=0, loss_sum=0., seen=0, seconds=0., peak_gpu_gib=0.)


class Interrupted(Exception):
    pass


class OverflowOnceScaler:
    """Small CPU test double for GradScaler's skip-and-backoff behavior."""
    def __init__(self):
        self.value = 8.
        self.calls = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        self.calls += 1
        if self.calls > 1:
            optimizer.step()

    def update(self):
        if self.calls == 1:
            self.value *= .5


class ResumeTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def run_interruption_case(self, stop_update, workers):
        order = epoch_order(23, 42, 0)
        seed_all(); reference, ropt, rscale = make_objects()
        expected = train_epoch(reference, loader(order,0,workers), ropt, rscale, CFG,
                               'cpu',0,23,initial_progress(),10,2,lambda p: None)
        expected_next_draw = torch.rand(5)
        seed_all(); model, opt, scale = make_objects()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'last.pt'
            def save(progress):
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                save_atomic(snapshot(model,opt,scale,'test',0,False,order,progress,float('inf'),[]), path, True)
                if progress['step'] == stop_update:
                    raise Interrupted()
            with self.assertRaises(Interrupted):
                train_epoch(model,loader(order,0,workers),opt,scale,CFG,'cpu',0,23,
                            initial_progress(),10,2,save)
            first_visits = list(model.visited)
            # Simulate a completely new process constructing model and loaders.
            seed_all(); model, opt, scale = make_objects()
            saved = load_resume(path,'test')
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
            scale.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
            resumed_loader = loader(saved['order'],saved['progress']['next_batch'],workers)
            actual = train_epoch(model,resumed_loader,opt,scale,CFG,'cpu',0,23,
                                 saved['progress'],10,2,lambda p: None)
        self.assert_nested_equal(reference.state_dict(),model.state_dict())
        self.assert_nested_equal(ropt.state_dict(),opt.state_dict())
        torch.testing.assert_close(torch.rand(5),expected_next_draw,rtol=0,atol=0)
        self.assertEqual(first_visits+model.visited,reference.visited)
        self.assertEqual(reference.visited,order.tolist())
        for key in ['step','next_batch','loss_sum','seen']:
            self.assertEqual(actual[key],expected[key])

    def test_mid_epoch_with_prefetch(self):
        self.run_interruption_case(stop_update=2,workers=2)

    def test_short_final_accumulation_group(self):
        self.run_interruption_case(stop_update=3,workers=0)

    def test_training_done_validation_pending(self):
        self.run_interruption_case(stop_update=4,workers=2)

    def test_time_trigger_only_at_update_boundaries(self):
        seed_all(); model,opt,scale=make_objects(); saved=[]
        cfg = dict(CFG,checkpoint_every_updates=0,checkpoint_every_minutes=1e-12)
        train_epoch(model,loader(epoch_order(23,42,0),0,0),opt,scale,cfg,'cpu',0,23,
                    initial_progress(),10,2,lambda p: saved.append(p['next_batch']))
        self.assertEqual(saved,[3,6,9,12])

    def test_amp_overflow_skips_update_backs_off_and_advances_cursor(self):
        seed_all(); model,optimizer,_=make_objects(); scaler=OverflowOnceScaler(); saved=[]
        cfg=dict(CFG,amp=True,checkpoint_every_updates=0,checkpoint_every_minutes=0)
        real_clip=torch.nn.utils.clip_grad_norm_
        calls=iter([torch.tensor(float('inf'))]+[None]*3)
        def overflow_once(parameters,max_norm):
            result=next(calls)
            return real_clip(parameters,max_norm) if result is None else result
        with patch('training_state.torch.autocast') as autocast, \
             patch('training_state.torch.nn.utils.clip_grad_norm_',side_effect=overflow_once):
            autocast.return_value.__enter__.return_value=None
            progress=train_epoch(model,loader(epoch_order(23,42,0),0,0),optimizer,scaler,
                                 cfg,'cpu',0,23,initial_progress(),10,2,
                                 lambda p:saved.append(dict(p)))
        self.assertEqual(progress['next_batch'],12)
        self.assertEqual(progress['step'],3)
        self.assertEqual(progress['amp_skipped_updates'],1)
        self.assertEqual(scaler.get_scale(),4.)
        self.assertEqual(saved[0]['next_batch'],3)

    def test_bad_cursor_rejected(self):
        with self.assertRaises(ValueError):
            ResumeBatchSampler(epoch_order(23,42,0),2,1,3)

    def test_failed_save_preserves_recovery_and_bad_latest_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'
            old=dict(format_version=2,fingerprint='test',value=1)
            new=dict(old,value=2)
            save_atomic(old,path,True)
            with patch('training_state.torch.save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_atomic(new,path,True)
            self.assertEqual(load_resume(path,'test')['value'],1)
            save_atomic(new,path,True)
            path.write_bytes(b'interrupted/corrupt')
            self.assertEqual(load_resume(path,'test')['value'],1)
            self.assertFalse(path.exists())  # corrupt last cannot replace good previous
            save_atomic(new,path,True)
            self.assertEqual(load_resume(path,'test')['value'],2)
            self.assertEqual(torch.load(path.with_name('previous.pt'),weights_only=False)['value'],1)
            with self.assertRaises(ValueError):
                load_resume(path,'different experiment')

    def test_publish_failure_after_rotation_recovers_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'
            state=dict(format_version=2,fingerprint='test',value=1)
            save_atomic(state,path,True)
            replace=os.replace
            def fail_publish(source,target):
                if str(source).endswith('.tmp'):
                    raise OSError('disconnected during final rename')
                replace(source,target)
            with patch('training_state.os.replace',side_effect=fail_publish):
                with self.assertRaises(OSError):
                    save_atomic(dict(state,value=2),path,True)
            self.assertEqual(load_resume(path,'test')['value'],1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
