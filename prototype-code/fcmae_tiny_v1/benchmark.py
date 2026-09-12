"""Measure the assigned GPU using the real FCMAE/data/optimizer pipeline.

The model is disposable. This command never reads or changes production resume
checkpoints. Only lightweight reports remain in its unique benchmark directory.
"""
import argparse
import hashlib
import itertools
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark_report import projections, write_reports
from data_utils import ECGDataset, prepare_splits, sha256
from training_state import train_epoch, snapshot, save_atomic


def benchmark(cfg, options, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    report=dict(status='running',units_per_hour=options.get('units_per_hour'),
                configuration=cfg,benchmark_options=options,notes=[
        'Measurements apply only to the GPU allocated to this session; no extrapolation between GPU models.',
        'Benchmarks use real ECG images, sparse FCMAE, backward passes, AdamW and the configured accumulation/precision.',
        'Repeated training/validation timings use a populated resized-image cache and persistent workers.',
        'Production restarts loader workers each epoch; that small recurring startup overhead is not extrapolated from these warm-worker measurements.',
        'Fast/slow scenarios are observed round extrema, not confidence intervals or guaranteed runtime bounds.',
        'First-pass cache preparation is measured on a sample and extrapolated separately; adding it serially is conservative because real training can overlap preprocessing.',
        'Planning totals include one recorded preparation session and one extrapolated cache build, but exclude benchmarking cost, future restarts, idle time and unmeasured cloud-sync delays.',
        'Checkpoint/encoder/preview writes are measured on mounted Drive; best-SSL export is conservatively budgeted every epoch.',
        'Compare only reports with the same comparison_key. Compute-unit rates are manually supplied, never inferred.',
        'The benchmark updates a disposable model; its progress is not part of pretraining.'
    ])
    write_reports(report,directory)
    try:
        rate=options.get('units_per_hour')
        if rate is not None and (not math.isfinite(rate) or rate < 0):
            raise ValueError('Compute-unit rate must be nonnegative or null')
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA GPU assigned; select T4, L4 or A100 in Colab')
        report.update(gpu_name=torch.cuda.get_device_name(),
                      gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                      torch_version=torch.__version__,cuda_version=torch.version.cuda)
        report['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
        seed=cfg['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        rounds=options.get('rounds',3)
        warmup_updates=options.get('warmup_updates',2)
        measured_updates=options.get('measured_updates',6)
        if rounds < 2 or not 1 <= warmup_updates <= measured_updates:
            raise ValueError('Use at least 2 rounds, and 1 <= warmup_updates <= measured_updates')
        train_frame,val_frame,audit=prepare_splits(cfg['index'],cfg['image_root'],directory/'audit',
                                                  seed,cfg.get('holdout_csv'),cfg['ssl_val_fraction'])
        report['split_audit']=audit
        sample_count=cfg['batch_size']*cfg['accumulation']*measured_updates
        if len(train_frame) < sample_count:
            raise ValueError('Insufficient images for configured benchmark sample; reduce measured_updates')
        sampled=train_frame.sample(n=sample_count,random_state=seed).reset_index(drop=True)
        sampled_val=val_frame.sample(n=min(len(val_frame),options.get('validation_images',64)),random_state=seed).reset_index(drop=True)
        sample_ids=dict(train=sampled.fcmae_record_id.tolist(),validation=sampled_val.fcmae_record_id.tolist())
        source_hashes={str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')}
        keys=['input_size','batch_size','accumulation','workers','amp','seed','learning_rate',
              'epochs','warmup_epochs','save_every','checkpoint_every_updates','checkpoint_every_minutes',
              'clip_grad','ssl_val_fraction']
        comparison=dict(config={k:cfg[k] for k in keys},source_hashes=source_hashes,
                        index_sha256=audit['index_sha256'],holdout_sha256=audit['holdout_sha256'],
                        checkpoint_sha256=sha256(cfg['checkpoint']),sample_ids=sample_ids,
                        rounds=rounds,warmup_updates=warmup_updates,measured_updates=measured_updates,
                        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                                     timm=importlib.metadata.version('timm'),
                                     minkowski=importlib.metadata.version('MinkowskiEngine')))
        report['comparison_key']=hashlib.sha256(json.dumps(comparison,sort_keys=True).encode()).hexdigest()
        report['comparison_inputs']=comparison
        write_reports(report,directory)
        from rect_fcmae import RectFCMAE, load_initial, export_encoder
        from train import preflight, visualize
        model=RectFCMAE().cuda()
        report['initialization']=load_initial(model,cfg['checkpoint'])
        preflight(model,'cuda')
        report['sparse_preflight_passed']=True
        decay,no_decay=[],[]
        for name,param in model.named_parameters():
            (no_decay if param.ndim<=1 or name.endswith('bias') or 'grn' in name else decay).append(param)
        optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':.05},{'params':no_decay,'weight_decay':0.}],
                                    lr=cfg['learning_rate'],betas=(.9,.95))
        scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'])
        bench_cfg=dict(cfg,checkpoint_every_updates=0,checkpoint_every_minutes=0)
        updates_per_epoch=math.ceil(len(train_frame)/(cfg['batch_size']*cfg['accumulation']))
        total_steps=cfg['epochs']*updates_per_epoch
        warmup=min(total_steps-1,cfg['warmup_epochs']*updates_per_epoch)
        def progress(step=0):
            return dict(next_batch=0,step=step,loss_sum=0.,seen=0,seconds=0.,peak_gpu_gib=0.)
        # New scratch directory ensures a cold resized-image cache every GPU run.
        # Original PNGs may already be in the OS cache; this is disclosed below.
        with tempfile.TemporaryDirectory(prefix='fcmae_benchmark_') as scratch:
            cache=Path(scratch)/'cache'
            train_ds=ECGDataset(sampled,cfg['image_root'],cfg['input_size'],cache)
            val_ds=ECGDataset(sampled_val,cfg['image_root'],cfg['input_size'],cache)
            def loader(dataset,offset):
                return DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,
                                  num_workers=cfg['workers'],pin_memory=True,
                                  persistent_workers=cfg['workers']>0,
                                  generator=torch.Generator().manual_seed(seed+offset))
            train_loader=loader(train_ds,300000); val_loader=loader(val_ds,400000)
            cache_timings=[]
            for label,data_loader in [('train',train_loader),('validation',val_loader)]:
                t=time.perf_counter()
                for _ in data_loader:
                    pass
                cache_timings.append(dict(split=label,images=len(data_loader.dataset),seconds=time.perf_counter()-t))
            cache_rate=sum(r['seconds'] for r in cache_timings)/sum(r['images'] for r in cache_timings)
            report['cache_preparation']=dict(samples=cache_timings,
                 estimated_full_seconds=cache_rate*(len(train_frame)+len(val_frame)),
                 limitation='Fresh resized cache; master-image OS cache cannot be guaranteed cold. Includes sample worker startup.')
            warm_samples=warmup_updates*cfg['batch_size']*cfg['accumulation']
            print(f'GPU warmup: {warmup_updates} optimizer updates',flush=True)
            state=train_epoch(model,itertools.islice(train_loader,warmup_updates*cfg['accumulation']),
                              optimizer,scaler,bench_cfg,'cuda',0,warm_samples,progress(),total_steps,warmup,lambda p:None)
            step=state['step']; training=[]; validation=[]; peak=0.
            for round_index in range(rounds):
                torch.cuda.synchronize(); t=time.perf_counter()
                state=train_epoch(model,train_loader,optimizer,scaler,bench_cfg,'cuda',0,
                                  len(train_ds),progress(step),total_steps,warmup,lambda p:None)
                torch.cuda.synchronize(); duration=time.perf_counter()-t
                step=state['step']
                training.append(dict(round=round_index+1,images=len(train_ds),seconds=duration,
                                     seconds_per_image=duration/len(train_ds)))
                peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
                model.eval(); torch.cuda.synchronize(); t=time.perf_counter()
                val_rng=torch.Generator(device='cuda').manual_seed(seed+1000)
                with torch.no_grad():
                    for images,valid,_ in val_loader:
                        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
                            loss,_,_=model(images.cuda(),valid.cuda(),generator=val_rng)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite benchmark validation loss')
                torch.cuda.synchronize(); duration=time.perf_counter()-t
                validation.append(dict(round=round_index+1,images=len(val_ds),seconds=duration,
                                       seconds_per_image=duration/len(val_ds)))
                peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
                print(f"Round {round_index+1}: train={training[-1]['seconds']:.2f}s, validation={duration:.2f}s",flush=True)
            report.update(training_rounds=training,validation_rounds=validation,peak_gpu_gib=peak)
            # Exercise actual mounted-Drive serialization, then remove disposable
            # benchmark weights. They are never named as production checkpoints.
            io_dir=directory/'disposable_io'; io_dir.mkdir()
            try:
                sample_state=snapshot(model,optimizer,scaler,'BENCHMARK_NOT_A_TRAINING_CHECKPOINT',
                                      0,False,torch.arange(len(train_frame)),state,float('inf'),[])
                checkpoint_times=[]
                for _ in range(2):
                    torch.cuda.synchronize(); t=time.perf_counter()
                    save_atomic(sample_state,io_dir/'last.pt',keep_previous=True)
                    checkpoint_times.append(time.perf_counter()-t)
                checkpoint_bytes=(io_dir/'last.pt').stat().st_size
                torch.cuda.synchronize(); t=time.perf_counter()
                save_atomic({'model':export_encoder(model),'kind':'DISPOSABLE BENCHMARK'},io_dir/'encoder.pt')
                export_seconds=time.perf_counter()-t
                torch.cuda.synchronize(); t=time.perf_counter()
                visualize(model,val_ds,io_dir/'preview.png',seed+2000,'cuda')
                torch.cuda.synchronize(); preview_seconds=time.perf_counter()-t
                io=dict(checkpoint_seconds=sum(checkpoint_times)/len(checkpoint_times),
                        checkpoint_trials_seconds=checkpoint_times,checkpoint_bytes=checkpoint_bytes,
                        export_seconds=export_seconds,preview_seconds=preview_seconds)
            finally:
                for path in io_dir.iterdir():
                    path.unlink()
                io_dir.rmdir()
            report['drive_io']=io
            preparation=options.get('preparation_timings',{})
            report['preparation_timings']=preparation
            report['projections']=projections(cfg,audit,[r['seconds_per_image'] for r in training],
                    [r['seconds_per_image'] for r in validation],io,sum(preparation.values()),
                    report['cache_preparation']['estimated_full_seconds'],rate)
            # End persistent workers before removing their scratch cache.
            del train_loader,val_loader
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report['benchmark_wall_seconds']=time.perf_counter()-started
        report['benchmark_compute_units']=None if report['units_per_hour'] is None else report['benchmark_wall_seconds']/3600*report['units_per_hour']
        write_reports(report,directory)
        print('Benchmark report:',directory/'summary.md',flush=True)
    return report['status']=='complete'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--options',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    ok=benchmark(json.loads(Path(args.config).read_text()),json.loads(Path(args.options).read_text()),args.output)
    raise SystemExit(0 if ok else 1)
