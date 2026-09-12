"""Pure-Python timing/cost projections; no GPU prices or speed ratios assumed."""
import csv
import json
import math
from pathlib import Path


def project(cfg, n_train, n_val, train_seconds_per_image, val_seconds_per_image,
            checkpoint_seconds, export_seconds, preview_seconds, epochs,
            units_per_hour=None):
    if n_train <= 0 or n_val <= 0 or epochs <= 0:
        raise ValueError('Positive dataset counts and epochs required')
    if any(v < 0 for v in [train_seconds_per_image, val_seconds_per_image,
                           checkpoint_seconds, export_seconds, preview_seconds]):
        raise ValueError('Timings cannot be negative')
    if units_per_hour is not None and (not math.isfinite(units_per_hour) or units_per_hour < 0):
        raise ValueError('Enter a nonnegative observed compute-unit rate, or leave blank')
    group = cfg['batch_size']*cfg['accumulation']
    updates = math.ceil(n_train/group)
    update_interval = cfg.get('checkpoint_every_updates',100)
    time_interval = cfg.get('checkpoint_every_minutes',15)*60
    global_step, saves, exports, previews = 0, 0, 1, 0  # Initial control export.
    milestones = 0
    for epoch in range(epochs):
        since_save = 0.
        for update in range(updates):
            since_save += min(group,n_train-update*group)*train_seconds_per_image
            global_step += 1
            if ((update_interval and global_step%update_interval == 0)
                or (time_interval and since_save >= time_interval)
                or update+1 == updates):
                saves += 1
                since_save = 0.
        saves += 1  # Completed epoch, after validation/artifacts.
        exports += 1  # Conservative assumption: best-SSL improves every epoch.
        if (epoch+1)%cfg['save_every'] == 0 or epoch+1 == cfg['epochs']:
            saves += 1; exports += 1; previews += 1; milestones += 1
    training = n_train*train_seconds_per_image*epochs
    validation = n_val*val_seconds_per_image*epochs
    writes = saves*checkpoint_seconds + exports*export_seconds + previews*preview_seconds
    seconds = training+validation+writes
    return dict(epochs=epochs, training_seconds=training, validation_seconds=validation,
                checkpoint_and_artifact_seconds=writes, checkpoint_writes=saves,
                milestone_checkpoints=milestones, encoder_exports=exports,
                preview_writes=previews, steady_total_hours=seconds/3600,
                average_epoch_minutes=seconds/epochs/60,
                steady_compute_units=None if units_per_hour is None else seconds/3600*units_per_hour)


def projections(cfg, counts, train_rates, val_rates, io, preparation_seconds,
                cache_seconds, units_per_hour):
    rows=[]
    for epochs in sorted({1,min(20,cfg['epochs']),cfg['epochs']}):
        for label, reducer in [('fast_observed',min),('mean_observed',lambda xs:sum(xs)/len(xs)),('slow_observed',max)]:
            row=project(cfg,counts['ssl_train'],counts['ssl_validation'],
                        reducer(train_rates),reducer(val_rates),io['checkpoint_seconds'],
                        io['export_seconds'],io['preview_seconds'],epochs,units_per_hour)
            row['scenario']=label
            row['planning_total_hours']=row['steady_total_hours']+(preparation_seconds+cache_seconds)/3600
            row['planning_compute_units']=None if units_per_hour is None else row['planning_total_hours']*units_per_hour
            rows.append(row)
    return rows


def write_reports(report, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    (directory/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    rows=report.get('projections',[])
    if rows:
        with open(directory/'estimates.csv','w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# FCMAE GPU benchmark', '', f"Status: **{report['status']}**",'',
           f"GPU: {report.get('gpu_name','unavailable')}",
           f"Compute units/hour (entered by user): {report.get('units_per_hour')}",
           f"Comparison group: `{report.get('comparison_key','unavailable')}`",'']
    if report.get('error'):
        lines += [f"Error: {report['error']}",'']
    if rows:
        lines += ['| Epochs | Mean hours/epoch | Total hours incl. preparation/cache estimate | Compute units |',
                  '|---:|---:|---:|---:|']
        for r in rows:
            if r['scenario']=='mean_observed':
                units='not supplied' if r['planning_compute_units'] is None else f"{r['planning_compute_units']:.2f}"
                lines.append(f"| {r['epochs']} | {r['average_epoch_minutes']/60:.2f} | {r['planning_total_hours']:.2f} | {units} |")
    lines += ['', 'Notes:', '']+[f'- {note}' for note in report.get('notes',[])]
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')


def comparison_rows(root):
    rows=[]
    for path in sorted(Path(root).glob('*/report.json')):
        report=json.loads(path.read_text())
        row=dict(gpu=report.get('gpu_name'),status=report['status'],
                 comparison_key=report.get('comparison_key'),units_per_hour=report.get('units_per_hour'),
                 peak_gpu_gib=report.get('peak_gpu_gib'),epoch_minutes=None,
                 epoch_compute_units=None,planned_hours=None,planned_compute_units=None,
                 report=str(path))
        means=[r for r in report.get('projections',[]) if r['scenario']=='mean_observed']
        if means:
            first=next(r for r in means if r['epochs']==1)
            final=max(means,key=lambda r:r['epochs'])
            row.update(epoch_minutes=first['average_epoch_minutes'],
                       epoch_compute_units=first['steady_compute_units'],
                       planned_hours=final['planning_total_hours'],planned_compute_units=final['planning_compute_units'])
        rows.append(row)
    return rows
