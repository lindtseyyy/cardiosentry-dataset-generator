"""Freeze canonical exclusion IDs from the current supervised V2 split."""
import hashlib
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO/'dataset-combination-code/ecg_dataset_pipeline/data/releases/v2/harmonized/harmonized_ecg_metadata.csv'


def main():
    frame = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    held = frame.loc[frame['split'].isin(['val','test'])].copy()
    if held.empty:
        raise ValueError('No downstream holdouts')
    if set(held.source_dataset)-{'PTBXL','CHAPMAN','STEMI'}:
        raise ValueError('New source: define its canonical namespace before rebuilding')
    output = pd.DataFrame()
    # PTBXL and Chapman mappings match the FCMAE corpus builder. STEMI has no
    # established cross-source mapping and retains its own explicit namespace.
    output['source_dataset'] = held.source_dataset.map({'PTBXL':'ptb-xl', 'CHAPMAN':'chapman_shaoxing_ningbo', 'STEMI':'STEMI'})
    output['canonical_record_id'] = [str(int(v)) if s == 'PTBXL' else v
                                     for s, v in zip(held.source_dataset, held.original_record_id)]
    known = held.patient_id_is_synthetic.eq('0') & held.patient_id.ne('')
    output['patient_id'] = held.patient_id.where(known, '')
    output['patient_id_available'] = known.astype(int)
    output['downstream_split'] = held['split']
    output.to_csv(HERE/'downstream_holdouts_v2.csv', index=False)
    report = dict(source=str(SOURCE.relative_to(REPO)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  holdout_records=len(output), counts=output.source_dataset.value_counts().to_dict(),
                  limitation='Only established source namespaces are matched. Unknown and cross-source patient linkage remains unavailable.')
    (HERE/'holdout_provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
