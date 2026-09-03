AF+LVH test-case image review pack
==================================
Source: ResNet-50 @768 (hybrid_v1/resnet50_hires) frozen test predictions on
        prototype_1152  (LVH decision threshold = 0.47, tuned on val, frozen)

Contents
--------
correct/     18 test images where the model predicted LVH=1 (correct)
             among the 28 AF+LVH test cases
incorrect/   10 test images where the model predicted LVH=0 (LVH false
             negatives) among the 28 AF+LVH test cases

Images are exact copies of the rendered ECGs the model was evaluated on
(prototype_1152/images/test/... 1686x1311 RGB jpg). Nothing re-rendered.

Filename convention
-------------------
<sample_id>__<OK|MISS>__LVHp<prob>.jpg

  ECG003330__MISS__LVHp0.4297.jpg
     ^         ^          ^
     |         |          L-- model LVH probability (sigmoid head output)
     |         +------------- OK  = LVH predicted positive (correct)
     |                        MISS = LVH predicted negative (false negative)
     +----------------------- prototype sample id (= row `record` in index.csv)

Label details and probabilities
-------------------------------
Every folder contains case_details.csv with one row per image:

  filename             image file in this folder
  sample_id            prototype sample id (ECG00xxxx)
  global_record_id     pipeline id (PTBXL_xxxxxx)
  source               original dataset (PTBXL for all 28)
  orig_record_id       PTB-XL ecg_id  (raw ptbxl_database.csv row)
  patient_id           PTB-XL patient id
  age / sex
  prototype_labels     labels the model was trained on (AF+LVH)
  original_labels      FULL original annotation set (scp_codes), verbatim
  original_readable    readable names
  additional_conditions  original labels beyond AF+LVH (unmodeled)
  report_text          PTB-XL free-text ECG report excerpt
  LVH_prob / LVH_pred  model LVH probability / binary prediction
  LVH_correct          correct | INCORRECT
  AF_prob / AF_pred    model AF probability / binary prediction
  image_rel_path       path inside prototype_1152
  signal_path          original PTB-XL .hea/.dat signal on disk

Full 28-case analysis table:
  ../test_af_lvh_case_table.csv
Training-set audit (202 AF+LVH):
  ../train_af_lvh_extra_condition_audit.csv

Frozen model artifacts were NOT modified.
