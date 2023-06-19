1. Download Pfam-A.seed
2. Prepare Data
`python3 prepare_data.py`
3. Train Model(gpu=xx)
`CUDA_VISIBLE_DEVICES=xx python3 train_msa.py`
4. Test Model(gpu=xx)
- Get align.txt `CUDA_VISIBLE_DEVICES=xx python3 test_msa.py`
- Compute Precision/Recall/F1 `python3 compute_f1_score.py`
