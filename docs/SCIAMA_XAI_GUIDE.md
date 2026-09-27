# SCIAMA Supercomputer Execution Guide: XAI Attribution Evaluation

**Tasks Addressed:** Task 10 & Task 19 (Reviewer 1 Finding 4, Major Comments 9 & 10)
**Target Output:** `outputs/xai_evaluation_results.json` (populates Table 8 & Section 5 in manuscript)
**Cluster:** University of Portsmouth SCIAMA HPC (`gpu.q` partition, NVIDIA L40 / A100 GPU nodes)

---

## 1. Why SCIAMA is Required for this Task

1. **Authentic Checkpoints:** The 5-round incremental training checkpoint (`best.pt`) for YOLO11n pretrained and
   the 100-epoch reference baseline checkpoint (`best.pt`) are 16+ MB each and are archived on the SCIAMA Lustre
   storage (`/mnt/lustre2/mres/ghahrem/`), not in the git repository.
2. **GPU Backpropagation:** Integrated Gradients requires 20 interpolation steps and backward passes per image
   across weapon classes, which requires GPU acceleration (`torch.cuda`).
3. **Data Integrity:** Running against authentic checkpoints ensures 100% data provenance without running ad-hoc
   local reruns that risk overwriting training logs.

---

## 2. Quick-Start Execution on SCIAMA

### Option A: Submit via SLURM (Recommended)

1. SSH into SCIAMA (via University VPN):

   ```bash
   ssh sciama
   ```

2. Navigate to the repository on Lustre and pull latest changes:

   ```bash
   cd /mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark
   git pull origin main
   ```

3. Submit the batch job:

   ```bash
   sbatch scripts/slurm/submit_xai_eval.slurm
   ```

4. Monitor job status:

   ```bash
   squeue -u ghahrem
   # View log once started:
   tail -f runs/slurm_logs/xai_eval_*.out
   ```

### Option B: Interactive GPU Node Execution

1. Request an interactive GPU session:

   ```bash
   srun --partition=gpu.q --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
   ```

2. Load required environment modules:

   ```bash
   module purge
   module load system/sciama services/slurm anaconda3 uv git
   cd /mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark
   ```

3. Run the evaluation script:

   ```bash
   uv run python scripts/evaluate_xai_test_set.py --device 0
   ```

---

## 3. CLI Arguments Reference

`scripts/evaluate_xai_test_set.py` accepts the following optional arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--inc-ckpt` | Path to incremental round 5 `best.pt` | Auto-detected from `runs/yolo11n_pretrained_100ep/...` |
| `--base-ckpt` | Path to one-shot baseline `best.pt` | Auto-detected from `runs/yolo11n_baseline_pretrained_100ep/...` |
| `--data-root` | Path to `Weapon_Detection-1` directory | Auto-detected from Lustre / repo candidates |
| `--test-list` | Path to `test_fixed.txt` split file | `config/data/test_fixed.txt` |
| `--output` | Path to output JSON | `outputs/xai_evaluation_results.json` |
| `--n-samples` | Number of test images to evaluate | `20` (use `0` for all 506 images) |
| `--device` | Device index or name | `0` (or `cuda` / `cpu`) |

---

## 4. Syncing Results Back to Local Machine

Once the job finishes on SCIAMA, sync the output JSON back to your local repository:

```bash
# From your local machine:
scp sciama:/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/outputs/xai_evaluation_results.json ./outputs/
```

Or using `rsync`:

```bash
rsync -avz sciama:/mnt/lustre2/mres/ghahrem/yolo-threat-detection-benchmark/outputs/xai_evaluation_results.json \
    ./outputs/xai_evaluation_results.json
```

Once `outputs/xai_evaluation_results.json` is updated, the agent can immediately parse the verified numbers into
Table 8 and Section 5 in `submission/sn-article.tex`.
