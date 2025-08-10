# Set Consistency Energy

**Official code for the ACL 2025 Oral paper:**  
_[Introducing Verification Task of Set Consistency with Set-Consistency Energy Networks]_  
**ACL Anthology:** https://aclanthology.org/2025.acl-long.1599/ · **arXiv:** https://arxiv.org/abs/2503.10695


## Environment

Create and activate the conda environment:

```bash
conda env create -f env.yml
conda activate set_consistency
```

If PyTorch installation fails, install the GPU-enabled build following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

## Datasets

You can download the prepared dataset from Google Drive:  
👉 [Download dataset (Google Drive)](https://drive.google.com/file/d/1vF19VRmwjQd5BqdjrzE5sP05h6hHzp4Z/view?usp=sharing)

## Model Weights

You can download the trained model from Google Drive:  
👉 **[Download model (Google Drive)](https://drive.google.com/file/d/1w81S9Ut6Fg3EtYz4F_f99bJN0uhcXbb1/view?usp=sharing)**

## Training

Example command for training an energy model:

```bash
  python train_wdb_Set_Contrastive.py \
    --task vqa \
    --dataset lconvqa \
    --loss_type margin \
    --decomposition no \
    --repre_model roberta
```

The resulting model is stored in `results/task/dataset/job_id/`.

## Evaluation

To evaluate a trained energy model:

```bash
  python evaluate.py --task vqa --dataset lconvqa --loss_type margin --decomposition no --repre_model roberta
```

Baseline models can be evaluated with:

```bash
  python evaluate_baseline.py --task vqa --dataset lconvqa --type llm --model gpt-4o-mini --shot_num 5
```

## Locate

To run the locate procedure on a trained energy model:

```bash
  python locate.py --task vqa --dataset lconvqa --loss_type margin --decomposition no
```

## SLURM Example

Example `sbatch` script for a single GPU on Linux:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

module load anaconda
conda activate set_consistency
  python train_wdb_Set_Contrastive.py --task vqa --dataset lconvqa --loss_type margin --decomposition no --repre_model roberta
```

## License

This project is released under the MIT License.
