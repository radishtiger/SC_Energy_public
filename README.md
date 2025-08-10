# Set Consistency Energy

This repository provides the official implementation for the ACL paper on Set Consistency Energy Networks. The code enables training, evaluation, and analysis of energy-based models that operate on sets of examples.

## Environment

Create and activate the conda environment:

```bash
conda env create -f env.yml
conda activate set_consistency
```

If PyTorch installation fails, install the GPU-enabled build following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

## Datasets

Download the required datasets and place them under the `datasets` directory:

  - **L-ConVQA**: [Logical ConVQA](https://arijitray1993.github.io/ConVQA/Logical_ConVQA.zip)

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
  python evaluate_baseline.py --task vqa --dataset lconvqa --type llm --model gpt-4o-mini --shot_num 1
```

## Locate

To run the locate procedure on a trained energy model:

```bash
  python locate.py --task vqa --dataset lconvqa --loss_type margin --decomposition no --time_key <time-key-of-model>
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
