#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16384
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0
module load cudnn/10.0-7.4.2.24

DATA_DIR=/tudelft.net/staff-umbrella/punchbag/msmarco/data
SCRIPT_DIR=/tudelft.net/staff-umbrella/punchbag/msmarco/scripts
PYTHON=/home/nfs/acamara/venvs/bert/bin/python

cd $SCRIPT_DIR

srun $PYTHON end-to-end-training.py --data_dir $DATA_DIR --train_file $DATA_DIR/train-triples.top100 --dev_file $DATA_DIR/dev-triples.top100 --bert_model bert-base-uncased
