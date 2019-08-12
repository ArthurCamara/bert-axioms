#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16384
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0
module load cudnn/10.0-7.4.2.24

source /home/nfs/acamara/venvs/bert/bin/activate

DATA_DIR=/tudelft.net/staff-umbrella/punchbag/msmarco/data
SCRIPT_DIR=/tudelft.net/staff-umbrella/punchbag/msmarco/scripts

cd $SCRIPT_DIR

srun python sentence_level_classifier.py --task_name msmarco --do_train --do_eval --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 512 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir $DATA_DIR/models/

