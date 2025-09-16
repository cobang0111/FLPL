#!/bin/bash
#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --job-name=train_job
#SBATCH --output=slurm-%j.out

gpustat

echo "### JOB STARTED: $(date)"
echo "### NODE: $(hostname)"
echo "### CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source ~/anaconda3/etc/profile.d/conda.sh && conda activate flpl

export HF_HOME=""
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
echo $HF_HOME

export WANDB_MODE=online
export WANDB_PROJECT=FLPL_p2
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name='meta-llama/Llama-3.1-8B-Instruct'

model_type=$1
augment_type="84"

# Variational Preference Learning (VPL): vpl
if [ ${model_type} == "vpl" ];
then
python -m config.train_llm_vpl_model \
        --model_name=${model_name} \
        --data_path="data/P_survey_16/llama-3.1-8B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=vpl \
        --data_subset=all \
        --log_dir="logs/llama-3.1-8B-instruct_P_survey_16" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 4096 \
        --decoder_embed_dim 4096 \
        --max_length 1024 \
        --learning_rate 1e-4 \
        --use_annealing True \
        --kl_loss_weight 3e-6 \
        --guiding False \
        --controversial_only True \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --up_sampling False \
        --other_subsets ${augment_type} \
        --use_last_token_embedding True \
        --seed 31

# Inverse Autoregressive Flow + Variational Preference Learning (IAF-VPL): ivpl
elif [ ${model_type} == "ivpl" ];
then
python -m config.train_llm_ivpl_model \
        --model_name=${model_name} \
        --data_path="data/P_survey_16/llama-3.1-8B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=ivpl \
        --data_subset=all \
        --log_dir="logs/llama-3.1-8B-instruct_P_survey_16" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 4096 \
        --decoder_embed_dim 4096 \
        --max_length 1024 \
        --learning_rate 1e-4 \
        --use_annealing True \
        --kl_loss_weight 3e-6 \
        --guiding False \
        --guiding_weight 5e-4 \
        --controversial_only True \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --up_sampling False \
        --other_subsets ${augment_type} \
        --use_last_token_embedding True \
        --seed 31 \
        --use_iaf True \
        --num_iaf_flows 2

# Flow Latent Preference Learning (FLPL): flpl
elif [ ${model_type} == "flpl" ];
then
python -m config.train_llm_flpl_model \
        --model_name=${model_name} \
        --data_path="data/P_survey_16/llama-3.1-8B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=flpl \
        --data_subset=all \
        --log_dir="logs/llama-3.1-8B-instruct_P_survey_16" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 4096 \
        --decoder_embed_dim 4096 \
        --max_length 1024 \
        --learning_rate 1e-4 \
        --use_annealing True \
        --kl_loss_weight 3e-6 \
        --guiding True \
        --guiding_weight 5e-4 \
        --controversial_only True \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --up_sampling False \
        --other_subsets ${augment_type} \
        --use_last_token_embedding True \
        --seed 31 \
        --use_iaf True \
        --num_iaf_flows 2 

# DPL (Distributional Preference Learning): catergorical / mean_and_variance
# BTL (Bradley-Terry-Luce RLHF): base
else
python -m config.train_llm_preference_model \
        --model_name=${model_name} \
        --data_path="data/P_survey_16/llama-3.1-8B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=${model_type} \
        --data_subset=all \
        --log_dir="logs/llama-3.1-8B-instruct_P_survey_16" \
        --bf16 True \
        --fp16 False \
        --max_length 1024 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --controversial_only True \
        --up_sampling False \
        --other_subsets ${augment_type} \
        --gradient_checkpointing True \
        --seed 31
fi


echo "### JOB ENDED: $(date)"