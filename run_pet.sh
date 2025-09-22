export WANDB_MODE=online
export WANDB_PROJECT=SPL_pet

# huggingface original name
model_name='meta-llama/Llama-3.1-8B-Instruct'

python -m config.train_llm_spl_model \
        --model_name=${model_name} \
        --data_path="data/simple_pets/llama-3.1-8B-instruct" \
        --num_train_epochs=2 \
        --reward_model_type=spl \
        --data_subset=both \
        --log_dir="logs/llama-3.1-8B-instruct_simple_pets" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --latent_dim 1024 \
        --hidden_dim 1024 \
        --encoder_embed_dim 4096 \
        --decoder_embed_dim 4096 \
        --max_length 1024 \
        --learning_rate 3e-4 \
        --use_annealing True \
        --kl_loss_weight 1e-4 \
        --guiding True \
        --guiding_weight 1e-5 \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --use_last_token_embedding True \
        --up_sampling True \
        --controversial_only False \
        --seed 31 \
        --use_iaf True \
        --num_iaf_flows 2 

