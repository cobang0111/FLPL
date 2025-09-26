<h1 align="center">Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback</h1>
<h2 align="center"></h2>

## ✨Summary
This repository provides the implementation of **Swap-guided Preference Learning (SPL)**.

<p align="center">
  <img src="fig/spl.png" alt="Overview of SPL" width="1000"/>
</p>

## ✨Environment Setting
Clone this repository and run:

```bash
conda create -n spl python=3.10
conda activate spl
pip install -r requirements.txt
```
## ✨Dataset Generation
Run the following commands with the model of your choice.
You may change the model identifiers in the script files if needed (default: Llama 3 3B).
You need an approved Hugging Face access token to download Llama 3 the first time.
```bash
# Put your authorized Hugging Face token
huggingface-cli login
```

And then, follow below command to generate datasets.
```bash
# For Pets (Dataset size: 3B ≈ 3GB, 8B ≈ 4GB)
bash generate_llm_embeddings_pets.sh

# For UF-P-2 (Dataset size: 3B ≈ 43GB, 8B ≈ 56GB)
python -m config.data_utils.ultrafeedback_augment -a 84 -n P
bash generate_llm_embeddings_UF_P_2.sh

# For UF-P-4 (Dataset size: 3B ≈ 61GB, 8B ≈ 79GB)
python -m config.data_utils.ultrafeedback_augment -a single -n P_4 -c
bash generate_llm_embeddings_UF_P_4.sh

```

## ✨Experiments

You can evaluate SPL on the generated datasets, alongside all baseline models included in the paper.

For example:

- **run_pets.sh** runs SPL on the Pets dataset with Llama-3.1-8B-instruct.

- **run_p4.sh** runs SPL on the UF-P-4 dataset with Llama-3.2-3B-instruct.
```bash
# FLPL in Pets
bash run_pets.sh
# FLPL in UF-P-4
bash run_p4.sh 
```
Additional examples for other datasets and model sizes can be found in the corresponding sbatch_[data]_[model_size].sh files.

You can check all the result metrics in Weights & Biases (wandb).
For first-time use, you’ll need to enter your wandb API key.