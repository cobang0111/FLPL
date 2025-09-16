# This file is used to generate synthetic language dataset
from typing import cast

from transformers import (
    HfArgumentParser,
)

import torch
import random

from config.data_utils.data_processing import (
    ScriptArguments,
    generate_embeddings_with_llm,
    generate_contexts
)

from config.data_utils.simple_templates import *

from config.train_llm_preference_model import (
    DataSubset,
    get_hh_rlhf_dataset,
)

import numpy as np


def generate_synthetic_dataset(args):
    data_subset = cast(DataSubset, args.data_subset)
    input_dataset = get_hh_rlhf_dataset(
        data_subset,
        args.data_split,
        args.dataset_size,
        data_path=args.data_path,
        use_subset_as_dir=True
    )
    def generate_simple_data_point(example):
        prompt_length = 1
        prompt = 'Human: Please talk about one kind of pets.'
        if args.data_split == 'train':
            helpful_harmless = bird_sentences[:80]
            helpful_harmful = dog_sentences[:80]
            harmless_unhelpful = cat_sentences[:80]
            harmful_unhelpful = rabbit_sentences[:80]
        else:
            helpful_harmless = bird_sentences[80:]
            helpful_harmful = dog_sentences[80:]
            harmless_unhelpful = cat_sentences[80:]
            harmful_unhelpful = rabbit_sentences[80:]
        pair_type = np.random.randint(10)   # set to 6 previously
        if pair_type == 0:
            chosen = np.random.choice(helpful_harmless)
            rejected = np.random.choice(helpful_harmful)
        elif pair_type == 1:
            chosen = np.random.choice(harmless_unhelpful)
            rejected = np.random.choice(harmful_unhelpful)
        elif pair_type == 2:
            chosen = np.random.choice(helpful_harmless)
            rejected = np.random.choice(harmless_unhelpful)
        elif pair_type == 3:
            chosen = np.random.choice(helpful_harmful)
            rejected = np.random.choice(harmful_unhelpful)
        elif pair_type == 4:
            chosen = np.random.choice(helpful_harmless)
            rejected = np.random.choice(harmful_unhelpful)
        else:
            if script_args.data_subset == 'helpful':
                chosen = np.random.choice(helpful_harmful)
                rejected = np.random.choice(harmless_unhelpful)
            else:
                chosen = np.random.choice(harmless_unhelpful)
                rejected = np.random.choice(helpful_harmful)
        chosen_repeated = ' '.join([chosen] * prompt_length)
        rejected_repeated = ' '.join([rejected] * prompt_length)
        return_dict = {'prompt': prompt, 'chosen': prompt + '\n\n' + 'Assistant: ' + chosen_repeated,
                       'rejected': prompt + '\n\n' + 'Assistant: ' + rejected_repeated}
        if example['label'] == 0:
            return_dict['responses'] = [chosen_repeated, rejected_repeated]
        else:
            return_dict['responses'] = [rejected_repeated, chosen_repeated]
        if pair_type >= 5:
            return_dict['controversial'] = True
        else:
            return_dict['controversial'] = False
        return return_dict

    input_dataset = input_dataset.map(generate_simple_data_point)
    print(len(input_dataset.filter(lambda x: x['controversial'] == True)))
    return input_dataset


if __name__ == "__main__":
    # default setting on synthetic language dataset, please iterate over data subsets and data splits
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    dataset = generate_synthetic_dataset(script_args)
    if script_args.with_embeddings:
        dataset = generate_embeddings_with_llm(script_args, dataset)
    generate_contexts(script_args, dataset)
