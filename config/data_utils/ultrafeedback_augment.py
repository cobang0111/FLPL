import argparse
import random

import torch
from datasets import load_dataset, Dataset
import numpy as np
import os


def random_argmax(values):
    """ a random tie-breaking argmax """
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


def random_greater_than_zero(values):
    return (np.random.randn(values.shape[0]) * (values == 0) > 0.0) | (values > 0.0)


def array_to_type(arr):
    return str(int(np.dot(arr, np.array([8, 4, 2, 1]))))


def get_user_type(chosen_ratings, rejected_ratings, augment_type):
    keys = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    chosen_rating_values = list()
    rejected_rating_values = list()
    for key in keys:
        chosen_rating_values.append(chosen_ratings[key])
        rejected_rating_values.append(rejected_ratings[key])
    chosen_values = np.asarray(chosen_rating_values)
    rejected_values = np.asarray(rejected_rating_values)
    is_equal = list(chosen_values == rejected_values)
    if augment_type == 'single' or augment_type == '84':
        data_subsets = ['8', '4', '2', '1']
        reversed_labels = {data_subsets[idx]: list(random_greater_than_zero(rejected_values - chosen_values))[idx] for
                           idx in range(len(data_subsets))}
        is_equal = {data_subsets[idx]: is_equal[idx] for idx in range(len(data_subsets))}
        return data_subsets, reversed_labels, is_equal
    else:
        raise ValueError('Invalid augment_type')


def inner_join(original, binarized, augment_type, users, two_two_only=False, filter_equal=False):
    agreed_counter = 0
    controversial_counter = 0
    keys = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    user_counter = {key: 0 for key in users.keys()}
    reversed_counter = {key: 0 for key in users.keys()}
    dumb_baseline = {key: 0 for key in users.keys()}
    dumb_controversial_baseline = {key: 0 for key in users.keys()}
    orig_idx = 0
    out_idx = 0
    dataset_dict = {
        'Index': list(),
        'original_idx': list(),
        'prompt': list(),
        'chosen': list(),
        'rejected': list(),
        'data_subset': list(),
        'controversial': list(),
        'reversed': list(),
        'satisfied_subset': list(),
        'survey_options': list(),
    }
    for bin_idx in range(len(binarized)):
        while binarized[bin_idx]['prompt'] != original[orig_idx]['instruction']:
            orig_idx += 1
        prompt = binarized[bin_idx]['prompt']
        chosen = binarized[bin_idx]['chosen'][1]['content']
        rejected = binarized[bin_idx]['rejected'][1]['content']
        if chosen == '' or rejected == '':
            continue
        chosen_ratings = dict()
        rejected_ratings = dict()
        flag = True
        for c in original[orig_idx]['completions']:
            if c['response'] == chosen:
                for key in keys:
                    r = c['annotations'][key]['Rating']
                    if r == 'N/A':
                        flag = False
                        continue
                    chosen_ratings[key] = int(r)
            elif c['response'] == rejected:
                for key in keys:
                    r = c['annotations'][key]['Rating']
                    if r == 'N/A':
                        flag = False
                        continue
                    rejected_ratings[key] = int(r)
            else:
                continue
        if not flag or len(chosen_ratings) != 4 or len(rejected_ratings) != 4:
            continue
        #data_subsets, reversed_labels, is_equal = get_user_type(chosen_ratings, rejected_ratings, augment_type, users)
        data_subsets, reversed_labels, is_equal = get_user_type(chosen_ratings, rejected_ratings, augment_type)
        if filter_equal:
            reversed_labels = {key: reversed_labels[key] for key in data_subsets if not is_equal[key]}
            data_subsets = [key for key in data_subsets if not is_equal[key]]
            is_equal = {key: False for key in data_subsets}
            if augment_type == '84' and len(is_equal.keys()) != 2:
                continue
        for data_subset in users.keys():
            if data_subset not in data_subsets:
                dumb_baseline[data_subset] += 0.5 * len(data_subsets)
                if True in reversed_labels.values() and False in reversed_labels.values():
                    dumb_controversial_baseline[data_subset] += 0.5 * len(data_subsets)
                continue
            user_counter[data_subset] += 1
            if True in reversed_labels.values() and False in reversed_labels.values():
                is_controversial = True
                controversial_counter += 1
            else:
                is_controversial = False
                agreed_counter += 1
            if reversed_labels[data_subset]:
                reversed_counter[data_subset] += 1
                dumb_baseline[data_subset] += list(reversed_labels.values()).count(True)
                if is_controversial:
                    dumb_controversial_baseline[data_subset] += list(reversed_labels.values()).count(True)
            else:
                dumb_baseline[data_subset] += list(reversed_labels.values()).count(False)
                if is_controversial:
                    dumb_controversial_baseline[data_subset] += list(reversed_labels.values()).count(False)
            dataset_dict['Index'].append(out_idx)
            dataset_dict['original_idx'].append(orig_idx)
            dataset_dict['prompt'].append(prompt)
            if not reversed_labels[data_subset]:
                dataset_dict['chosen'].append('Human: ' + prompt + '\n\nAssistant: ' + chosen)
                dataset_dict['rejected'].append('Human: ' + prompt + '\n\nAssistant: ' + rejected)
            else:
                dataset_dict['chosen'].append('Human: ' + prompt + '\n\nAssistant: ' + rejected)
                dataset_dict['rejected'].append('Human: ' + prompt + '\n\nAssistant: ' + chosen)
            dataset_dict['data_subset'].append(data_subset)
            dataset_dict['controversial'].append(is_controversial)
            dataset_dict['reversed'].append(reversed_labels[data_subset])
            satisfied_subset = set([key for key in users.keys() if key not in data_subsets or reversed_labels[key] == reversed_labels[data_subset]])
            dataset_dict['satisfied_subset'].append(satisfied_subset)
            dataset_dict['survey_options'].append(is_controversial and len(data_subsets) == len(users.keys()))
            out_idx += 1
    print(out_idx, agreed_counter, controversial_counter)
    print("User counter:", user_counter)
    print("Reversed counter:", reversed_counter)
    print("Dumb baseline:", dumb_baseline)
    print("Dumb controversial baseline:", dumb_controversial_baseline)
    return Dataset.from_dict(dataset_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-a', '--augment_type', type=str, default='single', help='How to augment data')
    parser.add_argument('-c', '--controversial_only', action='store_true', help='Whether to only generate controversial data')
    parser.add_argument('-n', '--name', type=str, default='P_4', help='name of dataset')
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if args.augment_type == 'single' or args.augment_type == '84':
        user_types = {
            '8': (1, 0, 0, 0),
            '4': (0, 1, 0, 0),
            '2': (0, 0, 1, 0),
            '1': (0, 0, 0, 1),
        }
    else:
        raise ValueError('Invalid augment_type')

    ultra_feedback = load_dataset('openbmb/UltraFeedback')
    binarized_cleaned = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned')
    length = len(binarized_cleaned['train'])
    print(length)
    test_ids = list(np.random.choice(length, int(length * 0.1), replace=False))
    train_split = binarized_cleaned['train'].filter(lambda example, idx: idx not in test_ids, with_indices=True)
    test_split = binarized_cleaned['train'].filter(lambda example, idx: idx in test_ids, with_indices=True)
    print(len(train_split), len(test_split))
    print("start processing train split")
    joined_dataset_train = inner_join(ultra_feedback['train'], train_split, args.augment_type, user_types)
    print("start processing test split")
    joined_dataset_test = inner_join(ultra_feedback['train'], test_split, args.augment_type, user_types)

    output_dir = os.path.join('data', 'UltraFeedback_{}_{}'.format(args.augment_type, args.name))
    for user_type in user_types.keys():
        train_subset = joined_dataset_train.filter(lambda x: x['data_subset'] == user_type)
        test_subset = joined_dataset_test.filter(lambda x: x['data_subset'] == user_type)
        if args.controversial_only:
            train_subset = train_subset.filter(lambda x: x['controversial'] == True)
            test_subset = test_subset.filter(lambda x: x['controversial'] == True)
        print(user_types[user_type], len(train_subset), len(test_subset))
        train_subset.to_json(os.path.join(output_dir, user_type, 'train.jsonl'))
        test_subset.to_json(os.path.join(output_dir, user_type, 'test.jsonl'))

# python -m hidden_context.data_utils.ultrafeedback_augment -a single -n P_4 -c

# python -m hidden_context.data_utils.ultrafeedback_augment -a 84 -n P
