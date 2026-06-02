# Parts of this script are adapted from 
# https://github.com/zlin7/UQ-NLG/

import copy
import gc
from typing import List

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from torch.nn import DataParallel
from tqdm import tqdm

import data


def fill_off_diagonals(a: int, fill_tensor: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(
        a, a, 3, dtype=fill_tensor.dtype, device=fill_tensor.device
    )
    eye = torch.eye(a, dtype=torch.bool, device=fill_tensor.device)
    result[~eye] = fill_tensor
    return result


def first_indices(seq):
    seen = set()
    indices = []
    for i, value in enumerate(seq):
        if value not in seen:
            seen.add(value)
            indices.append(i)
    return indices


def create_semantic_sets(entry):
    generated_texts = entry["mapping"]
    semantic_mat = entry["semantic_mat"].argmax(axis=-1)
    unique_generated_texts = sorted(list(set(generated_texts)))
    semantic_set_ids = {ans: i for i, ans in enumerate(unique_generated_texts)}
    for i, ans_i in enumerate(unique_generated_texts):
        for j, ans_j in enumerate(unique_generated_texts[i + 1 :], i + 1):
            if semantic_mat[ans_i, ans_j] + semantic_mat[ans_j, ans_i] >= 2:
                semantic_set_ids[ans_j] = semantic_set_ids[ans_i]
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    relabel = {}
    ret = []
    for ans in list_of_semantic_set_ids:
        if ans not in relabel:
            relabel[ans] = len(relabel)
        ret.append(relabel[ans])
    return ret


def cluster_answers(
    generation_greedy,
    generation_sample,
    *,
    nli_model: str,
    nli_batch_size_per_gpu: int,
) -> List[List[str]]:
    generation_plus = [
        copy.deepcopy(generation_greedy[i]["output"])
        + copy.deepcopy(s["output"])
        for i, s in enumerate(generation_sample)
    ]
    generation_plus_clean = [
        [a for a in row if a.strip() != ""] for row in generation_plus
    ]

    sample_flat = []
    mapping_list = []
    for gen_row in tqdm(generation_plus, desc="building NLI pairs"):
        answers = [out.strip() for out in gen_row if out.strip() != ""]
        unique_ans = sorted(list(set(answers)))
        ans_to_id = {ans: i for i, ans in enumerate(unique_ans)}
        mapping_list.append([ans_to_id[a] for a in answers])
        for i, ans_i in enumerate(unique_ans):
            for j, ans_j in enumerate(unique_ans):
                if i == j:
                    continue
                sample_flat.append((ans_i, ans_j))

    cross_encoder = CrossEncoder(model_name=nli_model)
    if torch.cuda.device_count() > 1:
        cross_encoder.model = DataParallel(cross_encoder.model)
    cross_encoder.model.eval()

    num_gpu = max(1, torch.cuda.device_count())
    nli_init_batch_size = nli_batch_size_per_gpu * num_gpu

    if len(sample_flat) > 0:
        nli_logits = cross_encoder.predict(
            sample_flat,
            batch_size=nli_init_batch_size,
            convert_to_tensor=False,
            show_progress_bar=False,
        )
        nli_logits = torch.from_numpy(nli_logits)
    else:
        nli_logits = torch.zeros(0, 3)

    semantic_mat_list = []
    for mapping_row in tqdm(mapping_list, desc="filling NLI matrices"):
        num_valid = len(set(mapping_row))
        if num_valid > 0:
            num_pairs = num_valid * (num_valid - 1)
            sample_logits = nli_logits[:num_pairs].clone()
            semantic_mat = fill_off_diagonals(num_valid, sample_logits)
            nli_logits = nli_logits[num_pairs:]
        else:
            semantic_mat = torch.zeros(0, 0, 3)
        semantic_mat_list.append(
            {"mapping": mapping_row, "semantic_mat": semantic_mat}
        )

    del cross_encoder
    gc.collect()
    torch.cuda.empty_cache()

    semantic_set = [
        create_semantic_sets(s) if len(s["mapping"]) > 0 else []
        for s in semantic_mat_list
    ]
    unique_set = [first_indices(s) for s in semantic_set]

    unique_set_choice = [
        [generation_plus_clean[i][j].strip() for j in unique_set[i]]
        + [data.NONE_OF_THE_ABOVE]
        for i in range(len(generation_greedy))
    ]
    return unique_set_choice
