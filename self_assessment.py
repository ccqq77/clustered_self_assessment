import string
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import data


def choices_to_string(choices):
    labels = string.ascii_uppercase[: len(choices)]
    lines = [f"({lbl}) {item}" for lbl, item in zip(labels, choices)]
    return "\n".join(lines)


def ragged_gather(
    x: List[List[int]], y: torch.Tensor
) -> List[torch.Tensor]:
    if len(x) != y.size(0):
        raise ValueError(f"len(x) = {len(x)} but y has {y.size(0)} rows.")
    out: List[torch.Tensor] = []
    for row_index, col_indices in enumerate(x):
        idx = torch.as_tensor(col_indices, dtype=torch.long, device=y.device)
        out.append(y[row_index].index_select(0, idx))
    return out


def forward_last_logits(model, input_ids, attention_mask):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    return out.logits[:, -1, :].detach().cpu()


def self_assess(
    generation_greedy,
    unique_set_choice,
    *,
    judge_model: str,
    dataset: str,
    judge_batch_size: int,
    huggingface_token: str = None,
) -> List[float]:
    model = AutoModelForCausalLM.from_pretrained(
        judge_model,
        device_map="balanced_low_0",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        token=huggingface_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        judge_model,
        padding_side="left",
        trust_remote_code=True,
        token=huggingface_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    choice_tokens_all = [
        tokenizer.convert_tokens_to_ids(
            list(string.ascii_uppercase[: len(c)])
        )
        for c in unique_set_choice
    ]

    judge_prompts = []
    choice_tokens_valid = []
    for sample_idx, gen in enumerate(generation_greedy):
        if gen["output"][0].strip() == "":
            continue
        choice_str = choices_to_string(unique_set_choice[sample_idx])
        none_label = string.ascii_uppercase[len(unique_set_choice[sample_idx]) - 1]

        if dataset == "xsum":
            text_part = data.extract_xsum_text(gen["input"])
            prompt = data.build_xsum_judge_prompt(
                text_part, choice_str, none_label
            )
        else:
            question_part = data.extract_question_part(gen["input"])
            prompt = data.build_qa_judge_prompt(
                question_part, choice_str, none_label
            )

        judge_prompts.append(prompt)
        choice_tokens_valid.append(choice_tokens_all[sample_idx])

    probs_list = []
    for start in tqdm(
        range(0, len(judge_prompts), judge_batch_size), desc="judge forward"
    ):
        batch = judge_prompts[start : start + judge_batch_size]
        choice_tokens_slc = choice_tokens_valid[start : start + judge_batch_size]

        batch_token = tokenizer(batch, return_tensors="pt", padding=True).to(
            "cuda"
        )
        with torch.inference_mode():
            output_logits = forward_last_logits(
                model,
                batch_token.input_ids,
                batch_token.attention_mask,
            )
        probs = torch.softmax(output_logits, dim=-1)
        probs_list.extend(ragged_gather(choice_tokens_slc, probs))

        del batch_token, output_logits

    scores = []
    idx = 0
    for gen in generation_greedy:
        if gen["output"][0].strip() != "":
            scores.append(probs_list[idx][0].item())
            idx += 1
        else:
            scores.append(0.0)
    return scores
