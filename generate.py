import argparse
import copy
import gc
import os
import pickle
import types

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

import data

torch._dynamo.config.disable = True


def find_newline_token_ids(tokenizer):
    newline_token = tokenizer.convert_ids_to_tokens(tokenizer.encode("\n"))[-1]
    newline_token_ids = []
    for token, token_id in tokenizer.get_vocab().items():
        if newline_token in token:
            newline_token_ids.append(token_id)
    return newline_token_ids


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_arguments()

    set_seed(args.random_seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        token=args.huggingface_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        trust_remote_code=True,
        token=args.huggingface_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    additional_eos = find_newline_token_ids(tokenizer)
    all_eos = list(set(additional_eos + [tokenizer.eos_token_id]))

    input_prompts = data.load_input_prompts(args.dataset)

    generations = []
    torch.cuda.empty_cache()

    if args.temperature == 0 and args.num_return_sequences > 1:
        print(
            "Warning: temperature=0 uses greedy decoding; "
            "num_return_sequences is ignored (always 1)"
        )

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=all_eos,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        num_beams=1,
        do_sample=args.temperature > 0,
        num_return_sequences=(
            args.num_return_sequences if args.temperature > 0 else 1
        ),
        **(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
            if args.temperature > 0
            else {}
        ),
    )

    for i in tqdm(range(0, len(input_prompts), args.batch_size)):
        x = input_prompts[i : i + args.batch_size]
        x_token = tokenizer(x, return_tensors="pt", padding=True).to("cuda")

        with torch.inference_mode():
            try:
                output = model.generate(
                    inputs=x_token.input_ids,
                    attention_mask=x_token.attention_mask,
                    generation_config=generation_config,
                )
            except RuntimeError as e:
                num_return_seq = (
                    args.num_return_sequences if args.temperature > 0 else 1
                )
                if num_return_seq <= 1:
                    raise

                print(f"[retry] generation failed: {e}")
                del x_token
                try:
                    del output
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

                print(
                    f"[retry] falling back to num_return_sequences=1 "
                    f"(looping {num_return_seq}x)"
                )

                generation_config_single = copy.deepcopy(generation_config)
                generation_config_single.num_return_sequences = 1

                bs_in = len(x)
                total_flat = bs_in * num_return_seq
                all_gen_seqs = [None] * total_flat

                x_token = tokenizer(x, return_tensors="pt", padding=True).to(
                    "cuda"
                )
                fallback_prompt_len = x_token.input_ids.size(1)

                for seq_idx in range(num_return_seq):
                    sub_output = model.generate(
                        inputs=x_token.input_ids,
                        attention_mask=x_token.attention_mask,
                        generation_config=generation_config_single,
                    )
                    gen_part = sub_output.sequences[:, fallback_prompt_len:].to(
                        "cpu"
                    ).detach()

                    for local_b in range(bs_in):
                        j = local_b * num_return_seq + seq_idx
                        all_gen_seqs[j] = gen_part[local_b : local_b + 1]

                    del sub_output
                    gc.collect()
                    torch.cuda.empty_cache()

                max_gen_len = max(s.size(1) for s in all_gen_seqs)
                padded_gen = []
                for s in all_gen_seqs:
                    if s.size(1) < max_gen_len:
                        pad_tensor = torch.full(
                            (1, max_gen_len - s.size(1)),
                            tokenizer.pad_token_id,
                            dtype=s.dtype,
                        )
                        s = torch.cat([s, pad_tensor], dim=1)
                    padded_gen.append(s)
                gen_combined = torch.cat(padded_gen, dim=0)

                prompt_expanded = x_token.input_ids.to("cpu").repeat_interleave(
                    num_return_seq, dim=0
                )
                combined_seqs = torch.cat([prompt_expanded, gen_combined], dim=1)

                output = types.SimpleNamespace(sequences=combined_seqs)

        prompt_len = x_token.input_ids.size(1)
        bs = x_token.input_ids.size(0)
        seqs_per_prompt = output.sequences.size(0) // bs

        seqs_cpu = output.sequences.to("cpu").detach()
        seqs_after = seqs_cpu[:, prompt_len:]

        decoded_flat = tokenizer.batch_decode(
            seqs_after, skip_special_tokens=True
        )
        decoded_txt = [
            decoded_flat[k : k + seqs_per_prompt]
            for k in range(0, len(decoded_flat), seqs_per_prompt)
        ]

        del output

        for prompt, sample_outputs in zip(x, decoded_txt):
            generations.append({"input": prompt, "output": sample_outputs})

        del x_token

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "generations.pkl"), "wb") as f:
        pickle.dump(generations, f)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="access token for gated HuggingFace models",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trivia_qa",
        choices=["nq", "trivia_qa", "xsum"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_greedy",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
