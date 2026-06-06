import argparse
import os
import pickle

import answer_clustering
import self_assessment


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_arguments()

    with open(os.path.join(args.greedy_dir, "generations.pkl"), "rb") as f:
        generation_greedy = pickle.load(f)
    with open(os.path.join(args.sample_dir, "generations.pkl"), "rb") as f:
        generation_sample = pickle.load(f)

    if len(generation_greedy) != len(generation_sample):
        raise ValueError(
            f"greedy ({len(generation_greedy)}) and sample "
            f"({len(generation_sample)}) lengths differ"
        )

    unique_set_choice = answer_clustering.cluster_answers(
        generation_greedy,
        generation_sample,
        nli_model=args.nli_model,
        nli_batch_size_per_gpu=args.nli_batch_size_per_gpu,
    )

    scores = self_assessment.self_assess(
        generation_greedy,
        unique_set_choice,
        assess_model=args.assess_model,
        dataset=args.dataset,
        assess_batch_size=args.assess_batch_size,
        huggingface_token=args.huggingface_token,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "score.pkl"), "wb") as f:
        pickle.dump(scores, f)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--greedy_dir", type=str, default="./output_greedy")
    parser.add_argument("--sample_dir", type=str, default="./output_sample")
    parser.add_argument(
        "--dataset",
        type=str,
        default="trivia_qa",
        choices=["nq", "trivia_qa", "xsum"],
    )
    parser.add_argument("--assess_model", type=str, required=True)
    parser.add_argument(
        "--nli_model", type=str, default="microsoft/deberta-large-mnli"
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="access token for gated HuggingFace models",
    )
    parser.add_argument("--assess_batch_size", type=int, default=64)
    parser.add_argument(
        "--nli_batch_size_per_gpu",
        type=int,
        default=64,
        help="initial NLI batch size per GPU; total = this * num_gpu",
    )
    parser.add_argument("--output_dir", type=str, default="./score")

    return parser.parse_args()


if __name__ == "__main__":
    main()
