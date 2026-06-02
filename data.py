# Parts of this script are adapted from 
# https://github.com/zlin7/UQ-NLG/

import functools

import datasets
import numpy as np

NQ_PATH = "google-research-datasets/nq_open"
TRIVIA_QA_PATH = "mandarjoshi/trivia_qa"
TRIVIA_QA_CONFIG = "rc.nocontext"
XSUM_PATH = "LM-Polygraph/xsum"
XSUM_CONFIG = "continuation"

NQ_FEWSHOT_SEED = 42
NQ_FEWSHOT_K = 5
TRIVIA_QA_FEWSHOT = ("In Scotland a bothy/bothie is a?", "House")

QA_HEADER = "Answer these questions:\n\n"
QA_QUESTION_DELIMITER = "\n\nQuestion:\n"
QA_ANSWER_TAG = "\nAnswer:"
QA_CHOICES_TAG = "\n\nChoices:"

XSUM_TEXT_MARKER = (
    "Here's the text and it's short one-sentence summary.\n\nText:"
)
XSUM_SUMMARY_MARKER = "Summary (one sentence):"

NONE_OF_THE_ABOVE = "None of the above"


def format_qa_shot(question, answer):
    return f"Question:\n{question}\nAnswer:\n{answer}\n\n"


def format_qa_query(question):
    return f"Question:\n{question}\nAnswer:\n"


def extract_question_part(prompt_input):
    return prompt_input.split(QA_QUESTION_DELIMITER)[-1].replace(
        QA_ANSWER_TAG, QA_CHOICES_TAG
    )


def extract_xsum_text(prompt_input):
    return (
        prompt_input.split(XSUM_TEXT_MARKER)[1]
        .split(XSUM_SUMMARY_MARKER)[0]
        .strip()
    )


def build_qa_judge_prompt(question_part, choice_str, none_label):
    return (
        "Task:\nSelect the one correct answer to the question from "
        "the choices provided. If none of the provided choices is "
        f"correct, select the final choice ({none_label}) {NONE_OF_THE_ABOVE}"
        ".\n\n"
        f"Question:\n{question_part}{choice_str}\n\n"
        "Answer:\nThe answer is ("
    )


def build_xsum_judge_prompt(text_part, choice_str, none_label):
    return (
        "Task:\nSelect the one correct summary for the text from the "
        "choices provided. If none of the provided choices is "
        f"correct, select the final choice ({none_label}) {NONE_OF_THE_ABOVE}"
        ".\n\n"
        f"Text:\n{text_part}\n\nChoices:\n{choice_str}\n\n"
        "Answer:\nThe summary is ("
    )


def load_input_prompts(dataset):
    if dataset == "nq":
        @functools.lru_cache()
        def get_fewshot_prefix():
            train = datasets.load_dataset(
                NQ_PATH, split="train", trust_remote_code=True
            )
            indices = np.random.RandomState(NQ_FEWSHOT_SEED).choice(
                len(train), NQ_FEWSHOT_K
            )
            prefix = ""
            for i in indices.tolist():
                prefix += format_qa_shot(
                    train[i]["question"], train[i]["answer"][0]
                )
            return prefix

        def sample_to_prompt(sample):
            if isinstance(sample["question"], list):
                return [
                    sample_to_prompt({"question": q})
                    for q in sample["question"]
                ]
            return (
                QA_HEADER
                + get_fewshot_prefix()
                + format_qa_query(sample["question"])
            )

        val = datasets.load_dataset(
            NQ_PATH, split="validation", trust_remote_code=True
        )
        return [sample_to_prompt(sample) for sample in val]

    if dataset == "trivia_qa":
        raw = datasets.load_dataset(
            TRIVIA_QA_PATH,
            TRIVIA_QA_CONFIG,
            split="validation",
            trust_remote_code=True,
        )
        id_mem = set()

        def remove_dups(batch):
            if batch["question_id"][0] in id_mem:
                return {col: [] for col in batch.keys()}
            id_mem.add(batch["question_id"][0])
            return batch

        deduped = raw.map(
            remove_dups, batch_size=1, batched=True, load_from_cache_file=False
        )
        shot_q, shot_a = TRIVIA_QA_FEWSHOT
        prefix = QA_HEADER + format_qa_shot(shot_q, shot_a)
        return [prefix + format_qa_query(sample["question"]) for sample in deduped]

    if dataset == "xsum":
        test = datasets.load_dataset(XSUM_PATH, XSUM_CONFIG, split="test")
        return list(test["input"])

    raise ValueError(f"unknown dataset: {dataset!r}")
