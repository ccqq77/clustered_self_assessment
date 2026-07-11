# Clustered Self-Assessment (CSA)

Code for the ACL 2026 Findings paper
[**Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models**](https://aclanthology.org/2026.findings-acl.1531/)
([arXiv:2606.03846](https://arxiv.org/abs/2606.03846)).

CSA is a simple yet effective uncertainty quantification method for LLMs. It quantifies uncertainty by eliciting the model's self-assessment: sampled answers are clustered into the semantically distinct options of a multiple-choice question, and the probability the model assigns to its own answer serves as the confidence score.

## Requirements

The direct dependencies are listed in `requirements.txt`. The code has been tested with Python 3.12, PyTorch 2.7, Transformers 4.53, and FlashAttention 2 on CUDA GPUs.

## Pipeline

The pipeline has three steps. The generation scripts share `--model` and every script takes `--dataset` (one of `nq`, `trivia_qa`, `xsum`). Steps 1 and 2 generate answers with the model and write a `generations.pkl`; step 3 clusters those answers and computes the CSA confidence score. Outputs are written under `./output_greedy`, `./output_sample`, and `./score` by default.

For gated HuggingFace models, pass an access token via `--huggingface_token` to `generate.py` or `score.py`, or run `huggingface-cli login` once beforehand.

### 1. Generate the answer to assess (greedy decoding)

```bash
python generate.py \
    --model Qwen/Qwen2.5-7B \
    --dataset trivia_qa \
    --temperature 0.0 \
    --num_return_sequences 1 \
    --max_new_tokens 256 \
    --batch_size 64 \
    --output_dir ./output_greedy
```

### 2. Generate additional samples (temperature sampling)

```bash
python generate.py \
    --model Qwen/Qwen2.5-7B \
    --dataset trivia_qa \
    --temperature 0.5 \
    --top_k 32 \
    --top_p 0.95 \
    --num_return_sequences 8 \
    --max_new_tokens 256 \
    --batch_size 8 \
    --output_dir ./output_sample
```

Using the same script with temperature sampling, this draws additional answers per question that provide the diversity needed for clustering. The number of supplementary samples is set with `--num_return_sequences`.

### 3. Cluster answers and compute CSA scores

```bash
python score.py \
    --greedy_dir ./output_greedy \
    --sample_dir ./output_sample \
    --dataset trivia_qa \
    --assess_model Qwen/Qwen2.5-7B \
    --nli_model microsoft/deberta-large-mnli \
    --assess_batch_size 64 \
    --nli_batch_size_per_gpu 64 \
    --output_dir ./score
```

This uses the greedy and sampled answers and runs the two-stage CSA procedure:

- **Answer clustering.** For each question, the union of answers is grouped into semantically distinct clusters using bidirectional NLI predictions from `--nli_model` (default `microsoft/deberta-large-mnli`). Each cluster contributes one option, and a final "None of the above" option is appended.
- **Self-assessment.** The resulting MCQ is presented to the original LLM (`--assess_model`), and the probability it assigns to the option corresponding to the greedy answer (label "A") is recorded as the confidence score.

## Citation

```bibtex
@inproceedings{cao-etal-2026-clustered,
    title = "Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models",
    author = "Cao, Qi  and
      Kojima, Takeshi  and
      Gambardella, Andrew  and
      Peng, Helinyi  and
      Matsuo, Yutaka  and
      Iwasawa, Yusuke",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1531/",
    doi = "10.18653/v1/2026.findings-acl.1531",
    pages = "30666--30680",
    ISBN = "979-8-89176-395-1"
}
```
