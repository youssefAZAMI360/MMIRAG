# MMIRAG

**Multimodal Medical Iterative RAG**
*API prompts, pipeline, and per-question outputs*

---

## 1. Overview

MMIRAG is a Multimodal Medical Iterative Retrieval-Augmented Generation pipeline for medical multiple-choice question answering. It combines dense text retrieval, iterative query refinement (RAG-SEQ), and PubMedCLIP-based image-similarity search to ground model reasoning in domain-specific medical literature.

**Experimental conditions.** The primary 360-question benchmark (300 text-only, 60 image-paired) evaluates **two** conditions: a **No-RAG** parametric baseline and the full **RAG-SEQ** iterative pipeline. The **single-pass RAG** condition is isolated separately on the external **MedMCQA** replication (1,101 questions), where it is compared directly against both No-RAG and RAG-SEQ. Per-question outputs for all of these are in this repository (see §5).

---

## 2. Model configuration

| Model | API identifier | Provider | Modality |
|-------|---------------|----------|----------|
| GPT-5 | `gpt-5` | OpenAI | Text + Image |
| Gemini-2.5 | `gemini-2.5-pro` | Google | Text + Image |
| Grok-4 | `grok-4` | xAI | Text + Image |

**Decoding parameters.** Grok-4 and Gemini-2.5 were queried at `temperature = 0` for deterministic output. The `gpt-5` endpoint accepts only its default temperature (1.0), so GPT-5 was run at that default; all other parameters were held constant across models. Queries use the ChatCompletion-style API; images are base64-encoded in the user message content array.

**Retrieval parameters.**

| Parameter | Value |
|-----------|-------|
| Text embedding model | `all-mpnet-base-v2` (Sentence-BERT, **768-dimensional**) |
| Index / distance | FAISS `IndexFlatL2` (L2 / Euclidean) |
| Chunking | 180 words, 40-word overlap |
| Top-k documents | 5 per retrieval pass |
| Max RAG-SEQ iterations | 3 |
| Image similarity model | PubMedCLIP |

---

## 3. System prompt (all conditions)

See [`prompts/system_prompt.txt`](prompts/system_prompt.txt).

## 4. Condition prompts

- **No-RAG baseline:** [`prompts/no_rag_prompt.txt`](prompts/no_rag_prompt.txt)
- **Single-pass RAG** (MedMCQA replication): [`prompts/rag_single_pass_prompt.txt`](prompts/rag_single_pass_prompt.txt)
- **RAG-SEQ** iterative refinement: [`prompts/rag_seq_iteration1.txt`](prompts/rag_seq_iteration1.txt) (initial retrieval + GAP analysis) and [`prompts/rag_seq_iterationN.txt`](prompts/rag_seq_iterationN.txt) (query-refinement iterations). The loop emits `GAP_IDENTIFIED: <...>` to continue and terminates on a direct answer or after 3 iterations.

The multimodal (PubMedCLIP) image-retrieval content-array structure is documented inline in the pipeline notebook.

---

## 5. Data and results

```
data/
├── FINALE.xlsx               # Primary benchmark: 300 text + 60 image MCQs,
│                             #   per-question No-RAG and RAG-SEQ answers +
│                             #   explanations, for GPT-5 / Grok-4 / Gemini-2.5
└── medmcqa_results.csv       # MedMCQA replication (1,101 Q): per-question
                              #   No-RAG, single-pass RAG, and RAG-SEQ answers
results/
├── retriever_ablation_summary.csv      # general vs biomedical vs MedCPT vs hybrid
├── retriever_ablation_perquestion.csv  # per-question correctness + overlap
├── cap_sweep_summary.csv               # RAG-SEQ cap = 1/2/3: acc, rounds, cost
└── cap_sweep_perquestion.csv           # per-question rounds / stop reason / tokens
```

`FINALE.xlsx` columns: `number`, `Pathology` (domain), `Description` (stem), `A`–`E` (options), `Correct answer`, then answer blocks **No-RAG** and **RAG-SEQ** (one column per model) and the matching explanation blocks.

---

## 6. Output parsing

See [`evaluation/parse_results.py`](evaluation/parse_results.py). The answer letter is the first standalone `A`–`E` at the start of the response or after a newline.

---

## 7. Repository structure

```
MMIRAG/
├── prompts/           # System + per-condition prompt templates (§3–§4)
│   ├── system_prompt.txt
│   ├── no_rag_prompt.txt
│   ├── rag_single_pass_prompt.txt
│   ├── rag_seq_iteration1.txt
│   └── rag_seq_iterationN.txt
├── retrieval/
│   └── README.md      # Text + image retrieval description
├── evaluation/
│   ├── Pipeline.ipynb # Full multimodal pipeline (PubMedCLIP image retrieval)
│   ├── pipeline.py    # Extracted Python source of the notebook
│   └── parse_results.py
├── data/
│   ├── FINALE.xlsx
│   └── medmcqa_results.csv
├── results/
│   ├── retriever_ablation_summary.csv
│   ├── retriever_ablation_perquestion.csv
│   ├── cap_sweep_summary.csv
│   └── cap_sweep_perquestion.csv
└── README.md
```

---

## 8. Citation

```bibtex
@article{azami2026mmirag,
  title   = {RAG Sensitivity in Medical Question Answering:
             Domain-Aware Evaluation Using the MMIRAG Sequential
             Retrieval-Augmented Generation Pipeline},
  author  = {Azami, Youssef and Riffi, Jamal},
  year    = {2026}
}
```

🔗 https://github.com/youssefAZAMI360/MMIRAG
