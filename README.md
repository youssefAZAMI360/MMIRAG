# MMIRAG

**Multimodal Medical Iterative RAG**
*Prompts, pipeline, per-question outputs and controlled experiments*

---

## 1. Overview

MMIRAG is a Multimodal Medical Iterative Retrieval-Augmented Generation pipeline for medical multiple-choice question answering. It combines dense text retrieval, iterative query refinement (RAG-SEQ), and PubMedCLIP-based image-similarity search to ground model reasoning in domain-specific medical literature.

The repository contains two groups of experiments:

1. **Frontier models on the primary benchmark** — GPT-5, Gemini-2.5 and Grok-4 on 250 text-only pathology questions (five domains) and 60 image-paired questions, under No-RAG and RAG-SEQ (`data/FINALE.xlsx`).
2. **Controlled experiments** — one reader (`gpt-4.1-mini`) on the same 250 questions and on 1,000 public MedMCQA pathology questions, under No-RAG, single-pass RAG, RAG-SEQ with caps of 1–3 rounds and a decoupled stopping model, with every API reply saved and an independent verification script (`controlled_experiments/`).

---

## 2. Model configuration

| Model | API identifier | Provider | Modality |
|-------|---------------|----------|----------|
| GPT-5 | `gpt-5` | OpenAI | Text + Image |
| Gemini-2.5 | `gemini-2.5-pro` | Google | Text + Image |
| Grok-4 | `grok-4` | xAI | Text + Image |
| Controlled-experiment reader | `gpt-4.1-mini` (snapshot `gpt-4.1-mini-2025-04-14`) | OpenAI | Text |
| Stopping model / faithfulness judge | `gpt-4o-mini` (snapshot `gpt-4o-mini-2024-07-18`) | OpenAI | Text |

**Decoding parameters.** Grok-4, Gemini-2.5 and the controlled-experiment models were queried at `temperature = 0`. The `gpt-5` endpoint accepts only its default temperature (1.0), so GPT-5 was run at that default; all other parameters were held constant across models. Queries use the ChatCompletion-style API; images are base64-encoded in the user message content array.

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

## 3. Prompts

- **System prompt (all conditions):** [`prompts/system_prompt.txt`](prompts/system_prompt.txt)
- **No-RAG baseline:** [`prompts/no_rag_prompt.txt`](prompts/no_rag_prompt.txt)
- **Single-pass RAG:** [`prompts/rag_single_pass_prompt.txt`](prompts/rag_single_pass_prompt.txt)
- **RAG-SEQ:** [`prompts/rag_seq_iteration1.txt`](prompts/rag_seq_iteration1.txt) (initial retrieval + gap analysis) and [`prompts/rag_seq_iterationN.txt`](prompts/rag_seq_iterationN.txt) (refinement rounds). The loop emits `GAP_IDENTIFIED: <...>` to continue and stops on a direct answer or after 3 rounds.

In the controlled experiments the system prompt additionally asks for a last line `FINAL ANSWER: <letter>`, which is read by a strict parser; the exact prompts are printed in the notebooks (`controlled_experiments/notebooks/`). The multimodal (PubMedCLIP) image-retrieval content-array structure is documented inline in the pipeline notebook.

---

## 4. Data and results

```
data/
├── FINALE.xlsx                  # Primary benchmark: text + image MCQs with per-question No-RAG and
│                                #   RAG-SEQ answers and explanations for GPT-5 / Grok-4 / Gemini-2.5
└── medmcqa_results.csv          # Earlier MedMCQA run (1,101 questions) — superseded, see §6
controlled_experiments/
├── README.md                    # Settings and file descriptions
├── MANIFEST.md                  # SHA-256 fingerprint of every file, API usage totals
├── verify_results.py            # Independent recomputation of the main results
├── files/                       # Questions, per-question outputs, all API replies, tables
└── notebooks/                   # The notebooks as they were run, with their outputs
results/
├── retriever_ablation_summary.csv       # general vs biomedical vs MedCPT vs hybrid retriever
├── retriever_ablation_perquestion.csv   # per-question correctness + overlap
├── cap_sweep_summary.csv                # earlier cap analysis — superseded, see §6
└── cap_sweep_perquestion.csv
```

`FINALE.xlsx` columns: `number`, `Pathology` (domain), `Description` (stem), `A`–`E` (options), `Correct answer`, then answer blocks **No-RAG** and **RAG-SEQ** (one column per model) and the matching explanation blocks. Its text sheet has 300 rows: the 50 rows labelled Atherosclerosis duplicate the 50 Immunopathology rows, so all analyses use the 250 unique questions (listed in `controlled_experiments/files/clean250.csv`). The retriever ablation files cover the same 300 rows; restrict them to these 250 questions.

---

## 5. Checking the results

```bash
cd controlled_experiments
python verify_results.py files
```

The script needs only the Python standard library and is independent of the notebooks. It checks every file against its SHA-256 fingerprint in `MANIFEST.md` and recomputes, from the raw replies, the accuracies, the McNemar tests, the sensitivity analyses, the faithfulness counts and the API usage totals.

---

## 6. Output parsing and superseded files

The frontier-model answers in `FINALE.xlsx` were read with [`evaluation/parse_results.py`](evaluation/parse_results.py): the answer is the first standalone `A`–`E` at the start of the response or after a newline. This rule can mistake a sentence-initial article ("A 45-year-old patient ...") for option A. It affected `data/medmcqa_results.csv`, whose accuracies are therefore not valid; the MedMCQA results are those in `controlled_experiments/`, where the prompt asks for an explicit `FINAL ANSWER` line that is read by a strict parser. `results/cap_sweep_*.csv` come from an earlier cap analysis on the 300-row file and are superseded by the cap results in `controlled_experiments/`. Both are kept for reference only.

---

## 7. Repository structure

```
MMIRAG/
├── prompts/                  # System + per-condition prompt templates (§3)
├── retrieval/README.md       # Text + image retrieval description
├── evaluation/
│   ├── Pipeline.ipynb        # Full multimodal pipeline (PubMedCLIP image retrieval)
│   ├── pipeline.py           # Extracted Python source of the notebook
│   └── parse_results.py
├── image_retrieval/          # Image-retrieval notebook with the saved run shown in Figure 3
├── data/                     # Primary benchmark (§4)
├── controlled_experiments/   # gpt-4.1-mini experiments, all API replies, verification (§4–§5)
├── results/                  # Retriever ablation (§4)
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
