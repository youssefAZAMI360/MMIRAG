# MMIRAG

**Multimodal Medical Iterative RAG**  
*API Prompts & Pipeline Documentation*

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model Configuration](#2-model-configuration)
3. [System Prompt (All Conditions)](#3-system-prompt-all-conditions)
4. [Condition 1: No-RAG Baseline](#4-condition-1-no-rag-baseline-prompt)
5. [Condition 2: Single-Pass RAG](#5-condition-2-single-pass-rag-prompt)
6. [Condition 3: RAG-SEQ — Iterative Refinement](#6-condition-3-rag-seq--iterative-refinement-prompts)
7. [Multimodal RAG Retrieval — Image Similarity](#7-multimodal-rag-retrieval--image-similarity-prompt)
8. [Output Parsing](#8-output-parsing)
9. [Repository Structure](#9-repository-structure)
10. [Citation](#10-citation)

---

## 1. Overview

MMIRAG is a **Multimodal Medical Iterative Retrieval-Augmented Generation** pipeline designed to enhance LLM performance in medical multiple-choice question answering. The pipeline combines dense vector retrieval, iterative query refinement, and PubMedCLIP-based image similarity search to ground model reasoning in verified, domain-specific medical literature.

This document provides the exact API prompts used in all experimental conditions reported in the paper: **No-RAG (baseline)**, **single-pass RAG**, and **RAG-SEQ** (sequential iterative retrieval).

---

## 2. Model Configuration

### 2.1 Models Used

All experiments were conducted via the official API endpoints of the following models:

| Model | API Identifier | Provider | Modality |
|-------|---------------|----------|----------|
| GPT-5 | `gpt-5` | OpenAI | Text + Image |
| Gemini-2.5 | `gemini-2.5-pro` | Google | Text + Image |
| Grok-4 | `grok-4` | xAI | Text + Image |

### 2.2 Shared Parameters

The following parameters were kept identical across all models and all experimental conditions to ensure fair comparison:

- `temperature`: `0` — deterministic output
- `max_tokens`: `1024`
- `top_p`: `1.0`
- All queries submitted via ChatCompletion-style API (`POST /v1/chat/completions` or equivalent)
- Images encoded in base64 and included in the user message content array

---

## 3. System Prompt (All Conditions)

The following system prompt was used identically across No-RAG, RAG, and RAG-SEQ conditions for all three models:

**System Role Prompt:**

```
You are an expert medical professional specialising in pathology.

Your task is to answer multiple-choice questions about medical pathology.

You must reason step by step through the clinical and pathological evidence
before selecting your answer.

Rules:
- Read the full question stem and all answer options carefully.
- If a pathological image is provided, integrate its visual findings
  into your reasoning.
- Return ONLY a single capital letter (A, B, C, D, or E) as your answer,
  followed by a concise medical justification of 2-4 sentences.
- Do not add any text before the answer letter.
- Format: [LETTER] [Justification]
```

---

## 4. Condition 1: No-RAG Baseline Prompt

In the No-RAG condition, the model receives the question and options with **no external context**. This establishes the parametric knowledge baseline.

**User Message (Text-only question):**

```
Answer the following pathology multiple-choice question.

Patient History:
{patient_history}

Question:
{question_stem}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}  [if applicable]

Provide your answer as a single letter followed by your justification.
```

**User Message Content Array (Image-paired question):**

```json
[
  {
    "type": "image_url",
    "image_url": {
      "url": "data:image/jpeg;base64,{base64_encoded_image}"
    }
  },
  {
    "type": "text",
    "text": "Answer the following pathology MCQ.\n\nPatient History:\n{patient_history}\n\nQuestion:\n{question_stem}\n\nOptions:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n\nAnalyse the image and the clinical context carefully before answering."
  }
]
```

---

## 5. Condition 2: Single-Pass RAG Prompt

In the single-pass RAG condition, the top-k retrieved document chunks are injected into the prompt after the question. Retrieval uses **Sentence-BERT (768-dim)** with L2 distance ranking. For image-paired questions, PubMedCLIP retrieval additionally provides figure captions and surrounding text.

**User Message (Single-pass RAG, text-only):**

```
Answer the following pathology multiple-choice question.
Use the retrieved context below to support your reasoning.

--- RETRIEVED CONTEXT (from verified medical literature) ---
{retrieved_chunk_1}

{retrieved_chunk_2}

{retrieved_chunk_3}
--- END RETRIEVED CONTEXT ---

Patient History:
{patient_history}

Question:
{question_stem}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}  [if applicable]

Provide your answer as a single letter followed by your justification.
Ground your justification in the retrieved context where relevant.
```

---

## 6. Condition 3: RAG-SEQ — Iterative Refinement Prompts

RAG-SEQ uses a **multi-turn iterative loop**. Each iteration consists of three steps: (1) retrieve, (2) GAP analysis, (3) query refinement. The loop repeats until the context is focused or a maximum iteration count is reached.

### 6.1 Iteration 1 — Initial Retrieval + GAP Analysis

Iteration 1 uses the same retrieval prompt as single-pass RAG, then appends a GAP analysis instruction:

**GAP Analysis Prompt (appended after initial context):**

```
You have retrieved the context above.

Before answering, perform a GAP ANALYSIS:

1. List any key pathological concepts in the question that are NOT
   addressed by the retrieved context.
2. List any contradictions between retrieved passages.
3. Rate your confidence in answering with current context: LOW / MEDIUM / HIGH.

If confidence is LOW or MEDIUM, respond with:
GAP_IDENTIFIED: [describe the missing information in one sentence]

If confidence is HIGH, proceed directly to your answer.

Format: [LETTER] [Justification]
```

### 6.2 Iteration N — Query Refinement Retrieval

If a GAP is identified, the system generates a refined sub-query and retrieves again. The following prompt is used for all subsequent iterations:

**Refinement Prompt (Iteration 2, 3, ...):**

```
You previously identified the following gap in your knowledge:

GAP: {gap_description_from_previous_iteration}

New targeted context has been retrieved to address this gap:

--- ADDITIONAL RETRIEVED CONTEXT ---
{new_retrieved_chunk_1}

{new_retrieved_chunk_2}
--- END ADDITIONAL CONTEXT ---

Combined with your previous context, now answer the question:

Patient History:
{patient_history}

Question:
{question_stem}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}  [if applicable]

If you still have LOW confidence, respond with another GAP_IDENTIFIED.
Otherwise provide your final answer.

Format: [LETTER] [Justification]
```

### 6.3 Stopping Criterion

The iterative loop terminates under any of the following conditions:

- The model returns a direct answer (letter + justification) without `GAP_IDENTIFIED`
- Maximum iterations reached
- Confidence rating returns `HIGH` in the GAP analysis step

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max iterations | 3 | Balances quality vs. API cost |
| Top-k documents | 5 | Per retrieval pass |
| Embedding model | `all-MiniLM-L6-v2` (Sentence-BERT) | 768-dim dense vectors |
| Distance metric | L2 (Euclidean) | As per pipeline specification |
| Image similarity model | PubMedCLIP | Medical-domain image encoder |
| Context window (image) | 2–15 lines/side | Adaptive based on similarity score |

---

## 7. Multimodal RAG Retrieval — Image Similarity Prompt

For image-paired MCQs, the PubMedCLIP retrieval module runs first. The retrieved figure caption and surrounding text are injected before the standard RAG context block. The following shows the full content array structure:

**User Message Content Array (RAG-SEQ, image-paired):**

```json
[
  {
    "type": "image_url",
    "image_url": { "url": "data:image/jpeg;base64,{base64_query_image}" }
  },
  {
    "type": "text",
    "text": "--- IMAGE RETRIEVAL CONTEXT (PubMedCLIP similarity: {score}%) ---\n
             Figure Caption: {retrieved_figure_caption}\n
             Text before image:\n{text_before}\n
             Text after image:\n{text_after}\n
             --- END IMAGE CONTEXT ---\n\n
             --- TEXT RETRIEVAL CONTEXT ---\n
             {retrieved_text_chunk_1}\n\n
             {retrieved_text_chunk_2}\n
             --- END TEXT CONTEXT ---\n\n
             Patient History:\n{patient_history}\n\n
             Question:\n{question_stem}\n\n
             Options:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n\n
             Integrate the visual and textual evidence in your reasoning."
  }
]
```

---

## 8. Output Parsing

Model outputs are parsed using the following regex pattern to extract the answer letter:

**Python parsing snippet:**

```python
import re

def parse_answer(response_text: str) -> str:
    """
    Extracts the answer letter from model output.
    Returns the first capital letter A-E found at the
    start of the response or after a newline.
    """
    match = re.search(r"(?:^|\n)\s*([A-E])\b", response_text)
    if match:
        return match.group(1)
    # Fallback: any standalone A-E in the response
    match = re.search(r"\b([A-E])\b", response_text)
    return match.group(1) if match else "PARSE_ERROR"
```

---

## 9. Repository Structure

```
MMIRAG/
├── prompts/
│   ├── system_prompt.txt           # Section 3
│   ├── no_rag_prompt.txt            # Section 4
│   ├── rag_single_pass_prompt.txt   # Section 5
│   ├── rag_seq_iteration1.txt       # Section 6.1
│   └── rag_seq_iterationN.txt       # Section 6.2
├── retrieval/
│   ├── text_retrieval.py            # Sentence-BERT + L2
│   └── image_retrieval.py           # PubMedCLIP pipeline
├── evaluation/
│   ├── run_experiment.py            # All 3 conditions
│   └── parse_results.py             # Output parsing
├── data/
│   ├── mcq_text_only/               # 300 text MCQs
│   └── mcq_image_paired/            # 60 image MCQs
└── README.md                        # This document
```

---

## 10. Citation

If you use MMIRAG in your research, please cite:

```bibtex
@article{azami2026mmirag,
  title   = {RAG Sensitivity in Medical Question Answering:
             Domain-Aware Evaluation Using the MMIRAG
             Sequential Retrieval-Augmented Generation Pipeline},
  author  = {Azami, Youssef and Riffi, Jamal},
  journal = {Knowledge-Based Systems},
  year    = {2026}
}
```

🔗 **GitHub:** [https://github.com/youssefAZAMI360/MMIRAG](https://github.com/youssefAZAMI360/MMIRAG)
