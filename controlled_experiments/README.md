# Controlled experiments

One reader on the 250 primary questions and on 1,000 MedMCQA pathology questions, with every API reply saved.

| Setting | Value |
|---|---|
| Reader | `gpt-4.1-mini` (snapshot `gpt-4.1-mini-2025-04-14`), temperature 0, max 600 output tokens |
| Stopping model (decoupled variant) and faithfulness judge | `gpt-4o-mini` (snapshot `gpt-4o-mini-2024-07-18`) |
| Corpus | *Robbins and Cotran Atlas of Pathology*, 4th ed. (1,398 chunks of 180 words, 40-word overlap) |
| Retriever | `all-mpnet-base-v2`, FAISS exact L2, top-5 |
| Conditions | No-RAG; single-pass RAG; RAG-SEQ with caps of 1, 2 and 3 rounds (read from one trajectory); decoupled stopping model |
| RAG-SEQ cap | a reader still flagging a gap at the cap is asked once for a final answer |
| Answer rule | the system prompt asks for a last line `FINAL ANSWER: <letter>`, read by a strict parser (`parse_new` in the notebooks) |

## Files

`files/` — every input and output, fingerprinted in [`MANIFEST.md`](MANIFEST.md). File names are those written by the notebooks.

| File | Content |
|---|---|
| `clean250.csv` | the 250 unique primary questions |
| `medmcqa_set.csv` | 250 MedMCQA questions per domain in random order; the analysed set is the first 200 of each domain |
| `r3_gpt-4.1-mini_primary.jsonl`, `r3_gpt-4.1-mini_medmcqa.jsonl` | per-question replies under every condition (No-RAG, single-pass, RAG-SEQ rounds with forced answers, decoupled) |
| `r3_relevance_primary.jsonl` | relevance labels of the top-10 retrieved passages, primary set |
| `r3A_forced_answers_medmcqa.jsonl` | follow-up request for a final answer for the 19 MedMCQA replies without a parseable answer |
| `r3B_faithfulness_primary.jsonl`, `r3B_faithfulness_medmcqa.jsonl` | faithfulness judgement of every RAG-SEQ justification |
| `api_cache.jsonl`, `api_cache_runs_A_B.jsonl` | every API reply: text, token counts, cost, latency and the model snapshot that served it |
| `MMIRAG_R3_results.xlsx`, `*.csv`, `r3_runs_A_B_summary.txt` | tables produced by the notebooks |

`notebooks/` — the notebooks as they were run, with their outputs:

| Notebook | Content |
|---|---|
| `controlled_experiments.ipynb` | all conditions on both question sets, retrieval relevance, statistics (27 October 2026) |
| `followup_and_faithfulness.ipynb` | follow-up answers for unparsed replies and the faithfulness audit (29 October 2026) |
| `leakage_and_image_checks.ipynb` | n-gram leakage check against a corpus and image-overlap check (no API access needed) |

## Checking the results

```bash
python verify_results.py files
```

The script is independent of the notebooks and needs only the Python standard library. It checks every file
against its SHA-256 fingerprint in `MANIFEST.md` and recomputes, from the raw replies, the accuracies, the McNemar
tests, the sensitivity analyses, the faithfulness counts and the API usage totals.

No API key is stored in any file; the notebooks read it from the Colab secrets panel.
