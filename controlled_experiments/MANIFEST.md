# MMIRAG controlled experiments: provenance manifest

Controlled experiments run on 27 October 2026 (UTC) in Google Colab with the OpenAI API (notebook `notebooks/controlled_experiments.ipynb`);
runs A and B on 29 October 2026 (UTC) with `notebooks/followup_and_faithfulness.ipynb`, which only reads the files of 27 October and writes new ones.
SHA-256 fingerprints identify each file exactly: changing a single character changes the fingerprint.
Run `python verify_results.py files` to check the files and recompute the main results from the raw outputs.

| File | Bytes | Content | SHA-256 |
|---|---|---|---|
| `MMIRAG_R3_results.xlsx` | 289766 | all summary tables | `90093a3c3e39eee9b71addc93bfbf26b917ccaec0d1968f6a38c39148bdf6951` |
| `api_cache.jsonl` | 6497760 | every API reply: text, token counts, cost, latency, model snapshot that served it | `63d322ce68b7659548faeb93df66c5df7c3250eff87545e7ddcdf5a56d76b1b0` |
| `clean250.csv` | 142765 | the 250 primary questions (clean set) | `68f4adb8796e7a1efcfc7260acb1ec357e00e3584254efd6161dafd6562cf8c1` |
| `medmcqa_set.csv` | 296220 | 1,250 sampled MedMCQA questions (the 1,000 used are the first 200 per domain, in file order) | `11fc05dc7452d7eb76e2e05c5b3577a839e310e38c2c1d37b08a6158e6c68eba` |
| `r3_gpt-4.1-mini_medmcqa.jsonl` | 4698524 | per-question outputs, all conditions, MedMCQA | `8d4753e1a58da9fc390331af54e1d78fa4d668eb8c0dd298d5f4e45ce1ddcca4` |
| `r3_gpt-4.1-mini_medmcqa_contrasts.csv` | 878 | derived table | `79048fb822645bbd993336993f398cdb1faef63af04c15009bd7d7a9fa89885a` |
| `r3_gpt-4.1-mini_medmcqa_domains.csv` | 852 | derived table | `2cffa5b6c25e90c69c13cd919997aa064b68849807d15217effc15a3da62204b` |
| `r3_gpt-4.1-mini_medmcqa_perquestion.csv` | 520086 | derived table | `54030cbb06d1fadfdad5e79a786a15d03e95f25bc7dc85a8859815838f1f4f2f` |
| `r3_gpt-4.1-mini_medmcqa_summary.csv` | 864 | derived table | `36cd696fddd6e8ea6e581d04ba7e6f0715ade1e7744a27d22e58399d4a117cd5` |
| `r3_gpt-4.1-mini_primary.jsonl` | 1335932 | per-question outputs, all conditions, primary set | `a5f0740df109d226bbf457f3c7a17a0620229ef66d3f958462b8d979b00172a1` |
| `r3_gpt-4.1-mini_primary_contrasts.csv` | 627 | derived table | `9453473858f7267a1d776344fd9841953896ca2cb4bc6806d93840c0de5b0b9f` |
| `r3_gpt-4.1-mini_primary_domains.csv` | 847 | derived table | `14735f56bd5d2721b05576309e1c354c542802b9eeb83e50ac341f07ac685ca0` |
| `r3_gpt-4.1-mini_primary_perquestion.csv` | 120389 | derived table | `06d85bf8e2fb5259f371978993e9773dadfaa6b53969615b6ceb2e9e2ecc459f` |
| `r3_gpt-4.1-mini_primary_summary.csv` | 854 | derived table | `87614c10a6f634709b7d37366d036e95a9b34c281ba9bf4014019af048f40a29` |
| `r3_headroom_medmcqa_domain.csv` | 540 | derived table | `0114c9c2f2f28a30e229c1247108855450617af2830ecbfe2cc62123a11a1fe0` |
| `r3_headroom_primary_domain.csv` | 1997 | derived table | `204b727789dfd47d14cb3e06cc67956142ae5338bb981b4c3aefe7a307471bc2` |
| `r3_relevance_primary.jsonl` | 21676 | relevance labels of the top-10 passages, primary set | `699c0551094533e197e5ff73aefa370b9c9c2105e02a793816e09145e6b7e6ac` |
| `r3_relevance_summary.csv` | 263 | derived table | `8327be98395f2563a89319f7fb1a3d8262286bb8bfacd3e6b1bad3517556839b` |

## API usage recorded in the outputs

These totals should agree with the OpenAI usage page and the billing history of the account for 27 October 2026 (the usage page may show slightly more requests if any call was retried).

| Model snapshot | Requests | Input tokens | of which cached | Output tokens | Cost at list price |
|---|---|---|---|---|---|
| gpt-4.1-mini-2025-04-14 | 8,930 | 15,328,942 | 1,508,864 | 616,882 | US$6.67 |
| gpt-4o-mini-2024-07-18 | 2,959 | 8,414,786 | 3,193,088 | 61,065 | US$1.06 |

## Runs A and B (29 October 2026)

| File | Bytes | Content | SHA-256 |
|---|---|---|---|
| `r3A_forced_answers_medmcqa.jsonl` | 13355 | run A: follow-up final answer for the 19 unparsed MedMCQA replies (No-RAG, single-pass, decoupled) | `5716d6430d6eb72378906f721affdf34a400c338dfcd52e95de466de3807bfd9` |
| `r3B_faithfulness_primary.jsonl` | 117482 | run B: faithfulness judgement of every RAG-SEQ (cap 3) justification, primary set | `629127281a0014c35ca69cda1b757925c5963ea9e16ac00428825c3d4636140c` |
| `r3B_faithfulness_medmcqa.jsonl` | 492493 | run B: faithfulness judgement of every RAG-SEQ (cap 3) justification, MedMCQA | `7a8d88b49e39c3ad7e48f4b64bf8cb68ab53d62592bb6ed7f6efff779be5d6e0` |
| `api_cache_runs_A_B.jsonl` | 789204 | every API reply of runs A and B (kept separate so that api_cache.jsonl is unchanged) | `92688f2605a1ca5ba253c1afb1885a0f2572afcb3bddc255e18d8f278dd8ef5b` |
| `r3_runs_A_B_summary.txt` | 2346 | printed summary of runs A and B | `7ebe9b4d54e7a3300e28672447d63b34d7523ddc9836adcf386f6f72883c85f4` |

Every follow-up of run A continues the original conversation; its rebuilt request was found in `api_cache.jsonl` in all 19 cases
(`original_call_matched`). For run B the rebuilt RAG-SEQ context matched the saved requests for all 1,250 questions (`context_matched`).

| Model snapshot | Requests | Input tokens | of which cached | Output tokens | Cost at list price |
|---|---|---|---|---|---|
| gpt-4.1-mini-2025-04-14 | 19 | 39,959 | 0 | 1,497 | US$0.02 |
| gpt-4o-mini-2024-07-18 | 1,250 | 3,401,788 | 55,424 | 80,912 | US$0.55 |

Total for all controlled experiments: US$8.30 (US$7.73 on 27 October, US$0.57 on 29 October).
