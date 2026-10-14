# Retrieval

**Image retrieval (multimodal):** the full PubMedCLIP-based image-similarity
pipeline is implemented in [`../evaluation/Pipeline.ipynb`](../evaluation/Pipeline.ipynb)
(and its extracted form `../evaluation/pipeline.py`). It encodes the query
figure with PubMedCLIP and retrieves the most similar figure–caption pairs from
the knowledge base, returning the caption and adjacent text lines.

**Text retrieval:** dense retrieval over the chunked knowledge base
(chunk = 180 words, 40-word overlap) using `all-mpnet-base-v2` (768-dimensional
Sentence-BERT embeddings) with a FAISS `IndexFlatL2` (L2 / Euclidean distance),
top-k = 5. The reranking and hybrid-retrieval ablations reported in the revision
(biomedical bi-encoder, MedCPT cross-encoder, BM25 + dense RRF) are summarised in
`../results/retriever_ablation_summary.csv`.
