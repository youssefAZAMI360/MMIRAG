# Retrieval

**Image retrieval (multimodal):** the full PubMedCLIP-based image-similarity
pipeline is implemented in [`../evaluation/Pipeline.ipynb`](../evaluation/Pipeline.ipynb)
(and its extracted form `../evaluation/pipeline.py`). It encodes the query
figure with PubMedCLIP and retrieves the most similar figure–caption pairs from
the knowledge base, returning the caption and adjacent text lines.

**Text retrieval:** dense retrieval over the chunked knowledge base
(chunk = 180 words, 40-word overlap) using `all-mpnet-base-v2` (768-dimensional
Sentence-BERT embeddings) with a FAISS `IndexFlatL2` (L2 / Euclidean distance),
top-k = 5. The reranking and hybrid-retrieval ablations reported in the paper
(biomedical bi-encoder, MedCPT cross-encoder, BM25 + dense RRF) are summarised in
`../results/retriever_ablation_summary.csv`.

**Figure 3 example:** [`../evaluation/Pipeline_figure3_run.ipynb`](../evaluation/Pipeline_figure3_run.ipynb)
is the same notebook with the saved output of the run shown in Figure 3 of the paper:
query image from the WebPath Cellular Injury image quiz (question 1), image corpus
Chapter 1 of *Pathologic Basis of Veterinary Disease* (6th ed., 2017; 1,396 extracted
images). Figure 3 shows the third-ranked match (page 25, cosine similarity 92.77%) and
the raw text dump printed by the notebook for it. The text around a matched image is
collected as the `n_lines` lines above and below it, ordered by their vertical position
on the page regardless of columns (`get_page_text_blocks`), so on two-column pages lines
of the two columns are interleaved.
