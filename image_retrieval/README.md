# MMIRAG image retrieval

Image-retrieval component of MMIRAG, the multimodal sequential retrieval-augmented generation pipeline for medical question answering.

## What the notebook does

`MMIRAG_image_retrieval.ipynb` (written for Google Colab):

1. Extracts every embedded image of a PDF document with PyMuPDF (images smaller than 50 × 50 px are skipped).
2. Encodes the query image and each extracted image with PubMedCLIP (`flaviagiammarino/pubmed-clip-vit-base-patch32`). OpenCLIP, then ResNet-50, are used only if PubMedCLIP cannot be loaded.
3. Ranks the extracted images by cosine similarity to the query (L2-normalised embeddings).
4. For each match, takes the figure caption detected below the image and the N text lines before and after the image on its page. N depends on the similarity (`SIMILARITY_WINDOW`): 15 lines per side at 0.90 or above, 10 at 0.75–0.90, 7 at 0.60–0.75, 4 at 0.45–0.60 and 2 below 0.45. Lines are ordered by their vertical position on the page (`get_page_text_blocks`), regardless of columns.
5. Shows each match next to the query with its text, and prints the raw text that is passed to the language model ("RAW TEXT DUMP").

## Saved run

The notebook is saved with the outputs of the run shown in Figure 3 of the MMIRAG article.

- Query image: question 1 of the WebPath Cellular Injury image quiz, https://webpath.med.utah.edu/EXAM/IMGQUIZ/clfrm.htm
- PDF: Miller MA, Zachary JF. Mechanisms and Morphology of Cellular Injury, Adaptation, and Death. In: Zachary JF, ed. *Pathologic Basis of Veterinary Disease*, 6th ed. Mosby; 2017:2–43. Freely available at https://pmc.ncbi.nlm.nih.gov/articles/PMC7171462/
- 1,396 images were extracted. The ten best matches have cosine similarities between 92.0% and 94.0%. Figure 3 shows the match ranked third (page 25, Figure 1-18B, 92.77%) with the text retrieved for it.

The query image and the PDF are not included here; both are available at the links above.

In the saved figures, a text line that wraps onto several display lines can be drawn over the next line in the text panel (`show_match_with_text`). This affects the display only; the raw text dump printed at the end of the run gives the retrieved lines in full.

## Running

Open the notebook in Google Colab and run the two cells. The second cell asks you to upload the query image and the PDF; `TOP_K`, `MIN_SIM` and `MAX_DISPLAY` are set in the "Upload files & run" section. Outside Colab, remove the `google.colab` import and set `QUERY_IMAGE_PATH` and `PDF_PATH` directly.

Requirements: Python 3, PyMuPDF, Pillow, PyTorch, torchvision, transformers, NumPy, pandas and matplotlib (open-clip-torch only for the OpenCLIP fallback).
