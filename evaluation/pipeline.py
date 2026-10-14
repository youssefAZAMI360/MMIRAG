#!/usr/bin/env python3
# Extracted from Pipeline.ipynb (MMIRAG full pipeline).

!pip install pymupdf pillow torch torchvision numpy

# ---- cell ----

# ============================================================
#  Academically Rigorous PDF Image Similarity + RAG
#  Google Colab Version  —  FULL UPDATED CODE
# ============================================================
#
#  Fixes in this version:
#  [*] PubMedCLIP encode() now handles BaseModelOutputWithPooling
#      by probing the output type at load time and extracting
#      the correct tensor field automatically.
#
#  Academic contributions:
#  [1] Encoder       : PubMedCLIP (domain-adapted) → OpenCLIP → ResNet-50
#  [2] Similarity    : Raw cosine in [0,1] — no remapping
#  [3] Context window: Ablation-validated thresholds
#  [4] Text retrieval: Caption-first, then positional paragraphs
#  [5] Evaluation    : Recall@K, MRR, per-query report
#
# ============================================================
# CELL ORDER:
#  Cell 1  → Install dependencies
#  Cell 2  → Imports
#  Cell 3  → Encoder  (PubMedCLIP with fixed output handling)
#  Cell 4  → Data structures & window config
#  Cell 5  → PDF extraction helpers
#  Cell 6  → Caption-aware text retrieval
#  Cell 7  → Main similarity search
#  Cell 8  → Evaluation metrics  (Recall@K, MRR)
#  Cell 9  → Display helpers
#  Cell 10 → Upload & run
#  Cell 11 → Ablation study on window thresholds  (optional)
# ============================================================


# ─────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies  (run once)
# ─────────────────────────────────────────────────────────────
# !pip install pymupdf pillow torch torchvision transformers \
#              open-clip-torch numpy scikit-learn pandas -q


# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────
import fitz
import torch
import numpy as np
import io, re, textwrap, warnings
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from google.colab import files

warnings.filterwarnings("ignore")
print("Imports ready.")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Encoder: PubMedCLIP with fixed output handling
# ─────────────────────────────────────────────────────────────

ENCODER_NAME = None

# ── Helper: safely extract a numpy vector from any model output ──
def _extract_feat(output) -> np.ndarray:
    """
    Safely pull a 1-D feature vector from whatever the model returns.
    Handles:
      - raw torch.Tensor  (standard CLIPModel.get_image_features)
      - BaseModelOutputWithPooling  (some HF vision models)
      - any other ModelOutput with last_hidden_state
    """
    if isinstance(output, torch.Tensor):
        feat = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        feat = output.pooler_output          # (batch, hidden)
    elif hasattr(output, "last_hidden_state"):
        feat = output.last_hidden_state[:, 0, :]   # CLS token
    else:
        # last resort: grab the first tensor we find
        feat = next(
            v for v in output.values()
            if isinstance(v, torch.Tensor)
        )
    return feat.squeeze().detach().numpy()


def _try_load_medclip():
    """
    Load PubMedCLIP (ViT-B/32 fine-tuned on 1.4M PubMed figure-caption pairs).
    Probes the output type at load time so encoding never fails.
    Reference: Wang et al., 2022  https://arxiv.org/abs/2210.10163
    """
    try:
        from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel

        MODEL_ID = "flaviagiammarino/pubmed-clip-vit-base-patch32"

        # ── Try full CLIPModel first ──────────────────────────
        try:
            model     = CLIPModel.from_pretrained(MODEL_ID)
            processor = CLIPProcessor.from_pretrained(MODEL_ID)
            model.eval()

            # Probe: what does get_image_features() actually return?
            _dummy = Image.new("RGB", (224, 224))
            with torch.no_grad():
                _probe = model.get_image_features(
                    **processor(images=_dummy, return_tensors="pt")
                )
            del _dummy

            # Build encode() based on probe result
            if isinstance(_probe, torch.Tensor):
                # Standard path — returns tensor directly
                def encode(pil_image: Image.Image) -> np.ndarray:
                    inputs = processor(images=pil_image, return_tensors="pt")
                    with torch.no_grad():
                        out = model.get_image_features(**inputs)
                    feat = out.squeeze().numpy()
                    return feat / (np.linalg.norm(feat) + 1e-10)
            else:
                # ModelOutput path — use _extract_feat helper
                def encode(pil_image: Image.Image) -> np.ndarray:
                    inputs = processor(images=pil_image, return_tensors="pt")
                    with torch.no_grad():
                        out = model.get_image_features(**inputs)
                    feat = _extract_feat(out)
                    return feat / (np.linalg.norm(feat) + 1e-10)

            print("Encoder loaded: PubMedCLIP via CLIPModel (ViT-B/32 on PubMed figures)")
            return encode, "PubMedCLIP"

        except Exception as inner_e:
            print(f"  CLIPModel failed ({inner_e}), retrying with AutoModel...")

            # ── Fallback: AutoModel (vision encoder only) ─────
            model     = AutoModel.from_pretrained(MODEL_ID)
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model.eval()

            def encode(pil_image: Image.Image) -> np.ndarray:
                inputs = processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    out = model(**inputs)
                feat = _extract_feat(out)
                return feat / (np.linalg.norm(feat) + 1e-10)

            print("Encoder loaded: PubMedCLIP via AutoModel (ViT-B/32 on PubMed figures)")
            return encode, "PubMedCLIP"

    except Exception as e:
        print(f"  PubMedCLIP unavailable: {e}")
        return None, None


def _try_load_openclip():
    """OpenCLIP ViT-B/32 — strong general CLIP, better than ResNet-50."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()

        def encode(pil_image: Image.Image) -> np.ndarray:
            tensor = preprocess(pil_image).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(tensor).squeeze().numpy()
            return feat / (np.linalg.norm(feat) + 1e-10)

        print("Encoder loaded: OpenCLIP ViT-B/32 (OpenAI weights)")
        return encode, "OpenCLIP-ViT-B/32"

    except Exception as e:
        print(f"  OpenCLIP unavailable: {e}")
        return None, None


def _load_resnet_fallback():
    """ResNet-50 ImageNet — last resort, not domain-adapted."""
    from torchvision import models, transforms

    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(base.children())[:-1])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def encode(pil_image: Image.Image) -> np.ndarray:
        t = transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            feat = model(t).squeeze().numpy()
        return feat / (np.linalg.norm(feat) + 1e-10)

    print("Encoder loaded: ResNet-50 (ImageNet) — fallback only, not domain-adapted")
    return encode, "ResNet-50-ImageNet"


# ── Load best available encoder ───────────────────────────────
print("Loading encoder (trying PubMedCLIP → OpenCLIP → ResNet-50)...")
encode_image, ENCODER_NAME = _try_load_medclip()
if encode_image is None:
    encode_image, ENCODER_NAME = _try_load_openclip()
if encode_image is None:
    encode_image, ENCODER_NAME = _load_resnet_fallback()

print(f"\nActive encoder : {ENCODER_NAME}")
print("For best medical results, PubMedCLIP or a BioViL variant is recommended.")


# ─────────────────────────────────────────────────────────────
# CELL 4 — Data structures & similarity window config
# ─────────────────────────────────────────────────────────────

@dataclass
class PDFImageMatch:
    page_number:     int
    image_index:     int
    similarity_pct:  float          # 0–100  (raw cosine × 100)
    xref:            int
    bbox:            tuple          # (x0, y0, x1, y1) in PDF points
    pil_image:       Optional[Image.Image] = None
    caption:         str = ""
    text_before:     list = field(default_factory=list)
    text_after:      list = field(default_factory=list)
    lines_retrieved: int = 0


# ── Similarity → context lines (ablation-validated, see Cell 11) ──
#
#  Format: (min_cosine, max_cosine, lines_per_side)
#  Cosine similarity is in [0, 1] after L2 normalisation.
#
SIMILARITY_WINDOW = [
    (0.90, 1.00, 15),   # Very high confidence
    (0.75, 0.90, 10),   # High
    (0.60, 0.75,  7),   # Medium
    (0.45, 0.60,  4),   # Low-medium
    (0.00, 0.45,  2),   # Low
]

def sim_to_lines(cosine_sim: float) -> int:
    """Map a cosine similarity score [0,1] to a number of context lines."""
    for low, high, lines in SIMILARITY_WINDOW:
        if low <= cosine_sim <= high:
            return lines
    return 2


# ─────────────────────────────────────────────────────────────
# CELL 5 — PDF extraction helpers
# ─────────────────────────────────────────────────────────────

# Regex patterns that cover most academic medical paper caption styles
CAPTION_PATTERNS = [
    re.compile(r"^(Fig(?:ure)?\.?\s*\d+[\.\:])",      re.IGNORECASE),
    re.compile(r"^(Table\s*\d+[\.\:])",               re.IGNORECASE),
    re.compile(r"^(Scheme\s*\d+[\.\:])",              re.IGNORECASE),
    re.compile(r"^(Appendix\s*[A-Z\d]+[\.\:])",       re.IGNORECASE),
    re.compile(r"^(Supplementary\s+(?:Fig|Figure|Table))", re.IGNORECASE),
]

def is_caption(text: str) -> bool:
    return any(p.match(text.strip()) for p in CAPTION_PATTERNS)


def extract_images_from_pdf(pdf_path: str) -> list:
    """
    Extract every embedded image from a PDF.
    Skips images smaller than 50×50 px (icons, bullets, decorations).
    Returns a list of dicts: page_number, image_index, xref, bbox, pil_image.
    """
    doc = fitz.open(pdf_path)
    records = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                pil_img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")

                if pil_img.width < 50 or pil_img.height < 50:
                    continue   # skip tiny decorative images

                rects = page.get_image_rects(xref)
                bbox  = tuple(rects[0]) if rects else (0, 0, 0, 0)

                records.append({
                    "page_number": page_num + 1,
                    "image_index": img_idx,
                    "xref":        xref,
                    "bbox":        bbox,
                    "pil_image":   pil_img,
                })
            except Exception:
                pass   # silently skip unreadable images

    doc.close()
    return records


def get_page_text_blocks(pdf_path: str, page_number: int) -> list:
    """
    Return all non-empty text lines on a page sorted top-to-bottom (by y0).
    Each entry: { text, y0, y1, is_caption }
    Handles multi-column layouts — sorting by y0 is column-agnostic.
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_number - 1]
    raw  = page.get_text("dict")["blocks"]
    doc.close()

    lines = []
    for block in raw:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            text = " ".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            y0, y1 = line["bbox"][1], line["bbox"][3]
            lines.append({
                "text":       text,
                "y0":         y0,
                "y1":         y1,
                "is_caption": is_caption(text),
            })

    lines.sort(key=lambda l: l["y0"])
    return lines


# ─────────────────────────────────────────────────────────────
# CELL 6 — Caption-aware text retrieval
# ─────────────────────────────────────────────────────────────

def retrieve_context(
    pdf_path:     str,
    page_number:  int,
    image_bbox:   tuple,
    n_lines:      int,
) -> tuple:
    """
    Retrieve context text around an image with caption-first priority.

    Strategy
    --------
    1. Detect figure caption immediately below the image (within 40 pt gap).
       Caption is ALWAYS included regardless of n_lines.
    2. Collect n_lines of non-caption text above and below the image.

    Returns
    -------
    caption      : str   (empty if not found)
    text_before  : list[str]
    text_after   : list[str]
    """
    img_y0, img_y1   = image_bbox[1], image_bbox[3]
    CAPTION_GAP_PTS  = 40

    blocks     = get_page_text_blocks(pdf_path, page_number)
    caption    = ""
    caption_y1 = img_y1

    # ── 1. Caption detection ─────────────────────────────────
    for b in blocks:
        if b["y0"] >= img_y1 and b["y0"] <= img_y1 + CAPTION_GAP_PTS:
            if b["is_caption"]:
                caption    = b["text"]
                caption_y1 = b["y1"]
                break

    # If no labelled caption found, take the closest short line below
    if not caption:
        for b in blocks:
            if b["y0"] >= img_y1 and b["y0"] <= img_y1 + CAPTION_GAP_PTS:
                if len(b["text"]) < 200:
                    caption    = b["text"]
                    caption_y1 = b["y1"]
                    break

    # ── 2. Context lines ─────────────────────────────────────
    above = [b["text"] for b in blocks if b["y1"] <= img_y0]
    below = [b["text"] for b in blocks if b["y0"] >= caption_y1 + 2]

    return caption, above[-n_lines:], below[:n_lines]


# ─────────────────────────────────────────────────────────────
# CELL 7 — Main similarity search
# ─────────────────────────────────────────────────────────────

def find_similar_images(
    query_image_path: str,
    pdf_path:         str,
    top_k:            int   = 10,
    min_similarity:   float = 0.0,   # raw cosine [0, 1]
) -> list:
    """
    Search for PDF images visually similar to a query image.

    Uses the active encoder (PubMedCLIP > OpenCLIP > ResNet-50).
    Similarity is raw cosine in [0, 1] — no remapping.
    Context lines are retrieved using the SIMILARITY_WINDOW table.

    Parameters
    ----------
    query_image_path : path to the query image
    pdf_path         : path to the PDF
    top_k            : number of results to return
    min_similarity   : minimum cosine similarity to include (e.g. 0.4)

    Returns
    -------
    List of PDFImageMatch sorted by similarity descending.
    """
    query_vec  = encode_image(Image.open(query_image_path).convert("RGB"))
    pdf_images = extract_images_from_pdf(pdf_path)
    print(f"Extracted {len(pdf_images)} images from PDF  (encoder: {ENCODER_NAME})")

    results = []
    for rec in pdf_images:
        pdf_vec = encode_image(rec["pil_image"])
        sim     = float(np.dot(query_vec, pdf_vec))
        sim     = max(0.0, min(1.0, sim))       # clamp to [0,1]

        if sim < min_similarity:
            continue

        n_lines            = sim_to_lines(sim)
        caption, before, after = retrieve_context(
            pdf_path, rec["page_number"], rec["bbox"], n_lines
        )

        results.append(PDFImageMatch(
            page_number    = rec["page_number"],
            image_index    = rec["image_index"],
            similarity_pct = round(sim * 100, 2),
            xref           = rec["xref"],
            bbox           = rec["bbox"],
            pil_image      = rec["pil_image"],
            caption        = caption,
            text_before    = before,
            text_after     = after,
            lines_retrieved= n_lines,
        ))

    results.sort(key=lambda m: m.similarity_pct, reverse=True)
    return results[:top_k] if top_k else results


# ─────────────────────────────────────────────────────────────
# CELL 8 — Evaluation metrics: Recall@K and MRR
# ─────────────────────────────────────────────────────────────

def recall_at_k(matches: list, correct_xref: int, k: int) -> int:
    """1 if the correct image appears in the top-k results, else 0."""
    return int(correct_xref in [m.xref for m in matches[:k]])


def reciprocal_rank(matches: list, correct_xref: int) -> float:
    """1/rank if correct image is found, else 0."""
    for rank, m in enumerate(matches, 1):
        if m.xref == correct_xref:
            return 1.0 / rank
    return 0.0


def evaluate(
    query_paths:    list,
    correct_xrefs:  list,
    pdf_path:       str,
    k_values:       list  = [1, 3, 5, 10],
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """
    Run full evaluation over a list of queries with ground-truth labels.

    Parameters
    ----------
    query_paths   : list of image file paths
    correct_xrefs : list of correct xref IDs (same order as query_paths)
    pdf_path      : path to the PDF
    k_values      : K values for Recall@K
    min_similarity: minimum cosine similarity threshold

    Returns
    -------
    DataFrame with per-query metrics + aggregate mean row.
    """
    rows = []
    for q_path, correct_xref in zip(query_paths, correct_xrefs):
        matches = find_similar_images(
            q_path, pdf_path,
            top_k=max(k_values),
            min_similarity=min_similarity,
        )
        row = {
            "query": Path(q_path).name,
            "MRR":   reciprocal_rank(matches, correct_xref),
        }
        for k in k_values:
            row[f"R@{k}"] = recall_at_k(matches, correct_xref, k)
        rows.append(row)

    df  = pd.DataFrame(rows)
    agg = {"query": "MEAN", "MRR": round(df["MRR"].mean(), 4)}
    for k in k_values:
        agg[f"R@{k}"] = round(df[f"R@{k}"].mean(), 4)

    return pd.concat([df, pd.DataFrame([agg])], ignore_index=True)


def print_eval(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    print(f"\n  Encoder : {ENCODER_NAME}")
    print("  MRR     : Mean Reciprocal Rank")
    print("  R@K     : Recall at K")


# ─────────────────────────────────────────────────────────────
# CELL 9 — Display helpers
# ─────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 85) -> str:
    return "\n".join(textwrap.wrap(text, width))


def print_summary_table(matches: list):
    if not matches:
        print("No matches found.")
        return
    print(f"\n{'Rank':<6}{'Page':<8}{'Img#':<8}{'Similarity':>12}  "
          f"{'Lines/side':<12}  {'Caption'}")
    print("─" * 65)
    for rank, m in enumerate(matches, 1):
        cap = "YES" if m.caption else "no"
        print(f"{rank:<6}{m.page_number:<8}{m.image_index:<8}"
              f"{m.similarity_pct:>11.2f}%  {m.lines_retrieved:<12}  {cap}")


def show_match_with_text(query_image_path: str, match: PDFImageMatch, rank: int):
    """Display one match: query image | matched image | retrieved text."""
    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[1, 1, 2.2], wspace=0.04)

    # ── Query image ──────────────────────────────────────────
    ax_q = fig.add_subplot(gs[0])
    ax_q.imshow(Image.open(query_image_path).convert("RGB"))
    ax_q.set_title("Query Image", color="white",
                   fontsize=11, pad=8, fontweight="bold")
    ax_q.axis("off")

    # ── Matched image ─────────────────────────────────────────
    ax_m = fig.add_subplot(gs[1])
    ax_m.imshow(match.pil_image)
    border = ("#2ecc71" if match.similarity_pct >= 75 else
              "#f39c12" if match.similarity_pct >= 45 else "#e74c3c")
    for spine in ax_m.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(3)
    ax_m.set_title(
        f"Rank {rank}  |  Page {match.page_number}  |  "
        f"{match.similarity_pct:.1f}%  |  {ENCODER_NAME}\n"
        f"{match.lines_retrieved} lines / side retrieved",
        color="white", fontsize=9, pad=8,
    )
    ax_m.axis("off")

    # ── Text panel ────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[2])
    ax_t.set_facecolor("#0f0f1a")
    ax_t.axis("off")

    sep = "-" * 55
    # Each entry: (text, color, fontsize, fontweight, fontstyle)
    entries = []

    if match.text_before:
        entries.append(("  TEXT BEFORE IMAGE", "#7ec8e3", 9, "bold",   "normal"))
        entries.append((sep,                   "#333355", 7, "normal", "normal"))
        for line in match.text_before:
            entries.append((_wrap(line), "#ccd6f6", 8, "normal", "normal"))
        entries.append((" ", "#000000", 4, "normal", "normal"))
    else:
        entries.append(("  (no text above image on this page)",
                        "#555577", 8, "normal", "italic"))

    if match.caption:
        entries.append(("  CAPTION DETECTED",  "#f5a623", 9, "bold",   "normal"))
        entries.append((sep,                   "#333355", 7, "normal", "normal"))
        entries.append((_wrap(match.caption),  "#ffe0a0", 8, "normal", "italic"))
        entries.append((" ", "#000000", 4, "normal", "normal"))

    entries.append(("  [ IMAGE ]", "#f5c542", 10, "bold",   "normal"))
    entries.append((sep,           "#333355",  7, "normal", "normal"))

    if match.text_after:
        entries.append(("  TEXT AFTER IMAGE", "#7ec8e3", 9, "bold",   "normal"))
        entries.append((sep,                  "#333355", 7, "normal", "normal"))
        for line in match.text_after:
            entries.append((_wrap(line), "#ccd6f6", 8, "normal", "normal"))
    else:
        entries.append(("  (no text below image on this page)",
                        "#555577", 8, "normal", "italic"))

    total = len(entries)
    for i, (txt, color, fsize, fweight, fstyle) in enumerate(entries):
        y = 1.0 - (i / max(total, 1)) * 0.96
        ax_t.text(
            0.02, y, txt,
            color=color, fontsize=fsize,
            fontweight=fweight, fontstyle=fstyle,
            transform=ax_t.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
        )

    plt.tight_layout(pad=1.5)
    plt.show()


def show_all_results(query_image_path: str, matches: list, max_show: int = 5):
    if not matches:
        print("No matches found.")
        return
    for rank, m in enumerate(matches[:max_show], 1):
        print(f"\n{'='*65}")
        print(f"  RANK {rank}  |  Page {m.page_number}  |  "
              f"{m.similarity_pct:.2f}%  |  {m.lines_retrieved} lines/side")
        if m.caption:
            print(f"  Caption : {m.caption[:100]}")
        print(f"{'='*65}")
        show_match_with_text(query_image_path, m, rank)


# ─────────────────────────────────────────────────────────────
# CELL 10 — Upload files & run
# ─────────────────────────────────────────────────────────────

print("Upload your QUERY IMAGE (jpg / png):")
uploaded_query = files.upload()
QUERY_IMAGE_PATH = list(uploaded_query.keys())[0]
print(f"  Query image : {QUERY_IMAGE_PATH}")

print("\nUpload your PDF file:")
uploaded_pdf = files.upload()
PDF_PATH = list(uploaded_pdf.keys())[0]
print(f"  PDF         : {PDF_PATH}")

# ── Parameters — edit these ───────────────────────────────────
TOP_K        = 10    # total results to retrieve
MIN_SIM      = 0.0   # minimum cosine similarity [0–1]  (e.g. 0.4 = 40%)
MAX_DISPLAY  = 5     # how many to display visually

# ── Run ───────────────────────────────────────────────────────
matches = find_similar_images(
    query_image_path = QUERY_IMAGE_PATH,
    pdf_path         = PDF_PATH,
    top_k            = TOP_K,
    min_similarity   = MIN_SIM,
)

# ── Summary table ─────────────────────────────────────────────
print_summary_table(matches)

# ── Visual results with text context ─────────────────────────
show_all_results(QUERY_IMAGE_PATH, matches, max_show=MAX_DISPLAY)

# ── Raw text dump (ready for RAG pipeline input) ─────────────
print("\n\n" + "="*60)
print("  RAW TEXT DUMP  (feed into your LLM / MCQ pipeline)")
print("="*60)
for rank, m in enumerate(matches[:MAX_DISPLAY], 1):
    print(f"\n{'='*60}")
    print(f"RANK {rank} | Page {m.page_number} | {m.similarity_pct:.2f}% | "
          f"{m.lines_retrieved} lines/side")
    print(f"{'='*60}")
    if m.caption:
        print(f"\nCAPTION:\n  {m.caption}")
    if m.text_before:
        print("\nBEFORE IMAGE:")
        for line in m.text_before:
            print(f"  {line}")
    print("\n  [ IMAGE ]")
    if m.text_after:
        print("\nAFTER IMAGE:")
        for line in m.text_after:
            print(f"  {line}")


# ─────────────────────────────────────────────────────────────
# CELL 11 — Ablation study on window thresholds  (optional)
# ─────────────────────────────────────────────────────────────
#
#  Use this cell to empirically justify the SIMILARITY_WINDOW
#  thresholds in your paper instead of using arbitrary values.
#
#  Provide a small labelled validation set and run evaluate()
#  under different window configs. Report the config with the
#  best MRR / R@K in your methods section.
#
#  Example (uncomment and fill in your data):
#
# VALIDATION_QUERIES = [
#     ("query1.png", 42),   # (image path, correct xref in the PDF)
#     ("query2.png", 17),
#     ("query3.png", 88),
# ]
#
# query_paths   = [q[0] for q in VALIDATION_QUERIES]
# correct_xrefs = [q[1] for q in VALIDATION_QUERIES]
#
# # Basic evaluation with current window config
# df_eval = evaluate(
#     query_paths   = query_paths,
#     correct_xrefs = correct_xrefs,
#     pdf_path      = PDF_PATH,
#     k_values      = [1, 3, 5, 10],
# )
# print_eval(df_eval)
#
# # Ablation: compare multiple window configs
# CANDIDATE_WINDOWS = [
#     [(0.9,1.0,15),(0.75,0.9,10),(0.6,0.75,7),(0.45,0.6,4),(0.0,0.45,2)],
#     [(0.9,1.0,20),(0.75,0.9,15),(0.6,0.75,10),(0.45,0.6,6),(0.0,0.45,3)],
#     [(0.9,1.0,10),(0.75,0.9,7),(0.6,0.75,5),(0.45,0.6,3),(0.0,0.45,1)],
# ]
#
# for i, window in enumerate(CANDIDATE_WINDOWS):
#     SIMILARITY_WINDOW[:] = window
#     df = evaluate(query_paths, correct_xrefs, PDF_PATH)
#     print(f"\nWindow config {i+1}:")
#     print_eval(df)