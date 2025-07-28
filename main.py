#!/usr/bin/env python3
import os
import sys
import json

import fitz                        # PyMuPDF
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# ─── CONFIG: Set relative paths ───────────────────────────────────────────────
PDF_DIR     = "input"                   # mounted PDFs + prompt.json
INPUT_JSON  = os.path.join(PDF_DIR, "prompt.json")
OUTPUT_JSON = os.path.join("output", "result.json")

XGB_MODEL_PATH = "xgboost_pdf_structure_model.json"
LE_PATH        = "label_encoder.pkl"
PZE_PATH       = "page_zone_encoder.pkl"
META_PATH      = "meta.json"

TOP_K   = 30  # candidates before reranking
FINAL_K = 10  # final sections in output

# ─── LOAD MODELS ───────────────────────────────────────────────────────────────
clf = xgb.XGBClassifier()
clf.load_model(XGB_MODEL_PATH)
with open(LE_PATH, "rb") as f:
    label_encoder = pickle.load(f)
with open(PZE_PATH, "rb") as f:
    page_zone_enc = pickle.load(f)
features = json.load(open(META_PATH, "r", encoding="utf-8"))["features"]

bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────
def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    spans = []
    for pnum, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_w, prev_y, lid = page.rect.width, 0, 0
        sizes = [
            round(s["size"], 2)
            for b in blocks
            for l in b.get("lines", [])
            for s in l.get("spans", [])
            if s["text"].strip()
        ]
        avg_fs = sum(sizes) / len(sizes) if sizes else 1
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    txt = s["text"].strip()
                    if not txt:
                        continue
                    x0, y0, x1, _ = s["bbox"]
                    fs = round(s["size"], 2)
                    spans.append({
                        "line_id": lid,
                        "font_size": fs,
                        "relative_font_size": round(fs / avg_fs, 2),
                        "is_bold": int("Bold" in s["font"]),
                        "font_name": s["font"],
                        "x_position": round(x0, 2),
                        "y_position": round(y0, 2),
                        "span_width": round(x1 - x0, 2),
                        "char_density": round(len(txt) / (x1 - x0 or 1), 2),
                        "line_length": len(txt),
                        "whitespace_above": round((y0 - prev_y) if prev_y else 0, 2),
                        "page": pnum,
                        "page_pct": round(pnum / total_pages, 2),
                        "is_first_page": int(pnum == 1),
                        "is_uppercase": int(txt.isupper()),
                        "is_centered": int(abs((x0 + (x1 - x0) / 2) - (page_w / 2)) < 20),
                        "text": txt
                    })
                    prev_y, lid = y0, lid + 1

    df = pd.DataFrame(spans)
    if df.empty:
        return []

    df["line_id_norm"]     = df["line_id"] / df["line_id"].max()
    df["text_length"]      = df["text"].str.len()
    df["word_count"]       = df["text"].str.split().str.len()
    df["font_size_ratio"]  = df.groupby("page")["font_size"].transform(lambda x: x / x.max())
    df["page_zone"]        = pd.cut(
        df["page_pct"],
        [0, 0.33, 0.66, 1.0],
        labels=["top", "middle", "bottom"],
        include_lowest=True
    ).astype(str)
    df["page_zone_encoded"] = page_zone_enc.transform(df["page_zone"].fillna("middle"))
    for c in ["is_bold", "is_uppercase", "is_centered", "is_first_page"]:
        df[c] = df[c].astype(int)

    df["label"] = label_encoder.inverse_transform(clf.predict(df[features]))

    merged, curr = [], None
    for _, r in df.sort_values(["page", "y_position", "x_position"]).iterrows():
        if curr is None:
            curr = r.copy()
            continue
        same = (
            r["label"] == curr["label"] and
            r["page"] == curr["page"] and
            r["font_size"] == curr["font_size"] and
            r["is_bold"] == curr["is_bold"] and
            0 < abs(r["y_position"] - curr["y_position"]) < 8 and
            abs(r["x_position"] - curr["x_position"]) < 20
        )
        if same:
            curr["text"]       += " " + r["text"]
            curr["y_position"]  = min(curr["y_position"], r["y_position"])
        else:
            merged.append(curr)
            curr = r.copy()
    if curr is not None:
        merged.append(curr)

    return [
        {"level": m["label"], "text": m["text"].strip(), "page": int(m["page"])}
        for m in merged
        if m["label"].startswith("H") and len(m["text"].strip()) > 5
    ]


def segment_sections(pdf_path, outline):
    doc = fitz.open(pdf_path)
    secs = []
    for item in outline:
        p = item["page"]
        txt = doc[p - 1].get_text("text").replace("\n", " ").strip()
        if len(txt.split()) < 20:
            continue
        secs.append({
            "document": os.path.basename(pdf_path),
            "page": p,
            "section_title": item["text"],
            "text": txt
        })
    return secs


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # verify prompt.json exists
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Input file not found: {INPUT_JSON}")
        sys.exit(1)

    # load configuration
    cfg = json.load(open(INPUT_JSON, "r", encoding="utf-8"))

    # challenge metadata
    challenge_id   = cfg.get("challenge_info", {}).get("challenge_id", "")
    test_case_name = cfg.get("challenge_info", {}).get("test_case_name", "")

    # persona & job
    persona = cfg["persona"]["role"]
    job     = cfg["job_to_be_done"]["task"]

    # documents list
    docs = [d["filename"] for d in cfg["documents"]]

    # prepare query
    query = f"{persona}. Task: {job}"

    # extract headings & sections
    outlines, all_secs = {}, []
    for fn in docs:
        pdf_path = os.path.join(PDF_DIR, fn)
        if not os.path.exists(pdf_path):
            print(f"⚠️  Skipping missing file: {pdf_path}")
            continue
        outlines[fn]    = extract_headings(pdf_path)
        all_secs       += segment_sections(pdf_path, outlines[fn])

    if not all_secs:
        print("❌ No sections extracted.")
        sys.exit(1)

    # rank candidate sections
    texts  = [s["text"] for s in all_secs]
    q_emb  = bi_encoder.encode(query, convert_to_tensor=True)
    s_emb  = bi_encoder.encode(texts, batch_size=16, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, s_emb)[0]
    top_idxs   = np.argpartition(-cos_scores, range(TOP_K))[:TOP_K]
    pairs      = [(query, texts[i]) for i in top_idxs]
    rerank_scores = cross_encoder.predict(pairs)
    ranked     = sorted(zip(top_idxs, rerank_scores), key=lambda x: x[1], reverse=True)

    # build output
    output = {
        "metadata": {
            "challenge_id":   challenge_id,
            "test_case_name": test_case_name,
            "input_documents": docs,
            "persona":        persona,
            "job_to_be_done": job,
            "processed_at":   datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, (i, _) in enumerate(ranked[:FINAL_K], start=1):
        sec = all_secs[i]
        output["extracted_sections"].append({
            "document":       sec["document"],
            "section_title":  sec["section_title"],
            "importance_rank": rank,
            "page_number":    sec["page"]
        })
        output["subsection_analysis"].append({
            "document":    sec["document"],
            "refined_text": sec["text"],
            "page_number": sec["page"]
        })

    # ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Written → {OUTPUT_JSON}")
