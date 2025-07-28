# Round 1B: Persona‑Driven Document Intelligence

This toolkit ingests a collection of PDFs plus a `prompt.json` (defining persona & job), then extracts and ranks the most relevant sections from each document for that persona’s task.

---
prompt.json ← defines challenge_info, documents, persona & job 
## 📂 Repository Layout

round1b/
├── Dockerfile
├── README.md
├── src/
│ └── main.py ← your extraction & ranking script
├── input/
│ ├── prompt.json ← defines challenge_info, documents, persona & job
│ ├── South of France - Cities.pdf
│ ├── South of France - Cuisine.pdf
│ └── … other PDFs …
├── output/
│ └── result.json ← produced by the script
├── xgboost_pdf_structure_model.json ← pre‑trained XGBoost for headings
├── label_encoder.pkl ← maps model outputs to labels
├── page_zone_encoder.pkl ← encodes top/middle/bottom page zones
└── meta.json ← list of features used by XGBoost