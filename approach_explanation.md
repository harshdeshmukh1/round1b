

## 🧠 Approach

1. **Prompt Parsing**  
   - Load `input/prompt.json`, which contains:
     - `challenge_info` (ID/test case)  
     - `documents[]` (filenames + titles)  
     - `persona.role`  
     - `job_to_be_done.task`

2. **Heading Extraction**  
   - For each PDF in `input/`, use PyMuPDF to extract every text span’s font, position, and style.  
   - Compute features: relative font size, character density, whitespace above, page‑zone, uppercase/bold/centered flags.

3. **Structure Classification**  
   - Apply a pre‑trained XGBoost multiclass model to label spans as `title`, `H1`/`H2`/`H3`, or body text.  
   - Merge adjacent spans with the same label, font, and close coordinates to reconstruct full headings.

4. **Section Segmentation**  
   - For each extracted heading, grab the first ~100 words from its page as a candidate section.

5. **Relevance Scoring**  
   - Formulate a query: `"{persona}. Task: {job}"`.  
   - Use a bi‑encoder (`all-MiniLM-L6-v2`) to compute embedding for query and each section.  
   - Pre‑select TOP_K candidates via cosine similarity, then rerank with a cross‑encoder (`ms-marco-MiniLM-L-6-v2`).

6. **Output Assembly**  
   - Choose the top FINAL_K sections.  
   - Produce `output/result.json` with:
     ```json
     {
       "metadata": { … },
       "extracted_sections": [
         { "document": "…", "section_title": "…", "importance_rank": 1, "page_number": 3 },
         … 
       ],
       "subsection_analysis": [
         { "document": "…", "refined_text": "…", "page_number": 3 },
         …
       ]
     }
     ```

This pipeline runs in **≤ 60 seconds** on CPU and fits within the **1 GB** model‑size limit.

---

## 🚀 Quickstart

1. **Build the Docker image**  
   ```bash
   cd round1b
   docker build --platform linux/amd64 -t pdf-intel-1b .
