# Verbatim RAG — Minimal PDF Q&A 

A tiny wrapper around [`verbatim-rag`](https://github.com/KRLabsOrg/verbatim-rag) plus a super-simple Streamlit UI.

- **`verbatim.py`** – helper functions to build the index and ask questions
- **`streamlit_app.py`** – upload PDFs, build the index, and ask questions in a browser

---

- Python **3.10+** (3.11 recommended)
- macOS, Linux, or Windows

Install Python deps:

```bash
# from your project folder
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Set your OpenAI API key before using the system:
export OPENAI_API_KEY=your_api_key_here


# To run the app while being in VERBATIM_RAG folder
streamlit run app.py
