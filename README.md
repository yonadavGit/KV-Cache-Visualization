# KV Cache Visualization

A Streamlit app for visualizing attention, KV caching, MHA, MQA, GQA, MLA, and post-context transformer flows.

## Run Locally

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit plotly numpy
streamlit run app.py
```

Then open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

## Stop The Server

Press `Ctrl+C` in the terminal running Streamlit.

## Run Again Later

```bash
source .venv/bin/activate
streamlit run app.py
```
