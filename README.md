# SocratisAI — Multi-Agent Discussion Simulator
Configure debating agents with YAML, run a structured multi-round discussion, visualize agreement with embeddings, and export a polished PDF report — all from a friendly Streamlit app.

> TL;DR: Point it at your OpenAI key, pick your agents & topic, press Run, download PDF.

## ✨ Features

- Streamlit UI — run everything from the browser.
- Configurable agents & moderator — name, role, goal, model, temperature (YAML).
- Inline editors or file uploads — edit YAML and prompt templates right in the app or upload files.
- Memory window — each agent sees the last N rounds + the latest moderator summary.
- Embeddings analytics — per-round cosine-similarity heatmap and per-agent stability over time.
- One-click reports — Markdown + LaTeX-backed PDF with embedded plots.
- Convenient downloads — download items individually or as a ZIP.
- API key in UI — enter your OPENAI_API_KEY directly (or load from .env).

## 🚀 Quick start
```bash
# 1) Clone & enter
git clone https://github.com/paveberanek91/socratisai.git
cd socratisai

# 2) Python env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt

# 4) Run Streamlit
streamlit run streamlit_app.py
```

API key: either create a .env with: 
```
OPENAI_API_KEY=sk-...
```

… or paste your key into the 🔐 OpenAI API key field in the sidebar (session-only).

> PDF export uses Pandoc + LaTeX. pypandoc can download Pandoc automatically; for LaTeX (pdflatex) you’ll need a TeX distribution (e.g., TeX Live / MikTeX).

## 🧩 Configuration
You can upload files or edit inline in the UI. Default paths:

```bash
agent_configurations/agent_conf.yml
agent_configurations/moderator_conf.yml
prompt_templates/agent.template
prompt_templates/moderator.template
```

### Agents (YAML)
```yaml
- name: Alice
  role: Optimistic Researcher
  goal: Bring optimistic long-term views with evidence
  model: gpt-4o-mini
  temperature: 0.7

- name: Bob
  role: Pragmatic Engineer
  goal: Challenge assumptions and ask for concrete trade-offs
  model: gpt-4o-mini
  temperature: 0.3
```

### Moderator (YAML)
```yaml
name: Moderator
model: gpt-4o-mini
temperature: 0.2
```

### Templates

```
agent.template has access to {name}, {role}, {goal}, {topic}, {history}.
moderator.template has {round_num}, {history}.
```

Example agent.template:
```
You are {name}, acting as {role}.
Your goal: {goal}

Discussion topic: {topic}

Recent context (may include summaries and last N rounds):
{history}

Respond briefly, insightfully, and stay on topic.
```

## 🧭 How it runs

1. You choose topic, rounds, memory window, and configs.
2. For each round:
   * Every agent produces a message (seeing the last moderator summary + last N rounds).
   * The moderator summarizes the round.
   * (Optional) A cosine-similarity heatmap is saved as PNG.
3. At the end:
   * A stability over time plot is saved as PNG.
   * A Markdown report is generated and converted to PDF (if enabled).
4. From the UI, download PDF/MD/PNGs individually or as a ZIP.

## 📂 Project structure
```
.
├── streamlit_app.py          # UI wrapper
├── app.py                    # core loop (CLI-style entry point)
├── agent_tools.py            # YAML loading, prompt building, LLM chains
├── metrics.py                # embeddings + plots
├── report.py                 # Markdown + PDF export (Pandoc/LaTeX)
├── agent_configurations/
│   ├── agent_conf.yml
│   └── moderator_conf.yml
└── prompt_templates/
    ├── agent.template
    └── moderator.template
```

## 🖼️ Screenshots

![Streamlit UI](docs/ui_main.png) 
![Per-round similarity](docs/heatmap.png) 
![Stability over time](docs/stability.png)

## ⚠️ Troubleshooting

- PDF didn’t generate: ensure a LaTeX distribution is installed (pdflatex available). Pandoc will auto-download if missing.
- OpenAI auth: set OPENAI_API_KEY in .env or enter it in the UI.
- Long contexts: adjust Memory window in the sidebar (default 10 rounds).
- Model availability: pick models available to your account/region.

## 🗺️ Roadmap (ideas)

- Profiles for configs (switch between folders).
- Run snapshots (runs/<timestamp>/…) with auto-ZIP.
- Stable agent ordering in plots per YAML order.
- Seed/reproducibility and richer run metadata in the PDF.

## 🤝 Contributing

Issues and PRs welcome! Please include:
- steps to reproduce,
- environment info,
- and a minimal YAML/template sample if relevant.

## 📜 License
MIT (or your preferred license).