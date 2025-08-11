# streamlit_app.py
import io, zipfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv, find_dotenv
import yaml

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings

# Tvoje moduly
from agent_tools import (
    create_agents, create_moderator,
    AGENTS_CONF_PATH, MODERATOR_CONF_PATH,
    AGENT_TEMPLATE_PATH, MODERATOR_TEMPLATE_PATH,
    validate_config,
)
from metrics import (
    plot_cosine_similarity_between_agents,
    plot_cosine_similarity_over_time_for_agent,
)
from reports import generate_markdown_report, convert_markdown_to_pdf


# -----------------------------
# Pomocné funkce
# -----------------------------
def ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file, dest_path: str) -> bool:
    if uploaded_file is None:
        return False
    ensure_parent(dest_path)
    Path(dest_path).write_bytes(uploaded_file.read())
    return True

def write_text_file(dest_path: str | Path, content: str):
    ensure_parent(dest_path)
    Path(dest_path).write_text(content, encoding="utf-8")

def read_text_file(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""

def read_file_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()

def select_recent_rounds(history: List[BaseMessage], current_round: int, window_rounds: int) -> List[BaseMessage]:
    if window_rounds <= 0:
        return []
    start_round = max(0, current_round - (window_rounds - 1))
    recent = []
    for msg in history:
        if isinstance(msg, AIMessage):
            r = msg.response_metadata.get("round")
            if isinstance(r, int) and start_round <= r <= current_round:
                recent.append(msg)
    return recent

def run_turn(round_idx, agent, discussion_topic, history, summary_message, memory_window_rounds=10):
    recent_rounds = select_recent_rounds(history, round_idx, memory_window_rounds)
    short_history = [summary_message] + recent_rounds

    response = agent["chain"].invoke({
        "role": agent["role"],
        "name": agent["name"],
        "topic": discussion_topic,
        "goal": agent["goal"],
        "history": short_history
    })
    message = AIMessage(
        content=response.content,
        response_metadata={"agent_name": agent["name"], "round": round_idx}
    )
    history.append(message)

def run_moderator(moderator_chain, round_idx, history, summary_history):
    round_history = [msg for msg in history if msg.response_metadata.get("round") == round_idx]
    response = moderator_chain.invoke({
        "round_num": round_idx + 1,
        "history": round_history
    })
    summary = f"[Shrnutí kola {round_idx + 1}] {response.content}"
    summary_history.append(AIMessage(content=summary))

def validate_agents_yaml_text(text: str):
    data = yaml.safe_load(text)
    if not isinstance(data, list):
        raise ValueError("agent_conf.yml musí být list (seznam agentů).")
    for i, agent in enumerate(data, start=1):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent #{i} není objekt.")
        validate_config(agent, {"name", "role", "goal", "model", "temperature"}, f"Agent {i}")

def validate_moderator_yaml_text(text: str):
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("moderator_conf.yml musí být objekt (mapa klíč→hodnota).")
    validate_config(data, {"name", "model", "temperature"}, "Moderator")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="SocratisAI – Multi-Agent Simulation", layout="wide")
st.title("SocratisAI – Multi-Agent Simulation")
st.caption("Streamlit UI: nastavení simulace, konfigurace/šablony (upload nebo inline), běh, grafy a export.")

with st.sidebar:
    st.header("⚙️ Nastavení")
    if st.button("Načíst .env", use_container_width=True):
        load_dotenv(find_dotenv(), override=True)
        st.success("Načteno z .env")

    discussion_topic = st.text_input("Téma diskuze", value="Budoucnost umělé inteligence")
    conversation_rounds = st.slider("Počet kol", min_value=1, max_value=50, value=10, step=1)
    memory_window = st.slider("Paměťové okno (počet kol)", min_value=1, max_value=50, value=10, step=1)

    st.divider()
    st.subheader("Embeddings")
    embedding_model_name = st.text_input("OpenAI embeddings model", value="text-embedding-3-small")

    st.divider()
    st.subheader("Konfigurace & šablony")
    config_mode = st.radio(
        "Způsob úprav",
        options=["Nahrát soubory", "Upravit inline v UI"],
        index=0,
        horizontal=True,
    )

    if config_mode == "Nahrát soubory":
        st.caption("Nahraj YAML konfigurace a/nebo .template soubory. Uloží se na očekávané cesty v projektu.")
        up_agents = st.file_uploader("agent_conf.yml", type=["yml", "yaml"])
        up_moderator = st.file_uploader("moderator_conf.yml", type=["yml", "yaml"])
        up_agent_tmpl = st.file_uploader("agent.template", type=["template", "txt"])
        up_moderator_tmpl = st.file_uploader("moderator.template", type=["template", "txt"])

        if st.button("💾 Uložit nahrané soubory", use_container_width=True):
            saved = []
            if up_agents:    saved.append(("agents YAML", save_uploaded_file(up_agents, AGENTS_CONF_PATH)))
            if up_moderator: saved.append(("moderator YAML", save_uploaded_file(up_moderator, MODERATOR_CONF_PATH)))
            if up_agent_tmpl: saved.append(("agent template", save_uploaded_file(up_agent_tmpl, AGENT_TEMPLATE_PATH)))
            if up_moderator_tmpl: saved.append(("moderator template", save_uploaded_file(up_moderator_tmpl, MODERATOR_TEMPLATE_PATH)))

            if any(s for _, s in saved):
                st.success("Konfigurace/šablony uloženy. Při dalším spuštění simulace se použijí.")
            else:
                st.info("Nic nebylo nahráno/uloženo.")

    else:
        # Inline editory s předvyplněním z aktuálních souborů
        with st.expander("🧩 agent_conf.yml (YAML)", expanded=True):
            default_agents_yaml = read_text_file(AGENTS_CONF_PATH) or """\
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
"""
            agents_yaml_text = st.text_area(
                "Obsah agent_conf.yml",
                value=default_agents_yaml,
                height=220,
                key="agents_yaml_text",
                help="Seznam agentů: name, role, goal, model, temperature.",
            )

        with st.expander("🧩 moderator_conf.yml (YAML)", expanded=True):
            default_moderator_yaml = read_text_file(MODERATOR_CONF_PATH) or """\
name: Moderator
model: gpt-4o-mini
temperature: 0.2
"""
            moderator_yaml_text = st.text_area(
                "Obsah moderator_conf.yml",
                value=default_moderator_yaml,
                height=120,
                key="moderator_yaml_text",
                help="Nastavení moderátora: name, model, temperature.",
            )

        with st.expander("📝 agent.template", expanded=False):
            default_agent_tmpl = read_text_file(AGENT_TEMPLATE_PATH) or """\
You are {name}, acting as {role}.
Your goal: {goal}

Discussion topic: {topic}

Recent context (may include summaries and last N rounds):
{history}

Respond briefly but insightfully, keeping focus on the topic.
"""
            agent_tmpl_text = st.text_area(
                "Obsah agent.template",
                value=default_agent_tmpl,
                height=220,
                key="agent_tmpl_text",
                help="Dostupné placeholdery: {name}, {role}, {goal}, {topic}, {history}",
            )

        with st.expander("📝 moderator.template", expanded=False):
            default_moderator_tmpl = read_text_file(MODERATOR_TEMPLATE_PATH) or """\
You are the Moderator. Summarize round {round_num}.
Base your summary only on the following messages:
{history}

Output a neutral, concise summary (bullets welcome).
"""
            moderator_tmpl_text = st.text_area(
                "Obsah moderator.template",
                value=default_moderator_tmpl,
                height=200,
                key="moderator_tmpl_text",
                help="Dostupné placeholdery: {round_num}, {history}",
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Validovat YAML", use_container_width=True):
                try:
                    validate_agents_yaml_text(agents_yaml_text)
                    validate_moderator_yaml_text(moderator_yaml_text)
                    st.success("YAML konfigurace vypadají v pořádku.")
                except Exception as e:
                    st.error(f"Chyba ve validaci YAML: {e}")

        with c2:
            if st.button("💾 Uložit inline obsah", use_container_width=True):
                # Nejprve validace YAML
                try:
                    validate_agents_yaml_text(agents_yaml_text)
                    validate_moderator_yaml_text(moderator_yaml_text)
                except Exception as e:
                    st.error(f"YAML není validní, neukládám: {e}")
                else:
                    # Uložení YAML + templátů
                    try:
                        write_text_file(AGENTS_CONF_PATH, agents_yaml_text)
                        write_text_file(MODERATOR_CONF_PATH, moderator_yaml_text)
                        write_text_file(AGENT_TEMPLATE_PATH, agent_tmpl_text)
                        write_text_file(MODERATOR_TEMPLATE_PATH, moderator_tmpl_text)
                        st.success("Inline konfigurace a šablony uloženy.")
                    except Exception as e:
                        st.error(f"Chyba při ukládání: {e}")

    st.divider()
    st.subheader("Výstupy")
    do_graphs = st.checkbox("Generovat grafy", value=True)
    do_pdf = st.checkbox("Generovat PDF report", value=True)
    do_md = st.checkbox("Také uložit Markdown report", value=True)

    st.divider()
    run_btn = st.button("▶️ Spustit simulaci", use_container_width=True)


# -----------------------------
# Běh simulace
# -----------------------------
if run_btn:
    load_dotenv(find_dotenv(), override=True)

    # Embeddings (s volbou modelu)
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)

    # Inicializace agentů a moderátora podle aktuálních souborů
    try:
        agents = create_agents()
        moderator = create_moderator()
    except Exception as e:
        st.error(f"Chyba při načítání konfigurace: {e}")
        st.stop()

    init_prompt = f"Dnešní téma je: {discussion_topic}. Diskutujte."
    history: List[BaseMessage] = [HumanMessage(content=init_prompt)]
    summary_history: List[BaseMessage] = [
        HumanMessage(content="Zatím neproběhla žádná předchozí diskuze. Jedná se o první konverzační kolo.")
    ]

    progress = st.progress(0, text="Připravuji simulaci…")
    rounds_container = st.container()

    for round_idx in range(conversation_rounds):
        progress.progress((round_idx) / conversation_rounds, text=f"Kolo {round_idx + 1} / {conversation_rounds}")

        for agent in agents:
            run_turn(round_idx, agent, discussion_topic, history, summary_history[-1], memory_window)

        run_moderator(moderator, round_idx, history, summary_history)

        if do_graphs:
            plot_cosine_similarity_between_agents(history, round_idx, embedding_model)

        with rounds_container.expander(f"🌀 Kolo {round_idx + 1}", expanded=False):
            st.markdown("**Příspěvky agentů:**")
            for msg in history:
                if isinstance(msg, AIMessage) and msg.response_metadata.get("round") == round_idx:
                    st.markdown(f"- **{msg.response_metadata.get('agent_name', 'Agent')}**: {msg.content}")

            st.markdown("**Shrnutí moderátorem:**")
            st.info(summary_history[-1].content)

            if do_graphs:
                img_path = Path(f"Interagentni podobnost kolo {round_idx + 1}.png")
                if img_path.exists():
                    st.image(str(img_path), caption=f"Interagentní podobnost – kolo {round_idx + 1}", use_container_width=True)

    if do_graphs:
        plot_cosine_similarity_over_time_for_agent(history, embedding_model)
        evol_path = Path("Vyvoj podobnosti nazoru agentu.png")
        if evol_path.exists():
            st.image(str(evol_path), caption="Vývoj stability odpovědí agentů", use_container_width=True)

    # Reporty
    st.subheader("📄 Export")
    try:
        md_path = generate_markdown_report(init_prompt, history, summary_history) if (do_pdf or do_md) else None

        if do_md and md_path:
            st.download_button(
                label="⬇️ Stáhnout Markdown report",
                data=read_file_bytes(md_path),
                file_name=Path(md_path).name,
                mime="text/markdown"
            )

        if do_pdf and md_path:
            pdf_out = "report.pdf"
            convert_markdown_to_pdf(md_path, pdf_out)
            if Path(pdf_out).exists():
                st.download_button(
                    label="⬇️ Stáhnout PDF report",
                    data=read_file_bytes(pdf_out),
                    file_name=pdf_out,
                    mime="application/pdf"
                )
            else:
                st.warning("PDF se nepodařilo vytvořit (zkontroluj pandoc/latex).")
    except Exception as e:
        st.error(f"Chyba při generování reportu: {e}")

    # PNG ke stažení
    if do_graphs:
        st.markdown("**📊 Obrázky ke stažení:**")
        imgs = sorted(Path(".").glob("Interagentni podobnost kolo *.png"))
        for p in imgs:
            st.download_button(
                label=f"Stáhnout {p.name}",
                data=read_file_bytes(p),
                file_name=p.name,
                mime="image/png",
                key=f"dl_{p.name}"
            )
        evol_path = Path("Vyvoj podobnosti nazoru agentu.png")
        if evol_path.exists():
            st.download_button(
                label=f"Stáhnout {evol_path.name}",
                data=read_file_bytes(evol_path),
                file_name=evol_path.name,
                mime="image/png",
                key="dl_evol_png"
            )

    progress.progress(1.0, text="Hotovo ✅")

else:
    st.info("Nastav parametry vlevo a klikni na **Spustit simulaci**.")
