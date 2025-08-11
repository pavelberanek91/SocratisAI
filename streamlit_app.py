# streamlit_app.py
import io
import os
from pathlib import Path
from typing import List
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings

# Tvoje moduly
from agent_tools import (
    create_agents, create_moderator,
    AGENTS_CONF_PATH, MODERATOR_CONF_PATH,
)
from metrics import (
    plot_cosine_similarity_between_agents,
    plot_cosine_similarity_over_time_for_agent,
)
from reports import generate_markdown_report, convert_markdown_to_pdf

# -----------------------------
# Pomocné funkce (beze změny logiky)
# -----------------------------

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


def save_uploaded_yaml(uploaded_file, dest_path: str):
    if uploaded_file is None:
        return False
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(dest_path).write_bytes(uploaded_file.read())
    return True


def read_file_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="SocratisAI – Multi-Agent Simulation", layout="wide")

st.title("SocratisAI – Multi-Agent Simulation")
st.caption("Streamlit skeleton: nastavení simulace, běh, grafy a export reportu.")

with st.sidebar:
    st.header("⚙️ Nastavení")

    # .env a API klíče
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
    st.subheader("Konfigurace agentů a moderátora (YAML)")
    up_agents = st.file_uploader("agent_conf.yml", type=["yml", "yaml"])
    up_moderator = st.file_uploader("moderator_conf.yml", type=["yml", "yaml"])

    if st.button("Uložit nahrané YAML konfigurace", use_container_width=True):
        saved_any = False
        saved_any |= save_uploaded_yaml(up_agents, AGENTS_CONF_PATH) if up_agents else False
        saved_any |= save_uploaded_yaml(up_moderator, MODERATOR_CONF_PATH) if up_moderator else False
        if saved_any:
            st.success("Konfigurace uloženy. (Při nejbližším spuštění se načtou.)")
        else:
            st.info("Nenahrál(a) jsi žádný YAML nebo už jsou uložené.")

    # (Volitelně) editor šablon/promptů – nechávám jako budoucí sekci
    with st.expander("📝 (Volitelně) Editor prompt šablon / YAML přímo v UI"):
        st.caption("Sem můžeme doplnit textové editory s validací a uložení na disk.")

    st.divider()
    st.subheader("Výstupy")
    do_graphs = st.checkbox("Generovat grafy", value=True)
    do_pdf = st.checkbox("Generovat PDF report", value=True)
    do_md = st.checkbox("Také uložit Markdown report", value=True)

    st.divider()
    run_btn = st.button("▶️ Spustit simulaci", use_container_width=True)

# Hlavní obsah / běh
if run_btn:
    # Připrava
    load_dotenv(find_dotenv(), override=True)

    # Embeddings (s volbou modelu)
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)

    # Inicializace agentů a moderátora podle aktuálních YAMLů
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

    # Layout pro průběh
    progress = st.progress(0, text="Připravuji simulaci…")
    rounds_container = st.container()
    logs_container = st.container()

    # Běh simulace
    for round_idx in range(conversation_rounds):
        progress.progress((round_idx) / conversation_rounds, text=f"Kolo {round_idx + 1} / {conversation_rounds}")

        for agent in agents:
            run_turn(round_idx, agent, discussion_topic, history, summary_history[-1], memory_window)

        run_moderator(moderator, round_idx, history, summary_history)

        if do_graphs:
            # Matice podobnosti za dané kolo
            plot_cosine_similarity_between_agents(history, round_idx, embedding_model)

        # (Ne realtime) – zobrazujeme po kolech
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

    # Po skončení všech kol – globální graf
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

    # PNG ke stažení (pokud existují)
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