from agent_tools import create_agents, create_moderator
from metrics import plot_cosine_similarity_between_agents, plot_cosine_similarity_over_time_for_agent
from reports import generate_markdown_report, convert_markdown_to_pdf
from langchain_openai import OpenAIEmbeddings
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv


def select_recent_rounds(history: list[BaseMessage], current_round: int, window_rounds: int) -> list[BaseMessage]:
    """
    Vrátí zprávy agentů z posledních `window_rounds` kol včetně aktuálního.
    Spoléhá se na response_metadata["round"] u AIMessage.
    """
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


def run_turn(round_idx, agent, discussion_topic, history, summary_history, memory_window_rounds=10):
    
    # because limited context lenght of models agents only remember last N rounds and summary from moderator
    recent_rounds = select_recent_rounds(history, round_idx, memory_window_rounds)
    short_history = [summary_history] + recent_rounds

    response = agent["chain"].invoke({
        "role": agent["role"],
        "name": agent["name"],
        "topic": discussion_topic,
        "goal": agent["goal"],
        "history": short_history
    })
    message = AIMessage(
        content=response.content,
        response_metadata={"agent_name": agent["name"],"round": round_idx}
    )
    history.append(message)


# ✍️ Funkce pro shrnutí kola moderátorem
def run_moderator(moderator_chain, round_idx, history, summary_history):
    round_history = [msg for msg in history if msg.response_metadata.get("round") == round_idx]
    response = moderator_chain.invoke({
        "round_num": round_idx + 1,
        "history": round_history
    })
    summary = f"[Shrnutí kola {round_idx + 1}] {response.content}"
    summary_history.append(AIMessage(content=summary))


def main():

    CONVERSATION_ROUNDS = 10
    MEMORY_WINDOW_ROUNDS = 10

    load_dotenv(find_dotenv(), override=True)
    embedding_model = OpenAIEmbeddings()

    agents = create_agents()
    moderator = create_moderator()

    #tady to pak opravit, pri vypisu jsou posunute indexy, protoze tohle je pak sumarizace prvniho kola
    summary_history = [HumanMessage(content="Zatím neproběhla žádná předchozí diskuze. Jedná se o první konverzační kolo.")]

    discussion_topic = "Budoucnost umělé inteligence"
    init_prompt = f"Dnešní téma je: {discussion_topic}. Diskutujte."
    history = [HumanMessage(content=init_prompt)]

    for round_idx in range(CONVERSATION_ROUNDS):
        print(f"\n🔁 Kolo {round_idx + 1} / {CONVERSATION_ROUNDS}")

        for agent in agents:
            run_turn(round_idx, agent, discussion_topic, history, summary_history[-1], MEMORY_WINDOW_ROUNDS)

        run_moderator(moderator, round_idx, history, summary_history)        
        plot_cosine_similarity_between_agents(history, round_idx, embedding_model)
    
    plot_cosine_similarity_over_time_for_agent(history, embedding_model)
    md_path = generate_markdown_report(init_prompt, history, summary_history)
    convert_markdown_to_pdf(md_path, "report.pdf")

if __name__ == "__main__":
    main()
