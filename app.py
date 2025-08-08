from agent_tools import create_agents, create_moderator
from metrics import plot_cosine_similarity_between_agents, plot_cosine_similarity_over_time_for_agent
from reports import generate_markdown_report, convert_markdown_to_pdf
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv


# 🔁 Funkce pro jedno vystoupení agenta
def run_turn(round_idx, agent, discussion_topic, history, summary_history):
    
    # because limited context lenght of models agents only remember last N rounds and summary from moderator
    #recent_rounds = history[-(5 * len(agents_history)):]
    recent_rounds = history[-(10 * len(history)):]
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
    #agents_history[agent["name"]].append(response.content)


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
    load_dotenv(find_dotenv(), override=True)
    embedding_model = OpenAIEmbeddings()

    agents = create_agents()
    moderator = create_moderator()

    #agents_history = {agent["name"]: [] for agent in agents} #TODO: ted se potrebuji zbavit tohohle

    #tady to pak opravit, pri vypisu jsou posunute indexy, protoze tohle je pak sumarizace prvniho kola
    summary_history = [HumanMessage(content="Zatím neproběhla žádná předchozí diskuze. Jedná se o první konverzační kolo.")]

    discussion_topic = "Budoucnost umělé inteligence"
    init_prompt = f"Dnešní téma je: {discussion_topic}. Diskutujte."
    history = [HumanMessage(content=init_prompt)]
    conversation_rounds = 10

    for round_idx in range(conversation_rounds):
        print(f"\n🔁 Kolo {round_idx + 1} / {conversation_rounds}")

        for agent in agents:
            run_turn(round_idx, agent, discussion_topic, history, summary_history[-1])

        run_moderator(moderator, round_idx, history, summary_history)        
        plot_cosine_similarity_between_agents(history, round_idx, embedding_model)
    
    plot_cosine_similarity_over_time_for_agent(history, embedding_model)
    #md_path = generate_markdown_report(init_prompt, agents_history, summary_history, conversation_rounds)
    #convert_markdown_to_pdf(md_path, "report.pdf")

if __name__ == "__main__":
    main()
