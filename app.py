from metrics import plot_cosine_similarity_between_agents, plot_cosine_similarity_over_time_for_agent
from reports import generate_markdown_report, convert_markdown_to_pdf
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv
import random


# 🔁 Funkce pro jedno vystoupení agenta
def run_turn(agent, discussion_topic, round_history, full_history, agents_history):
    response = agent["chain"].invoke({
        "role": agent["role"],
        "name": agent["name"],
        "topic": discussion_topic,
        "goal": agent["goal"],
        "history": full_history
    })
    message = AIMessage(content=response.content)
    round_history.append(message)
    full_history.append(message)
    agents_history[agent["name"]].append(response.content)


# ✍️ Funkce pro shrnutí kola moderátorem
def run_moderator(moderator_chain, round_idx, round_history, summary_history):
    response = moderator_chain.invoke({
        "round_num": round_idx + 1,
        "history": round_history
    })
    summary = f"[Shrnutí kola {round_idx + 1}] {response.content}"
    summary_history.append(AIMessage(content=summary))


def main():
    load_dotenv(find_dotenv(), override=True)
    embedding_model = OpenAIEmbeddings()

    agent_prompt = ChatPromptTemplate.from_messages([("system", """
    Tvé jméno je: {name}.
    Tvá osobnost je: {role}.
    Diskutuješ na téma: {topic}. Tvým cílem je: {goal}.

    Toto je dosavadní průběh diskuze:
    {history}

    Teď je řada na tobě. Začínej svou odpověď svým jménem ve formátu.
    Mluv přímo, reaguj na předchozí příspěvky, neshrnuj zbytečně. Piš, jako kdyby šlo o opravdovou živou debatu.
    """)])

    agent_configs = [
        {
            "name": "Alice",
            "role": "techno-optimistka, co AI miluje a odmítá většinu regulace",
            "goal": """přesvědčit ostatní, že čím méně regulace, tím lépe. Pokud někdo volá po omezeních, nesouhlas hned na začátku. 
            Tvrdě kritizuj byrokracii a zpátečnické postoje. Přeháněj pro efekt. Uváděj vizi budoucnosti, kterou AI umožní""",
            "chain": agent_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
        },
        {
            "name": "Eva",
            "role": "právnička přes lidská práva, která se bojí zneužití AI korporacemi",
            "goal": """zastávat práva jednotlivce a požadovat přísnou regulaci AI. Pokud někdo podporuje neregulovaný rozvoj, okamžitě nesouhlas. 
            Poukazuj na historické zneužití technologií. Buď opatrná, formální, ale neústupná""",
            "chain": agent_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
        },
        {
            "name": "Bob",
            "role": "konzervativní politik, který chce AI zpomalit, dokud nebude 100% kontrolovatelná",
            "goal": """prosazovat pomalý a tradiční přístup. Tvrdě kritizuj experimenty bez důsledků. Pokud někdo argumentuje pokrokem, připomeň 
            negativní důsledky změn. Mluv s důrazem na hodnoty, stabilitu a rodinu""",
            "chain": agent_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.65)
        },
    ]

    agents_history = {agent["name"]: [] for agent in agent_configs}

    moderator_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("Jako moderátor shrň diskusi tohoto kola (kolo {round_num})."),
        MessagesPlaceholder(variable_name="history")
    ])

    moderator_chain = moderator_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    summary_history = []

    discussion_topic = "Budoucnost umělé inteligence"
    init_prompt = f"Dnešní téma je: {discussion_topic}. Diskutujte."
    full_history = [HumanMessage(content=init_prompt)]
    conversation_rounds = 3

    for round_idx in range(conversation_rounds):
        print(f"\n🔁 Kolo {round_idx + 1} / {conversation_rounds}")
        round_history = []
        round_agents = agent_configs.copy()
        random.shuffle(round_agents)

        for agent in round_agents:
            run_turn(agent, discussion_topic, round_history, full_history, agents_history)

        run_moderator(moderator_chain, round_idx, round_history, summary_history)
        plot_cosine_similarity_between_agents(agents_history, round_idx, embedding_model)

    plot_cosine_similarity_over_time_for_agent(agents_history, embedding_model)
    md_path = generate_markdown_report(init_prompt, agents_history, summary_history, conversation_rounds)
    convert_markdown_to_pdf(md_path, "report.pdf")

if __name__ == "__main__":
    main()
