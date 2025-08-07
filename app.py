from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv, find_dotenv

# 🔐 Načtení API klíče
load_dotenv(find_dotenv(), override=True)

# Inicializace embedding modelu
embedding_model = OpenAIEmbeddings()

# ⚙️ Nastavení jednotlivých agentů a jejich rolí
agent_configs = [
    {
        "name": "Alice",
        "llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
        "role": "Jsi technologický optimista. Zdůrazňuješ přínosy umělé inteligence a nesouhlasíš s přehnanými obavami.",
    },
    {
        "name": "Bob",
        "llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
        "role": "Jsi etický skeptik. Poukazuješ na rizika a slabiny AI. Vyvracíš přehnaný optimismus.",
    },
    {
        "name": "Eva",
        "llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
        "role": "Jsi akademický vyvažovač. Snažíš se obě předchozí názory zasadit do vědeckého kontextu a zpochybňuješ jejich argumenty.",
    },
]

# 🧠 Moderátor – samostatný model (může být jiný)
moderator = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


# Uchování embedding historie pro každého agenta
agent_embeddings_history = {agent["name"]: [] for agent in agent_configs}

# Uchování podobností mezi koly
agent_similarity_over_time = {agent["name"]: [] for agent in agent_configs}

# 📌 Téma a historie
init_prompt = "Dnešní téma je: Budoucnost umělé inteligence. Diskutujte."
full_history = [HumanMessage(content=init_prompt)]
summary_history = []

# 🔄 Počet kol
conversation_rounds = 10


# 🗨️ Funkce pro jedno vystoupení agenta
def run_turn(agent, round_history, full_history):
    system_prompt = HumanMessage(
        content=f"{agent['role']}\nDiskutuj k tématu a reaguj na předchozí příspěvky."
    )
    response = agent["llm"].invoke(round_history + [system_prompt])
    #print(f"\n🧠 {agent['name']}: {response.content}")
    message = AIMessage(content=response.content)
    round_history.append(message)
    full_history.append(message)

     # Výpočet embeddingu a podobnosti s předchozím kolem
    current_embedding = embedding_model.embed_documents([response.content])[0]
    name = agent["name"]
    history = agent_embeddings_history[name]

    if history:
        prev_embedding = history[-1]
        sim = cosine_similarity([current_embedding], [prev_embedding])[0][0]
        agent_similarity_over_time[name].append(sim)
    else:
        agent_similarity_over_time[name].append(1.0)  # první výskyt – podobnost se sebou samým

    history.append(current_embedding)


# 🧾 Funkce pro shrnutí kola
def run_moderator(moderator_llm, round_idx, round_history, summary_history):
    summary_prompt = HumanMessage(
        content=f"Jako moderátor shrň diskusi tohoto kola (kolo {round_idx + 1})."
    )
    response = moderator_llm.invoke(round_history + [summary_prompt])
    summary = f"[Shrnutí kola {round_idx + 1}] {response.content}"
    #print(f"\n✍️ Moderátor (shrnutí kola {round_idx + 1}): {response.content}")
    summary_history.append(AIMessage(content=summary))


# Výpočet kosinové podobnosti mezi agentovými výstupy
def plot_cosine_similarity_between_agents(agent_names, agent_outputs, round_idx):
    # Získání embeddingů
    embeddings = embedding_model.embed_documents(agent_outputs)

    # Výpočet párových podobností
    similarity_matrix = cosine_similarity(embeddings)

    # Vykreslení heatmapy
    plt.figure(figsize=(6, 5))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Kosinová podobnost")
    plt.xticks(ticks=np.arange(len(agent_names)), labels=agent_names)
    plt.yticks(ticks=np.arange(len(agent_names)), labels=agent_names)
    plt.title(f"Kosinová podobnost – Kolo {round_idx + 1}")
    plt.tight_layout()
    plt.show()


def plot_similarity_over_time(agent_similarity_over_time):
    # Vykreslení vývoje podobnosti odpovědí v čase
    plt.figure(figsize=(10, 5))
    for name, similarities in agent_similarity_over_time.items():
        plt.plot(range(1, len(similarities) + 1), similarities, marker='o', label=name)

    plt.title("Stabilita odpovědí agentů mezi koly (intra-agent similarity)")
    plt.xlabel("Kolo")
    plt.ylabel("Kosinová podobnost s předchozím kolem")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 🚀 Spuštění simulace
    for round_idx in range(conversation_rounds):
        print(f"\n🔁 Kolo {round_idx + 1}")
        round_history = []

        for agent in agent_configs:
            run_turn(agent, round_history, full_history)

        # Shrnutí kola moderátorem
        run_moderator(moderator, round_idx, round_history, summary_history)

        # Spočítej a vykresli podobnosti
        agent_names = [agent["name"] for agent in agent_configs]
        agent_outputs = [msg.content for msg in round_history]
        plot_cosine_similarity_between_agents(agent_names, agent_outputs, round_idx)
    
    plot_similarity_over_time(agent_similarity_over_time)

    
if __name__ == "__main__":
    main()