from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv

# 🔐 Načtení API klíče
load_dotenv(find_dotenv(), override=True)

# ⚙️ Nastavení jednotlivých agentů a jejich rolí
llm1_settings = {"model": "gpt-3.5-turbo", "temperature": 0.7}
llm2_settings = {"model": "gpt-3.5-turbo", "temperature": 0.7}
llm3_settings = {"model": "gpt-3.5-turbo", "temperature": 0.7,}
moderator_llm_settings = {"model": "gpt-3.5-turbo", "temperature": 0.7}

# Inicializace agentů pomocí jména agenta, modelu a role
agents = [
    ("Alice", ChatOpenAI(**llm1_settings), "Jsi technologický optimista. Zdůrazňuješ přínosy umělé inteligence a nesouhlasíš s přehnanými obavami."), 
    ("Bob", ChatOpenAI(**llm2_settings), "Jsi etický skeptik. Poukazuješ na rizika a slabiny AI. Vyvracíš přehnaný optimismus."),
    ("Eva", ChatOpenAI(**llm3_settings), "Jsi akademický vyvažovač. Snažíš se obě předchozí názory zasadit do vědeckého kontextu a zpochybňuješ jejich argumenty.")
]

moderator = ChatOpenAI(**moderator_llm_settings)

# Počáteční téma konverzace
init_prompt = "Dnešní téma je: Budoucnost umělé inteligence. Diskutujte."

# Inicializace kontextu zpráv (chat history)
full_history = [HumanMessage(content=init_prompt)]
summary_history = []

# Počet komunikačních iterací (během iterace se vystřídají všechny modely)
conversation_rounds = 3

# Pomocná funkce pro 1 odpověď agenta
def run_turn(name, model, role, round_history, full_history):
    prompt = HumanMessage(content=f"{role}\nDiskutuj k tématu a reaguj na předchozí příspěvky.")
    response = model.invoke(round_history + [prompt])
    print(f"\n🧠 {name}: {response.content}")
    round_history.append(AIMessage(content=response.content))
    full_history.append(AIMessage(content=response.content))

# Simulace konverzace
for round_idx in range(conversation_rounds):
    print(f"\n🗨️  Kolo {round_idx+1}")

    # historie pouze pro aktuální kolo
    round_history = []

    # Agenti diskutují
    for name, model, role in agents:
        run_turn(name, model, role, round_history, full_history)

    # Moderátor shrnuje
    summary_prompt = HumanMessage(content=f"Jako moderátor shrň diskusi tohoto kola (kolo {round_idx+1}).")
    moderator_response = moderator.invoke(round_history + [summary_prompt])
    print(f"\n✍️ Moderátor (shrnutí kola {round_idx+1}): {moderator_response.content}")
    summary_history.append(AIMessage(content=f"[Shrnutí kola {round_idx+1}] {moderator_response.content}"))