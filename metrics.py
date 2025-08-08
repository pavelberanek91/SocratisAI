from langchain.schema import AIMessage
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def plot_cosine_similarity_between_agents(history, round_idx, embedding_model):
    """Visualization of cosine similarity between agent responses in embeddings form
    Args:
        agent_names (List[str]): used as tick names on x and y axis of matrix
        agent_responses (List[str]): outputs of agent LLMs to be compared
        round_idx (int): used in title to identify round of conversation
        embedding_model (OpenAIEmbeddings): model to be used for embeddings calculation of agent_responses
    """
    last_round_conversation = {
        agent_name: msg.content
        for msg in history
        if (agent_name := msg.response_metadata.get("agent_name")) is not None
        and msg.response_metadata.get("round") == round_idx
    }
    agent_names = last_round_conversation.keys()
    last_round_agent_responses = last_round_conversation.values()
    
    embeddings = embedding_model.embed_documents(last_round_agent_responses)
    similarity_matrix = cosine_similarity(embeddings)

    plt.figure(figsize=(6, 5))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Kosinová podobnost")
    plt.xticks(ticks=np.arange(len(agent_names)), labels=agent_names)
    plt.yticks(ticks=np.arange(len(agent_names)), labels=agent_names)
    plt.title(f"Kosinová podobnost mezi agenty - Kolo {round_idx + 1}")
    plt.tight_layout()
    plt.savefig(f'Interagentni podobnost kolo {round_idx+1}.png')
    plt.close()


def plot_cosine_similarity_over_time_for_agent(history, embedding_model):
    """Visualization of cosine similarity of agent responses evolution over conversation rounds

    Args:
        agents_history (Dict[str, List[str]]): dictionary of agent responses in rounds
        embedding_model (OpenAIEmbeddings): model to be used for embeddings calculation of agent_responses
    """

    # 1.0 for init cosine similarity (first embedding doesnt have comparable partner)
    agent_similarity_over_time = {
        agent_name: [1.0] 
        for msg in history
        if (agent_name := msg.response_metadata.get("agent_name")) is not None
        and msg.response_metadata.get("round") == 0
    }

    agents_history = defaultdict(list)
    for msg in history:
        if not isinstance(msg, AIMessage):
            continue
        agent_name = msg.response_metadata.get("agent_name")
        if agent_name:
            agents_history[agent_name].append(msg.content)
    
    for agent_name, agent_responses in agents_history.items():
        agent_embeddings = embedding_model.embed_documents(agent_responses)
        prev_embedding = agent_embeddings[0]
        for agent_embedding in agent_embeddings[1:]:
            current_embedding = agent_embedding
            between_rounds_similarity = cosine_similarity([current_embedding], [prev_embedding])[0][0]
            agent_similarity_over_time[agent_name].append(between_rounds_similarity)
            prev_embedding = current_embedding

    plt.figure(figsize=(10, 5))
    for name, cosine_similarities in agent_similarity_over_time.items():
        plt.plot(range(1, len(cosine_similarities) + 1), cosine_similarities, marker='o', label=name)
    plt.title("Stabilita odpovědí agentů mezi koly")
    plt.xlabel("Kolo")
    plt.ylabel("Kosinová podobnost s předchozím kolem")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Vyvoj podobnosti nazoru agentu.png')
    plt.close()