from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pathlib import Path
import yaml


def create_agents():
    """_summary_

    Returns:
        _type_: _description_
    """
    agent_template_path = Path("prompt_templates/agent.template")
    agent_prompt_template = agent_template_path.read_text(encoding="utf-8")
    agent_prompt = ChatPromptTemplate.from_messages([("system", agent_prompt_template)])

    with open("agent_configurations/agent_conf.yml", encoding="utf-8") as f:
        agent_configs = yaml.safe_load(f)

    agents = []
    for agent in agent_configs:
        agents.append({
            "name": agent["name"],
            "role": agent["role"],
            "goal": agent["goal"],
            "chain": agent_prompt | ChatOpenAI(model=agent["model"], temperature=agent["temperature"])
        })

    return agents


def create_moderator():
    """_summary_

    Returns:
        _type_: _description_
    """
    moderator_template_path = Path("prompt_templates/moderator.template")
    moderator_prompt_template = moderator_template_path.read_text(encoding="utf-8")
    moderator_prompt = ChatPromptTemplate.from_messages([("system", moderator_prompt_template)])

    with open("agent_configurations/moderator_conf.yml", encoding="utf-8") as f:
        moderator_config = yaml.safe_load(f)

    moderator = moderator_prompt | ChatOpenAI(
        model=moderator_config["model"],
        temperature=moderator_config["temperature"]
    )

    return moderator