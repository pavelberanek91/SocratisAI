from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import TypedDict
from pathlib import Path
import yaml


AGENT_TEMPLATE_PATH = "prompt_templates/agent.template"
AGENTS_CONF_PATH = "agent_configurations/agent_conf.yml"
MODERATOR_CONF_PATH = "agent_configurations/moderator_conf.yml"
MODERATOR_TEMPLATE_PATH = "prompt_templates/moderator.template"


class AgentDict(TypedDict):
    name: str
    role: str
    goal: str
    chain: Runnable[dict, str]


def load_prompt(path: str) -> ChatPromptTemplate:
    """
    Loads a prompt template from a file and returns it as a ChatPromptTemplate.

    Args:
        path (str): Path to the prompt template file.

    Returns:
        ChatPromptTemplate: LangChain chat prompt with a system message from the file.
    """
    prompt_text = Path(path).read_text(encoding="utf-8")
    return ChatPromptTemplate.from_messages([("system", prompt_text)])


def load_yaml(path: str) -> dict | list:
    """
    Loads YAML content from a file and returns it as a Python dictionary or list.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict | list: Parsed YAML data.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def validate_config(config: dict, required_keys: set, label: str) -> None:
    """
    Validates that the configuration dictionary contains all required keys.

    Args:
        config (dict): Configuration dictionary to validate.
        required_keys (set): Set of keys that must be present in the config.
        label (str): Label used in the error message for context.

    Raises:
        ValueError: If any required key is missing.
    """
    missing = required_keys - config.keys()
    if missing:
        raise ValueError(f"{label} missing keys: {missing}")
    

def build_chain(prompt_path: str, model: str, temperature: float) -> Runnable[dict, str]:
    """
    Builds a LangChain runnable by combining a prompt template and a language model.

    Args:
        prompt_path (str): Path to the prompt template file.
        model (str): OpenAI model name (e.g., "gpt-3.5-turbo").
        temperature (float): Temperature setting for the model.

    Returns:
        Runnable[dict, str]: A runnable chain that takes a dict input and produces a string output.
    """
    prompt = load_prompt(prompt_path)
    return prompt | ChatOpenAI(model=model, temperature=temperature)


def create_agents() -> list[AgentDict]:
    """
    Creates agents with a given name, role, goal, and a runnable LangChain chain based on a configuration file.

    Returns:
        list[AgentDict]: A list of agent dictionaries configured to participate in a discussion on a given topic.
    """
    agent_configs = load_yaml(AGENTS_CONF_PATH)
    for agent_idx, agent in enumerate(agent_configs, start=1):
        validate_config(agent, {"name", "role", "goal", "model", "temperature"}, f"Agent {agent_idx}")

    agents = [
        {
            "name": agent_config["name"],
            "role": agent_config["role"],
            "goal": agent_config["goal"],
            "chain": build_chain(AGENT_TEMPLATE_PATH, agent_config["model"], agent_config["temperature"])
        }
        for agent_config in agent_configs
    ]

    return agents


def create_moderator() -> Runnable[dict, str]:
    """
    Creates a moderator agent that summarizes the discussion between agents in a given round.

    Returns:
        Runnable[dict, str]: A runnable LangChain chain for the moderator, based on a configuration file.
    """
    moderator_config = load_yaml(MODERATOR_CONF_PATH)
    validate_config(moderator_config, {"name", "model", "temperature"}, "Moderator")

    moderator = build_chain(MODERATOR_TEMPLATE_PATH, moderator_config["model"], moderator_config["temperature"])

    return moderator