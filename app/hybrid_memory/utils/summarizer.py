

from app.models.interaction import Interaction


def summarize_interaction(interaction: Interaction) -> str:
    user_part = f"User: {interaction.user_prompt if interaction.user_prompt else '[No prompt provided]'}"
    agent_part = f"Agent: {interaction.target_output or interaction.actual_output}"
    return f"{user_part} | {agent_part}"