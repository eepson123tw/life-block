from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool
import numpy as np
import time
import datetime


login()

# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

# agent.run("tell me the average years old for the taiwan old man.")


# Tool to suggest a menu based on the occasion
# @tool
# def suggest_menu(occasion: str) -> str:
#     """
#     Suggests a menu based on the occasion.
#     Args:
#         occasion: The type of occasion for the party.
#     """
#     if occasion == "casual":
#         return "Pizza, snacks, and drinks."
#     elif occasion == "formal":
#         return "3-course dinner with wine and dessert."
#     elif occasion == "superhero":
#         return "Buffet with high-energy and healthy food."
#     else:
#         return "Custom menu for the butler."

# # Alfred, the butler, preparing the menu for the party
# agent = CodeAgent(tools=[suggest_menu], model=HfApiModel())

# # Preparing the menu for the party
# agent.run("Prepare a formal menu for the party.")



agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['datetime'])

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    3. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)
