from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    GoogleSearchTool,
    LiteLLMModel,
    DuckDuckGoSearchTool,
)
import os
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


openai_api_key = os.environ.get("OPENAI_API_KEY")

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    

# Initialize the LLM model
model = LiteLLMModel(
    model_id="gpt-4o",
    api_key=openai_api_key
)
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=5,
    planning_interval=20,
    name="web_search_agent",
    description="Runs web searches for you.",
)
