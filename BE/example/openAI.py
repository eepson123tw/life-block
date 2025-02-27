import sys
import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel,DuckDuckGoSearchTool

# Add parent directory to the Python path to find Gradio_UI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Gradio_UI import GradioUI

# Load environment variables
load_dotenv()

# Get API key from environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in environment.")

# Initialize the LLM model
model = LiteLLMModel(
    model_id="gpt-4o", 
    api_key=openai_api_key
)  # Could use 'gpt-4o'

# Create the agent
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model, 
    add_base_tools=True, 
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name='OPENAIAGENT',
    description=None,
)

# Launch the Gradio UI
GradioUI(agent).launch()
