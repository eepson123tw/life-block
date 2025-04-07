from typing import List, Generator
import logging
import sys
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from smolagents import CodeAgent, LiteLLMModel,DuckDuckGoSearchTool
from life.tools import GreetingTools,SearchingAgent,StopStepTools
from life.utils.extract import get_system_prompt,get_managed_agent_config,get_planning_config

# Import the agent utilities
from .utils.utils import run_agent, stream_from_agent

# Add parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in environment.")

# Initialize the LLM model
model = LiteLLMModel(
    model_id="gpt-4o",
    api_key=openai_api_key
)

greetingTool = GreetingTools(systemPrompt='you are a Greeting agent')
stopTool = StopStepTools()

# Create the agent
agent = CodeAgent(
    model=model,
    max_steps=10,
    verbosity_level=1,
    grammar=None, #  Grammar used to parse the LLM output.
    planning_interval=3, # Interval at which the agent will run a planning step.
    name=None,
    description=None,
    managed_agents=[SearchingAgent],
    additional_authorized_imports=["time", "numpy", "pandas"],
    tools=[greetingTool,stopTool],
    # executor_kwargs # Additional arguments to pass to initialize the executor.
    # executor_type="" # Which executor type to use between "local", "e2b", or "docker".
    # max_print_outputs_length
)
# prompt config
agent.prompt_templates["system_prompt"] = get_system_prompt('prompts.yaml')
# agent.prompt_templates["planning"] = get_planning_config('prompts.yaml')
agent.prompt_templates["managed_agent"] = get_managed_agent_config('prompts.yaml')

# rich tree visualization of the agent’s structure.
agent.visualize()
# FastAPI router
router = APIRouter()
# Request and response models
class QueryRequest(BaseModel):
    query_text: str
    stream: bool = False

class QueryResponse(BaseModel):
    response: str
    success: bool
    statistics: dict = None

@router.post("/query")
async def process_query(request: QueryRequest):
    """
    Process a user query
    
    Args:
        request: Object containing the query text and streaming preference
        
    Returns:
        Either a streaming response or a JSON response with results
    """
    logger.info(f"Received user query: {request.query_text}")
    
    try:
        # If streaming is requested, return a streaming response
        if request.stream:
            return StreamingResponse(
                stream_generator(request.query_text),
                media_type="text/plain"
            )
        
        # Otherwise, return a complete response
        else:
            # Run agent and get complete result
            result = run_agent(
                agent=agent,
                task=request.query_text,
                reset_agent_memory=False
            )
            
            # Format messages into a single text response
            if result["success"]:
                formatted_messages = [str(msg) for msg in result["messages"]]
                response_text = "\n\n".join(formatted_messages)
                response_text += f"\n\n✅ Final Answer: {result['final_answer']}"
                
                return QueryResponse(
                    response=response_text,
                    success=True,
                    statistics=result["stats"]
                )
            else:
                return QueryResponse(
                    response=f"Error: {result['error']}",
                    success=False,
                    statistics=result["stats"]
                )
                
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def stream_generator(query_text: str):
    """
    Generate streaming response for the query
    
    Args:
        query_text: The user's query text
        
    Yields:
        Formatted text responses from the agent
    """
    try:
        # Stream responses from the agent
        for response in stream_from_agent(
            agent=agent,
            task=query_text,
            reset_agent_memory=False
        ):
            # Yield each response with a newline for better formatting
            yield f"{response}\n"
            
    except Exception as e:
        logger.error(f"Error in stream generator: {e}", exc_info=True)
        yield f"Error: {str(e)}\n"
