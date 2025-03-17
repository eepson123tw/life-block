from typing import List, Generator
import logging
import sys
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from smolagents import CodeAgent, LiteLLMModel,DuckDuckGoSearchTool

# Import the agent utilities
from .utils import run_agent, stream_from_agent

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
