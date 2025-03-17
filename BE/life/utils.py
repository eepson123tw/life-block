import re
import time
import logging
from typing import Optional, List, Dict, Any, Generator, Union

from smolagents.agent_types import handle_agent_output_types
from smolagents.agents import ActionStep
from smolagents.memory import MemoryStep

# Configure logging
logger = logging.getLogger(__name__)

def format_code(content: str) -> str:
    """Format code blocks properly by cleaning up tags and ensuring proper markdown"""
    # Remove any end code tags
    content = re.sub(r"```.*?\n", "", content)  # Remove existing code block start tags
    content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
    content = content.strip()
    
    # If it's not already marked as Python code, add the markdown
    if not content.startswith("```python"):
        content = f"```python\n{content}\n```"
    
    return content


def clean_model_output(output: str) -> str:
    """Clean up the model output by removing artifacts and extra tags"""
    if not output:
        return ""
        
    # Clean up the LLM output
    output = output.strip()
    
    # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
    output = re.sub(r"```\s*<end_code>", "```", output)  # handles ```<end_code>
    output = re.sub(r"<end_code>\s*```", "```", output)  # handles <end_code>```
    output = re.sub(r"```\s*\n\s*<end_code>", "```", output)  # handles ```\n<end_code>
    
    # Ensure code blocks are properly formatted
    output = re.sub(r"```(?!\w+)(\s*\n)", "```python\n", output)  # Add language to unlabeled code blocks
    
    return output.strip()


class AgentMessage:
    """A simple class to represent agent messages without UI dependencies"""
    def __init__(self, content: str, message_type: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.message_type = message_type
        self.metadata = metadata or {}

    def __str__(self):
        """Format the message as a string based on its type"""
        if self.message_type == "step_header":
            return f"### {self.content}"
        
        elif self.message_type == "tool_call":
            tool_name = self.metadata.get("tool_name", "unknown_tool")
            tool_emoji = {
                "python_interpreter": "🐍",
                "search": "🔍",
                "calculator": "🧮",
                "file_reader": "📄",
            }.get(tool_name, "🛠️")
            return f"{tool_emoji} Used tool: {tool_name}\n{self.content}"
        
        elif self.message_type == "execution_logs":
            return f"📝 Execution Logs:\n{self.content}"
        
        elif self.message_type == "error":
            return f"💥 Error: {self.content}"
        
        elif self.message_type == "step_footer":
            return f"ℹ️ {self.content}"
        
        elif self.message_type == "final_answer":
            return f"✅ Final Answer: {self.content}"
        
        elif self.message_type == "summary":
            return f"📊 {self.content}"
        
        # Default case
        return self.content


def extract_messages_from_step(step_log: MemoryStep) -> List[AgentMessage]:
    """Extract messages from agent steps with relevant metadata"""
    messages = []

    if isinstance(step_log, ActionStep):
        # Step header
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Processing"
        messages.append(AgentMessage(
            content=step_number,
            message_type="step_header"
        ))

        # Model reasoning/thoughts
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = clean_model_output(step_log.model_output)
            messages.append(AgentMessage(
                content=model_output,
                message_type="reasoning"
            ))

        # Tool calls
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None and step_log.tool_calls:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            
            # Process arguments
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            # Format code if needed
            if used_code:
                content = format_code(content)

            # Tool call message
            tool_name = first_tool_call.name
            messages.append(AgentMessage(
                content=content,
                message_type="tool_call",
                metadata={"tool_name": tool_name}
            ))

            # Execution logs
            if hasattr(step_log, "observations") and step_log.observations is not None and step_log.observations.strip():
                log_content = step_log.observations.strip()
                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                if log_content:
                    messages.append(AgentMessage(
                        content=log_content,
                        message_type="execution_logs"
                    ))

            # Error messages
            if hasattr(step_log, "error") and step_log.error is not None:
                messages.append(AgentMessage(
                    content=str(step_log.error),
                    message_type="error"
                ))

        # Standalone errors
        elif hasattr(step_log, "error") and step_log.error is not None:
            messages.append(AgentMessage(
                content=str(step_log.error),
                message_type="error"
            ))

        # Step footer with metadata
        step_info = []
        if step_log.step_number is not None:
            step_info.append(f"Step {step_log.step_number}")
            
        if (hasattr(step_log, "input_token_count") and 
            hasattr(step_log, "output_token_count") and 
            step_log.input_token_count and 
            step_log.output_token_count):
            step_info.append(
                f"Input tokens: {step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
            )
                
        if hasattr(step_log, "duration") and step_log.duration:
            step_info.append(f"Duration: {round(float(step_log.duration), 2)}s")
            
        if step_info:
            footer = " | ".join(step_info)
            messages.append(AgentMessage(
                content=footer,
                message_type="step_footer"
            ))

    return messages


def stream_from_agent(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
) -> Generator[str, None, None]:
    """
    Stream responses from an agent.
    
    Args:
        agent: Agent object
        task: Task description string
        reset_agent_memory: Whether to reset agent memory
        additional_args: Additional arguments dictionary
    
    Returns:
        Generator yielding formatted message strings
    """
    # Track token counts
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Record start time
    start_time = time.time()

    try:
        # Run agent and stream responses
        for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
            # Track token counts if available
            if hasattr(agent.model, "last_input_token_count"):
                total_input_tokens += agent.model.last_input_token_count
                total_output_tokens += agent.model.last_output_token_count
                
                # If it's an action step, record token counts
                if isinstance(step_log, ActionStep):
                    step_log.input_token_count = agent.model.last_input_token_count
                    step_log.output_token_count = agent.model.last_output_token_count

            # Extract and yield messages
            for message in extract_messages_from_step(step_log):
                yield str(message)

        # Extract final answer from the last step
        final_answer = handle_agent_output_types(step_log)
        
        # Yield final answer
        final_answer_message = AgentMessage(
            content=str(final_answer),
            message_type="final_answer"
        )
        yield str(final_answer_message)
        
        # Calculate total duration
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = f"Total tokens: {total_input_tokens + total_output_tokens:,} | Duration: {round(total_duration, 2)}s"
        summary_message = AgentMessage(
            content=summary,
            message_type="summary"
        )
        yield str(summary_message)
            
    except Exception as e:
        # Handle errors
        error_message = f"Agent execution error: {str(e)}"
        logger.error(error_message, exc_info=True)
        yield f"💥 Error: {error_message}"


def run_agent(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run a task with an agent and return structured result data
    
    Args:
        agent: Agent object
        task: Task description string
        reset_agent_memory: Whether to reset agent memory
        additional_args: Additional arguments dictionary
    
    Returns:
        Dictionary with results, messages, and statistics
    """
    # Track metrics
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()
    all_messages = []
    final_step = None

    try:
        # Run agent and process each step
        for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
            # Track token counts if available
            if hasattr(agent.model, "last_input_token_count"):
                total_input_tokens += agent.model.last_input_token_count
                total_output_tokens += agent.model.last_output_token_count
                
                if isinstance(step_log, ActionStep):
                    step_log.input_token_count = agent.model.last_input_token_count
                    step_log.output_token_count = agent.model.last_output_token_count

            # Process step and collect messages
            step_messages = extract_messages_from_step(step_log)
            all_messages.extend(step_messages)
            final_step = step_log
            
        # Final answer
        final_answer = handle_agent_output_types(final_step) if final_step else "No result"
        
        # Calculate duration
        total_duration = time.time() - start_time
        
        # Return structured result
        return {
            "success": True,
            "final_answer": final_answer,
            "messages": all_messages,
            "stats": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "duration_seconds": round(total_duration, 2)
            }
        }
            
    except Exception as e:
        # Handle errors
        error_message = f"Agent execution error: {str(e)}"
        logger.error(error_message, exc_info=True)
        return {
            "success": False,
            "error": error_message,
            "messages": all_messages,
            "stats": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "duration_seconds": round(time.time() - start_time, 2)
            }
        }
