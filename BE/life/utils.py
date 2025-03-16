import mimetypes
import os
import re
import shutil
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Generator

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available



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


def pull_messages_from_step(
    step_log: MemoryStep,
    theme: UITheme = None,
) -> Generator:
    """Extract ChatMessage objects from agent steps with proper nesting and styling"""
    import gradio as gr
    
    theme = theme or UITheme()

    if isinstance(step_log, ActionStep):
        # Output the step number with improved formatting
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Processing"
        yield gr.ChatMessage(
            role="assistant", 
            content=f"**{step_number}**",
            metadata={"type": "step_header"}
        )

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = clean_model_output(step_log.model_output)
            yield gr.ChatMessage(
                role="assistant", 
                content=model_output,
                metadata={"type": "reasoning"}
            )

        # For tool calls, create a parent message with improved formatting
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None and step_log.tool_calls:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Process tool call arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            # Handle code formatting specifically for Python interpreter
            if used_code:
                content = format_code(content)

            # Create tool call message with status indicator
            tool_emoji = {
                "python_interpreter": "🐍",
                "search": "🔍",
                "calculator": "🧮",
                "file_reader": "📄",
            }.get(first_tool_call.name, "🛠️")
            
            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"{tool_emoji} Used tool: {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                    "type": "tool_call",
                    "tool_name": first_tool_call.name,
                },
            )
            yield parent_message_tool

            # Nesting execution logs with better formatting
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"{log_content}",
                        metadata={
                            "title": "📝 Execution Logs", 
                            "parent_id": parent_id, 
                            "status": "done",
                            "type": "execution_logs",
                        },
                    )

            # Nesting errors with better formatting
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={
                        "title": "💥 Error", 
                        "parent_id": parent_id, 
                        "status": "done",
                        "type": "error",
                    },
                )

            # Update parent message metadata to done status
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(
                role="assistant", 
                content=str(step_log.error), 
                metadata={
                    "title": "💥 Error",
                    "type": "error",
                }
            )

        # Calculate duration and token information with improved formatting
        step_footnote = []
        if step_log.step_number is not None:
            step_footnote.append(f"Step {step_log.step_number}")
            
        if (hasattr(step_log, "input_token_count")
            and hasattr(step_log, "output_token_count")
            and step_log.input_token_count
            and step_log.output_token_count):
                step_footnote.append(
                    f"Input tokens: {step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
                )
                
        if hasattr(step_log, "duration") and step_log.duration:
            step_footnote.append(f"Duration: {round(float(step_log.duration), 2)}s")
            
        if step_footnote:
            footnote_html = f"""<div class="footnote">{' | '.join(step_footnote)}</div>"""
            yield gr.ChatMessage(role="assistant", content=footnote_html)
            
        # Add a visual separator between steps
        yield gr.ChatMessage(
            role="assistant", 
            content="<div class='step-separator'></div>",
            metadata={"type": "separator"}
        )



def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
    theme: UITheme = None,
) -> Generator:
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr
    
    theme = theme or UITheme()
    
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    try:
        for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
            # Track tokens if model provides them
            if hasattr(agent.model, "last_input_token_count"):
                total_input_tokens += agent.model.last_input_token_count
                total_output_tokens += agent.model.last_output_token_count
                if isinstance(step_log, ActionStep):
                    step_log.input_token_count = agent.model.last_input_token_count
                    step_log.output_token_count = agent.model.last_output_token_count

            for message in pull_messages_from_step(step_log, theme=theme):
                yield message

        # Extract the final answer from the last step
        final_answer = step_log  # Last log is the run's final_answer
        final_answer = handle_agent_output_types(final_answer)

        # Format the final answer based on its type
        if isinstance(final_answer, AgentText):
            yield gr.ChatMessage(
                role="assistant",
                content=f"<div class='final-answer'><strong>Final answer:</strong><br>{final_answer.to_string()}</div>",
                metadata={"type": "final_answer"}
            )
        elif isinstance(final_answer, AgentImage):
            yield gr.ChatMessage(
                role="assistant",
                content={"path": final_answer.to_string(), "mime_type": "image/png"},
                metadata={"type": "final_answer", "media_type": "image"}
            )
        elif isinstance(final_answer, AgentAudio):
            yield gr.ChatMessage(
                role="assistant",
                content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
                metadata={"type": "final_answer", "media_type": "audio"}
            )
        else:
            yield gr.ChatMessage(
                role="assistant", 
                content=f"<div class='final-answer'><strong>Final answer:</strong><br>{str(final_answer)}</div>",
                metadata={"type": "final_answer"}
            )
        
        # Display summary information
        total_duration = time.time() - start_time
        summary = [
            f"Total tokens: {total_input_tokens + total_output_tokens:,}",
            f"Total time: {round(total_duration, 2)}s"
        ]
        yield gr.ChatMessage(
            role="assistant",
            content=f"""<div class="footnote">{' | '.join(summary)}</div>""",
            metadata={"type": "summary"}
        )
            
    except Exception as e:
        logger.exception("Error in agent execution")
        yield gr.ChatMessage(
            role="assistant",
            content=f"<div class='error-message'><strong>Error in agent execution:</strong><br>{str(e)}</div>",
            metadata={"type": "error"}
        )
