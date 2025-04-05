#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available

# Set up logging
logger = logging.getLogger("smolagents.ui")

@dataclass
class UITheme:
    """Customizable theme settings for the Gradio UI"""
    primary_color: str = "#7864FA"  # A pleasing purple as default
    secondary_color: str = "#F0F0F5"
    background_color: str = "#FFFFFF"
    text_color: str = "#2C2C2C"
    font_family: str = "Inter, system-ui, sans-serif"
    border_radius: str = "8px"
    step_separator_color: str = "#E0E0E5"
    
    # Message styling
    user_avatar: str = None
    assistant_avatar: str = "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png"
    
    # Status indicators
    success_color: str = "#4CAF50"
    error_color: str = "#F44336"
    warning_color: str = "#FFC107"
    info_color: str = "#2196F3"


def create_css_styles(theme: UITheme) -> str:
    """Generate CSS styles based on the theme"""
    return f"""
    .gradio-container {{
        font-family: {theme.font_family};
        color: {theme.text_color};
        background-color: {theme.background_color};
    }}
    
    .chatbot-message {{
        border-radius: {theme.border_radius};
        border: 1px solid {theme.secondary_color};
    }}
    
    .step-separator {{
        border-top: 1px dashed {theme.step_separator_color};
        margin: 12px 0;
    }}
    
    .tool-call {{
        background-color: {theme.secondary_color};
        border-left: 3px solid {theme.primary_color};
        padding: 8px;
        margin: 4px 0;
        border-radius: {theme.border_radius};
    }}
    
    .execution-logs {{
        font-family: monospace;
        background-color: #2C2C2C;
        color: #F0F0F5;
        padding: 12px;
        border-radius: {theme.border_radius};
        overflow-x: auto;
    }}
    
    .error-message {{
        background-color: #FEE;
        border-left: 3px solid {theme.error_color};
        padding: 8px;
        margin: 4px 0;
        border-radius: {theme.border_radius};
    }}
    
    .final-answer {{
        background-color: #E0F2F1;
        border-left: 3px solid {theme.success_color};
        padding: 12px;
        margin: 8px 0;
        border-radius: {theme.border_radius};
        font-weight: 500;
    }}
    
    .footnote {{
        color: #BBBBC2;
        font-size: 12px;
        text-align: right;
        margin-top: 4px;
    }}
    """


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


class GradioUI:
    """An enhanced interface to launch your agent in Gradio with customization options"""

    def __init__(
        self, 
        agent: MultiStepAgent, 
        file_upload_folder: str | None = None,
        theme: Optional[UITheme] = None,
        allowed_file_types: Optional[List[str]] = None,
        max_file_size_mb: int = 10,
        enable_debug: bool = False,
        custom_css: Optional[str] = None,
    ):
        """
        Initialize the Gradio UI for the agent
        
        Args:
            agent: The SmoLAgent to use for this interface
            file_upload_folder: Directory to store uploaded files (created if it doesn't exist)
            theme: Custom UI theme settings
            allowed_file_types: List of allowed MIME types for file uploads
            max_file_size_mb: Maximum allowed file size in MB
            enable_debug: Enable debug logging
            custom_css: Additional custom CSS to apply to the UI
        """
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
            
        # Set up logging
        log_level = logging.DEBUG if enable_debug else logging.INFO
        logging.basicConfig(level=log_level)
        
        self.agent = agent
        self.theme = theme or UITheme()
        self.custom_css = custom_css
        self.max_file_size_mb = max_file_size_mb
        self.allowed_file_types = allowed_file_types or [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/csv",
            "application/json",
            "image/jpeg",
            "image/png",
        ]
        
        # Set up file upload directory
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            upload_dir = Path(file_upload_folder)
            upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"File uploads will be stored in: {upload_dir.absolute()}")

    def interact_with_agent(self, prompt, messages, history=None):
        """Process user input and stream agent responses"""
        import gradio as gr

        # Add user message to the chat
        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        
        # Process through the agent and stream responses
        for msg in stream_to_gradio(
            self.agent, 
            task=prompt, 
            reset_agent_memory=False,
            theme=self.theme
        ):
            messages.append(msg)
            yield messages

    def upload_file(self, file, file_uploads_log):
        """
        Handle file uploads with improved validation and security
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            # Check file size
            file_size_mb = os.path.getsize(file.name) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return gr.Textbox(
                    f"File exceeds maximum size limit of {self.max_file_size_mb}MB", 
                    visible=True
                ), file_uploads_log

            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file.name)
            if not mime_type:
                mime_type = "application/octet-stream"  # Default if type can't be determined
                
            if mime_type not in self.allowed_file_types:
                allowed_types_str = ", ".join(self.allowed_file_types)
                return gr.Textbox(
                    f"File type '{mime_type}' not allowed. Supported types: {allowed_types_str}", 
                    visible=True
                ), file_uploads_log

            # Sanitize file name to prevent path traversal and other issues
            original_name = os.path.basename(file.name)
            sanitized_name = re.sub(r"[^\w\-.]", "_", original_name)
            
            # Ensure unique filenames to prevent overwriting
            timestamp = int(time.time())
            base_name, extension = os.path.splitext(sanitized_name)
            unique_filename = f"{base_name}_{timestamp}{extension}"
            
            # Save the uploaded file to the specified folder
            file_path = os.path.join(self.file_upload_folder, unique_filename)
            shutil.copy(file.name, file_path)
            
            # Log the successful upload
            logger.info(f"File uploaded: {file_path}")
            
            return gr.Textbox(
                f"✅ File uploaded successfully: {os.path.basename(file_path)}", 
                visible=True
            ), file_uploads_log + [file_path]
            
        except Exception as e:
            logger.exception("Error during file upload")
            return gr.Textbox(f"Error during file upload: {str(e)}", visible=True), file_uploads_log

    def log_user_message(self, text_input, file_uploads_log):
        """Format the user's message with file context information"""
        if not text_input:
            return "", ""
            
        message = text_input
        
        # Add file context if files have been uploaded
        if file_uploads_log:
            file_names = [os.path.basename(path) for path in file_uploads_log]
            files_str = ", ".join(file_names)
            message += f"\n\nYou have access to these files: {files_str}"
            
        return message, ""

    def launch(self, **kwargs):
        """Launch the Gradio UI with customized styling and features"""
        import gradio as gr

        # Generate CSS styles
        css = create_css_styles(self.theme)
        if self.custom_css:
            css += f"\n{self.custom_css}"

        with gr.Blocks(css=css, theme=gr.themes.Soft(), fill_height=True) as demo:
            # Initialize state variables
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Agent Conversation",
                        type="messages",
                        avatar_images=(
                            self.theme.user_avatar,
                            self.theme.assistant_avatar,
                        ),
                        height=600,
                        resizeable=True,
                        scale=1,
                        elem_id="agent-chatbot",
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=8):
                            text_input = gr.Textbox(
                                lines=2, 
                                label="Your message",
                                placeholder="Ask the agent...",
                                scale=3,
                                elem_id="user-input",
                            )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("Send", variant="primary")
                
                # File upload section if enabled
                if self.file_upload_folder is not None:
                    with gr.Column(scale=1):
                        gr.Markdown("### File Upload")
                        upload_file = gr.File(
                            label="Upload a file",
                            file_types=self.allowed_file_types,
                            file_count="multiple",
                        )
                        upload_status = gr.Textbox(
                            label="Upload Status", 
                            interactive=False, 
                            visible=True
                        )
                        gr.Markdown(
                            f"Supported file types: {', '.join(ext for ext in self.allowed_file_types)}\n"
                            f"Max file size: {self.max_file_size_mb}MB"
                        )
                        
                        # Display list of uploaded files
                        gr.Markdown("### Uploaded Files")
                        file_list = gr.Dataframe(
                            headers=["Filename"],
                            datatype=["str"],
                            row_count=(5, "fixed"),
                            col_count=(1, "fixed"),
                            interactive=False,
                        )
                        
                        # Update file list when uploads change
                        def update_file_list(file_paths):
                            if not file_paths:
                                return [[""]]
                            return [[os.path.basename(path)] for path in file_paths]
                        
                        file_uploads_log.change(
                            update_file_list,
                            [file_uploads_log],
                            [file_list],
                        )
                        
                        # Handle file upload
                        upload_file.change(
                            self.upload_file,
                            [upload_file, file_uploads_log],
                            [upload_status, file_uploads_log],
                        )
            
            # Set up event handlers for chat interaction
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(
                self.interact_with_agent, 
                [stored_messages, chatbot], 
                [chatbot]
            )
            
            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(
                self.interact_with_agent, 
                [stored_messages, chatbot], 
                [chatbot]
            )
            
            # Add additional UI elements
            with gr.Accordion("About", open=False):
                gr.Markdown(f"""
                # SmoLAgents Gradio UI
                
                This interface allows you to interact with a SmoLAgent.
                
                **Agent Model:** {getattr(self.agent.model, 'model_name', 'Unknown')}
                
                **Capabilities:**
                - Chat with the agent to accomplish tasks
                - Upload files for the agent to analyze
                - See the agent's step-by-step reasoning
                
                Made with ❤️ using SmoLAgents and Gradio
                """)

        # Launch the demo with customized settings
        launch_kwargs = {
            "debug": True,
            "share": True,
            "show_api": False,
        }
        launch_kwargs.update(kwargs)
        
        demo.launch(**launch_kwargs)


__all__ = ["stream_to_gradio", "GradioUI", "UITheme"]
