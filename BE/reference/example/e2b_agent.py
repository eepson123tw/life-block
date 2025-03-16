import os
import sys
import base64
from typing import Any, Optional
from dotenv import load_dotenv
from smolagents import LiteLLMModel, CodeAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e2b_code_interpreter import Sandbox
from Gradio_UI import GradioUI

# Load environment variables
load_dotenv()

# Validate environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
e2b_api_key = os.getenv("E2B_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in environment.")
if not e2b_api_key:
    raise ValueError("E2B_API_KEY not set in environment.")

# Create the LLM model
model = LiteLLMModel(
    model_id="gpt-4o", 
    api_key=openai_api_key
)

# Create the sandbox
sandbox = Sandbox()

# Install required visualization packages
sandbox.commands.run("pip install matplotlib plotly pandas numpy seaborn pillow")

from smolagents.tools import Tool

class DrawTool(Tool):
    name = "draw"
    description = """
    Execute drawing/visualization code in the E2B sandbox when the user asks for any visual representation.
    
    This tool should be used whenever the user requests any kind of visual output such as:
    - Charts, graphs, or plots
    - Diagrams or flowcharts
    - Visual representations of data or concepts
    - Animations or visual simulations
    """
    inputs = {
        'code': {
            'type': 'string', 
            'description': 'Python code string that generates visualizations'
        },
        'verbose': {
            'type': 'boolean',
            'description': 'Whether to include detailed logs',
            'default': False,
            'nullable': True
        }
    }
    output_type = "string"
    
    def __init__(self, sandbox_instance=None):
        """Initialize with a sandbox instance"""
        super().__init__()
        self.sandbox = sandbox_instance

    def forward(self, code: str, verbose: bool = False) -> str:
        """The main method that will be called by the agent framework"""
        # Use the instance sandbox
        if not self.sandbox:
            raise ValueError("No sandbox instance provided to the DrawTool")
            
        # Setup visualization environment with common imports
        setup_code = """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import io
import base64
from PIL import Image

# Configure matplotlib for non-interactive environment
plt.switch_backend('Agg')

# Function to convert matplotlib figure to base64 string
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str
"""
        
        # Run the setup code
        setup_execution = self.sandbox.run_code(setup_code)
        if setup_execution.error:
            raise ValueError(f"Error setting up visualization environment: {setup_execution.error.traceback}")
        
        # If the code uses matplotlib but doesn't save the figure, add code to capture it
        if "plt." in code and "plt.savefig" not in code and "plt.show()" not in code:
            code += """
# Capture the current figure
img_base64 = fig_to_base64()
print(f"IMAGE_GENERATED: First 50 chars of base64: {img_base64[:50]}...")
plt.close()
"""
        
        # Execute the visualization code
        execution = self.sandbox.run_code(code)
        
        if execution.error:
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            logs = execution_logs
            logs += execution.error.traceback
            raise ValueError(logs)
        
        output_logs = "\n".join([str(log) for log in execution.logs.stdout])
        
        # Check if an image was generated
        if "IMAGE_GENERATED" in output_logs:
            return "Visualization generated successfully. In a UI environment, this image would be displayed.\n\n" + output_logs
        
        return output_logs

# Create an instance of the DrawTool class, passing the sandbox
draw_tool_instance = DrawTool(sandbox_instance=sandbox)

# Create the agent with our tool instance
agent = CodeAgent(
    model=model,
    use_e2b_executor=True,
    tools=[draw_tool_instance],
)

GradioUI(agent).launch()


