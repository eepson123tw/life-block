from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

# from Gradio_UI import GradioUI

@tool
def execute_python_code(code: str) -> str:
    """A tool that safely executes a snippet of Python code and returns its output.
    
    Args:
        code: A string containing the Python code to execute.
    """
    try:
        import io
        import sys
        # Create a buffer to capture stdout
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        
        # Execute the code in a new, isolated global environment
        exec(code, {})
        
        # Reset stdout back to normal
        sys.stdout = old_stdout
        
        output = buffer.getvalue()
        if output.strip() == "":
            return "Code executed successfully with no output."
        return f"Output:\n{output}"
    except Exception as e:
        # Ensure stdout is reset in case of error
        sys.stdout = old_stdout
        return f"Error executing code: {str(e)}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def calculate_bmi(weight: float, height: float) -> str:
    """A tool that calculates the Body Mass Index (BMI) given weight and height,
    and provides health recommendations based on the BMI value.
    
    Args:
        weight: Your weight in kilograms.
        height: Your height in meters or centimeters. 
                If the value is greater than 3, it is assumed to be in centimeters.
    """
    try:
        # If height is provided in centimeters (i.e., greater than 3), convert to meters.
        if height > 3:
            height = height / 100.0
        
        bmi = weight / (height ** 2)
        bmi = round(bmi, 2)
        
        if bmi < 18.5:
            category = "Underweight"
            recommendation = (
                "Consider increasing your calorie intake with nutrient-dense foods, "
                "and possibly consult a healthcare provider for a tailored plan."
            )
        elif bmi < 25:
            category = "Normal weight"
            recommendation = (
                "Great job maintaining a healthy weight! Keep up with your balanced diet "
                "and regular physical activity."
            )
        elif bmi < 30:
            category = "Overweight"
            recommendation = (
                "A balanced diet and regular exercise might help in managing your weight. "
                "It could be helpful to consult with a nutritionist for personalized advice."
            )
        else:
            category = "Obese"
            recommendation = (
                "It's advisable to consult a healthcare professional for personalized recommendations "
                "regarding nutrition and exercise."
            )
        
        return f"Your BMI is {bmi} ({category}). {recommendation}"
    except Exception as e:
        return f"Error calculating BMI: {str(e)}"

final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',  # it is possible that this model may be overloaded
    custom_role_conversions=None,
)

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Add all your tools here (ensure final_answer remains included)
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, execute_python_code, calculate_bmi],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# GradioUI(agent).launch()
