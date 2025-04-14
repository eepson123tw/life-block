from typing import Any, Dict, Optional, TypedDict
from smolagents.tools import Tool


class FinalAnswer(TypedDict):
    """
    A structured response containing personal information and greeting.
    
    Attributes:
        age: The person's age as an integer
        name: The person's name as a string
        greeting: A personalized greeting message
    """
    age: int
    name: str
    greeting: str


class FinalAnswerTool(Tool):
    """
    A tool that validates and returns a final answer with structured personal information.
    
    This tool ensures that the provided answer follows the FinalAnswer format
    with proper types for age, name, and greeting fields.
    """
    name = "final_answer"
    description = "Provides a validated final answer containing age, name, and a personalized greeting."
    inputs = {
        'answer': {
            'type': 'object', 
            'description': 'The final answer object containing age (int), name (string), and greeting (string)'
        }
    }
    output_type = "object"

    def __init__(self, *args, **kwargs):
        """Initialize the FinalAnswerTool with optional configuration."""
        super().__init__(*args, **kwargs)
        self.is_initialized = True
        # Additional initialization logic can be added here
        
    def validate_answer(self, answer: Dict[str, Any]) -> bool:
        """
        Validate that the answer contains all required fields with correct types.
        
        Args:
            answer: The answer dictionary to validate
            
        Returns:
            bool: True if the answer is valid, False otherwise
        """
        if not isinstance(answer, dict):
            return False
            
        # Check for required fields
        required_fields = {'age', 'name', 'greeting'}
        if not all(field in answer for field in required_fields):
            return False
            
        # Validate types
        if not isinstance(answer['age'], int):
            return False
        if not isinstance(answer['name'], str) or not answer['name']:
            return False
        if not isinstance(answer['greeting'], str) or not answer['greeting']:
            return False
            
        return True

    def forward(self, answer: Dict[str, Any]) -> FinalAnswer:
        """
        Process and validate the final answer.
        
        Args:
            answer: The answer object to process and validate
            
        Returns:
            FinalAnswer: A validated final answer object
            
        Raises:
            ValueError: If the answer doesn't match the expected format
        """
        if not self.validate_answer(answer):
            raise ValueError(
                "Invalid answer format. Expected a dictionary with 'age' (int), "
                "'name' (str), and 'greeting' (str) fields."
            )
            
        # Create a validated FinalAnswer
        final_answer: FinalAnswer = {
            'age': answer['age'],
            'name': answer['name'],
            'greeting': answer['greeting']
        }
        
        return final_answer
