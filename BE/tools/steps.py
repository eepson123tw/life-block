from smolagents import Tool
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union

class StopStepTools(Tool):
	"""To stop generate and have error"""
	inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
	output_type = "any"
	name = "stopTool"
	description = "Provides a final error answer to the given problem."

	def __init__(self):
		self.is_initialized = False  

	def forward(self, answer: Any) -> Any:
		print(f"{answer} and has error with any issue")
		return answer
