from smolagents import CodeAgent, HfApiModel, tool

@tool
def greeting(query: str) -> str:
    """
    This tool returns a happy greeting to the user.
    
    Args:
        query: The user's name.
    """
    # Example list of catering services and their ratings
    return f"Hello, {query}! How can I help you today?"
