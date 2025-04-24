
"""
YAML-like file parsing utilities

This module provides functions for extracting sections from YAML-like files,
especially those with non-standard formatting or multi-line string blocks.
"""

def extract_section(filename, section_name):
    """
    Extract a specific section from a YAML-like file.
    
    Args:
        filename (str): Path to the YAML-like file
        section_name (str): Name of the section to extract
        
    Returns:
        str: The content of the section with indentation removed
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find section marker (without quotes, followed by |- for multi-line string)
    section_marker = f"{section_name}: |-"
    
    start_index = content.find(section_marker)
    
    if start_index == -1:
        # Try alternative formats in case |- is not present
        alt_markers = [f"{section_name}:", f"{section_name}: >"]
        for marker in alt_markers:
            pos = content.find(marker)
            if pos != -1:
                start_index = pos
                section_marker = marker
                break
    
    if start_index == -1:
        return f"Section '{section_name}' not found in file."
    
    # Move to the start of the actual content (after the marker and newline)
    start_index = content.find('\n', start_index) + 1
    
    # Find where the section ends (next section or end of file)
    remaining_content = content[start_index:]
    
    # Look for the next section marker (a word followed by a colon at the beginning of a line)
    import re
    next_section_matches = list(re.finditer(r'^[a-zA-Z_][a-zA-Z0-9_]*:', remaining_content, re.MULTILINE))
    
    if next_section_matches:
        # Use the first match as end of our section
        end_index = start_index + next_section_matches[0].start()
        section_content = content[start_index:end_index].strip()
    else:
        # If no next section found, take until the end
        section_content = remaining_content.strip()
    
    # Process the content based on indentation
    lines = section_content.split('\n')
    
    # Determine the indentation level (usually 2 spaces in YAML)
    indent_level = 0
    for line in lines:
        if line.strip():  # Skip empty lines when determining indentation
            indent_level = len(line) - len(line.lstrip())
            break
    
    # Remove the consistent indentation
    processed_lines = []
    for line in lines:
        if not line.strip():  # Keep empty lines as is
            processed_lines.append(line)
        elif line.startswith(' ' * indent_level):
            processed_lines.append(line[indent_level:])
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)
def get_system_prompt(filename):
    """
    Extract the system prompt from a configuration file.
    
    Args:
        filename (str): Path to the configuration file
        
    Returns:
        str: The system prompt content
    """
    return extract_section(filename, "system_prompt")

def get_planning_config(filename):
    """
    Extract the planning configuration from a configuration file and return it as a dictionary.
    
    Args:
        filename (str): Path to the configuration file
        
    Returns:
        dict: The planning configuration as a dictionary with keys for each sub-section
    """
    # First, extract the entire planning section
    planning_section = extract_section(filename, "planning")
    
    # Define the keys we want to extract based on PlanningPromptTemplate
    planning_keys = [
        "initial_plan",
        "initial_facts",
        "update_facts_pre_messages",
        "update_facts_post_messages",
        "update_plan_pre_messages",
        "update_plan_post_messages"
    ]
    
    # Create a dictionary to store the results
    planning_dict = {}
    
    # Extract each subsection from the raw content
    for key in planning_keys:
        # The content might be directly in the planning section
        # Look for patterns like "initial_plan : |-" within the planning section
        content = extract_subsection(planning_section, key)
        if content and not content.startswith("Section"):
            planning_dict[key] = content
        else:
            # If not found as a subsection, try to extract directly from the file
            direct_content = extract_section(filename, f"planning.{key}")
            if not direct_content.startswith("Section"):
                planning_dict[key] = direct_content
            else:
                # Set empty string as default if section not found
                planning_dict[key] = ""
    
    return planning_dict

def extract_subsection(content, subsection_name):
    """
    Extract a subsection from a larger content string.
    
    Args:
        content (str): The content to search within
        subsection_name (str): Name of the subsection to extract
        
    Returns:
        str: The content of the subsection
    """
    import re
    
    # Look for the subsection marker
    patterns = [
        f"{subsection_name} : |-",
        f"{subsection_name}: |-",
        f"{subsection_name} :",
        f"{subsection_name}:"
    ]
    
    start_index = -1
    for pattern in patterns:
        pos = content.find(pattern)
        if pos != -1:
            start_index = pos
            marker = pattern
            break
    
    if start_index == -1:
        return f"Section '{subsection_name}' not found."
    
    # Move to the start of the actual content
    start_index = content.find('\n', start_index) + 1
    
    # Find where the subsection ends (next subsection or end of content)
    remaining_content = content[start_index:]
    
    # Look for the next subsection marker
    next_subsection_matches = list(re.finditer(r'^[a-zA-Z_][a-zA-Z0-9_]*:', remaining_content, re.MULTILINE))
    
    if next_subsection_matches:
        end_index = start_index + next_subsection_matches[0].start()
        subsection_content = content[start_index:end_index].strip()
    else:
        subsection_content = remaining_content.strip()
    
    # Process indentation
    lines = subsection_content.split('\n')
    
    # Determine the indentation level
    indent_level = 0
    for line in lines:
        if line.strip():
            indent_level = len(line) - len(line.lstrip())
            break
    
    # Remove the consistent indentation
    processed_lines = []
    for line in lines:
        if not line.strip():
            processed_lines.append(line)
        elif line.startswith(' ' * indent_level):
            processed_lines.append(line[indent_level:])
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def get_managed_agent_config(filename):
    """
    Extract the managed agent configuration from a configuration file.
    
    Args:
        filename (str): Path to the configuration file
        
    Returns:
        str: The managed agent configuration content
    """
    return extract_section(filename, "managed_agent")
