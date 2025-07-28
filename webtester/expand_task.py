import json
import os
import re
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

from expand_task_logger import setup_task_expansion_logging, log_task_expansion, log_task_expansion_error

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_task_expansion_logging()

class TaskExpander:
    """
    A tool to automatically expand complex tasks into smaller, manageable subtasks
    using an LLM (Large Language Model).
    """

    def __init__(self, model_name: str = 'gemini-pro'):
        """
        Initialize the TaskExpander with a specific LLM.
        
        :param model_name: Name of the generative AI model to use
        """
        # Configure the generative AI
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(model_name)

    def expand_task(
        self, 
        task_description: str, 
        num_subtasks: int = 3, 
        focus_prompt: Optional[str] = None,
        use_research: bool = False
    ) -> List[str]:
        """
        Expand a complex task into subtasks.
        
        :param task_description: The main task to be expanded
        :param num_subtasks: Number of subtasks to generate
        :param focus_prompt: Optional additional context for task expansion
        :param use_research: Whether to use research-backed generation
        :return: List of subtask descriptions
        """
        try:
            # Construct the prompt
            prompt = f"""
            Break down the following task into {num_subtasks} specific, actionable subtasks:

            Main Task: {task_description}

            {f'Additional Focus: {focus_prompt}' if focus_prompt else ''}

            Guidelines:
            - Each subtask should be clear and specific
            - Subtasks should be logically sequenced
            - Focus on concrete, implementable steps
            - Ensure subtasks collectively cover the main task's objectives

            Output Format: JSON array of subtask descriptions
            """

            # Generate subtasks
            response = self.model.generate_content(prompt)
            
            # Extract JSON from the response
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group(0))
            else:
                # Fallback parsing if JSON extraction fails
                subtasks = [task.strip() for task in response.text.split('\n') if task.strip()]

            # Log the task expansion
            log_task_expansion(
                logger, 
                task_description, 
                subtasks, 
                num_subtasks, 
                focus_prompt
            )

            return subtasks[:num_subtasks]

        except Exception as e:
            # Log the error
            log_task_expansion_error(
                logger, 
                task_description, 
                e, 
                num_subtasks
            )
            
            # Provide a default fallback if generation fails
            fallback_subtasks = [
                f"Subtask 1 for {task_description}",
                f"Subtask 2 for {task_description}",
                f"Subtask 3 for {task_description}"
            ]
            
            return fallback_subtasks

def expand_task(
    task_description: str, 
    num_subtasks: int = 3, 
    focus_prompt: Optional[str] = None,
    use_research: bool = False
) -> Dict[str, Any]:
    """
    Public interface for task expansion.
    
    :param task_description: The main task to be expanded
    :param num_subtasks: Number of subtasks to generate
    :param focus_prompt: Optional additional context for task expansion
    :param use_research: Whether to use research-backed generation
    :return: Dictionary with task expansion results
    """
    expander = TaskExpander()
    subtasks = expander.expand_task(
        task_description, 
        num_subtasks, 
        focus_prompt, 
        use_research
    )
    
    return {
        "original_task": task_description,
        "subtasks": subtasks
    }

# Example usage and testing
if __name__ == "__main__":
    test_task = "Develop a new web application for task management"
    result = expand_task(test_task)
    print(json.dumps(result, indent=2))