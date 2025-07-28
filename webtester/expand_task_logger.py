import logging
import os
from datetime import datetime

def setup_task_expansion_logging(log_dir='logs'):
    """
    Set up logging for task expansion operations.
    
    :param log_dir: Directory to store log files
    :return: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'task_expansion_{timestamp}.log')
    
    # Configure logger
    logger = logging.getLogger('TaskExpansionLogger')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_task_expansion(
    logger, 
    original_task: str, 
    subtasks: list, 
    num_subtasks: int, 
    focus_prompt: str = None
):
    """
    Log details of a task expansion operation.
    
    :param logger: Logging instance
    :param original_task: The original task description
    :param subtasks: List of generated subtasks
    :param num_subtasks: Number of subtasks requested
    :param focus_prompt: Optional focus prompt used
    """
    logger.info(f"Task Expansion Operation")
    logger.info(f"Original Task: {original_task}")
    logger.info(f"Requested Subtask Count: {num_subtasks}")
    
    if focus_prompt:
        logger.info(f"Focus Prompt: {focus_prompt}")
    
    logger.info("Generated Subtasks:")
    for i, subtask in enumerate(subtasks, 1):
        logger.info(f"  {i}. {subtask}")

def log_task_expansion_error(
    logger, 
    original_task: str, 
    error: Exception, 
    num_subtasks: int
):
    """
    Log errors during task expansion.
    
    :param logger: Logging instance
    :param original_task: The original task description
    :param error: Exception that occurred
    :param num_subtasks: Number of subtasks requested
    """
    logger.error(f"Task Expansion Error")
    logger.error(f"Original Task: {original_task}")
    logger.error(f"Requested Subtask Count: {num_subtasks}")
    logger.error(f"Error Details: {str(error)}")
    logger.exception(error)

# Example usage
if __name__ == "__main__":
    logger = setup_task_expansion_logging()
    
    # Simulate a successful task expansion
    original_task = "Develop a new web application for task management"
    subtasks = [
        "Design the database schema",
        "Create user authentication system",
        "Implement task CRUD operations"
    ]
    
    log_task_expansion(
        logger, 
        original_task, 
        subtasks, 
        num_subtasks=3, 
        focus_prompt="Focus on security and scalability"
    )
    
    # Simulate an error scenario
    try:
        raise ValueError("Example error during task expansion")
    except Exception as e:
        log_task_expansion_error(
            logger, 
            original_task, 
            e, 
            num_subtasks=3
        )