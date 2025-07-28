import unittest
import json
from expand_task import expand_task

class TestTaskExpander(unittest.TestCase):
    def test_default_expansion(self):
        """
        Test default task expansion with no additional parameters
        """
        task = "Develop a new web application for task management"
        result = expand_task(task)
        
        # Check basic structure
        self.assertIn("original_task", result)
        self.assertIn("subtasks", result)
        self.assertEqual(result["original_task"], task)
        self.assertEqual(len(result["subtasks"]), 3)
        
        # Check subtask content
        for subtask in result["subtasks"]:
            self.assertTrue(isinstance(subtask, str))
            self.assertTrue(len(subtask) > 0)

    def test_custom_subtask_count(self):
        """
        Test task expansion with a custom number of subtasks
        """
        task = "Create a machine learning model for predictive analytics"
        result = expand_task(task, num_subtasks=4)
        
        self.assertEqual(len(result["subtasks"]), 4)

    def test_focus_prompt(self):
        """
        Test task expansion with a focus prompt
        """
        task = "Design an e-commerce platform"
        focus = "Prioritize user experience and security"
        result = expand_task(task, focus_prompt=focus)
        
        self.assertEqual(len(result["subtasks"]), 3)
        
        # Optional: You might want to add more specific checks about the focus

    def test_error_handling(self):
        """
        Test task expansion with an empty task description
        """
        with self.assertRaises(Exception):
            expand_task("")

if __name__ == '__main__':
    unittest.main()