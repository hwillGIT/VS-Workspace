"""
DSPy signatures for agent cognition.

This module defines the DSPy signatures used by different agents
for optimizable prompt engineering and structured outputs.
"""

from typing import Any, Dict, List, Optional, Type, Union
import logging

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    # Mock DSPy classes for fallback
    class MockDSPy:
        class Signature:
            pass
        class InputField:
            def __init__(self, desc=""):
                self.desc = desc
        class OutputField:
            def __init__(self, desc=""):
                self.desc = desc
    dspy = MockDSPy()


class AgentSignatures:
    """
    Repository of DSPy signatures for different agent types.
    
    This class maintains all the signature definitions used by agents
    for structured, optimizable interactions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._signatures: Dict[str, Type[dspy.Signature]] = {}
        
        if DSPY_AVAILABLE:
            self._initialize_signatures()
        else:
            self.logger.warning("DSPy not available, signatures will be mocked")
    
    def _initialize_signatures(self) -> None:
        """Initialize all agent signatures."""
        
        # Manager Agent Signatures
        self._register_manager_signatures()
        
        # Coder Agent Signatures  
        self._register_coder_signatures()
        
        # Reviewer Agent Signatures
        self._register_reviewer_signatures()
        
        # Researcher Agent Signatures
        self._register_researcher_signatures()
        
        # General Purpose Signatures
        self._register_general_signatures()
        
        self.logger.info(f"Initialized {len(self._signatures)} DSPy signatures")
    
    def _register_manager_signatures(self) -> None:
        """Register Manager Agent signatures."""
        
        class TaskDecomposition(dspy.Signature):
            """Break down complex tasks into manageable subtasks."""
            
            project_description = dspy.InputField(desc="The project or task to be broken down")
            requirements = dspy.InputField(desc="Specific requirements and constraints")
            context = dspy.InputField(desc="Additional context about the project")
            
            subtasks = dspy.OutputField(desc="JSON list of subtasks with id, title, description, agent_type, priority, dependencies, estimated_effort")
            approach = dspy.OutputField(desc="High-level approach and architectural decisions")
            risks = dspy.OutputField(desc="Identified risks and mitigation strategies")
        
        class TaskCoordination(dspy.Signature):
            """Coordinate task execution and agent assignment."""
            
            current_plan = dspy.InputField(desc="Current project plan state")
            available_agents = dspy.InputField(desc="List of available agents and their capabilities")
            completed_tasks = dspy.InputField(desc="Recently completed tasks and their results")
            
            next_actions = dspy.OutputField(desc="Next tasks to execute with agent assignments")
            plan_updates = dspy.OutputField(desc="Any updates needed to the project plan")
            coordination_notes = dspy.OutputField(desc="Notes about coordination decisions")
        
        class ProjectPlanning(dspy.Signature):
            """Create comprehensive project plans."""
            
            project_scope = dspy.InputField(desc="Project scope and objectives")
            resources = dspy.InputField(desc="Available resources and constraints")
            timeline = dspy.InputField(desc="Timeline requirements and milestones")
            
            project_plan = dspy.OutputField(desc="Detailed project plan with phases, tasks, and dependencies")
            resource_allocation = dspy.OutputField(desc="Resource allocation strategy")
            success_criteria = dspy.OutputField(desc="Success criteria and metrics")
        
        self._signatures.update({
            "task_decomposition": TaskDecomposition,
            "task_coordination": TaskCoordination,
            "project_planning": ProjectPlanning
        })
    
    def _register_coder_signatures(self) -> None:
        """Register Coder Agent signatures."""
        
        class CodeImplementation(dspy.Signature):
            """Implement code based on specifications."""
            
            specification = dspy.InputField(desc="Detailed specification of what to implement")
            requirements = dspy.InputField(desc="Technical requirements and constraints")
            context = dspy.InputField(desc="Existing codebase context and patterns to follow")
            
            implementation_plan = dspy.OutputField(desc="High-level implementation approach")
            code_files = dspy.OutputField(desc="JSON list of code files to create/modify with path, language, content, description")
            tests = dspy.OutputField(desc="Test cases and testing strategy")
            documentation = dspy.OutputField(desc="Documentation and usage examples")
        
        class CodeRefactoring(dspy.Signature):
            """Refactor existing code for improved quality."""
            
            existing_code = dspy.InputField(desc="Current code that needs refactoring")
            refactoring_goals = dspy.InputField(desc="What to improve (performance, readability, maintainability, etc.)")
            constraints = dspy.InputField(desc="Constraints and requirements to maintain")
            
            refactored_code = dspy.OutputField(desc="Improved code implementation")
            changes_summary = dspy.OutputField(desc="Summary of changes made and rationale")
            impact_analysis = dspy.OutputField(desc="Analysis of potential impacts and risks")
        
        class CodeDebugging(dspy.Signature):
            """Debug and fix code issues."""
            
            problematic_code = dspy.InputField(desc="Code with bugs or issues")
            error_description = dspy.InputField(desc="Description of the problem or error")
            context = dspy.InputField(desc="Additional context about when/how the error occurs")
            
            root_cause = dspy.OutputField(desc="Identified root cause of the issue")
            fixed_code = dspy.OutputField(desc="Corrected code implementation")
            testing_recommendations = dspy.OutputField(desc="Recommendations for testing the fix")
        
        class TestGeneration(dspy.Signature):
            """Generate comprehensive test cases."""
            
            code_to_test = dspy.InputField(desc="Code that needs test coverage")
            test_requirements = dspy.InputField(desc="Testing requirements and coverage goals")
            test_framework = dspy.InputField(desc="Preferred testing framework and conventions")
            
            test_plan = dspy.OutputField(desc="Comprehensive test plan and strategy")
            test_code = dspy.OutputField(desc="Generated test code with multiple test cases")
            coverage_analysis = dspy.OutputField(desc="Analysis of test coverage and gaps")
        
        self._signatures.update({
            "code_implementation": CodeImplementation,
            "code_refactoring": CodeRefactoring,
            "code_debugging": CodeDebugging,
            "test_generation": TestGeneration
        })
    
    def _register_reviewer_signatures(self) -> None:
        """Register Reviewer Agent signatures."""
        
        class CodeReview(dspy.Signature):
            """Conduct comprehensive code review."""
            
            code_content = dspy.InputField(desc="Code to be reviewed")
            file_path = dspy.InputField(desc="File path and context")
            review_criteria = dspy.InputField(desc="Specific criteria and standards to apply")
            
            overall_assessment = dspy.OutputField(desc="Overall code quality assessment and score (1-10)")
            quality_findings = dspy.OutputField(desc="JSON list of code quality findings with category, severity, title, description, line_number, suggestion")
            security_findings = dspy.OutputField(desc="JSON list of security-related findings")
            performance_findings = dspy.OutputField(desc="JSON list of performance-related findings")
            recommendations = dspy.OutputField(desc="High-level recommendations for improvement")
        
        class SecurityAnalysis(dspy.Signature):
            """Analyze code for security vulnerabilities."""
            
            code_content = dspy.InputField(desc="Code to analyze for security issues")
            language = dspy.InputField(desc="Programming language")
            context = dspy.InputField(desc="Application context and security requirements")
            
            vulnerability_assessment = dspy.OutputField(desc="Assessment of potential security vulnerabilities")
            security_score = dspy.OutputField(desc="Security score (1-10) with justification")
            critical_issues = dspy.OutputField(desc="JSON list of critical security issues")
            recommendations = dspy.OutputField(desc="Security improvement recommendations")
        
        class ArchitectureReview(dspy.Signature):
            """Review architectural and design decisions."""
            
            code_structure = dspy.InputField(desc="Code structure and architecture")
            design_patterns = dspy.InputField(desc="Design patterns and architectural decisions")
            requirements = dspy.InputField(desc="Functional and non-functional requirements")
            
            architecture_assessment = dspy.OutputField(desc="Assessment of architectural quality and design decisions")
            design_patterns_analysis = dspy.OutputField(desc="Analysis of design pattern usage and appropriateness")
            scalability_assessment = dspy.OutputField(desc="Assessment of scalability and maintainability")
            improvement_suggestions = dspy.OutputField(desc="Suggestions for architectural improvements")
        
        class QualityMetrics(dspy.Signature):
            """Calculate and analyze code quality metrics."""
            
            code_content = dspy.InputField(desc="Code to analyze for quality metrics")
            metrics_requirements = dspy.InputField(desc="Specific metrics to calculate and thresholds")
            
            quality_metrics = dspy.OutputField(desc="JSON object with calculated quality metrics")
            metric_analysis = dspy.OutputField(desc="Analysis of metrics and their implications")
            improvement_priorities = dspy.OutputField(desc="Prioritized list of quality improvements")
        
        self._signatures.update({
            "code_review": CodeReview,
            "security_analysis": SecurityAnalysis,
            "architecture_review": ArchitectureReview,
            "quality_metrics": QualityMetrics
        })
    
    def _register_researcher_signatures(self) -> None:
        """Register Researcher Agent signatures."""
        
        class SolutionResearch(dspy.Signature):
            """Research solutions for development challenges."""
            
            problem_description = dspy.InputField(desc="The problem or challenge to research")
            requirements = dspy.InputField(desc="Specific requirements and constraints")
            context = dspy.InputField(desc="Additional context about the domain and environment")
            
            solution_approaches = dspy.OutputField(desc="JSON list of potential solution approaches with pros/cons")
            technology_recommendations = dspy.OutputField(desc="Recommended technologies, frameworks, and libraries")
            implementation_considerations = dspy.OutputField(desc="Key considerations for implementation")
            risk_analysis = dspy.OutputField(desc="Potential risks and mitigation strategies")
        
        class TechnologyAnalysis(dspy.Signature):
            """Analyze technologies and frameworks."""
            
            technology_name = dspy.InputField(desc="Name of the technology to analyze")
            use_case = dspy.InputField(desc="Specific use case or application context")
            alternatives = dspy.InputField(desc="Alternative technologies to compare against")
            
            technology_overview = dspy.OutputField(desc="Overview of the technology and its capabilities")
            pros_and_cons = dspy.OutputField(desc="Detailed pros and cons analysis")
            use_cases = dspy.OutputField(desc="Best use cases and scenarios for this technology")
            comparison = dspy.OutputField(desc="Comparison with alternatives")
            recommendation = dspy.OutputField(desc="Recommendation with justification")
        
        class CodebaseAnalysis(dspy.Signature):
            """Analyze existing codebases."""
            
            codebase_structure = dspy.InputField(desc="Structure and organization of the codebase")
            code_samples = dspy.InputField(desc="Representative code samples")
            documentation = dspy.InputField(desc="Available documentation and comments")
            
            architecture_analysis = dspy.OutputField(desc="Analysis of the codebase architecture and patterns")
            technology_stack = dspy.OutputField(desc="Identified technology stack and dependencies")
            code_quality_assessment = dspy.OutputField(desc="Assessment of code quality and maintainability")
            improvement_opportunities = dspy.OutputField(desc="Identified opportunities for improvement")
            integration_points = dspy.OutputField(desc="Key integration points and APIs")
        
        class RequirementsAnalysis(dspy.Signature):
            """Analyze and clarify project requirements."""
            
            raw_requirements = dspy.InputField(desc="Initial requirements description")
            stakeholder_input = dspy.InputField(desc="Input from various stakeholders")
            constraints = dspy.InputField(desc="Known constraints and limitations")
            
            functional_requirements = dspy.OutputField(desc="Clear functional requirements list")
            non_functional_requirements = dspy.OutputField(desc="Performance, security, and other non-functional requirements")
            requirements_gaps = dspy.OutputField(desc="Identified gaps or ambiguities in requirements")
            clarification_questions = dspy.OutputField(desc="Questions to resolve requirements ambiguity")
        
        self._signatures.update({
            "solution_research": SolutionResearch,
            "technology_analysis": TechnologyAnalysis,
            "codebase_analysis": CodebaseAnalysis,
            "requirements_analysis": RequirementsAnalysis
        })
    
    def _register_general_signatures(self) -> None:
        """Register general-purpose signatures."""
        
        class ProblemSolving(dspy.Signature):
            """General problem-solving signature."""
            
            problem_statement = dspy.InputField(desc="Clear statement of the problem to solve")
            context = dspy.InputField(desc="Relevant context and background information")
            constraints = dspy.InputField(desc="Constraints and limitations to consider")
            
            problem_analysis = dspy.OutputField(desc="Analysis of the problem and its components")
            solution_approach = dspy.OutputField(desc="Recommended approach to solve the problem")
            implementation_steps = dspy.OutputField(desc="Step-by-step implementation plan")
            success_criteria = dspy.OutputField(desc="Criteria to evaluate solution success")
        
        class DecisionMaking(dspy.Signature):
            """Make decisions based on available information."""
            
            decision_context = dspy.InputField(desc="Context requiring a decision")
            available_options = dspy.InputField(desc="Available options and alternatives")
            evaluation_criteria = dspy.InputField(desc="Criteria for evaluating options")
            
            option_analysis = dspy.OutputField(desc="Analysis of each available option")
            recommended_decision = dspy.OutputField(desc="Recommended decision with rationale")
            risk_assessment = dspy.OutputField(desc="Assessment of risks associated with the decision")
            implementation_plan = dspy.OutputField(desc="Plan for implementing the decision")
        
        class InformationSynthesis(dspy.Signature):
            """Synthesize information from multiple sources."""
            
            information_sources = dspy.InputField(desc="Multiple sources of information to synthesize")
            synthesis_goals = dspy.InputField(desc="Goals and objectives for the synthesis")
            context = dspy.InputField(desc="Context for the synthesis task")
            
            synthesized_information = dspy.OutputField(desc="Coherent synthesis of the information")
            key_insights = dspy.OutputField(desc="Key insights derived from the synthesis")
            information_gaps = dspy.OutputField(desc="Identified gaps in the available information")
            conclusions = dspy.OutputField(desc="Conclusions drawn from the synthesized information")
        
        self._signatures.update({
            "problem_solving": ProblemSolving,
            "decision_making": DecisionMaking,
            "information_synthesis": InformationSynthesis
        })
    
    def get_signature(self, name: str) -> Optional[Type[dspy.Signature]]:
        """Get a signature by name."""
        return self._signatures.get(name)
    
    def get_all_signatures(self) -> Dict[str, Type[dspy.Signature]]:
        """Get all registered signatures."""
        return self._signatures.copy()
    
    def get_signatures_by_agent(self, agent_type: str) -> Dict[str, Type[dspy.Signature]]:
        """Get signatures for a specific agent type."""
        
        agent_signature_mapping = {
            "manager": [
                "task_decomposition", "task_coordination", "project_planning",
                "decision_making", "problem_solving"
            ],
            "coder": [
                "code_implementation", "code_refactoring", "code_debugging", 
                "test_generation", "problem_solving"
            ],
            "reviewer": [
                "code_review", "security_analysis", "architecture_review",
                "quality_metrics", "decision_making"
            ],
            "researcher": [
                "solution_research", "technology_analysis", "codebase_analysis",
                "requirements_analysis", "information_synthesis"
            ]
        }
        
        agent_signatures = agent_signature_mapping.get(agent_type, [])
        return {name: self._signatures[name] for name in agent_signatures if name in self._signatures}
    
    def register_custom_signature(self, name: str, signature_class: Type[dspy.Signature]) -> bool:
        """Register a custom signature."""
        
        try:
            if not DSPY_AVAILABLE:
                self.logger.warning("DSPy not available, cannot register custom signature")
                return False
            
            self._signatures[name] = signature_class
            self.logger.info(f"Registered custom signature: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register custom signature {name}: {e}")
            return False
    
    def validate_signature(self, name: str) -> bool:
        """Validate that a signature is properly defined."""
        
        if name not in self._signatures:
            return False
        
        try:
            signature_class = self._signatures[name]
            
            # Check if it's a proper DSPy signature
            if not issubclass(signature_class, dspy.Signature):
                return False
            
            # Additional validation could be added here
            return True
            
        except Exception as e:
            self.logger.error(f"Signature validation failed for {name}: {e}")
            return False
    
    def get_signature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a signature."""
        
        if name not in self._signatures:
            return None
        
        try:
            signature_class = self._signatures[name]
            
            # Extract field information
            input_fields = []
            output_fields = []
            
            for field_name, field in signature_class.__annotations__.items():
                if hasattr(signature_class, field_name):
                    field_obj = getattr(signature_class, field_name)
                    if hasattr(field_obj, 'desc'):
                        field_info = {
                            "name": field_name,
                            "description": field_obj.desc,
                            "type": str(field)
                        }
                        
                        if isinstance(field_obj, dspy.InputField):
                            input_fields.append(field_info)
                        elif isinstance(field_obj, dspy.OutputField):
                            output_fields.append(field_info)
            
            return {
                "name": name,
                "class_name": signature_class.__name__,
                "docstring": signature_class.__doc__,
                "input_fields": input_fields,
                "output_fields": output_fields
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get signature info for {name}: {e}")
            return None
    
    def list_signatures(self) -> List[str]:
        """List all available signature names."""
        return list(self._signatures.keys())
    
    def count_signatures(self) -> int:
        """Count total number of signatures."""
        return len(self._signatures)