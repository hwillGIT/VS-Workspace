"""
LLM-as-Judge implementation for agent evaluation.

This module provides sophisticated evaluation capabilities using
large language models as judges to assess agent performance,
code quality, and task completion.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .evaluation_types import (
    EvaluationType, EvaluationCriteria, EvaluationResult,
    EvaluationRequest, PerformanceMetrics
)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMJudge:
    """
    LLM-based evaluation system for comprehensive agent assessment.
    
    Uses sophisticated prompting techniques to evaluate agent outputs
    across multiple dimensions including code quality, reasoning,
    communication, and task completion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.provider = config.get("provider", "openai")  # openai, anthropic, local
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.1)  # Low temperature for consistent evaluation
        self.max_tokens = config.get("max_tokens", 2000)
        
        # Evaluation configuration
        self.scoring_scale = config.get("scoring_scale", "1-10")  # "1-10" or "0-1"
        self.require_reasoning = config.get("require_reasoning", True)
        self.use_rubrics = config.get("use_rubrics", True)
        self.multi_pass_evaluation = config.get("multi_pass_evaluation", False)
        
        # Quality control
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.7)
        self.enable_self_consistency = config.get("enable_self_consistency", True)
        self.consistency_samples = config.get("consistency_samples", 3)
        
        # Rate limiting and cost control
        self.max_concurrent_requests = config.get("max_concurrent_requests", 5)
        self.cost_limit_per_evaluation = config.get("cost_limit_per_evaluation", 1.0)
        
        # Clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Evaluation templates and rubrics
        self.evaluation_templates = {}
        self.evaluation_rubrics = {}
        
        # Performance tracking
        self.evaluation_stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_cost": 0.0,
            "avg_evaluation_time": 0.0,
            "avg_confidence_score": 0.0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the LLM judge."""
        
        try:
            # Initialize API clients
            if self.provider == "openai" and OPENAI_AVAILABLE:
                # Initialize OpenAI client (async client would be used in real implementation)
                self.openai_client = "initialized"  # Placeholder
                self.logger.info("OpenAI client initialized")
                
            elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                # Initialize Anthropic client
                self.anthropic_client = "initialized"  # Placeholder
                self.logger.info("Anthropic client initialized")
                
            else:
                self.logger.warning(f"Provider {self.provider} not available, using mock evaluation")
            
            # Load evaluation templates
            await self._load_evaluation_templates()
            
            # Load evaluation rubrics
            await self._load_evaluation_rubrics()
            
            self.initialized = True
            self.logger.info("LLM judge initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM judge: {e}")
            return False
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate an item using LLM-as-Judge.
        
        Args:
            request: Evaluation request with item and criteria
            
        Returns:
            Comprehensive evaluation result
        """
        
        start_time = datetime.now()
        
        try:
            self.evaluation_stats["total_evaluations"] += 1
            
            # Generate evaluation prompt
            prompt = await self._generate_evaluation_prompt(request)
            
            # Perform evaluation
            if self.enable_self_consistency:
                # Multiple evaluations for consistency
                evaluation_responses = []
                for _ in range(self.consistency_samples):
                    response = await self._call_llm(prompt, request.evaluation_type)
                    evaluation_responses.append(response)
                
                # Aggregate responses
                result = await self._aggregate_evaluations(evaluation_responses, request)
            else:
                # Single evaluation
                response = await self._call_llm(prompt, request.evaluation_type)
                result = await self._parse_evaluation_response(response, request)
            
            # Post-process result
            result.evaluation_duration = (datetime.now() - start_time).total_seconds()
            result.evaluation_method = f"llm_judge_{self.provider}"
            
            # Update statistics
            self.evaluation_stats["successful_evaluations"] += 1
            self._update_stats(result)
            
            self.logger.debug(f"Completed evaluation: {result.evaluation_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            self.evaluation_stats["failed_evaluations"] += 1
            
            # Return error result
            return EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                evaluation_type=request.evaluation_type,
                evaluated_item_id=request.item_id,
                evaluator_id=f"llm_judge_{self.provider}",
                created_at=datetime.now(),
                overall_score=0.0,
                summary=f"Evaluation failed: {str(e)}",
                confidence_score=0.0,
                reliability_score=0.0,
                evaluation_method=f"llm_judge_{self.provider}_failed"
            )
    
    async def _generate_evaluation_prompt(self, request: EvaluationRequest) -> str:
        """Generate a comprehensive evaluation prompt."""
        
        try:
            # Get base template for evaluation type
            template = self.evaluation_templates.get(
                request.evaluation_type.value,
                self.evaluation_templates.get("default", "")
            )
            
            # Get rubric if available
            rubric = self.evaluation_rubrics.get(
                request.evaluation_type.value,
                ""
            ) if self.use_rubrics else ""
            
            # Build criteria section
            criteria_section = ""
            if request.criteria:
                criteria_list = [f"- {criterion.value}" for criterion in request.criteria]
                criteria_section = f"\nEvaluate based on these specific criteria:\n" + "\n".join(criteria_list)
            
            if request.custom_criteria:
                custom_criteria_list = [f"- {name}: {desc}" for name, desc in request.custom_criteria.items()]
                criteria_section += f"\n\nAdditional custom criteria:\n" + "\n".join(custom_criteria_list)
            
            # Build context section
            context_section = ""
            if request.context:
                context_items = [f"- {key}: {value}" for key, value in request.context.items()]
                context_section = f"\nContext:\n" + "\n".join(context_items)
            
            # Build requirements section
            requirements_section = ""
            if request.requirements:
                requirements_list = [f"- {req}" for req in request.requirements]
                requirements_section = f"\nRequirements:\n" + "\n".join(requirements_list)
            
            # Build the complete prompt
            prompt = f"""You are an expert evaluator tasked with assessing the following item.

EVALUATION TYPE: {request.evaluation_type.value}

ITEM TO EVALUATE:
{request.item_to_evaluate}

{criteria_section}

{context_section}

{requirements_section}

{rubric}

EVALUATION INSTRUCTIONS:
1. Provide a comprehensive evaluation based on the specified criteria
2. Use a scoring scale of {self.scoring_scale}
3. Provide specific examples and evidence for your assessment
4. Include both strengths and areas for improvement
5. Give actionable recommendations
6. Rate your confidence in this evaluation (0.0 to 1.0)

Please structure your response as a JSON object with the following format:
{{
    "overall_score": <score>,
    "criteria_scores": {{
        "<criterion_name>": <score>,
        ...
    }},
    "summary": "<brief summary>",
    "strengths": ["<strength1>", "<strength2>", ...],
    "weaknesses": ["<weakness1>", "<weakness2>", ...],
    "recommendations": ["<recommendation1>", "<recommendation2>", ...],
    "detailed_feedback": "<detailed analysis>",
    "confidence_score": <0.0 to 1.0>,
    "evidence": [
        {{
            "criterion": "<criterion>",
            "observation": "<what you observed>",
            "score_rationale": "<why this score>"
        }},
        ...
    ],
    "examples": ["<example1>", "<example2>", ...]
}}

Begin your evaluation:"""
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation prompt: {e}")
            return f"Evaluate this item: {request.item_to_evaluate}"
    
    async def _call_llm(self, prompt: str, evaluation_type: EvaluationType) -> str:
        """Call the LLM for evaluation."""
        
        try:
            if self.provider == "openai" and self.openai_client:
                return await self._call_openai(prompt)
            elif self.provider == "anthropic" and self.anthropic_client:
                return await self._call_anthropic(prompt)
            else:
                # Mock evaluation for testing
                return await self._mock_evaluation(prompt, evaluation_type)
                
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        
        try:
            # This would be the actual OpenAI API call
            # For now, return mock response
            await asyncio.sleep(0.1)  # Simulate API call delay
            
            return self._generate_mock_response()
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        
        try:
            # This would be the actual Anthropic API call
            # For now, return mock response
            await asyncio.sleep(0.1)  # Simulate API call delay
            
            return self._generate_mock_response()
            
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            raise
    
    async def _mock_evaluation(self, prompt: str, evaluation_type: EvaluationType) -> str:
        """Generate mock evaluation response."""
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate contextually appropriate mock response
        base_score = 7.5  # Base score out of 10
        
        if evaluation_type == EvaluationType.CODE_QUALITY:
            return json.dumps({
                "overall_score": base_score,
                "criteria_scores": {
                    "correctness": 8.0,
                    "readability": 7.0,
                    "maintainability": 7.5,
                    "performance": 8.0,
                    "security": 7.0
                },
                "summary": "Good code quality with minor areas for improvement",
                "strengths": [
                    "Code is functionally correct",
                    "Good variable naming",
                    "Proper error handling"
                ],
                "weaknesses": [
                    "Could benefit from more comments",
                    "Some functions are slightly long"
                ],
                "recommendations": [
                    "Add more inline documentation",
                    "Consider breaking down larger functions",
                    "Add unit tests for edge cases"
                ],
                "detailed_feedback": "The code demonstrates solid programming practices with correct implementation of the required functionality. The logic is sound and the structure is generally well-organized.",
                "confidence_score": 0.85,
                "evidence": [
                    {
                        "criterion": "correctness",
                        "observation": "Code executes without errors and produces expected output",
                        "score_rationale": "All test cases pass successfully"
                    }
                ],
                "examples": [
                    "The error handling in the main function properly catches and manages exceptions",
                    "Variable names like 'user_input' and 'processed_data' are clear and descriptive"
                ]
            })
        
        elif evaluation_type == EvaluationType.TASK_COMPLETION:
            return json.dumps({
                "overall_score": base_score,
                "criteria_scores": {
                    "completeness": 8.5,
                    "requirement_adherence": 8.0,
                    "timeliness": 7.0
                },
                "summary": "Task completed successfully with all requirements met",
                "strengths": [
                    "All specified requirements implemented",
                    "Solution addresses the core problem effectively"
                ],
                "weaknesses": [
                    "Could have been completed faster",
                    "Some optional features not implemented"
                ],
                "recommendations": [
                    "Consider time management strategies",
                    "Prioritize core features first"
                ],
                "detailed_feedback": "The task was completed successfully with all core requirements satisfied. The solution is functional and meets the specified criteria.",
                "confidence_score": 0.9,
                "evidence": [],
                "examples": []
            })
        
        else:
            return json.dumps({
                "overall_score": base_score,
                "criteria_scores": {},
                "summary": f"Evaluation completed for {evaluation_type.value}",
                "strengths": ["Generally good performance"],
                "weaknesses": ["Some areas for improvement"],
                "recommendations": ["Continue current practices", "Focus on identified weaknesses"],
                "detailed_feedback": f"This {evaluation_type.value} evaluation shows satisfactory performance with room for improvement.",
                "confidence_score": 0.8,
                "evidence": [],
                "examples": []
            })
    
    def _generate_mock_response(self) -> str:
        """Generate a generic mock response."""
        
        return json.dumps({
            "overall_score": 7.5,
            "criteria_scores": {
                "quality": 7.5,
                "effectiveness": 8.0,
                "efficiency": 7.0
            },
            "summary": "Good performance with areas for improvement",
            "strengths": ["Strong foundation", "Good approach"],
            "weaknesses": ["Could be more efficient", "Minor quality issues"],
            "recommendations": ["Focus on optimization", "Address quality concerns"],
            "detailed_feedback": "The evaluation shows solid performance with some opportunities for enhancement.",
            "confidence_score": 0.8,
            "evidence": [],
            "examples": []
        })
    
    async def _parse_evaluation_response(
        self,
        response: str,
        request: EvaluationRequest
    ) -> EvaluationResult:
        """Parse LLM response into evaluation result."""
        
        try:
            # Try to parse JSON response
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError:
                # Fallback: extract JSON from response
                response_data = self._extract_json_from_text(response)
            
            # Convert scoring scale if needed
            overall_score = response_data.get("overall_score", 0.0)
            if self.scoring_scale == "1-10" and overall_score > 1.0:
                overall_score = overall_score / 10.0  # Normalize to 0-1
            
            criteria_scores = response_data.get("criteria_scores", {})
            if self.scoring_scale == "1-10":
                criteria_scores = {k: v/10.0 for k, v in criteria_scores.items()}
            
            # Create evaluation result
            result = EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                evaluation_type=request.evaluation_type,
                evaluated_item_id=request.item_id,
                evaluator_id=f"llm_judge_{self.provider}",
                created_at=datetime.now(),
                evaluation_context=request.context,
                overall_score=overall_score,
                criteria_scores=criteria_scores,
                summary=response_data.get("summary", ""),
                strengths=response_data.get("strengths", []),
                weaknesses=response_data.get("weaknesses", []),
                recommendations=response_data.get("recommendations", []),
                detailed_feedback=response_data.get("detailed_feedback", ""),
                confidence_score=response_data.get("confidence_score", 0.8),
                reliability_score=0.8,  # Default reliability
                evidence=response_data.get("evidence", []),
                examples=response_data.get("examples", [])
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse evaluation response: {e}")
            
            # Return minimal result
            return EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                evaluation_type=request.evaluation_type,
                evaluated_item_id=request.item_id,
                evaluator_id=f"llm_judge_{self.provider}",
                created_at=datetime.now(),
                overall_score=0.5,
                summary="Failed to parse evaluation response",
                confidence_score=0.1
            )
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response."""
        
        try:
            # Look for JSON block in the response
            start_markers = ["{", "```json"]
            end_markers = ["}", "```"]
            
            json_start = -1
            json_end = -1
            
            # Find JSON start
            for marker in start_markers:
                pos = text.find(marker)
                if pos != -1:
                    json_start = pos if marker == "{" else pos + len(marker)
                    break
            
            # Find JSON end
            if json_start != -1:
                for marker in end_markers:
                    pos = text.rfind(marker, json_start)
                    if pos != -1:
                        json_end = pos + (1 if marker == "}" else 0)
                        break
            
            if json_start != -1 and json_end != -1:
                json_text = text[json_start:json_end]
                return json.loads(json_text)
            
            # Fallback: return minimal structure
            return {
                "overall_score": 0.5,
                "summary": "Could not parse evaluation response",
                "confidence_score": 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract JSON: {e}")
            return {"overall_score": 0.0, "summary": "JSON extraction failed"}
    
    async def _aggregate_evaluations(
        self,
        responses: List[str],
        request: EvaluationRequest
    ) -> EvaluationResult:
        """Aggregate multiple evaluation responses for consistency."""
        
        try:
            parsed_responses = []
            
            # Parse all responses
            for response in responses:
                parsed = await self._parse_evaluation_response(response, request)
                parsed_responses.append(parsed)
            
            if not parsed_responses:
                raise ValueError("No valid responses to aggregate")
            
            # Aggregate scores
            overall_scores = [r.overall_score for r in parsed_responses]
            avg_overall_score = sum(overall_scores) / len(overall_scores)
            
            # Aggregate criteria scores
            all_criteria = set()
            for response in parsed_responses:
                all_criteria.update(response.criteria_scores.keys())
            
            aggregated_criteria_scores = {}
            for criterion in all_criteria:
                scores = [r.criteria_scores.get(criterion, 0) for r in parsed_responses]
                scores = [s for s in scores if s > 0]  # Remove missing scores
                if scores:
                    aggregated_criteria_scores[criterion] = sum(scores) / len(scores)
            
            # Aggregate qualitative feedback
            all_strengths = []
            all_weaknesses = []
            all_recommendations = []
            
            for response in parsed_responses:
                all_strengths.extend(response.strengths)
                all_weaknesses.extend(response.weaknesses)
                all_recommendations.extend(response.recommendations)
            
            # Remove duplicates while preserving order
            unique_strengths = list(dict.fromkeys(all_strengths))
            unique_weaknesses = list(dict.fromkeys(all_weaknesses))
            unique_recommendations = list(dict.fromkeys(all_recommendations))
            
            # Calculate consistency metrics
            score_variance = sum((s - avg_overall_score) ** 2 for s in overall_scores) / len(overall_scores)
            consistency_score = max(0.0, 1.0 - score_variance)
            
            # Use the first response as template and update with aggregated data
            result = parsed_responses[0]
            result.overall_score = avg_overall_score
            result.criteria_scores = aggregated_criteria_scores
            result.strengths = unique_strengths
            result.weaknesses = unique_weaknesses
            result.recommendations = unique_recommendations
            result.detailed_feedback = f"Aggregated evaluation from {len(responses)} samples. " + result.detailed_feedback
            result.confidence_score = consistency_score
            result.reliability_score = consistency_score
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate evaluations: {e}")
            # Return first response if aggregation fails
            return await self._parse_evaluation_response(responses[0], request) if responses else EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                evaluation_type=request.evaluation_type,
                evaluated_item_id=request.item_id,
                evaluator_id=f"llm_judge_{self.provider}",
                created_at=datetime.now(),
                overall_score=0.0
            )
    
    async def _load_evaluation_templates(self) -> None:
        """Load evaluation prompt templates."""
        
        # Default templates for different evaluation types
        self.evaluation_templates = {
            "code_quality": """
You are a senior software engineer evaluating code quality.
Focus on correctness, readability, maintainability, performance, and security.
Consider industry best practices and common code smells.
""",
            "task_completion": """
You are a project manager evaluating task completion.
Assess whether all requirements have been met, the quality of the solution,
and the effectiveness of the approach taken.
""",
            "communication": """
You are a communication expert evaluating clarity and effectiveness.
Focus on clarity, conciseness, tone, structure, and audience appropriateness.
""",
            "reasoning": """
You are a logic and reasoning expert evaluating thought processes.
Assess logical flow, evidence support, assumption validity, and conclusion soundness.
""",
            "default": """
You are an expert evaluator with deep knowledge in the relevant domain.
Provide a comprehensive assessment based on industry standards and best practices.
"""
        }
    
    async def _load_evaluation_rubrics(self) -> None:
        """Load detailed evaluation rubrics."""
        
        # Detailed rubrics for consistent evaluation
        self.evaluation_rubrics = {
            "code_quality": """
EVALUATION RUBRIC:
Correctness (25%): Does the code work as intended? Are there bugs or logical errors?
- Excellent (9-10): No bugs, handles all edge cases, produces correct output
- Good (7-8): Minor issues, handles most cases correctly
- Fair (5-6): Some bugs or logical errors, basic functionality works
- Poor (1-4): Major bugs, incorrect output, doesn't compile/run

Readability (20%): Is the code easy to understand?
- Excellent (9-10): Clear naming, good structure, self-documenting
- Good (7-8): Mostly clear, some unclear parts
- Fair (5-6): Somewhat difficult to follow, inconsistent style
- Poor (1-4): Very difficult to read, poor naming, no structure

Maintainability (20%): How easy would it be to modify or extend?
- Excellent (9-10): Modular, well-organized, follows SOLID principles
- Good (7-8): Generally well-structured, some tightly coupled parts
- Fair (5-6): Some structure, but modifications would be challenging
- Poor (1-4): Monolithic, tightly coupled, hard to modify

Performance (15%): How efficient is the code?
- Excellent (9-10): Optimal algorithms, efficient resource usage
- Good (7-8): Good performance, minor inefficiencies
- Fair (5-6): Acceptable performance, some optimization needed
- Poor (1-4): Poor performance, inefficient algorithms

Security (10%): Are there security vulnerabilities?
- Excellent (9-10): Secure coding practices, no vulnerabilities
- Good (7-8): Generally secure, minor issues
- Fair (5-6): Some security concerns
- Poor (1-4): Major security vulnerabilities

Documentation (10%): Are there appropriate comments and documentation?
- Excellent (9-10): Comprehensive documentation, clear comments
- Good (7-8): Good documentation, most areas covered
- Fair (5-6): Basic documentation, some areas unclear
- Poor (1-4): Little to no documentation
""",
            "task_completion": """
EVALUATION RUBRIC:
Completeness (40%): Are all requirements satisfied?
- Excellent (9-10): All requirements fully implemented
- Good (7-8): Most requirements met, minor gaps
- Fair (5-6): Core requirements met, some missing features
- Poor (1-4): Major requirements missing

Quality (30%): How well is the task executed?
- Excellent (9-10): High-quality solution, exceeds expectations
- Good (7-8): Good quality, meets expectations
- Fair (5-6): Acceptable quality, some issues
- Poor (1-4): Poor quality, many issues

Timeliness (15%): Was the task completed on time?
- Excellent (9-10): Completed early or exactly on time
- Good (7-8): Minor delay, still within reasonable time
- Fair (5-6): Moderate delay
- Poor (1-4): Significant delay

Approach (15%): Was the approach effective?
- Excellent (9-10): Optimal approach, innovative solution
- Good (7-8): Good approach, effective solution
- Fair (5-6): Reasonable approach, some inefficiencies
- Poor (1-4): Poor approach, many issues
"""
        }
    
    def _update_stats(self, result: EvaluationResult) -> None:
        """Update evaluation statistics."""
        
        # Update averages
        total_evals = self.evaluation_stats["successful_evaluations"]
        
        # Update average evaluation time
        current_avg_time = self.evaluation_stats["avg_evaluation_time"]
        new_avg_time = ((current_avg_time * (total_evals - 1)) + result.evaluation_duration) / total_evals
        self.evaluation_stats["avg_evaluation_time"] = new_avg_time
        
        # Update average confidence score
        current_avg_confidence = self.evaluation_stats["avg_confidence_score"]
        new_avg_confidence = ((current_avg_confidence * (total_evals - 1)) + result.confidence_score) / total_evals
        self.evaluation_stats["avg_confidence_score"] = new_avg_confidence
        
        # Update total cost (would be calculated from API usage in real implementation)
        self.evaluation_stats["total_cost"] += result.evaluation_cost
    
    async def evaluate_batch(self, requests: List[EvaluationRequest]) -> List[EvaluationResult]:
        """Evaluate multiple requests in batch."""
        
        try:
            # Process requests concurrently with rate limiting
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            async def evaluate_with_semaphore(request):
                async with semaphore:
                    return await self.evaluate(request)
            
            # Execute all evaluations
            tasks = [evaluate_with_semaphore(request) for request in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch evaluation {i} failed: {result}")
                else:
                    successful_results.append(result)
            
            return successful_results
            
        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            return []
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation performance statistics."""
        
        return {
            **self.evaluation_stats,
            "initialized": self.initialized,
            "provider": self.provider,
            "model": self.model,
            "config": {
                "scoring_scale": self.scoring_scale,
                "require_reasoning": self.require_reasoning,
                "use_rubrics": self.use_rubrics,
                "multi_pass_evaluation": self.multi_pass_evaluation,
                "enable_self_consistency": self.enable_self_consistency,
                "consistency_samples": self.consistency_samples
            },
            "success_rate": (
                self.evaluation_stats["successful_evaluations"] / 
                max(1, self.evaluation_stats["total_evaluations"])
            )
        }
    
    async def shutdown(self) -> None:
        """Shutdown the LLM judge."""
        
        try:
            # Close API clients
            self.openai_client = None
            self.anthropic_client = None
            
            self.initialized = False
            self.logger.info("LLM judge shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during LLM judge shutdown: {e}")