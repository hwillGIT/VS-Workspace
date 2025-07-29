"""
Researcher Agent implementation for information gathering and analysis.

The Researcher Agent is responsible for:
- Researching solutions, technologies, and best practices
- Analyzing existing codebases and documentation
- Gathering requirements and understanding problem contexts
- Investigating APIs, libraries, and frameworks
- Conducting competitive analysis and benchmarking
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

import dspy
from pydantic import BaseModel, Field

from .base_agent import BaseAgent


class ResearchType(Enum):
    """Types of research tasks."""
    SOLUTION_RESEARCH = "solution_research"
    TECHNOLOGY_ANALYSIS = "technology_analysis"
    CODEBASE_ANALYSIS = "codebase_analysis"
    REQUIREMENTS_GATHERING = "requirements_gathering"
    API_RESEARCH = "api_research"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    DOCUMENTATION_ANALYSIS = "documentation_analysis"


class InformationSource(Enum):
    """Sources of information for research."""
    CODEBASE = "codebase"
    DOCUMENTATION = "documentation"
    API_DOCS = "api_docs"
    EXTERNAL_SEARCH = "external_search"
    MEMORY = "memory"
    CONFIGURATION = "configuration"


@dataclass
class ResearchFinding:
    """Represents a research finding or piece of information."""
    title: str
    description: str
    source: InformationSource
    relevance_score: float
    confidence: float
    url: Optional[str] = None
    timestamp: datetime = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.tags is None:
            self.tags = []


class SolutionResearch(dspy.Signature):
    """DSPy signature for researching solutions to problems."""
    
    problem_description = dspy.InputField(desc="The problem or challenge to research")
    requirements = dspy.InputField(desc="Specific requirements and constraints")
    context = dspy.InputField(desc="Additional context about the domain and environment")
    
    solution_approaches = dspy.OutputField(desc="JSON list of potential solution approaches with pros/cons")
    technology_recommendations = dspy.OutputField(desc="Recommended technologies, frameworks, and libraries")
    implementation_considerations = dspy.OutputField(desc="Key considerations for implementation")
    risk_analysis = dspy.OutputField(desc="Potential risks and mitigation strategies")


class TechnologyAnalysis(dspy.Signature):
    """DSPy signature for analyzing technologies and frameworks."""
    
    technology_name = dspy.InputField(desc="Name of the technology to analyze")
    use_case = dspy.InputField(desc="Specific use case or application context")
    alternatives = dspy.InputField(desc="Alternative technologies to compare against")
    
    technology_overview = dspy.OutputField(desc="Overview of the technology and its capabilities")
    pros_and_cons = dspy.OutputField(desc="Detailed pros and cons analysis")
    use_cases = dspy.OutputField(desc="Best use cases and scenarios for this technology")
    comparison = dspy.OutputField(desc="Comparison with alternatives")
    recommendation = dspy.OutputField(desc="Recommendation with justification")


class CodebaseAnalysis(dspy.Signature):
    """DSPy signature for analyzing existing codebases."""
    
    codebase_structure = dspy.InputField(desc="Structure and organization of the codebase")
    code_samples = dspy.InputField(desc="Representative code samples")
    documentation = dspy.InputField(desc="Available documentation and comments")
    
    architecture_analysis = dspy.OutputField(desc="Analysis of the codebase architecture and patterns")
    technology_stack = dspy.OutputField(desc="Identified technology stack and dependencies")
    code_quality_assessment = dspy.OutputField(desc="Assessment of code quality and maintainability")
    improvement_opportunities = dspy.OutputField(desc="Identified opportunities for improvement")
    integration_points = dspy.OutputField(desc="Key integration points and APIs")


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent responsible for information gathering and analysis.
    
    This agent conducts comprehensive research on solutions, technologies,
    and codebases to inform development decisions and provide context
    for other agents.
    """
    
    def __init__(self, agent_id: str = "researcher", **kwargs):
        super().__init__(agent_id, **kwargs)
        
        # DSPy modules for research capabilities
        if self.dspy_enabled:
            self.solution_researcher = dspy.TypedChainOfThought(SolutionResearch)
            self.technology_analyzer = dspy.TypedChainOfThought(TechnologyAnalysis)
            self.codebase_analyzer = dspy.TypedChainOfThought(CodebaseAnalysis)
        
        # Research configuration
        self.max_search_results = self.config.get("max_search_results", 20)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.7)
        self.max_file_size = self.config.get("max_file_size", 1024 * 1024)  # 1MB
        
        # Research cache to avoid duplicate research
        self.research_cache: Dict[str, Dict[str, Any]] = {}
        
        # Common file extensions for different analysis types
        self.code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.h', '.cs', '.go', '.rs'}
        self.doc_extensions = {'.md', '.txt', '.rst', '.doc', '.docx', '.pdf'}
        self.config_extensions = {'.json', '.yaml', '.yml', '.xml', '.ini', '.toml'}
        
        self.logger.info("Researcher Agent initialized and ready for research tasks")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Researcher Agent."""
        return """You are the Researcher Agent in a Self-Reflecting Claude Code Agent system.

Your primary responsibilities are:
1. Researching solutions, technologies, and best practices for development challenges
2. Analyzing existing codebases to understand architecture, patterns, and conventions
3. Gathering and analyzing requirements to inform implementation decisions
4. Investigating APIs, libraries, and frameworks for suitability
5. Conducting competitive analysis and technology comparisons
6. Providing comprehensive, well-structured research findings

When conducting research, focus on:
- Thoroughness and accuracy of information
- Relevance to the specific problem or context
- Comparative analysis of alternatives
- Practical implementation considerations
- Risk assessment and mitigation strategies
- Clear documentation of sources and findings

Always provide evidence-based recommendations with clear rationale,
consider multiple perspectives, and identify potential risks or limitations.
Structure your findings clearly and prioritize information by relevance."""
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research task.
        
        Args:
            task: Task specification containing research type, topic, and context
            
        Returns:
            Result dictionary with research findings, analysis, and recommendations
        """
        start_time = datetime.now()
        
        try:
            research_type = task.get("type", ResearchType.SOLUTION_RESEARCH.value)
            self.logger.info(f"Processing {research_type} task: {task.get('title', 'Research Task')}")
            
            # Update state
            self.state.current_task = task.get('title', 'Research Task')
            
            # Check cache first
            cache_key = self._generate_cache_key(task)
            if cache_key in self.research_cache:
                self.logger.info("Returning cached research results")
                cached_result = self.research_cache[cache_key].copy()
                cached_result["from_cache"] = True
                cached_result["response_time"] = (datetime.now() - start_time).total_seconds()
                return cached_result
            
            # Route to appropriate research handler
            if research_type == ResearchType.SOLUTION_RESEARCH.value:
                result = await self._handle_solution_research(task)
            elif research_type == ResearchType.TECHNOLOGY_ANALYSIS.value:
                result = await self._handle_technology_analysis(task)
            elif research_type == ResearchType.CODEBASE_ANALYSIS.value:
                result = await self._handle_codebase_analysis(task)
            elif research_type == ResearchType.REQUIREMENTS_GATHERING.value:
                result = await self._handle_requirements_gathering(task)
            elif research_type == ResearchType.API_RESEARCH.value:
                result = await self._handle_api_research(task)
            else:
                result = await self._handle_general_research(task)
            
            # Cache results
            self.research_cache[cache_key] = result.copy()
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            success = result.get("status") == "completed"
            self.update_metrics(response_time, success)
            
            # Store in memory
            await self.update_memory(
                f"Completed {research_type}: {task.get('title', 'Research')}",
                {"task_type": research_type, "result": result}
            )
            
            result["response_time"] = response_time
            result["from_cache"] = False
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing research task: {str(e)}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return {
                "status": "failed",
                "error": str(e),
                "response_time": response_time
            }
    
    def _generate_cache_key(self, task: Dict[str, Any]) -> str:
        """Generate a cache key for the task."""
        
        key_parts = [
            task.get("type", ""),
            task.get("title", ""),
            str(hash(str(task.get("description", ""))))
        ]
        return "|".join(key_parts)
    
    async def _handle_solution_research(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solution research tasks."""
        
        problem_description = task.get("description", "")
        requirements = task.get("requirements", {})
        context = task.get("context", {})
        
        # Gather existing knowledge from memory
        memory_findings = await self._search_memory_for_solutions(problem_description)
        
        # Analyze codebase for existing patterns
        codebase_findings = await self._analyze_existing_patterns(problem_description)
        
        if self.dspy_enabled:
            try:
                result = self.solution_researcher(
                    problem_description=problem_description,
                    requirements=json.dumps(requirements, indent=2),
                    context=json.dumps(context, indent=2)
                )
                
                # Parse solution approaches
                try:
                    solution_approaches = json.loads(result.solution_approaches)
                except json.JSONDecodeError:
                    solution_approaches = [{"approach": result.solution_approaches, "pros": "", "cons": ""}]
                
                return {
                    "status": "completed",
                    "problem_description": problem_description,
                    "solution_approaches": solution_approaches,
                    "technology_recommendations": result.technology_recommendations,
                    "implementation_considerations": result.implementation_considerations,
                    "risk_analysis": result.risk_analysis,
                    "memory_findings": memory_findings,
                    "codebase_patterns": codebase_findings,
                    "research_quality": "high"
                }
                
            except Exception as e:
                self.logger.warning(f"DSPy solution research failed: {e}")
        
        # Fallback research
        fallback_solutions = await self._generate_fallback_solutions(problem_description, requirements)
        
        return {
            "status": "completed",
            "problem_description": problem_description,
            "solution_approaches": fallback_solutions,
            "memory_findings": memory_findings,
            "codebase_patterns": codebase_findings,
            "research_quality": "basic"
        }
    
    async def _handle_technology_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle technology analysis tasks."""
        
        technology_name = task.get("technology", "")
        use_case = task.get("use_case", "")
        alternatives = task.get("alternatives", [])
        
        if self.dspy_enabled:
            try:
                result = self.technology_analyzer(
                    technology_name=technology_name,
                    use_case=use_case,
                    alternatives=", ".join(alternatives) if alternatives else "General alternatives"
                )
                
                return {
                    "status": "completed",
                    "technology": technology_name,
                    "overview": result.technology_overview,
                    "pros_and_cons": result.pros_and_cons,
                    "use_cases": result.use_cases,
                    "comparison": result.comparison,
                    "recommendation": result.recommendation,
                    "analysis_depth": "comprehensive"
                }
                
            except Exception as e:
                self.logger.warning(f"DSPy technology analysis failed: {e}")
        
        # Fallback analysis
        basic_analysis = await self._basic_technology_analysis(technology_name, use_case)
        
        return {
            "status": "completed",
            "technology": technology_name,
            "analysis": basic_analysis,
            "analysis_depth": "basic"
        }
    
    async def _handle_codebase_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle codebase analysis tasks."""
        
        codebase_path = task.get("codebase_path", ".")
        focus_areas = task.get("focus_areas", [])
        
        # Analyze codebase structure
        structure_analysis = await self._analyze_codebase_structure(codebase_path)
        
        # Extract code samples
        code_samples = await self._extract_representative_code_samples(codebase_path)
        
        # Gather documentation
        documentation = await self._gather_codebase_documentation(codebase_path)
        
        if self.dspy_enabled:
            try:
                result = self.codebase_analyzer(
                    codebase_structure=json.dumps(structure_analysis, indent=2),
                    code_samples="\n\n".join(code_samples[:5]),  # Limit samples
                    documentation=documentation
                )
                
                return {
                    "status": "completed",
                    "codebase_path": codebase_path,
                    "structure_analysis": structure_analysis,
                    "architecture_analysis": result.architecture_analysis,
                    "technology_stack": result.technology_stack,
                    "code_quality_assessment": result.code_quality_assessment,
                    "improvement_opportunities": result.improvement_opportunities,
                    "integration_points": result.integration_points,
                    "documentation_summary": documentation,
                    "analysis_depth": "comprehensive"
                }
                
            except Exception as e:
                self.logger.warning(f"DSPy codebase analysis failed: {e}")
        
        # Fallback analysis
        return {
            "status": "completed",
            "codebase_path": codebase_path,
            "structure_analysis": structure_analysis,
            "code_samples_count": len(code_samples),
            "documentation_available": bool(documentation),
            "analysis_depth": "basic"
        }
    
    async def _handle_requirements_gathering(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requirements gathering tasks."""
        
        project_description = task.get("description", "")
        stakeholders = task.get("stakeholders", [])
        constraints = task.get("constraints", {})
        
        # Analyze the project description for requirements
        functional_requirements = await self._extract_functional_requirements(project_description)
        non_functional_requirements = await self._extract_non_functional_requirements(project_description)
        
        # Research similar projects for insights
        similar_projects = await self._find_similar_projects(project_description)
        
        return {
            "status": "completed",
            "project_description": project_description,
            "functional_requirements": functional_requirements,
            "non_functional_requirements": non_functional_requirements,
            "stakeholders": stakeholders,
            "constraints": constraints,
            "similar_projects": similar_projects,
            "requirements_completeness": await self._assess_requirements_completeness(
                functional_requirements, non_functional_requirements
            )
        }
    
    async def _handle_api_research(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API research tasks."""
        
        api_name = task.get("api_name", "")
        use_case = task.get("use_case", "")
        
        # This would typically involve actual API research
        # For now, we'll simulate the research process
        
        api_analysis = {
            "api_name": api_name,
            "use_case": use_case,
            "research_summary": f"Research conducted for {api_name} API usage in {use_case}",
            "key_endpoints": [],
            "authentication_methods": [],
            "rate_limits": "To be determined",
            "documentation_quality": "Requires further investigation",
            "integration_complexity": "Medium"
        }
        
        return {
            "status": "completed",
            **api_analysis
        }
    
    async def _handle_general_research(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general research tasks."""
        
        topic = task.get("topic", task.get("title", ""))
        description = task.get("description", "")
        
        # Conduct general research
        findings = []
        
        # Search memory for related information
        memory_results = await self.search_memory(topic, limit=10)
        for result in memory_results:
            findings.append(ResearchFinding(
                title=f"Memory: {result.get('content', '')[:50]}...",
                description=result.get('content', ''),
                source=InformationSource.MEMORY,
                relevance_score=0.8,
                confidence=0.9
            ))
        
        # Analyze local files if relevant
        if "codebase" in description.lower() or "code" in description.lower():
            code_findings = await self._search_local_codebase(topic)
            findings.extend(code_findings)
        
        return {
            "status": "completed",
            "topic": topic,
            "description": description,
            "findings": [self._finding_to_dict(f) for f in findings],
            "total_findings": len(findings),
            "research_scope": "general"
        }
    
    async def _search_memory_for_solutions(self, problem: str) -> List[Dict[str, Any]]:
        """Search memory for relevant solutions."""
        
        if not self.memory:
            return []
        
        # Extract key terms from the problem
        key_terms = self._extract_key_terms(problem)
        
        findings = []
        for term in key_terms[:3]:  # Limit searches
            results = await self.search_memory(term, limit=5)
            findings.extend(results)
        
        # Remove duplicates and return top results
        unique_findings = []
        seen_content = set()
        
        for finding in findings:
            content = finding.get('content', '')
            if content not in seen_content:
                seen_content.add(content)
                unique_findings.append(finding)
        
        return unique_findings[:10]
    
    async def _analyze_existing_patterns(self, problem: str) -> List[Dict[str, Any]]:
        """Analyze existing codebase for relevant patterns."""
        
        patterns = []
        
        # Search for relevant files based on problem description
        key_terms = self._extract_key_terms(problem)
        
        project_root = Path(".")  # Assume current directory
        if project_root.exists():
            for term in key_terms[:2]:  # Limit searches
                matching_files = await self._find_files_by_content(project_root, term)
                
                for file_path in matching_files[:3]:  # Limit files per term
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content) < self.max_file_size:
                                patterns.append({
                                    "file_path": str(file_path),
                                    "content_snippet": content[:500] + "..." if len(content) > 500 else content,
                                    "relevance": "Contains relevant patterns"
                                })
                    except (UnicodeDecodeError, IOError):
                        continue
        
        return patterns
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for searching."""
        
        # Simple keyword extraction
        # Remove common words and extract meaningful terms
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract words, filter out common words, and prioritize longer terms
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        key_terms = [word for word in words if word not in common_words and len(word) > 3]
        
        # Return unique terms sorted by length (longer terms first)
        return list(dict.fromkeys(sorted(key_terms, key=len, reverse=True)))
    
    async def _find_files_by_content(self, root_path: Path, search_term: str) -> List[Path]:
        """Find files containing the search term."""
        
        matching_files = []
        
        try:
            for file_path in root_path.rglob("*"):
                if (file_path.is_file() and 
                    file_path.suffix in self.code_extensions and
                    file_path.stat().st_size < self.max_file_size):
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if search_term.lower() in content.lower():
                                matching_files.append(file_path)
                                
                                if len(matching_files) >= 10:  # Limit results
                                    break
                    except (UnicodeDecodeError, IOError):
                        continue
        except Exception as e:
            self.logger.warning(f"Error searching files: {e}")
        
        return matching_files
    
    async def _generate_fallback_solutions(
        self, 
        problem: str, 
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate basic solution approaches when DSPy is not available."""
        
        solutions = []
        
        # Analyze problem for common solution patterns
        if "api" in problem.lower():
            solutions.append({
                "approach": "RESTful API Development",
                "description": "Build a RESTful API using standard HTTP methods",
                "pros": ["Well-established patterns", "Wide tooling support", "Scalable"],
                "cons": ["May be overkill for simple use cases", "Requires careful design"],
                "technologies": ["FastAPI", "Flask", "Express.js", "Spring Boot"]
            })
        
        if "database" in problem.lower() or "data" in problem.lower():
            solutions.append({
                "approach": "Database-Driven Solution",
                "description": "Use a database to store and manage data effectively",
                "pros": ["Data persistence", "ACID compliance", "Query capabilities"],
                "cons": ["Additional complexity", "Performance considerations"],
                "technologies": ["PostgreSQL", "MongoDB", "SQLite", "Redis"]
            })
        
        if "web" in problem.lower() or "frontend" in problem.lower():
            solutions.append({
                "approach": "Web Application Development",
                "description": "Create a web-based user interface",
                "pros": ["Cross-platform compatibility", "Easy deployment", "Rich UI capabilities"],
                "cons": ["Browser compatibility issues", "Security considerations"],
                "technologies": ["React", "Vue.js", "Angular", "vanilla JavaScript"]
            })
        
        # Default solution if no specific patterns detected
        if not solutions:
            solutions.append({
                "approach": "Modular Implementation",
                "description": "Break down the problem into smaller, manageable modules",
                "pros": ["Maintainable", "Testable", "Reusable components"],
                "cons": ["Initial complexity", "Requires good design"],
                "technologies": ["Python", "JavaScript", "Java", "Go"]
            })
        
        return solutions
    
    async def _basic_technology_analysis(self, technology: str, use_case: str) -> Dict[str, Any]:
        """Perform basic technology analysis."""
        
        # This would contain actual technology analysis logic
        # For now, provide a basic template
        
        return {
            "technology": technology,
            "use_case": use_case,
            "general_assessment": f"{technology} is being considered for {use_case}",
            "considerations": [
                "Evaluate learning curve and team expertise",
                "Assess community support and documentation",
                "Consider long-term maintenance and updates",
                "Review licensing and cost implications"
            ]
        }
    
    async def _analyze_codebase_structure(self, codebase_path: str) -> Dict[str, Any]:
        """Analyze the structure of a codebase."""
        
        root_path = Path(codebase_path)
        if not root_path.exists():
            return {"error": f"Path not found: {codebase_path}"}
        
        structure = {
            "total_files": 0,
            "code_files": 0,
            "doc_files": 0,
            "config_files": 0,
            "directories": 0,
            "languages": {},
            "file_sizes": {"small": 0, "medium": 0, "large": 0},
            "directory_structure": {}
        }
        
        try:
            for item in root_path.rglob("*"):
                if item.is_file():
                    structure["total_files"] += 1
                    
                    # Categorize by extension
                    suffix = item.suffix.lower()
                    if suffix in self.code_extensions:
                        structure["code_files"] += 1
                        structure["languages"][suffix] = structure["languages"].get(suffix, 0) + 1
                    elif suffix in self.doc_extensions:
                        structure["doc_files"] += 1
                    elif suffix in self.config_extensions:
                        structure["config_files"] += 1
                    
                    # Categorize by size
                    try:
                        size = item.stat().st_size
                        if size < 1000:
                            structure["file_sizes"]["small"] += 1
                        elif size < 10000:
                            structure["file_sizes"]["medium"] += 1
                        else:
                            structure["file_sizes"]["large"] += 1
                    except OSError:
                        pass
                        
                elif item.is_dir():
                    structure["directories"] += 1
        
        except Exception as e:
            structure["analysis_error"] = str(e)
        
        return structure
    
    async def _extract_representative_code_samples(self, codebase_path: str) -> List[str]:
        """Extract representative code samples from the codebase."""
        
        samples = []
        root_path = Path(codebase_path)
        
        if not root_path.exists():
            return samples
        
        try:
            # Find Python files (as an example)
            python_files = list(root_path.rglob("*.py"))[:10]  # Limit to 10 files
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) < 5000:  # Only include smaller files
                            samples.append(f"File: {file_path.name}\n{content}")
                        else:
                            # Include just the beginning
                            samples.append(f"File: {file_path.name} (truncated)\n{content[:1000]}...")
                except (UnicodeDecodeError, IOError):
                    continue
                    
                if len(samples) >= 5:  # Limit total samples
                    break
        
        except Exception as e:
            self.logger.warning(f"Error extracting code samples: {e}")
        
        return samples
    
    async def _gather_codebase_documentation(self, codebase_path: str) -> str:
        """Gather documentation from the codebase."""
        
        documentation_parts = []
        root_path = Path(codebase_path)
        
        if not root_path.exists():
            return ""
        
        # Look for common documentation files
        doc_files = ["README.md", "README.txt", "CHANGELOG.md", "CONTRIBUTING.md"]
        
        for doc_file in doc_files:
            file_path = root_path / doc_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documentation_parts.append(f"=== {doc_file} ===\n{content[:2000]}...")
                except (UnicodeDecodeError, IOError):
                    continue
        
        return "\n\n".join(documentation_parts)
    
    async def _extract_functional_requirements(self, description: str) -> List[str]:
        """Extract functional requirements from project description."""
        
        requirements = []
        
        # Look for action words and feature descriptions
        action_patterns = [
            r'should (be able to )?(.+?)(?:\.|$)',
            r'must (.+?)(?:\.|$)',
            r'will (.+?)(?:\.|$)',
            r'can (.+?)(?:\.|$)',
            r'needs to (.+?)(?:\.|$)'
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                requirement = match.group(1) if match.group(1) else match.group(2)
                if requirement and len(requirement.strip()) > 5:
                    requirements.append(requirement.strip())
        
        # Remove duplicates and return
        return list(dict.fromkeys(requirements))
    
    async def _extract_non_functional_requirements(self, description: str) -> List[str]:
        """Extract non-functional requirements from project description."""
        
        requirements = []
        
        # Look for performance, security, scalability keywords
        nfr_patterns = {
            "performance": r'(fast|quick|responsive|performance|speed|latency)',
            "scalability": r'(scale|scalable|users|concurrent|load)',
            "security": r'(secure|security|authentication|authorization|encryption)',
            "reliability": r'(reliable|availability|uptime|fault.tolerant)',
            "usability": r'(user.friendly|intuitive|easy.to.use|accessible)'
        }
        
        for category, pattern in nfr_patterns.items():
            if re.search(pattern, description, re.IGNORECASE):
                requirements.append(f"{category.title()} requirements mentioned")
        
        return requirements
    
    async def _find_similar_projects(self, description: str) -> List[Dict[str, Any]]:
        """Find similar projects based on description."""
        
        # This would typically search external sources
        # For now, return a placeholder
        
        similar_projects = []
        
        # Extract domain/technology keywords
        key_terms = self._extract_key_terms(description)
        
        if key_terms:
            # Simulate finding similar projects
            similar_projects.append({
                "name": f"Similar project using {key_terms[0]}",
                "description": f"A project that implements similar functionality using {key_terms[0]}",
                "relevance": "High",
                "source": "Research database"
            })
        
        return similar_projects
    
    async def _assess_requirements_completeness(
        self, 
        functional: List[str], 
        non_functional: List[str]
    ) -> Dict[str, Any]:
        """Assess the completeness of gathered requirements."""
        
        completeness = {
            "functional_count": len(functional),
            "non_functional_count": len(non_functional),
            "total_requirements": len(functional) + len(non_functional),
            "completeness_score": 0.0,
            "missing_areas": []
        }
        
        # Assess completeness based on typical requirement categories
        typical_nfr_categories = {
            "performance", "scalability", "security", "reliability", "usability"
        }
        
        found_categories = set()
        for nfr in non_functional:
            for category in typical_nfr_categories:
                if category in nfr.lower():
                    found_categories.add(category)
        
        missing_categories = typical_nfr_categories - found_categories
        completeness["missing_areas"] = list(missing_categories)
        
        # Calculate basic completeness score
        if completeness["total_requirements"] > 0:
            category_coverage = len(found_categories) / len(typical_nfr_categories)
            requirement_density = min(completeness["total_requirements"] / 10, 1.0)  # Assume 10 is ideal
            completeness["completeness_score"] = (category_coverage + requirement_density) / 2
        
        return completeness
    
    async def _search_local_codebase(self, topic: str) -> List[ResearchFinding]:
        """Search local codebase for topic-related information."""
        
        findings = []
        key_terms = self._extract_key_terms(topic)
        
        project_root = Path(".")
        
        for term in key_terms[:3]:  # Limit searches
            matching_files = await self._find_files_by_content(project_root, term)
            
            for file_path in matching_files[:2]:  # Limit files per term
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        findings.append(ResearchFinding(
                            title=f"Code reference: {file_path.name}",
                            description=f"Found '{term}' in {file_path.name}",
                            source=InformationSource.CODEBASE,
                            relevance_score=0.7,
                            confidence=0.8,
                            tags=[term, "codebase"]
                        ))
                except (UnicodeDecodeError, IOError):
                    continue
        
        return findings
    
    def _finding_to_dict(self, finding: ResearchFinding) -> Dict[str, Any]:
        """Convert ResearchFinding to dictionary for serialization."""
        
        return {
            "title": finding.title,
            "description": finding.description,
            "source": finding.source.value,
            "relevance_score": finding.relevance_score,
            "confidence": finding.confidence,
            "url": finding.url,
            "timestamp": finding.timestamp.isoformat(),
            "tags": finding.tags
        }
    
    async def search_external_sources(self, query: str) -> List[ResearchFinding]:
        """Search external sources for information (placeholder for actual implementation)."""
        
        # This would implement actual external search
        # For now, return empty list
        return []
    
    def clear_research_cache(self) -> None:
        """Clear the research cache."""
        self.research_cache.clear()
        self.logger.info("Research cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the research cache."""
        
        return {
            "cache_size": len(self.research_cache),
            "cache_keys": list(self.research_cache.keys()),
            "memory_usage": sum(len(str(result)) for result in self.research_cache.values())
        }