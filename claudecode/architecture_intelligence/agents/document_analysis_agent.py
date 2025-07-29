"""
Document Analysis Agent using Model Router
Delegates architecture document analysis to specialized models (including Gemini)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add paths for cross-project imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from self_reflecting_agent project
from self_reflecting_agent.routing.model_router import ModelRouter, TaskContext, TaskType, RoutingDecision
from self_reflecting_agent.routed_agent import RoutedSelfReflectingAgent, create_routed_agent

# Import from architecture intelligence
from ..core.knowledge_extractor import (
    ArchitectureKnowledgeExtractor, 
    ArchitectureDocument,
    DocumentRelevance,
    KnowledgeExtractionPlan
)


class DocumentAnalysisAgent:
    """
    Agent that uses model routing to analyze architecture documents.
    Can delegate to Gemini or other models based on task characteristics.
    """
    
    def __init__(self, router: Optional[ModelRouter] = None):
        self.logger = logging.getLogger(__name__)
        self.router = router or ModelRouter()
        self.knowledge_extractor = ArchitectureKnowledgeExtractor()
        
        # Create routed agent for advanced analysis
        self.routed_agent = None
        self._initialize_task = asyncio.create_task(self._initialize_routed_agent())
    
    async def _initialize_routed_agent(self):
        """Initialize routed agent asynchronously."""
        try:
            self.routed_agent = await create_routed_agent(
                project_path=Path.home() / '.architecture_intelligence',
                enable_rag=True,
                enable_semantic_search=True
            )
            self.logger.info("Routed agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize routed agent: {e}")
    
    async def ensure_initialized(self):
        """Ensure the routed agent is initialized."""
        if self._initialize_task:
            await self._initialize_task
            self._initialize_task = None
    
    async def analyze_document_with_gemini(
        self,
        document: ArchitectureDocument,
        extraction_goals: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document using Gemini model through the router.
        
        Args:
            document: Document to analyze
            extraction_goals: What to extract (patterns, principles, etc.)
            context: Optional project context
            
        Returns:
            Analysis results including extracted knowledge
        """
        await self.ensure_initialized()
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(document, extraction_goals, context)
        
        # Create task context for routing - prefer Gemini for document analysis
        task_context = TaskContext(
            task_type=TaskType.ANALYSIS,  # Document analysis task
            complexity="high",
            estimated_tokens=len(prompt) + 4000,  # Estimate response size
            requires_reasoning=True,
            requires_code=True,  # May need to understand code examples
            latency_sensitive=False,  # Thorough analysis is more important than speed
            cost_sensitive=False  # Quality over cost for knowledge extraction
        )
        
        # Route to appropriate model (will likely choose Gemini 2.0 Flash)
        routing_decision = await self.router.route_task(task_context)
        
        self.logger.info(f"Document analysis routed to: {routing_decision.selected_model}")
        self.logger.info(f"Routing reasoning: {routing_decision.reasoning}")
        
        # Execute analysis using routed agent
        if self.routed_agent:
            result = await self.routed_agent.execute_task(
                task_description=prompt,
                session_id=f"doc_analysis_{document.file_hash}",
                task_type=TaskType.ANALYSIS
            )
            
            # Parse and structure the results
            analyzed_content = self._parse_analysis_results(
                result['content'],
                document,
                extraction_goals
            )
            
            # Record performance metrics
            await self.router.record_task_result(
                model_name=result['model_used'],
                success=True,
                latency_ms=result.get('execution_time_ms', 0),
                cost=result.get('estimated_cost', 0.0)
            )
            
            return {
                'document': document.filename,
                'model_used': result['model_used'],
                'routing_decision': routing_decision,
                'extracted_knowledge': analyzed_content,
                'extraction_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'goals': extraction_goals,
                    'model_confidence': result.get('confidence', 0.8)
                }
            }
        else:
            raise RuntimeError("Routed agent not initialized")
    
    def _create_analysis_prompt(
        self,
        document: ArchitectureDocument,
        extraction_goals: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a detailed prompt for document analysis."""
        
        prompt_parts = [
            f"Analyze the architecture document: {document.filename}",
            f"Document relevance: {document.relevance.value} (score: {document.relevance_score:.2f})",
            f"Identified frameworks: {', '.join(document.frameworks_covered)}",
            f"\nExtraction Goals: {', '.join(extraction_goals)}",
            ""
        ]
        
        # Add specific extraction instructions
        if "patterns" in extraction_goals:
            prompt_parts.extend([
                "\n## Extract Architectural Patterns",
                "For each pattern found, provide:",
                "- Pattern name and category",
                "- Context where it applies",
                "- Problem it solves",
                "- Solution approach",
                "- Benefits and drawbacks",
                "- Implementation guidance",
                ""
            ])
        
        if "principles" in extraction_goals:
            prompt_parts.extend([
                "\n## Extract Design Principles",
                "For each principle found, provide:",
                "- Principle name and statement",
                "- Rationale and reasoning",
                "- Application examples",
                "- Relationship to other principles",
                ""
            ])
        
        if "frameworks" in extraction_goals:
            prompt_parts.extend([
                "\n## Extract Framework Knowledge",
                "For frameworks mentioned, extract:",
                "- Methodology and processes",
                "- Key artifacts and deliverables",
                "- Best practices and guidelines",
                "- Implementation strategies",
                ""
            ])
        
        # Add context if provided
        if context:
            prompt_parts.extend([
                "\n## Project Context",
                f"Domain: {context.get('domain', 'Unknown')}",
                f"Goals: {', '.join(context.get('goals', []))}",
                f"Tech Stack: {', '.join(context.get('technical_stack', []))}",
                "",
                "Focus extraction on elements relevant to this context."
            ])
        
        prompt_parts.append("\nProvide structured extraction results in JSON format where possible.")
        
        return "\n".join(prompt_parts)
    
    def _parse_analysis_results(
        self,
        raw_results: str,
        document: ArchitectureDocument,
        extraction_goals: List[str]
    ) -> Dict[str, Any]:
        """Parse analysis results from model response."""
        
        extracted_knowledge = {
            "patterns": [],
            "principles": [],
            "frameworks": {},
            "best_practices": [],
            "raw_insights": raw_results
        }
        
        # Try to extract JSON sections if present
        try:
            # Simple JSON extraction (would be more sophisticated in production)
            import re
            json_matches = re.findall(r'\{[^{}]*\}', raw_results, re.DOTALL)
            
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if 'pattern' in match.lower():
                        extracted_knowledge["patterns"].append(data)
                    elif 'principle' in match.lower():
                        extracted_knowledge["principles"].append(data)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            self.logger.warning(f"Could not parse JSON from results: {e}")
        
        # Fallback: Extract patterns from text
        if not extracted_knowledge["patterns"] and "patterns" in extraction_goals:
            extracted_knowledge["patterns"] = self._extract_patterns_from_text(raw_results)
        
        # Update document with extraction results
        document.extraction_status = "completed"
        document.extracted_patterns = [p.get("name", "Unknown") for p in extracted_knowledge["patterns"]]
        
        return extracted_knowledge
    
    def _extract_patterns_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract patterns from unstructured text."""
        patterns = []
        
        # Simple pattern extraction based on keywords
        pattern_indicators = [
            "pattern:", "the ... pattern", "architectural pattern",
            "design pattern", "uses pattern", "implements pattern"
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for indicator in pattern_indicators:
                if indicator.lower() in line.lower():
                    # Extract pattern name and context
                    pattern_name = line.strip()
                    context_lines = []
                    
                    # Get surrounding lines for context
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_lines = lines[start:end]
                    
                    patterns.append({
                        "name": pattern_name,
                        "context": "\n".join(context_lines),
                        "source": "text_extraction"
                    })
                    break
        
        return patterns
    
    async def analyze_document_library(
        self,
        library_path: Path,
        context: Optional[Dict[str, Any]] = None,
        max_documents: int = 5,
        prefer_gemini: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze an entire library of documents using intelligent routing.
        
        Args:
            library_path: Path to document library
            context: Project context for relevance
            max_documents: Maximum documents to analyze
            prefer_gemini: Whether to prefer Gemini for analysis
            
        Returns:
            Comprehensive library analysis results
        """
        await self.ensure_initialized()
        
        # Get document relevance scores
        pdf_files = list(library_path.glob("*.pdf"))
        analyzed_documents = []
        
        for pdf_file in pdf_files[:max_documents * 2]:  # Analyze more to filter
            try:
                document = await self.knowledge_extractor.analyze_document_relevance(
                    pdf_file, context
                )
                analyzed_documents.append(document)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {pdf_file}: {e}")
        
        # Sort by relevance and take top documents
        analyzed_documents.sort(key=lambda d: d.relevance_score, reverse=True)
        top_documents = analyzed_documents[:max_documents]
        
        # Create extraction plans
        extraction_goals = ["patterns", "principles", "frameworks"]
        extraction_plans = await self.knowledge_extractor.create_extraction_plan(
            top_documents, extraction_goals
        )
        
        # Analyze each document with routing
        analysis_results = []
        for plan in extraction_plans:
            if plan.document.relevance in [DocumentRelevance.ESSENTIAL, DocumentRelevance.HIGHLY_RELEVANT]:
                try:
                    # If preferring Gemini, update router priority temporarily
                    if prefer_gemini:
                        self.router.update_model_config(
                            'gemini-2.0-flash-exp',
                            {'priority': 15}  # Temporarily boost priority
                        )
                    
                    result = await self.analyze_document_with_gemini(
                        plan.document,
                        plan.extraction_goals,
                        context
                    )
                    analysis_results.append(result)
                    
                    # Reset priority
                    if prefer_gemini:
                        self.router.update_model_config(
                            'gemini-2.0-flash-exp',
                            {'priority': 7}  # Reset to default
                        )
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze {plan.document.filename}: {e}")
        
        # Get model performance stats
        router_status = self.router.get_model_status()
        
        return {
            "library_path": str(library_path),
            "documents_analyzed": len(analysis_results),
            "analysis_results": analysis_results,
            "model_usage": self._summarize_model_usage(analysis_results),
            "router_status": router_status,
            "knowledge_summary": self._summarize_extracted_knowledge(analysis_results)
        }
    
    def _summarize_model_usage(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize which models were used for analysis."""
        model_usage = {}
        for result in results:
            model = result.get('model_used', 'unknown')
            model_usage[model] = model_usage.get(model, 0) + 1
        return model_usage
    
    def _summarize_extracted_knowledge(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize all extracted knowledge."""
        summary = {
            "total_patterns": 0,
            "total_principles": 0,
            "frameworks_covered": set(),
            "pattern_categories": set(),
            "key_insights": []
        }
        
        for result in results:
            knowledge = result.get('extracted_knowledge', {})
            
            # Count patterns and principles
            patterns = knowledge.get('patterns', [])
            summary['total_patterns'] += len(patterns)
            
            principles = knowledge.get('principles', [])
            summary['total_principles'] += len(principles)
            
            # Collect frameworks
            frameworks = knowledge.get('frameworks', {})
            summary['frameworks_covered'].update(frameworks.keys())
            
            # Collect pattern categories
            for pattern in patterns:
                if 'category' in pattern:
                    summary['pattern_categories'].add(pattern['category'])
        
        # Convert sets to lists for JSON serialization
        summary['frameworks_covered'] = list(summary['frameworks_covered'])
        summary['pattern_categories'] = list(summary['pattern_categories'])
        
        return summary
    
    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        if self.routed_agent:
            await self.routed_agent.shutdown()


async def create_document_analysis_agent(
    prefer_models: Optional[List[str]] = None
) -> DocumentAnalysisAgent:
    """
    Create a document analysis agent with optional model preferences.
    
    Args:
        prefer_models: List of model names to prefer for analysis
        
    Returns:
        Configured document analysis agent
    """
    router = ModelRouter()
    
    # Update priorities if preferences specified
    if prefer_models:
        for i, model_name in enumerate(prefer_models):
            if model_name in router.models:
                router.update_model_config(model_name, {'priority': 20 - i})
    
    agent = DocumentAnalysisAgent(router)
    await agent.ensure_initialized()
    
    return agent