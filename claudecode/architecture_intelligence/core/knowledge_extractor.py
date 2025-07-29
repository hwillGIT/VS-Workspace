"""
Architecture Knowledge Extractor
Intelligent extraction and analysis of architectural knowledge from books and documents
"""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
from enum import Enum


class DocumentRelevance(Enum):
    """Relevance levels for architectural documents"""
    ESSENTIAL = "essential"  # Core architecture texts
    HIGHLY_RELEVANT = "highly_relevant"  # Directly applicable
    RELEVANT = "relevant"  # Useful concepts
    SUPPLEMENTARY = "supplementary"  # Background knowledge
    NOT_RELEVANT = "not_relevant"  # Outside scope


@dataclass
class ArchitectureDocument:
    """Represents an architecture-related document"""
    path: Path
    filename: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    document_type: str = "book"  # book, whitepaper, article, standard
    frameworks_covered: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    relevance: DocumentRelevance = DocumentRelevance.NOT_RELEVANT
    relevance_score: float = 0.0
    relevance_reasons: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    extracted_patterns: List[str] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)
    file_hash: Optional[str] = None
    analyzed_date: Optional[datetime] = None
    extraction_status: str = "pending"  # pending, analyzing, completed, failed


@dataclass
class KnowledgeExtractionPlan:
    """Plan for extracting knowledge from documents"""
    document: ArchitectureDocument
    extraction_goals: List[str]
    target_frameworks: List[str]
    extraction_depth: str  # quick_scan, standard, deep_analysis
    estimated_value: str  # high, medium, low
    priority: int  # 1-10
    specific_chapters: Optional[List[str]] = None
    pattern_focus_areas: Optional[List[str]] = None
    extraction_techniques: List[str] = field(default_factory=list)


class ArchitectureKnowledgeExtractor:
    """
    Intelligent extraction of architectural knowledge from books and documents
    
    Capabilities:
    - Identify relevant architecture books and documents
    - Extract patterns, principles, and best practices
    - Map content to framework knowledge base
    - Build contextual understanding from texts
    - Create actionable insights from theoretical knowledge
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Architecture book patterns - titles, authors, and keywords that indicate relevance
        self.architecture_indicators = {
            "essential_titles": [
                "domain-driven design",
                "clean architecture",
                "design patterns",
                "enterprise integration patterns",
                "building microservices",
                "software architecture in practice",
                "documenting software architectures",
                "pattern-oriented software architecture",
                "togaf",
                "archimate"
            ],
            "key_authors": [
                "eric evans",
                "martin fowler",
                "robert martin",
                "vaughn vernon",
                "sam newman",
                "gregor hohpe",
                "bobby woolf",
                "len bass",
                "paul clements",
                "simon brown"
            ],
            "architecture_keywords": [
                "architecture", "design patterns", "microservices", "domain-driven",
                "enterprise", "integration", "distributed systems", "scalability",
                "cloud native", "event-driven", "service-oriented", "api design",
                "system design", "software design", "architectural patterns",
                "reference architecture", "solution architecture", "togaf", "zachman"
            ],
            "framework_specific": {
                "togaf": ["togaf", "adm", "enterprise architecture", "architecture framework"],
                "ddd": ["domain-driven", "bounded context", "aggregate", "domain model"],
                "c4": ["c4 model", "simon brown", "context diagram", "container diagram"],
                "microservices": ["microservices", "service mesh", "api gateway", "distributed"],
                "patterns": ["design patterns", "architectural patterns", "enterprise patterns"]
            }
        }
        
        # Knowledge extraction rules
        self.extraction_rules = {
            "pattern_extraction": {
                "pattern_indicators": ["pattern:", "the ... pattern", "pattern name:"],
                "context_indicators": ["when to use", "problem", "context:"],
                "solution_indicators": ["solution:", "implementation", "how to"],
                "benefit_indicators": ["benefits:", "advantages:", "pros:"],
                "drawback_indicators": ["drawbacks:", "disadvantages:", "cons:"]
            },
            "principle_extraction": {
                "principle_indicators": ["principle:", "principles of", "key principle"],
                "rationale_indicators": ["because", "rationale:", "reasoning:"],
                "application_indicators": ["apply this", "in practice", "implementation"]
            },
            "framework_extraction": {
                "methodology_indicators": ["methodology", "framework", "approach"],
                "process_indicators": ["process:", "steps:", "phases:"],
                "artifact_indicators": ["deliverable:", "artifact:", "output:"]
            }
        }
        
        # Document analysis history
        self.analysis_history: Dict[str, ArchitectureDocument] = {}
        
        # Extracted knowledge base
        self.knowledge_base = {
            "patterns": {},
            "principles": {},
            "frameworks": {},
            "best_practices": {},
            "case_studies": {}
        }
    
    async def analyze_document_relevance(
        self,
        file_path: Path,
        context: Optional[Dict[str, Any]] = None
    ) -> ArchitectureDocument:
        """
        Analyze a document's relevance to architecture intelligence
        
        Args:
            file_path: Path to the document
            context: Optional context for relevance assessment
            
        Returns:
            ArchitectureDocument with relevance assessment
        """
        document = ArchitectureDocument(
            path=file_path,
            filename=file_path.name,
            file_hash=self._calculate_file_hash(file_path)
        )
        
        # Extract basic metadata from filename
        filename_lower = file_path.name.lower()
        
        # Check against essential titles
        relevance_score = 0.0
        relevance_reasons = []
        
        # Title matching
        for essential_title in self.architecture_indicators["essential_titles"]:
            if essential_title in filename_lower:
                relevance_score += 0.3
                relevance_reasons.append(f"Title matches essential architecture book: {essential_title}")
                document.frameworks_covered.extend(self._identify_frameworks_from_title(essential_title))
        
        # Author matching
        for author in self.architecture_indicators["key_authors"]:
            if author.replace(" ", "") in filename_lower.replace(" ", "").replace("-", "").replace("_", ""):
                relevance_score += 0.2
                relevance_reasons.append(f"By renowned architecture author: {author}")
        
        # Keyword matching
        keyword_matches = 0
        for keyword in self.architecture_indicators["architecture_keywords"]:
            if keyword in filename_lower:
                keyword_matches += 1
        
        if keyword_matches > 0:
            relevance_score += min(keyword_matches * 0.1, 0.3)
            relevance_reasons.append(f"Contains {keyword_matches} architecture keywords")
        
        # Framework-specific matching
        for framework, indicators in self.architecture_indicators["framework_specific"].items():
            for indicator in indicators:
                if indicator in filename_lower:
                    relevance_score += 0.2
                    relevance_reasons.append(f"Related to {framework.upper()} framework")
                    if framework not in document.frameworks_covered:
                        document.frameworks_covered.append(framework)
        
        # Context-based relevance
        if context:
            context_bonus = await self._assess_contextual_relevance(document, context)
            relevance_score += context_bonus
            if context_bonus > 0:
                relevance_reasons.append("Highly relevant to current context")
        
        # Set relevance level
        document.relevance_score = min(relevance_score, 1.0)
        document.relevance_reasons = relevance_reasons
        
        if relevance_score >= 0.8:
            document.relevance = DocumentRelevance.ESSENTIAL
        elif relevance_score >= 0.6:
            document.relevance = DocumentRelevance.HIGHLY_RELEVANT
        elif relevance_score >= 0.4:
            document.relevance = DocumentRelevance.RELEVANT
        elif relevance_score >= 0.2:
            document.relevance = DocumentRelevance.SUPPLEMENTARY
        else:
            document.relevance = DocumentRelevance.NOT_RELEVANT
        
        document.analyzed_date = datetime.now()
        
        # Store in history
        self.analysis_history[document.file_hash] = document
        
        return document
    
    async def create_extraction_plan(
        self,
        documents: List[ArchitectureDocument],
        goals: List[str],
        target_frameworks: Optional[List[str]] = None
    ) -> List[KnowledgeExtractionPlan]:
        """
        Create intelligent extraction plans for relevant documents
        
        Args:
            documents: List of analyzed documents
            goals: Extraction goals (e.g., "patterns", "principles", "frameworks")
            target_frameworks: Specific frameworks to focus on
            
        Returns:
            List of extraction plans prioritized by value
        """
        extraction_plans = []
        
        # Filter relevant documents
        relevant_docs = [
            doc for doc in documents 
            if doc.relevance in [
                DocumentRelevance.ESSENTIAL,
                DocumentRelevance.HIGHLY_RELEVANT,
                DocumentRelevance.RELEVANT
            ]
        ]
        
        for document in relevant_docs:
            # Determine extraction depth based on relevance
            if document.relevance == DocumentRelevance.ESSENTIAL:
                extraction_depth = "deep_analysis"
                priority = 10
            elif document.relevance == DocumentRelevance.HIGHLY_RELEVANT:
                extraction_depth = "standard"
                priority = 7
            else:
                extraction_depth = "quick_scan"
                priority = 4
            
            # Identify extraction goals for this document
            doc_extraction_goals = []
            
            if "patterns" in goals and any(
                keyword in document.filename.lower() 
                for keyword in ["pattern", "design", "architecture"]
            ):
                doc_extraction_goals.append("extract_architectural_patterns")
            
            if "principles" in goals and any(
                keyword in document.filename.lower()
                for keyword in ["principle", "clean", "solid"]
            ):
                doc_extraction_goals.append("extract_design_principles")
            
            if "frameworks" in goals and document.frameworks_covered:
                doc_extraction_goals.extend([
                    f"extract_{framework}_methodology" 
                    for framework in document.frameworks_covered
                ])
            
            # Create extraction plan
            plan = KnowledgeExtractionPlan(
                document=document,
                extraction_goals=doc_extraction_goals or ["general_architecture_knowledge"],
                target_frameworks=target_frameworks or document.frameworks_covered,
                extraction_depth=extraction_depth,
                priority=priority,
                estimated_value="high" if priority >= 7 else "medium" if priority >= 4 else "low",
                extraction_techniques=self._determine_extraction_techniques(document, goals)
            )
            
            # Add specific focus areas based on document type
            if "domain-driven" in document.filename.lower():
                plan.pattern_focus_areas = [
                    "bounded_contexts", "aggregates", "domain_events",
                    "strategic_design", "tactical_patterns"
                ]
            elif "microservices" in document.filename.lower():
                plan.pattern_focus_areas = [
                    "service_decomposition", "communication_patterns",
                    "data_management", "deployment_patterns"
                ]
            elif "togaf" in document.filename.lower():
                plan.pattern_focus_areas = [
                    "adm_phases", "architecture_artifacts",
                    "governance_processes", "reference_models"
                ]
            
            extraction_plans.append(plan)
        
        # Sort by priority
        extraction_plans.sort(key=lambda p: p.priority, reverse=True)
        
        return extraction_plans
    
    async def extract_knowledge(
        self,
        extraction_plan: KnowledgeExtractionPlan
    ) -> Dict[str, Any]:
        """
        Execute knowledge extraction based on plan
        
        Args:
            extraction_plan: Extraction plan to execute
            
        Returns:
            Extracted knowledge organized by type
        """
        extracted_knowledge = {
            "patterns": [],
            "principles": [],
            "frameworks": [],
            "best_practices": [],
            "case_studies": [],
            "metadata": {
                "source": extraction_plan.document.filename,
                "extraction_date": datetime.now().isoformat(),
                "extraction_depth": extraction_plan.extraction_depth,
                "frameworks": extraction_plan.target_frameworks
            }
        }
        
        # This would integrate with PDF parsing and NLP capabilities
        # For now, return structured placeholder based on document analysis
        
        if "extract_architectural_patterns" in extraction_plan.extraction_goals:
            extracted_knowledge["patterns"] = await self._extract_patterns_from_document(
                extraction_plan.document,
                extraction_plan.pattern_focus_areas
            )
        
        if "extract_design_principles" in extraction_plan.extraction_goals:
            extracted_knowledge["principles"] = await self._extract_principles_from_document(
                extraction_plan.document
            )
        
        # Extract framework-specific knowledge
        for goal in extraction_plan.extraction_goals:
            if goal.startswith("extract_") and goal.endswith("_methodology"):
                framework = goal.replace("extract_", "").replace("_methodology", "")
                extracted_knowledge["frameworks"][framework] = await self._extract_framework_knowledge(
                    extraction_plan.document,
                    framework
                )
        
        # Update document status
        extraction_plan.document.extraction_status = "completed"
        extraction_plan.document.extracted_patterns = [
            p["name"] for p in extracted_knowledge["patterns"]
        ]
        extraction_plan.document.key_concepts = self._extract_key_concepts(extracted_knowledge)
        
        # Store in knowledge base
        await self._update_knowledge_base(extracted_knowledge)
        
        return extracted_knowledge
    
    async def recommend_books_for_context(
        self,
        context: Dict[str, Any],
        current_knowledge: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend architecture books based on context and current knowledge gaps
        
        Args:
            context: Current architecture context
            current_knowledge: Existing knowledge base
            
        Returns:
            List of book recommendations with reasons
        """
        recommendations = []
        
        # Analyze context to identify knowledge needs
        needed_topics = []
        
        # Framework-specific needs
        frameworks = context.get("frameworks", [])
        for framework in frameworks:
            if framework.lower() == "togaf":
                recommendations.append({
                    "title": "TOGAF 9.2 Standard",
                    "reason": "Official TOGAF documentation for enterprise architecture",
                    "topics": ["enterprise_architecture", "adm", "architecture_governance"],
                    "priority": "essential"
                })
            elif framework.lower() == "ddd":
                recommendations.append({
                    "title": "Domain-Driven Design by Eric Evans",
                    "reason": "Foundational text for DDD strategic and tactical patterns",
                    "topics": ["bounded_contexts", "aggregates", "domain_modeling"],
                    "priority": "essential"
                })
                recommendations.append({
                    "title": "Implementing Domain-Driven Design by Vaughn Vernon",
                    "reason": "Practical DDD implementation guidance",
                    "topics": ["ddd_implementation", "event_sourcing", "cqrs"],
                    "priority": "highly_relevant"
                })
        
        # Architecture style needs
        if "microservices" in context.get("technical_stack", []):
            recommendations.append({
                "title": "Building Microservices by Sam Newman",
                "reason": "Comprehensive guide to microservices architecture",
                "topics": ["service_design", "deployment", "communication_patterns"],
                "priority": "essential"
            })
        
        # Quality attribute needs
        quality_attributes = context.get("quality_attributes", [])
        if "scalability" in quality_attributes:
            recommendations.append({
                "title": "Designing Data-Intensive Applications by Martin Kleppmann",
                "reason": "Deep insights into scalable system design",
                "topics": ["distributed_systems", "data_architecture", "consistency"],
                "priority": "highly_relevant"
            })
        
        # General architecture knowledge
        if not current_knowledge or len(current_knowledge.get("patterns", {})) < 10:
            recommendations.append({
                "title": "Software Architecture in Practice by Bass, Clements, and Kazman",
                "reason": "Comprehensive coverage of software architecture fundamentals",
                "topics": ["architecture_basics", "quality_attributes", "architecture_evaluation"],
                "priority": "essential"
            })
        
        # Remove duplicates and sort by priority
        seen_titles = set()
        unique_recommendations = []
        priority_order = {"essential": 1, "highly_relevant": 2, "relevant": 3}
        
        for rec in sorted(recommendations, key=lambda r: priority_order.get(r["priority"], 4)):
            if rec["title"] not in seen_titles:
                seen_titles.add(rec["title"])
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for identification"""
        return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _identify_frameworks_from_title(self, title: str) -> List[str]:
        """Identify frameworks mentioned in title"""
        frameworks = []
        
        framework_mappings = {
            "domain-driven": ["ddd"],
            "togaf": ["togaf"],
            "archimate": ["archimate"],
            "microservices": ["microservices"],
            "clean architecture": ["clean", "hexagonal"],
            "c4": ["c4"],
            "enterprise integration": ["integration_patterns"]
        }
        
        for keyword, framework_list in framework_mappings.items():
            if keyword in title:
                frameworks.extend(framework_list)
        
        return frameworks
    
    async def _assess_contextual_relevance(
        self,
        document: ArchitectureDocument,
        context: Dict[str, Any]
    ) -> float:
        """Assess additional relevance based on context"""
        bonus_score = 0.0
        
        # Check if document frameworks match context needs
        context_frameworks = context.get("frameworks", [])
        for framework in document.frameworks_covered:
            if framework in [f.lower() for f in context_frameworks]:
                bonus_score += 0.1
        
        # Check if document addresses context goals
        context_goals = context.get("goals", [])
        goal_keywords = {
            "scalability": ["scale", "distributed", "performance"],
            "maintainability": ["clean", "solid", "design"],
            "security": ["security", "authentication", "authorization"]
        }
        
        for goal in context_goals:
            if goal in goal_keywords:
                for keyword in goal_keywords[goal]:
                    if keyword in document.filename.lower():
                        bonus_score += 0.05
        
        return min(bonus_score, 0.3)  # Cap context bonus
    
    def _determine_extraction_techniques(
        self,
        document: ArchitectureDocument,
        goals: List[str]
    ) -> List[str]:
        """Determine extraction techniques based on document and goals"""
        techniques = ["metadata_extraction", "table_of_contents_analysis"]
        
        if "patterns" in goals:
            techniques.extend([
                "pattern_detection",
                "pattern_structure_extraction",
                "pattern_relationship_mapping"
            ])
        
        if "principles" in goals:
            techniques.extend([
                "principle_identification",
                "rationale_extraction"
            ])
        
        if document.frameworks_covered:
            techniques.extend([
                "framework_methodology_extraction",
                "artifact_identification",
                "process_mapping"
            ])
        
        return techniques
    
    async def _extract_patterns_from_document(
        self,
        document: ArchitectureDocument,
        focus_areas: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Extract architectural patterns from document"""
        # This would use actual PDF parsing and NLP
        # For now, return example patterns based on document type
        
        patterns = []
        
        if "microservices" in document.filename.lower():
            patterns.append({
                "name": "Database per Service",
                "category": "Data Management",
                "context": "Microservices needing data isolation",
                "problem": "Services sharing databases create coupling",
                "solution": "Each service owns its database",
                "benefits": ["Service independence", "Technology flexibility"],
                "drawbacks": ["Data consistency challenges", "Query complexity"],
                "source": document.filename,
                "page_references": ["Chapter 4", "Pages 87-102"]
            })
        
        return patterns
    
    async def _extract_principles_from_document(
        self,
        document: ArchitectureDocument
    ) -> List[Dict[str, Any]]:
        """Extract design principles from document"""
        principles = []
        
        if "clean" in document.filename.lower():
            principles.append({
                "name": "Dependency Inversion Principle",
                "category": "SOLID",
                "statement": "Depend on abstractions, not concretions",
                "rationale": "Reduces coupling and increases flexibility",
                "application": "Use interfaces between layers",
                "source": document.filename
            })
        
        return principles
    
    async def _extract_framework_knowledge(
        self,
        document: ArchitectureDocument,
        framework: str
    ) -> Dict[str, Any]:
        """Extract framework-specific knowledge"""
        framework_knowledge = {
            "methodology": {},
            "artifacts": [],
            "processes": [],
            "best_practices": []
        }
        
        if framework == "togaf":
            framework_knowledge["methodology"] = {
                "name": "Architecture Development Method",
                "phases": ["Preliminary", "A: Vision", "B: Business", "C: Information Systems"],
                "iteration_cycles": ["Architecture Cycle", "Transition Cycle"]
            }
        
        return framework_knowledge
    
    def _extract_key_concepts(self, extracted_knowledge: Dict[str, Any]) -> List[str]:
        """Extract key concepts from knowledge"""
        concepts = []
        
        # From patterns
        for pattern in extracted_knowledge.get("patterns", []):
            concepts.append(pattern.get("name", ""))
        
        # From principles  
        for principle in extracted_knowledge.get("principles", []):
            concepts.append(principle.get("name", ""))
        
        return list(set(concepts))  # Remove duplicates
    
    async def _update_knowledge_base(self, extracted_knowledge: Dict[str, Any]):
        """Update internal knowledge base with extracted knowledge"""
        # Update patterns
        for pattern in extracted_knowledge.get("patterns", []):
            pattern_id = pattern.get("name", "").lower().replace(" ", "_")
            self.knowledge_base["patterns"][pattern_id] = pattern
        
        # Update principles
        for principle in extracted_knowledge.get("principles", []):
            principle_id = principle.get("name", "").lower().replace(" ", "_")
            self.knowledge_base["principles"][principle_id] = principle
        
        # Update frameworks
        for framework, knowledge in extracted_knowledge.get("frameworks", {}).items():
            if framework not in self.knowledge_base["frameworks"]:
                self.knowledge_base["frameworks"][framework] = {}
            self.knowledge_base["frameworks"][framework].update(knowledge)
    
    async def export_knowledge_base(self, format: str = "json") -> str:
        """Export the accumulated knowledge base"""
        if format == "json":
            import json
            return json.dumps(self.knowledge_base, indent=2, default=str)
        elif format == "markdown":
            return self._export_knowledge_as_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_knowledge_as_markdown(self) -> str:
        """Export knowledge base as markdown"""
        md = "# Architecture Knowledge Base\n\n"
        
        md += f"## Patterns ({len(self.knowledge_base['patterns'])})\n\n"
        for pattern_id, pattern in self.knowledge_base["patterns"].items():
            md += f"### {pattern.get('name', pattern_id)}\n"
            md += f"**Category:** {pattern.get('category', 'General')}\n"
            md += f"**Context:** {pattern.get('context', 'N/A')}\n\n"
        
        md += f"## Principles ({len(self.knowledge_base['principles'])})\n\n"
        for principle_id, principle in self.knowledge_base["principles"].items():
            md += f"### {principle.get('name', principle_id)}\n"
            md += f"**Statement:** {principle.get('statement', 'N/A')}\n\n"
        
        return md


@dataclass
class BookRecommendation:
    """Structured book recommendation"""
    title: str
    authors: List[str]
    reason: str
    relevance_to_context: str
    topics_covered: List[str]
    frameworks: List[str]
    priority: str  # essential, highly_relevant, relevant, supplementary
    estimated_reading_time: str
    key_takeaways: List[str]
    prerequisites: Optional[List[str]] = None
    complementary_books: Optional[List[str]] = None