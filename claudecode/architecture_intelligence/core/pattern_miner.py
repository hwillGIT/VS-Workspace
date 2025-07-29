"""
Pattern Mining Engine
Cross-framework pattern detection and analysis with deep expertise
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
import logging


@dataclass
class ArchitecturePattern:
    """Represents a detected architecture pattern"""
    id: str
    name: str
    type: str  # structural, behavioral, creational, integration, etc.
    category: str  # enterprise, microservices, data, security, etc.
    description: str
    frameworks: List[str]  # Which frameworks recognize this pattern
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Supporting evidence
    context: Dict[str, Any]  # Context where pattern was found
    benefits: List[str]
    drawbacks: List[str]
    implementation_guidance: List[str]
    related_patterns: List[str]
    anti_patterns: List[str]  # Related anti-patterns to avoid
    quality_attributes: List[str]  # Affected quality attributes
    complexity_score: float  # Implementation complexity (0.0 to 1.0)
    maturity: str  # proven, emerging, experimental
    industry_adoption: str  # widespread, moderate, niche
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class PatternRelationship:
    """Relationship between patterns"""
    source_pattern: str
    target_pattern: str
    relationship_type: str  # requires, conflicts, enhances, specializes
    strength: float  # 0.0 to 1.0
    description: str


class PatternMiner:
    """
    Cross-framework pattern mining engine with deep architectural intelligence
    
    Capabilities:
    - Detect patterns across multiple frameworks
    - Mine emerging patterns from architectural contexts
    - Identify pattern relationships and conflicts
    - Provide intelligent pattern recommendations
    - Learn from pattern usage and feedback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Pattern knowledge base
        self.known_patterns: Dict[str, ArchitecturePattern] = {}
        self.pattern_relationships: List[PatternRelationship] = []
        
        # Pattern detection rules
        self.detection_rules: Dict[str, Dict[str, Any]] = {}
        
        # Learning data
        self.pattern_usage_history: List[Dict[str, Any]] = []
        self.feedback_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize with known architectural patterns
        asyncio.create_task(self._initialize_pattern_catalog())
    
    async def _initialize_pattern_catalog(self):
        """Initialize the pattern catalog with known architectural patterns"""
        await self._load_enterprise_patterns()
        await self._load_microservices_patterns()
        await self._load_data_patterns()
        await self._load_integration_patterns()
        await self._load_security_patterns()
        await self._load_cloud_patterns()
        await self._initialize_pattern_relationships()
        
        self.logger.info(f"Initialized pattern catalog with {len(self.known_patterns)} patterns")
    
    async def _load_enterprise_patterns(self):
        """Load enterprise architecture patterns"""
        patterns = [
            ArchitecturePattern(
                id="layered_architecture",
                name="Layered Architecture",
                type="structural",
                category="enterprise",
                description="Organizes system into horizontal layers with specific responsibilities",
                frameworks=["togaf", "archimate", "c4"],
                confidence=0.95,
                evidence=["layer separation", "abstraction levels", "dependency flow"],
                context={},
                benefits=[
                    "Clear separation of concerns",
                    "Maintainable and testable",
                    "Well-understood pattern",
                    "Support for distributed development"
                ],
                drawbacks=[
                    "Can become performance bottleneck",
                    "Tight coupling between layers",
                    "Architecture sinkhole anti-pattern risk"
                ],
                implementation_guidance=[
                    "Define clear layer responsibilities",
                    "Enforce one-way dependencies",
                    "Use dependency inversion for testability",
                    "Consider layer consolidation for performance"
                ],
                related_patterns=["hexagonal_architecture", "clean_architecture"],
                anti_patterns=["architecture_sinkhole", "cut_and_paste_programming"],
                quality_attributes=["maintainability", "testability", "modifiability"],
                complexity_score=0.3,
                maturity="proven",
                industry_adoption="widespread"
            ),
            
            ArchitecturePattern(
                id="service_oriented_architecture",
                name="Service-Oriented Architecture (SOA)",
                type="structural",
                category="enterprise",
                description="Architecture style that uses services as fundamental building blocks",
                frameworks=["togaf", "archimate", "dodaf"],
                confidence=0.90,
                evidence=["service interfaces", "service contracts", "service registry"],
                context={},
                benefits=[
                    "Service reusability",
                    "Platform independence",
                    "Improved maintainability",
                    "Business-IT alignment"
                ],
                drawbacks=[
                    "Complexity in service orchestration",
                    "Performance overhead",
                    "Governance challenges"
                ],
                implementation_guidance=[
                    "Define service contracts clearly",
                    "Implement service governance",
                    "Use enterprise service bus for integration",
                    "Plan for service versioning"
                ],
                related_patterns=["microservices", "event_driven_architecture"],
                anti_patterns=["chatty_interface", "shared_database"],
                quality_attributes=["reusability", "interoperability", "scalability"],
                complexity_score=0.7,
                maturity="proven",
                industry_adoption="widespread"
            ),
            
            ArchitecturePattern(
                id="enterprise_integration_patterns",
                name="Enterprise Integration Patterns",
                type="integration",
                category="enterprise",
                description="Collection of patterns for enterprise application integration",
                frameworks=["togaf", "archimate"],
                confidence=0.85,
                evidence=["message channels", "message transformation", "routing patterns"],
                context={},
                benefits=[
                    "Proven integration solutions",
                    "Consistent integration approach",
                    "Reduced integration complexity"
                ],
                drawbacks=[
                    "Learning curve for team",
                    "Tool dependency",
                    "Potential over-engineering"
                ],
                implementation_guidance=[
                    "Choose appropriate patterns for use case",
                    "Use integration platforms that support patterns",
                    "Document integration flows clearly",
                    "Plan for error handling and monitoring"
                ],
                related_patterns=["message_bus", "publish_subscribe"],
                anti_patterns=["point_to_point_integration", "shared_database"],
                quality_attributes=["interoperability", "maintainability", "reliability"],
                complexity_score=0.6,
                maturity="proven", 
                industry_adoption="widespread"
            )
        ]
        
        for pattern in patterns:
            self.known_patterns[pattern.id] = pattern
    
    async def _load_microservices_patterns(self):
        """Load microservices architecture patterns"""
        patterns = [
            ArchitecturePattern(
                id="microservices_architecture",
                name="Microservices Architecture",
                type="structural",
                category="microservices",
                description="Decomposes application into loosely coupled, independently deployable services",
                frameworks=["ddd", "c4", "reactive"],
                confidence=0.90,
                evidence=["service boundaries", "independent deployment", "decentralized data"],
                context={},
                benefits=[
                    "Independent scaling and deployment",
                    "Technology diversity",
                    "Fault isolation",
                    "Team autonomy"
                ],
                drawbacks=[
                    "Distributed system complexity",
                    "Network latency and reliability",
                    "Data consistency challenges",
                    "Operational overhead"
                ],
                implementation_guidance=[
                    "Start with modular monolith",
                    "Use domain-driven design for boundaries",
                    "Implement circuit breakers and retries",
                    "Invest in observability and monitoring"
                ],
                related_patterns=["domain_driven_design", "api_gateway", "saga_pattern"],
                anti_patterns=["distributed_monolith", "shared_database"],
                quality_attributes=["scalability", "maintainability", "deployability"],
                complexity_score=0.8,
                maturity="proven",
                industry_adoption="widespread"
            ),
            
            ArchitecturePattern(
                id="api_gateway",
                name="API Gateway",
                type="structural",
                category="microservices",
                description="Single entry point for client requests to microservices backend",
                frameworks=["c4", "microservices"],
                confidence=0.85,
                evidence=["single entry point", "request routing", "cross-cutting concerns"],
                context={},
                benefits=[
                    "Simplified client interaction",
                    "Centralized cross-cutting concerns",
                    "Protocol translation",
                    "Rate limiting and security"
                ],
                drawbacks=[
                    "Single point of failure risk",
                    "Performance bottleneck potential",
                    "Complexity in gateway logic"
                ],
                implementation_guidance=[
                    "Keep gateway logic lightweight",
                    "Implement gateway high availability",
                    "Use multiple gateways for different client types",
                    "Monitor gateway performance closely"
                ],
                related_patterns=["microservices_architecture", "backend_for_frontend"],
                anti_patterns=["god_gateway", "chatty_interface"],
                quality_attributes=["usability", "security", "performance"],
                complexity_score=0.5,
                maturity="proven",
                industry_adoption="widespread"
            ),
            
            ArchitecturePattern(
                id="saga_pattern",
                name="Saga Pattern",
                type="behavioral",
                category="microservices",
                description="Manages distributed transactions across microservices",
                frameworks=["ddd", "microservices", "event_driven"],
                confidence=0.80,
                evidence=["compensation logic", "transaction coordination", "eventual consistency"],
                context={},
                benefits=[
                    "Maintains data consistency across services",
                    "Avoids distributed locking",
                    "Fault tolerance through compensation"
                ],
                drawbacks=[
                    "Complex compensation logic",
                    "Eventual consistency model",
                    "Difficult debugging and monitoring"
                ],
                implementation_guidance=[
                    "Design idempotent operations",
                    "Implement comprehensive logging",
                    "Use choreography for simple flows",
                    "Use orchestration for complex flows"
                ],
                related_patterns=["microservices_architecture", "event_sourcing", "cqrs"],
                anti_patterns=["distributed_transaction", "shared_database"],
                quality_attributes=["consistency", "availability", "fault_tolerance"],
                complexity_score=0.9,
                maturity="emerging",
                industry_adoption="moderate"
            )
        ]
        
        for pattern in patterns:
            self.known_patterns[pattern.id] = pattern
    
    async def _load_data_patterns(self):
        """Load data architecture patterns"""
        patterns = [
            ArchitecturePattern(
                id="database_per_service",
                name="Database per Service",
                type="structural",
                category="data",
                description="Each microservice owns its data and database",
                frameworks=["ddd", "microservices"],
                confidence=0.85,
                evidence=["service-owned data", "data isolation", "independent schemas"],
                context={},
                benefits=[
                    "Service independence",
                    "Technology choice flexibility",
                    "Fault isolation",
                    "Scalability per service"
                ],
                drawbacks=[
                    "Data consistency challenges",
                    "Cross-service queries complexity",
                    "Increased operational overhead"
                ],
                implementation_guidance=[
                    "Define clear data ownership boundaries",
                    "Use event-driven patterns for data synchronization",
                    "Implement saga pattern for distributed transactions",
                    "Consider CQRS for complex queries"
                ],
                related_patterns=["microservices_architecture", "saga_pattern", "cqrs"],
                anti_patterns=["shared_database", "database_as_integration"],
                quality_attributes=["independence", "scalability", "maintainability"],
                complexity_score=0.7,
                maturity="proven",
                industry_adoption="widespread"
            ),
            
            ArchitecturePattern(
                id="cqrs",
                name="Command Query Responsibility Segregation (CQRS)",
                type="structural",
                category="data",
                description="Separates read and write operations using different models",
                frameworks=["ddd", "event_driven"],
                confidence=0.75,
                evidence=["separate read/write models", "command/query separation", "eventual consistency"],
                context={},
                benefits=[
                    "Optimized read and write models",
                    "Independent scaling of reads/writes",
                    "Flexibility in data storage"
                ],
                drawbacks=[
                    "Increased system complexity",
                    "Eventual consistency challenges",
                    "Data synchronization overhead"
                ],
                implementation_guidance=[
                    "Start simple, evolve to CQRS when needed",
                    "Use event sourcing for write model",
                    "Optimize read models for specific queries",
                    "Handle eventual consistency in UI"
                ],
                related_patterns=["event_sourcing", "database_per_service"],
                anti_patterns=["shared_database", "anemic_domain_model"],
                quality_attributes=["performance", "scalability", "flexibility"],
                complexity_score=0.8,
                maturity="emerging",
                industry_adoption="moderate"
            ),
            
            ArchitecturePattern(
                id="event_sourcing",
                name="Event Sourcing",
                type="behavioral",
                category="data",
                description="Stores domain events as the source of truth for application state",
                frameworks=["ddd", "event_driven"],
                confidence=0.75,
                evidence=["event store", "event replay", "immutable events"],
                context={},
                benefits=[
                    "Complete audit trail",
                    "Temporal queries possible",
                    "Natural fit with event-driven systems",
                    "Simplified debugging through replay"
                ],
                drawbacks=[
                    "Complex event schema evolution",
                    "Eventual consistency",
                    "Storage overhead",
                    "Learning curve for developers"
                ],
                implementation_guidance=[
                    "Design events for forward compatibility",
                    "Implement snapshot mechanism for performance",
                    "Use event versioning strategies",
                    "Plan for event store operational concerns"
                ],
                related_patterns=["cqrs", "saga_pattern", "event_driven_architecture"],
                anti_patterns=["mutable_event_store", "large_events"],
                quality_attributes=["auditability", "consistency", "debuggability"],
                complexity_score=0.9,
                maturity="emerging",
                industry_adoption="niche"
            )
        ]
        
        for pattern in patterns:
            self.known_patterns[pattern.id] = pattern
    
    async def _load_integration_patterns(self):
        """Load integration architecture patterns"""
        patterns = [
            ArchitecturePattern(
                id="event_driven_architecture",
                name="Event-Driven Architecture",
                type="behavioral",
                category="integration",
                description="Uses events as primary communication mechanism between components",
                frameworks=["reactive", "microservices", "event_driven"],
                confidence=0.85,
                evidence=["event publishing", "event consumption", "asynchronous communication"],
                context={},
                benefits=[
                    "Loose coupling between components",
                    "High scalability and responsiveness",
                    "Natural fault tolerance",
                    "Enables real-time processing"
                ],
                drawbacks=[
                    "Eventual consistency challenges",
                    "Complex debugging and tracing",
                    "Event ordering complexities"
                ],
                implementation_guidance=[
                    "Design events for forward compatibility",
                    "Implement comprehensive monitoring",
                    "Use event schemas and registries",
                    "Plan for event replay and recovery"
                ],
                related_patterns=["microservices_architecture", "cqrs", "saga_pattern"],
                anti_patterns=["event_chain", "event_message_confusion"],
                quality_attributes=["scalability", "responsiveness", "loose_coupling"],
                complexity_score=0.7,
                maturity="proven",
                industry_adoption="widespread"
            ),
            
            ArchitecturePattern(
                id="publish_subscribe",
                name="Publish-Subscribe",
                type="behavioral",
                category="integration",
                description="Publishers emit events to topics, subscribers receive relevant events",
                frameworks=["event_driven", "integration"],
                confidence=0.90,
                evidence=["topic-based routing", "subscriber registration", "asynchronous delivery"],
                context={},
                benefits=[
                    "Dynamic system composition",
                    "Scalable communication",
                    "Temporal decoupling"
                ],
                drawbacks=[
                    "Message delivery guarantees complexity",
                    "Potential message ordering issues",
                    "Subscriber management overhead"
                ],
                implementation_guidance=[
                    "Choose appropriate delivery guarantees",
                    "Implement dead letter queues",
                    "Use message deduplication",
                    "Monitor subscription health"
                ],
                related_patterns=["event_driven_architecture", "message_queue"],
                anti_patterns=["chatty_publisher", "slow_subscriber"],
                quality_attributes=["scalability", "flexibility", "decoupling"],
                complexity_score=0.5,
                maturity="proven",
                industry_adoption="widespread"
            )
        ]
        
        for pattern in patterns:
            self.known_patterns[pattern.id] = pattern
    
    async def _load_security_patterns(self):
        """Load security architecture patterns"""
        patterns = [
            ArchitecturePattern(
                id="zero_trust_architecture",
                name="Zero Trust Architecture",
                type="security",
                category="security",
                description="Security model that requires verification for every user and device",
                frameworks=["security", "enterprise"],
                confidence=0.80,
                evidence=["identity verification", "least privilege", "micro-segmentation"],
                context={},
                benefits=[
                    "Enhanced security posture",
                    "Reduced attack surface",
                    "Better compliance support"
                ],
                drawbacks=[
                    "Implementation complexity",
                    "User experience impact",
                    "Operational overhead"
                ],
                implementation_guidance=[
                    "Start with identity and access management",
                    "Implement network micro-segmentation",
                    "Use continuous monitoring",
                    "Plan phased rollout approach"
                ],
                related_patterns=["api_gateway", "identity_provider"],
                anti_patterns=["perimeter_security_only", "shared_credentials"],
                quality_attributes=["security", "auditability", "compliance"],
                complexity_score=0.8,
                maturity="emerging",
                industry_adoption="moderate"
            )
        ]
        
        for pattern in patterns:
            self.known_patterns[pattern.id] = pattern
    
    async def _load_cloud_patterns(self):
        """Load cloud architecture patterns"""
        patterns = [
            ArchitecturePattern(
                id="serverless_architecture",
                name="Serverless Architecture",
                type="deployment",
                category="cloud",
                description="Uses Function-as-a-Service for compute without server management",
                frameworks=["serverless", "cloud"],
                confidence=0.80,
                evidence=["function-based deployment", "event triggers", "auto-scaling"],
                context={},
                benefits=[
                    "No server management",
                    "Automatic scaling",
                    "Pay-per-use pricing",
                    "Fast deployment"
                ],
                drawbacks=[
                    "Cold start latency",
                    "Vendor lock-in risks",
                    "Limited execution time",
                    "Monitoring complexity"
                ],
                implementation_guidance=[
                    "Design for stateless functions",
                    "Optimize for cold start performance",
                    "Use appropriate trigger mechanisms",
                    "Implement comprehensive monitoring"
                ],
                related_patterns=["event_driven_architecture", "microservices_architecture"],
                anti_patterns=["monolithic_function", "stateful_function"],
                quality_attributes=["scalability", "cost_efficiency", "maintainability"],
                complexity_score=0.6,
                maturity="proven",
                industry_adoption="widespread"
            )
        ]
        
        for pattern in patterns:
            self.known_patterns[pattern.id] = pattern
    
    async def _initialize_pattern_relationships(self):
        """Initialize relationships between patterns"""
        relationships = [
            PatternRelationship(
                source_pattern="microservices_architecture",
                target_pattern="database_per_service",
                relationship_type="requires",
                strength=0.9,
                description="Microservices architecture requires database per service for true independence"
            ),
            PatternRelationship(
                source_pattern="microservices_architecture",
                target_pattern="api_gateway",
                relationship_type="enhances",
                strength=0.8,
                description="API Gateway enhances microservices by providing single entry point"
            ),
            PatternRelationship(
                source_pattern="cqrs",
                target_pattern="event_sourcing",
                relationship_type="enhances",
                strength=0.7,
                description="Event sourcing provides natural write model for CQRS"
            ),
            PatternRelationship(
                source_pattern="saga_pattern",
                target_pattern="event_driven_architecture",
                relationship_type="requires",
                strength=0.8,
                description="Saga pattern often requires event-driven communication"
            ),
            PatternRelationship(
                source_pattern="layered_architecture",
                target_pattern="microservices_architecture",
                relationship_type="conflicts",
                strength=0.6,
                description="Layered architecture can conflict with microservices boundaries"
            )
        ]
        
        self.pattern_relationships.extend(relationships)
    
    async def mine_patterns(
        self,
        framework_results: List[Dict[str, Any]],
        context: Dict[str, Any],
        depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Mine architectural patterns from framework analysis results
        
        Args:
            framework_results: Results from multiple framework analyses
            context: Architecture context
            depth: Mining depth (1=basic, 2=intermediate, 3=deep)
            
        Returns:
            List of detected patterns with confidence scores
        """
        self.logger.info(f"Mining patterns with depth {depth} across {len(framework_results)} frameworks")
        
        detected_patterns = []
        
        # Detect known patterns
        known_pattern_matches = await self._detect_known_patterns(framework_results, context)
        detected_patterns.extend(known_pattern_matches)
        
        # Mine emerging patterns if depth >= 2
        if depth >= 2:
            emerging_patterns = await self._discover_emerging_patterns(framework_results, context)
            detected_patterns.extend(emerging_patterns)
        
        # Analyze pattern relationships if depth >= 3
        if depth >= 3:
            pattern_relationships = await self._analyze_pattern_relationships(detected_patterns)
            # Add relationship information to patterns
            for pattern in detected_patterns:
                pattern["relationships"] = pattern_relationships.get(pattern["id"], [])
        
        # Calculate cross-framework confidence
        detected_patterns = await self._calculate_cross_framework_confidence(
            detected_patterns, framework_results
        )
        
        # Sort by confidence and relevance
        detected_patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        
        # Record pattern usage for learning
        await self._record_pattern_usage(detected_patterns, context)
        
        return detected_patterns
    
    async def _detect_known_patterns(
        self,
        framework_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect known patterns in the architecture"""
        detected = []
        
        for pattern_id, pattern in self.known_patterns.items():
            confidence = await self._calculate_pattern_confidence(
                pattern, framework_results, context
            )
            
            if confidence > 0.5:  # Threshold for pattern detection
                detected_pattern = {
                    "id": pattern.id,
                    "name": pattern.name,
                    "type": pattern.type,
                    "category": pattern.category,
                    "description": pattern.description,
                    "frameworks": pattern.frameworks,
                    "confidence": confidence,
                    "evidence": await self._gather_evidence(pattern, framework_results, context),
                    "benefits": pattern.benefits,
                    "drawbacks": pattern.drawbacks,
                    "implementation_guidance": pattern.implementation_guidance,
                    "quality_attributes": pattern.quality_attributes,
                    "complexity_score": pattern.complexity_score,
                    "maturity": pattern.maturity,
                    "industry_adoption": pattern.industry_adoption
                }
                detected.append(detected_pattern)
        
        return detected
    
    async def _calculate_pattern_confidence(
        self,
        pattern: ArchitecturePattern,
        framework_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for pattern detection"""
        confidence_factors = []
        
        # Framework alignment - check if frameworks that recognize this pattern found evidence
        framework_support = 0
        for result in framework_results:
            framework_name = result.get("framework", "")
            if framework_name in pattern.frameworks:
                # Check if this framework's results contain pattern indicators
                patterns_found = result.get("analysis", {}).get("patterns_identified", [])
                if any(pattern.name.lower() in p.get("name", "").lower() for p in patterns_found):
                    framework_support += 1
        
        if pattern.frameworks:
            framework_confidence = framework_support / len(pattern.frameworks)
            confidence_factors.append(framework_confidence)
        
        # Context matching - check if context contains pattern indicators
        context_text = json.dumps(context).lower()
        evidence_matches = 0
        for evidence in pattern.evidence:
            if evidence.lower() in context_text:
                evidence_matches += 1
        
        if pattern.evidence:
            context_confidence = evidence_matches / len(pattern.evidence)
            confidence_factors.append(context_confidence)
        
        # Technology stack alignment
        tech_stack = context.get("technical_stack", [])
        tech_alignment = await self._calculate_tech_alignment(pattern, tech_stack)
        confidence_factors.append(tech_alignment)
        
        # Goals alignment
        goals = context.get("goals", [])
        goal_alignment = await self._calculate_goal_alignment(pattern, goals)
        confidence_factors.append(goal_alignment)
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    async def _calculate_tech_alignment(
        self,
        pattern: ArchitecturePattern,
        tech_stack: List[str]
    ) -> float:
        """Calculate alignment between pattern and technology stack"""
        if not tech_stack:
            return 0.5  # Neutral if no tech stack info
        
        tech_keywords = {
            "microservices_architecture": ["docker", "kubernetes", "microservices", "api", "rest"],
            "serverless_architecture": ["lambda", "azure_functions", "serverless", "faas"],
            "event_driven_architecture": ["kafka", "rabbitmq", "eventbridge", "pubsub"],
            "api_gateway": ["api_gateway", "kong", "zuul", "istio"],
            "database_per_service": ["mongodb", "postgresql", "dynamodb", "cassandra"]
        }
        
        pattern_keywords = tech_keywords.get(pattern.id, [])
        if not pattern_keywords:
            return 0.5  # Neutral if no specific tech requirements
        
        matches = 0
        tech_stack_lower = [tech.lower() for tech in tech_stack]
        
        for keyword in pattern_keywords:
            if any(keyword in tech for tech in tech_stack_lower):
                matches += 1
        
        return matches / len(pattern_keywords) if pattern_keywords else 0.5
    
    async def _calculate_goal_alignment(
        self,
        pattern: ArchitecturePattern,
        goals: List[str]
    ) -> float:
        """Calculate alignment between pattern and business goals"""
        if not goals:
            return 0.5  # Neutral if no goals specified
        
        goal_keywords = {
            "scalability": ["microservices_architecture", "serverless_architecture", "event_driven_architecture"],
            "maintainability": ["layered_architecture", "microservices_architecture", "clean_architecture"],
            "performance": ["cqrs", "event_driven_architecture", "caching_patterns"],
            "security": ["zero_trust_architecture", "api_gateway"],
            "cost_optimization": ["serverless_architecture", "cloud_native_patterns"],
            "agility": ["microservices_architecture", "ci_cd_patterns", "devops_patterns"]
        }
        
        alignment_score = 0
        goals_lower = [goal.lower() for goal in goals]
        
        for goal in goals_lower:
            for keyword, supporting_patterns in goal_keywords.items():
                if keyword in goal and pattern.id in supporting_patterns:
                    alignment_score += 1
                    break
        
        return min(alignment_score / len(goals), 1.0) if goals else 0.5
    
    async def _gather_evidence(
        self,
        pattern: ArchitecturePattern,
        framework_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[str]:
        """Gather evidence supporting pattern detection"""
        evidence = []
        
        # Evidence from framework results
        for result in framework_results:
            framework_name = result.get("framework", "")
            if framework_name in pattern.frameworks:
                patterns_found = result.get("analysis", {}).get("patterns_identified", [])
                for found_pattern in patterns_found:
                    if pattern.name.lower() in found_pattern.get("name", "").lower():
                        evidence.append(f"{framework_name}: {found_pattern.get('description', 'Pattern detected')}")
        
        # Evidence from context
        context_str = json.dumps(context).lower()
        for pattern_evidence in pattern.evidence:
            if pattern_evidence.lower() in context_str:
                evidence.append(f"Context: Contains '{pattern_evidence}'")
        
        # Evidence from technical stack
        tech_stack = context.get("technical_stack", [])
        for tech in tech_stack:
            if any(keyword in tech.lower() for keyword in pattern.evidence):
                evidence.append(f"Technology: {tech} supports pattern")
        
        return evidence[:5]  # Limit to top 5 pieces of evidence
    
    async def _discover_emerging_patterns(
        self,
        framework_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Discover emerging or project-specific patterns"""
        emerging_patterns = []
        
        # Analyze common themes across framework results
        common_themes = await self._identify_common_themes(framework_results)
        
        for theme, frequency in common_themes.items():
            if frequency >= 2:  # Theme appears in multiple frameworks
                emerging_pattern = {
                    "id": f"emerging_{theme.replace(' ', '_').lower()}",
                    "name": f"Emerging Pattern: {theme.title()}",
                    "type": "emerging",
                    "category": "project_specific",
                    "description": f"Pattern identified through cross-framework analysis: {theme}",
                    "frameworks": [r["framework"] for r in framework_results if theme in str(r).lower()],
                    "confidence": min(frequency / len(framework_results), 0.8),  # Cap at 0.8 for emerging
                    "evidence": [f"Mentioned in {frequency} framework analyses"],
                    "maturity": "experimental",
                    "industry_adoption": "emerging"
                }
                emerging_patterns.append(emerging_pattern)
        
        return emerging_patterns
    
    async def _identify_common_themes(
        self,
        framework_results: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Identify common themes across framework results"""
        themes = Counter()
        
        # Extract key terms from all framework results
        all_text = ""
        for result in framework_results:
            all_text += json.dumps(result.get("analysis", {})).lower()
        
        # Look for architecture-related keywords
        architecture_keywords = [
            "api-first", "event-driven", "microservices", "serverless", "cloud-native",
            "containerization", "orchestration", "service-mesh", "data-mesh",
            "zero-trust", "multi-cloud", "edge-computing", "real-time",
            "automation", "self-healing", "observability", "chaos-engineering"
        ]
        
        for keyword in architecture_keywords:
            if keyword in all_text:
                themes[keyword] += all_text.count(keyword)
        
        return dict(themes.most_common(10))
    
    async def _analyze_pattern_relationships(
        self,
        detected_patterns: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze relationships between detected patterns"""
        relationships = {}
        
        detected_pattern_ids = {p["id"] for p in detected_patterns}
        
        for pattern in detected_patterns:
            pattern_id = pattern["id"]
            pattern_relationships = []
            
            # Find known relationships
            for rel in self.pattern_relationships:
                if (rel.source_pattern == pattern_id and rel.target_pattern in detected_pattern_ids):
                    pattern_relationships.append({
                        "target": rel.target_pattern,
                        "type": rel.relationship_type,
                        "strength": rel.strength,
                        "description": rel.description
                    })
                elif (rel.target_pattern == pattern_id and rel.source_pattern in detected_pattern_ids):
                    # Reverse relationship
                    reverse_type = self._reverse_relationship_type(rel.relationship_type)
                    pattern_relationships.append({
                        "target": rel.source_pattern,
                        "type": reverse_type,
                        "strength": rel.strength,
                        "description": f"Reverse: {rel.description}"
                    })
            
            relationships[pattern_id] = pattern_relationships
        
        return relationships
    
    def _reverse_relationship_type(self, relationship_type: str) -> str:
        """Get reverse relationship type"""
        reverse_map = {
            "requires": "required_by",
            "required_by": "requires",
            "enhances": "enhanced_by",
            "enhanced_by": "enhances",
            "conflicts": "conflicts",
            "specializes": "generalized_by",
            "generalized_by": "specializes"
        }
        return reverse_map.get(relationship_type, relationship_type)
    
    async def _calculate_cross_framework_confidence(
        self,
        detected_patterns: List[Dict[str, Any]],
        framework_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate confidence scores based on cross-framework agreement"""
        for pattern in detected_patterns:
            framework_support = 0
            total_relevant_frameworks = 0
            
            for result in framework_results:
                framework_name = result.get("framework", "")
                if framework_name in pattern.get("frameworks", []):
                    total_relevant_frameworks += 1
                    # Check if this framework actually found evidence of the pattern
                    if self._framework_supports_pattern(result, pattern):
                        framework_support += 1
            
            if total_relevant_frameworks > 0:
                cross_framework_confidence = framework_support / total_relevant_frameworks
                # Adjust original confidence based on cross-framework support
                original_confidence = pattern.get("confidence", 0.5)
                pattern["confidence"] = (original_confidence + cross_framework_confidence) / 2
            
            # Add metadata about framework support
            pattern["framework_support"] = {
                "supporting_frameworks": framework_support,
                "total_relevant_frameworks": total_relevant_frameworks,
                "support_ratio": framework_support / max(total_relevant_frameworks, 1)
            }
        
        return detected_patterns
    
    def _framework_supports_pattern(
        self,
        framework_result: Dict[str, Any],
        pattern: Dict[str, Any]
    ) -> bool:
        """Check if framework result supports the pattern"""
        analysis = framework_result.get("analysis", {})
        
        # Check if pattern name appears in framework's identified patterns
        identified_patterns = analysis.get("patterns_identified", [])
        for identified in identified_patterns:
            if pattern["name"].lower() in identified.get("name", "").lower():
                return True
        
        # Check if pattern evidence appears in framework recommendations
        recommendations = analysis.get("recommendations", [])
        for rec in recommendations:
            if any(evidence.lower() in rec.get("description", "").lower() 
                   for evidence in pattern.get("evidence", [])):
                return True
        
        return False
    
    async def _record_pattern_usage(
        self,
        detected_patterns: List[Dict[str, Any]],
        context: Dict[str, Any]
    ):
        """Record pattern usage for learning and improvement"""
        usage_record = {
            "timestamp": datetime.now().isoformat(),
            "context_hash": hash(json.dumps(context, sort_keys=True)),
            "patterns_detected": [p["id"] for p in detected_patterns],
            "pattern_confidences": {p["id"]: p["confidence"] for p in detected_patterns},
            "domain": context.get("domain", "unknown"),
            "organization_size": context.get("organization_size", "unknown"),
            "technical_stack": context.get("technical_stack", [])
        }
        
        self.pattern_usage_history.append(usage_record)
        
        # Limit history size
        if len(self.pattern_usage_history) > 1000:
            self.pattern_usage_history = self.pattern_usage_history[-1000:]
        
        self.logger.debug(f"Recorded pattern usage: {len(detected_patterns)} patterns detected")
    
    async def get_pattern_recommendations(
        self,
        context: Dict[str, Any],
        current_patterns: List[str],
        goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Get intelligent pattern recommendations based on context and goals"""
        recommendations = []
        
        current_pattern_ids = set(current_patterns)
        
        # Find patterns that support the goals
        for goal in goals:
            supporting_patterns = await self._find_patterns_for_goal(goal)
            
            for pattern_id in supporting_patterns:
                if pattern_id not in current_pattern_ids and pattern_id in self.known_patterns:
                    pattern = self.known_patterns[pattern_id]
                    
                    # Calculate recommendation score
                    score = await self._calculate_recommendation_score(
                        pattern, context, goals, current_pattern_ids
                    )
                    
                    if score > 0.6:  # Threshold for recommendations
                        recommendation = {
                            "pattern_id": pattern.id,
                            "pattern_name": pattern.name,
                            "description": pattern.description,
                            "relevance_score": score,
                            "supports_goals": [goal],
                            "benefits": pattern.benefits,
                            "implementation_effort": pattern.complexity_score,
                            "maturity": pattern.maturity,
                            "prerequisites": await self._get_pattern_prerequisites(pattern.id, current_pattern_ids),
                            "conflicts": await self._get_pattern_conflicts(pattern.id, current_pattern_ids),
                            "implementation_guidance": pattern.implementation_guidance[:3]  # Top 3 guidance items
                        }
                        recommendations.append(recommendation)
        
        # Remove duplicates and sort by relevance
        seen_patterns = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec["pattern_id"] not in seen_patterns:
                seen_patterns.add(rec["pattern_id"])
                unique_recommendations.append(rec)
        
        unique_recommendations.sort(key=lambda r: r["relevance_score"], reverse=True)
        
        return unique_recommendations[:10]  # Return top 10 recommendations
    
    async def _find_patterns_for_goal(self, goal: str) -> List[str]:
        """Find patterns that support a specific goal"""
        goal_lower = goal.lower()
        supporting_patterns = []
        
        for pattern_id, pattern in self.known_patterns.items():
            # Check if goal keywords appear in pattern benefits or quality attributes
            pattern_text = " ".join(pattern.benefits + pattern.quality_attributes).lower()
            
            if any(keyword in pattern_text for keyword in goal_lower.split()):
                supporting_patterns.append(pattern_id)
        
        return supporting_patterns
    
    async def _calculate_recommendation_score(
        self,
        pattern: ArchitecturePattern,
        context: Dict[str, Any],
        goals: List[str],
        current_patterns: Set[str]
    ) -> float:
        """Calculate recommendation score for a pattern"""
        score_factors = []
        
        # Goal alignment
        goal_score = await self._calculate_goal_alignment(pattern, goals)
        score_factors.append(goal_score * 0.4)  # 40% weight
        
        # Technology alignment
        tech_score = await self._calculate_tech_alignment(pattern, context.get("technical_stack", []))
        score_factors.append(tech_score * 0.3)  # 30% weight
        
        # Maturity and adoption
        maturity_scores = {"proven": 1.0, "emerging": 0.7, "experimental": 0.4}
        adoption_scores = {"widespread": 1.0, "moderate": 0.7, "niche": 0.4}
        
        maturity_score = maturity_scores.get(pattern.maturity, 0.5)
        adoption_score = adoption_scores.get(pattern.industry_adoption, 0.5)
        score_factors.append((maturity_score + adoption_score) / 2 * 0.2)  # 20% weight
        
        # Complexity consideration (lower complexity = higher score)
        complexity_score = 1.0 - pattern.complexity_score
        score_factors.append(complexity_score * 0.1)  # 10% weight
        
        return sum(score_factors)
    
    async def _get_pattern_prerequisites(
        self,
        pattern_id: str,
        current_patterns: Set[str]
    ) -> List[str]:
        """Get prerequisites for implementing a pattern"""
        prerequisites = []
        
        for relationship in self.pattern_relationships:
            if (relationship.target_pattern == pattern_id and 
                relationship.relationship_type == "requires" and
                relationship.source_pattern not in current_patterns):
                prerequisites.append(relationship.source_pattern)
        
        return prerequisites
    
    async def _get_pattern_conflicts(
        self,
        pattern_id: str,
        current_patterns: Set[str]
    ) -> List[str]:
        """Get patterns that conflict with the given pattern"""
        conflicts = []
        
        for relationship in self.pattern_relationships:
            if (relationship.relationship_type == "conflicts" and
                ((relationship.source_pattern == pattern_id and relationship.target_pattern in current_patterns) or
                 (relationship.target_pattern == pattern_id and relationship.source_pattern in current_patterns))):
                conflict_pattern = (relationship.target_pattern if relationship.source_pattern == pattern_id 
                                  else relationship.source_pattern)
                conflicts.append(conflict_pattern)
        
        return conflicts
    
    async def update_pattern_feedback(
        self,
        pattern_id: str,
        feedback: Dict[str, Any]
    ):
        """Update pattern based on user feedback"""
        self.feedback_data[pattern_id].append({
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Learn from feedback to improve pattern detection
        if feedback.get("accuracy") == "incorrect":
            # Lower confidence for this pattern in similar contexts
            self.logger.info(f"Received negative feedback for pattern {pattern_id}")
        elif feedback.get("accuracy") == "correct":
            # Increase confidence for this pattern in similar contexts
            self.logger.info(f"Received positive feedback for pattern {pattern_id}")
        
        # Update pattern based on feedback
        if pattern_id in self.known_patterns:
            await self._update_pattern_from_feedback(pattern_id, feedback)
    
    async def _update_pattern_from_feedback(
        self,
        pattern_id: str,
        feedback: Dict[str, Any]
    ):
        """Update pattern definition based on feedback"""
        pattern = self.known_patterns[pattern_id]
        
        # Update benefits if user provides additional ones
        if "additional_benefits" in feedback:
            new_benefits = feedback["additional_benefits"]
            for benefit in new_benefits:
                if benefit not in pattern.benefits:
                    pattern.benefits.append(benefit)
        
        # Update drawbacks if user identifies new ones
        if "additional_drawbacks" in feedback:
            new_drawbacks = feedback["additional_drawbacks"]
            for drawback in new_drawbacks:
                if drawback not in pattern.drawbacks:
                    pattern.drawbacks.append(drawback)
        
        # Update implementation guidance
        if "additional_guidance" in feedback:
            new_guidance = feedback["additional_guidance"]
            for guidance in new_guidance:
                if guidance not in pattern.implementation_guidance:
                    pattern.implementation_guidance.append(guidance)
        
        self.logger.info(f"Updated pattern {pattern_id} based on user feedback")
    
    async def export_pattern_analysis(
        self,
        detected_patterns: List[Dict[str, Any]],
        format: str = "json"
    ) -> str:
        """Export pattern analysis results"""
        if format == "json":
            return json.dumps(detected_patterns, indent=2, default=str)
        elif format == "markdown":
            return self._export_patterns_as_markdown(detected_patterns)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_patterns_as_markdown(self, patterns: List[Dict[str, Any]]) -> str:
        """Export patterns as markdown report"""
        md = "# Architecture Patterns Analysis\n\n"
        
        md += f"**Total Patterns Detected:** {len(patterns)}\n\n"
        
        # Group patterns by category
        by_category = defaultdict(list)
        for pattern in patterns:
            by_category[pattern.get("category", "unknown")].append(pattern)
        
        for category, category_patterns in by_category.items():
            md += f"## {category.title()} Patterns\n\n"
            
            for pattern in category_patterns:
                md += f"### {pattern['name']}\n"
                md += f"**Confidence:** {pattern['confidence']:.2f}\n"
                md += f"**Type:** {pattern['type']}\n\n"
                md += f"{pattern['description']}\n\n"
                
                if pattern.get("benefits"):
                    md += "**Benefits:**\n"
                    for benefit in pattern["benefits"][:3]:  # Top 3 benefits
                        md += f"- {benefit}\n"
                    md += "\n"
                
                if pattern.get("evidence"):
                    md += "**Evidence:**\n"
                    for evidence in pattern["evidence"][:3]:  # Top 3 evidence items
                        md += f"- {evidence}\n"
                    md += "\n"
        
        return md