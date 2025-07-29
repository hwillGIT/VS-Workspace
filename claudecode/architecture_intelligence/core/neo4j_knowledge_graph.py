"""
Neo4j Knowledge Graph for Architecture Intelligence

Manages architectural knowledge relationships, conflicts, and provenance using Neo4j.
Integrates with ChromaDB for semantic search capabilities.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json

try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    print("Neo4j driver not installed. Run: pip install neo4j")
    GraphDatabase = None


class ConflictType(Enum):
    """Types of architectural conflicts"""
    IMPLEMENTATION_APPROACH = "implementation_approach"
    PERFORMANCE_TRADEOFF = "performance_tradeoff"
    SECURITY_CONCERN = "security_concern"
    SCALABILITY_VIEW = "scalability_view"
    MAINTAINABILITY_OPINION = "maintainability_opinion"
    PHILOSOPHICAL_DIFFERENCE = "philosophical_difference"


class KnowledgeScope(Enum):
    """Scope of knowledge storage"""
    PROJECT = "project"
    GLOBAL = "global"
    BOTH = "both"


@dataclass
class ArchitecturalPattern:
    """Represents an architectural pattern in the knowledge graph"""
    name: str
    category: str
    description: str
    benefits: List[str]
    drawbacks: List[str]
    implementation_guidance: str
    source: str
    author: str
    confidence_score: float
    extraction_date: datetime = None

    def __post_init__(self):
        if self.extraction_date is None:
            self.extraction_date = datetime.now()


@dataclass
class ArchitecturalPrinciple:
    """Represents a design principle"""
    name: str
    statement: str
    rationale: str
    application_examples: List[str]
    related_patterns: List[str]
    source: str
    author: str
    confidence_score: float


@dataclass
class ArchitecturalConflict:
    """Represents a conflict between architectural viewpoints"""
    topic: str
    conflict_type: ConflictType
    source_a: str
    source_b: str
    position_a: str
    position_b: str
    reasoning_a: str
    reasoning_b: str
    detected_date: datetime = None
    resolved: bool = False
    resolution: Optional[str] = None

    def __post_init__(self):
        if self.detected_date is None:
            self.detected_date = datetime.now()


class Neo4jKnowledgeGraph:
    """
    Neo4j-powered knowledge graph for architectural intelligence.
    
    Manages relationships between patterns, principles, frameworks, and conflicts.
    Supports both project-specific and global knowledge scopes.
    """
    
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "architecture123",
        database: str = "neo4j"
    ):
        self.logger = logging.getLogger(__name__)
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        if GraphDatabase is None:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.driver = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Neo4j connection and create constraints/indexes"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=basic_auth(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            self.logger.info("Connected to Neo4j successfully")
            
            # Create schema
            asyncio.create_task(self._create_schema())
            
        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.logger.info("To start Neo4j:")
            self.logger.info("docker run --name neo4j-architecture -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/architecture123 neo4j:5.15")
            raise
    
    async def _create_schema(self):
        """Create Neo4j constraints and indexes for optimal performance"""
        schema_queries = [
            # Constraints for uniqueness
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT pattern_name IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE", 
            "CREATE CONSTRAINT principle_name IF NOT EXISTS FOR (pr:Principle) REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT framework_name IF NOT EXISTS FOR (f:Framework) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (proj:Project) REQUIRE proj.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX pattern_category IF NOT EXISTS FOR (p:Pattern) ON (p.category)",
            "CREATE INDEX document_source IF NOT EXISTS FOR (d:Document) ON (d.source)",
            "CREATE INDEX extraction_date IF NOT EXISTS FOR (p:Pattern) ON (p.extraction_date)",
            "CREATE INDEX conflict_type IF NOT EXISTS FOR (c:Conflict) ON (c.conflict_type)",
            "CREATE INDEX knowledge_scope IF NOT EXISTS FOR (n) ON (n.scope) WHERE n:Pattern OR n:Principle"
        ]
        
        with self.driver.session(database=self.database) as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    self.logger.debug(f"Executed schema query: {query}")
                except Exception as e:
                    self.logger.warning(f"Schema query failed (may already exist): {e}")
    
    async def add_pattern(
        self, 
        pattern: ArchitecturalPattern, 
        scope: KnowledgeScope = KnowledgeScope.PROJECT,
        project_id: Optional[str] = None
    ) -> str:
        """
        Add an architectural pattern to the knowledge graph.
        
        Returns:
            pattern_node_id: Neo4j node ID for the created pattern
        """
        
        with self.driver.session(database=self.database) as session:
            # Create pattern node
            pattern_query = """
            MERGE (p:Pattern {name: $name})
            SET p.category = $category,
                p.description = $description,
                p.benefits = $benefits,
                p.drawbacks = $drawbacks,
                p.implementation_guidance = $implementation_guidance,
                p.confidence_score = $confidence_score,
                p.extraction_date = datetime($extraction_date),
                p.scope = $scope
            
            // Create or connect to author
            MERGE (a:Author {name: $author})
            MERGE (a)-[:RECOMMENDS]->(p)
            
            // Create or connect to document
            MERGE (d:Document {id: $source})
            SET d.source = $source
            MERGE (d)-[:CONTAINS]->(p)
            
            // Connect to project if project scope
            WITH p
            WHERE $project_id IS NOT NULL
            MERGE (proj:Project {id: $project_id})
            MERGE (proj)-[:USES]->(p)
            
            RETURN p, id(p) as pattern_id
            """
            
            result = session.run(pattern_query, {
                "name": pattern.name,
                "category": pattern.category,
                "description": pattern.description,
                "benefits": pattern.benefits,
                "drawbacks": pattern.drawbacks,
                "implementation_guidance": pattern.implementation_guidance,
                "confidence_score": pattern.confidence_score,
                "extraction_date": pattern.extraction_date.isoformat(),
                "author": pattern.author,
                "source": pattern.source,
                "scope": scope.value,
                "project_id": project_id if scope in [KnowledgeScope.PROJECT, KnowledgeScope.BOTH] else None
            })
            
            record = result.single()
            pattern_id = record["pattern_id"]
            
            self.logger.info(f"Added pattern '{pattern.name}' with ID {pattern_id}")
            return str(pattern_id)
    
    async def detect_conflicts(self, new_pattern: ArchitecturalPattern) -> List[ArchitecturalConflict]:
        """
        Detect conflicts between new pattern and existing knowledge.
        
        Uses semantic analysis and explicit conflict rules to identify disagreements.
        """
        
        conflicts = []
        
        with self.driver.session(database=self.database) as session:
            # Find patterns in same category for conflict detection
            conflict_query = """
            MATCH (existing:Pattern {category: $category})
            WHERE existing.name <> $pattern_name
            RETURN existing.name as name, 
                   existing.description as description,
                   existing.benefits as benefits,
                   existing.drawbacks as drawbacks,
                   existing.author as author,
                   existing.source as source
            """
            
            results = session.run(conflict_query, {
                "category": new_pattern.category,
                "pattern_name": new_pattern.name
            })
            
            for record in results:
                # Simple conflict detection heuristics
                existing_benefits = set(record["benefits"])
                new_benefits = set(new_pattern.benefits)
                
                existing_drawbacks = set(record["drawbacks"])
                new_drawbacks = set(new_pattern.drawbacks)
                
                # Check for contradictory benefits/drawbacks
                if (existing_benefits & new_drawbacks) or (new_benefits & existing_drawbacks):
                    conflict = ArchitecturalConflict(
                        topic=new_pattern.name,
                        conflict_type=ConflictType.IMPLEMENTATION_APPROACH,
                        source_a=new_pattern.source,
                        source_b=record["source"],
                        position_a=f"Benefits: {', '.join(new_pattern.benefits)}",
                        position_b=f"Benefits: {', '.join(record['benefits'])}",
                        reasoning_a=new_pattern.description,
                        reasoning_b=record["description"]
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def save_conflict(self, conflict: ArchitecturalConflict) -> str:
        """Save a detected conflict to the knowledge graph"""
        
        with self.driver.session(database=self.database) as session:
            conflict_query = """
            CREATE (c:Conflict {
                topic: $topic,
                conflict_type: $conflict_type,
                position_a: $position_a,
                position_b: $position_b,
                reasoning_a: $reasoning_a,
                reasoning_b: $reasoning_b,
                detected_date: datetime($detected_date),
                resolved: false
            })
            
            // Connect to source documents
            MERGE (da:Document {id: $source_a})
            MERGE (db:Document {id: $source_b})
            MERGE (da)-[:HAS_CONFLICT]->(c)
            MERGE (db)-[:HAS_CONFLICT]->(c)
            
            RETURN id(c) as conflict_id
            """
            
            result = session.run(conflict_query, {
                "topic": conflict.topic,
                "conflict_type": conflict.conflict_type.value,
                "position_a": conflict.position_a,
                "position_b": conflict.position_b,
                "reasoning_a": conflict.reasoning_a,
                "reasoning_b": conflict.reasoning_b,
                "detected_date": conflict.detected_date.isoformat(),
                "source_a": conflict.source_a,
                "source_b": conflict.source_b
            })
            
            conflict_id = result.single()["conflict_id"]
            self.logger.info(f"Saved conflict on topic '{conflict.topic}' with ID {conflict_id}")
            return str(conflict_id)
    
    async def get_conflicts_for_review(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get unresolved conflicts for user review"""
        
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (c:Conflict {resolved: false})
            OPTIONAL MATCH (da:Document)-[:HAS_CONFLICT]->(c)
            OPTIONAL MATCH (db:Document)-[:HAS_CONFLICT]->(c)
            RETURN c, da.source as source_a, db.source as source_b
            ORDER BY c.detected_date DESC
            LIMIT $limit
            """
            
            results = session.run(query, {"limit": limit})
            
            conflicts = []
            for record in results:
                conflict_node = record["c"]
                conflicts.append({
                    "id": conflict_node.id,
                    "topic": conflict_node["topic"],
                    "conflict_type": conflict_node["conflict_type"],
                    "position_a": conflict_node["position_a"],
                    "position_b": conflict_node["position_b"],
                    "source_a": record["source_a"],
                    "source_b": record["source_b"],
                    "detected_date": conflict_node["detected_date"]
                })
            
            return conflicts
    
    async def request_synthesis_permission(self, conflicts: List[Dict[str, Any]]) -> bool:
        """Present conflicts to user and request synthesis permission"""
        
        if not conflicts:
            return True
        
        print(f"\n🚨 Found {len(conflicts)} conflicting architectural views:")
        print("=" * 60)
        
        for i, conflict in enumerate(conflicts, 1):
            print(f"\n{i}. Topic: {conflict['topic']}")
            print(f"   Type: {conflict['conflict_type'].replace('_', ' ').title()}")
            print(f"   Source A: {conflict['source_a']}")
            print(f"   Position: {conflict['position_a'][:100]}...")
            print(f"   Source B: {conflict['source_b']}")
            print(f"   Position: {conflict['position_b'][:100]}...")
        
        print("\n" + "=" * 60)
        response = input("Would you like me to synthesize these conflicts? (y/n/details): ").lower()
        
        if response == 'details':
            await self._show_conflict_details(conflicts)
            response = input("Proceed with synthesis? (y/n): ").lower()
        
        return response == 'y'
    
    async def _show_conflict_details(self, conflicts: List[Dict[str, Any]]):
        """Show detailed conflict information"""
        
        for i, conflict in enumerate(conflicts, 1):
            print(f"\n--- Conflict {i} Details ---")
            print(f"Topic: {conflict['topic']}")
            print(f"Type: {conflict['conflict_type']}")
            print(f"\nSource A ({conflict['source_a']}):")
            print(f"Position: {conflict['position_a']}")
            print(f"\nSource B ({conflict['source_b']}):")
            print(f"Position: {conflict['position_b']}")
            print("-" * 40)
    
    async def synthesize_conflicts(self, conflicts: List[Dict[str, Any]], synthesis: str) -> str:
        """Save conflict synthesis to knowledge graph"""
        
        with self.driver.session(database=self.database) as session:
            # Mark conflicts as resolved and add synthesis
            for conflict in conflicts:
                resolve_query = """
                MATCH (c:Conflict) WHERE id(c) = $conflict_id
                SET c.resolved = true,
                    c.resolution = $synthesis,
                    c.resolution_date = datetime()
                RETURN c
                """
                
                session.run(resolve_query, {
                    "conflict_id": conflict["id"],
                    "synthesis": synthesis
                })
        
        self.logger.info(f"Synthesized {len(conflicts)} conflicts")
        return "Conflicts resolved and synthesis saved"
    
    async def promote_to_global(self, pattern_name: str, project_id: str) -> bool:
        """Promote a project-specific pattern to global knowledge"""
        
        response = input(f"Add '{pattern_name}' to global knowledge base? (y/n): ").lower()
        
        if response == 'y':
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (proj:Project {id: $project_id})-[:USES]->(p:Pattern {name: $pattern_name})
                SET p.scope = 'global'
                RETURN p
                """
                
                result = session.run(query, {
                    "project_id": project_id,
                    "pattern_name": pattern_name
                })
                
                if result.single():
                    self.logger.info(f"Promoted '{pattern_name}' to global knowledge")
                    return True
        
        return False
    
    async def get_related_patterns(self, pattern_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get patterns related to the given pattern through various relationships"""
        
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (p:Pattern {name: $pattern_name})
            MATCH (related:Pattern)
            WHERE related.name <> p.name
            AND (
                related.category = p.category OR
                any(benefit in p.benefits WHERE benefit IN related.benefits) OR
                any(drawback in p.drawbacks WHERE drawback IN related.drawbacks)
            )
            RETURN related.name as name,
                   related.category as category, 
                   related.description as description,
                   related.confidence_score as confidence_score
            ORDER BY related.confidence_score DESC
            LIMIT $max_results
            """
            
            results = session.run(query, {
                "pattern_name": pattern_name,
                "max_results": max_results
            })
            
            return [dict(record) for record in results]
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")


# Convenience functions for common operations

async def create_knowledge_graph(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j", 
    password: str = "architecture123"
) -> Neo4jKnowledgeGraph:
    """Create and initialize a Neo4j knowledge graph"""
    
    kg = Neo4jKnowledgeGraph(uri, username, password)
    return kg


async def setup_neo4j_with_docker():
    """Helper to set up Neo4j using Docker"""
    
    import subprocess
    
    print("🚀 Starting Neo4j with Docker...")
    
    # Check if container already exists
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=neo4j-architecture"], 
            capture_output=True, text=True
        )
        
        if "neo4j-architecture" in result.stdout:
            print("📦 Neo4j container exists, starting...")
            subprocess.run(["docker", "start", "neo4j-architecture"])
        else:
            print("📦 Creating new Neo4j container...")
            subprocess.run([
                "docker", "run",
                "--name", "neo4j-architecture",
                "-p", "7474:7474", "-p", "7687:7687",
                "-d",
                "-v", "neo4j_data:/data",
                "-v", "neo4j_logs:/logs", 
                "-v", "neo4j_import:/var/lib/neo4j/import",
                "-e", "NEO4J_AUTH=neo4j/architecture123",
                "neo4j:5.15"
            ])
        
        print("✅ Neo4j is running!")
        print("🌐 Browser: http://localhost:7474")
        print("🔐 Username: neo4j | Password: architecture123")
        
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
        return False
    except Exception as e:
        print(f"❌ Failed to start Neo4j: {e}")
        return False
    
    return True