"""
Integrated Knowledge Manager

Combines Neo4j (relationships) and ChromaDB (semantic search) for comprehensive
architecture knowledge management with conflict detection and synthesis.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import json

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")
    chromadb = None

# Neo4j imports  
from .neo4j_knowledge_graph import (
    Neo4jKnowledgeGraph, 
    ArchitecturalPattern, 
    ArchitecturalPrinciple,
    ArchitecturalConflict,
    KnowledgeScope,
    ConflictType,
    create_knowledge_graph
)


class IntegratedKnowledgeManager:
    """
    Unified knowledge management combining Neo4j graph relationships 
    with ChromaDB semantic search capabilities.
    
    Features:
    - Project-specific and global knowledge scopes
    - Automatic conflict detection and synthesis
    - Semantic search across patterns and documents
    - Knowledge provenance and traceability
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        chroma_path: str = "./data/architecture_knowledge",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "architecture123"
    ):
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id
        self.chroma_path = Path(chroma_path)
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collections = {}
        self._initialize_chromadb()
        
        # Initialize Neo4j
        self.neo4j_kg = None
        self.neo4j_config = {
            "uri": neo4j_uri,
            "username": neo4j_username, 
            "password": neo4j_password
        }
        
        # Will be initialized async
        self._neo4j_ready = asyncio.Event()
        asyncio.create_task(self._initialize_neo4j())
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB collections for semantic search"""
        
        if chromadb is None:
            self.logger.error("ChromaDB not available")
            return
            
        try:
            # Create persistent client
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path)
            )
            
            # Initialize collections with different embedding functions
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            collection_configs = [
                ("documents", "Raw document chunks and metadata"),
                ("patterns", "Architectural patterns with descriptions"),
                ("principles", "Design principles and guidelines"),
                ("frameworks", "Framework knowledge and methodologies"),
                ("conflicts", "Conflicting architectural viewpoints")
            ]
            
            for collection_name, description in collection_configs:
                try:
                    collection = self.chroma_client.get_or_create_collection(
                        name=f"{collection_name}_{self.project_id or 'global'}",
                        embedding_function=embedding_function,
                        metadata={"description": description}
                    )
                    self.collections[collection_name] = collection
                    self.logger.debug(f"Initialized ChromaDB collection: {collection_name}")
                except Exception as e:
                    self.logger.error(f"Failed to create ChromaDB collection {collection_name}: {e}")
            
            self.logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
    
    async def _initialize_neo4j(self):
        """Initialize Neo4j knowledge graph"""
        
        try:
            self.neo4j_kg = await create_knowledge_graph(**self.neo4j_config)
            self._neo4j_ready.set()
            self.logger.info("Neo4j knowledge graph initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j: {e}")
            self.logger.info("Consider running: docker run --name neo4j-architecture -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/architecture123 neo4j:5.15")
    
    async def ensure_neo4j_ready(self):
        """Ensure Neo4j is ready for operations"""
        await self._neo4j_ready.wait()
        if self.neo4j_kg is None:
            raise RuntimeError("Neo4j knowledge graph not available")
    
    async def add_pattern(
        self, 
        pattern: ArchitecturalPattern,
        scope: KnowledgeScope = KnowledgeScope.PROJECT,
        check_conflicts: bool = True
    ) -> Dict[str, Any]:
        """
        Add architectural pattern with conflict detection and synthesis.
        
        Returns:
            Dictionary with pattern_id, conflicts detected, and synthesis status
        """
        
        await self.ensure_neo4j_ready()
        
        result = {
            "pattern_id": None,
            "conflicts_detected": [],
            "synthesis_performed": False,
            "added_to_global": False
        }
        
        # 1. Detect conflicts before adding
        if check_conflicts:
            conflicts = await self.neo4j_kg.detect_conflicts(pattern)
            result["conflicts_detected"] = conflicts
            
            # 2. Request synthesis permission if conflicts found
            if conflicts:
                conflicts_dict = [
                    {
                        "id": None,  # Not saved yet
                        "topic": c.topic,
                        "conflict_type": c.conflict_type,
                        "position_a": c.position_a,
                        "position_b": c.position_b,
                        "source_a": c.source_a,
                        "source_b": c.source_b
                    }
                    for c in conflicts
                ]
                
                should_synthesize = await self.neo4j_kg.request_synthesis_permission(conflicts_dict)
                
                if should_synthesize:
                    # Save conflicts and get synthesis from user
                    synthesis = await self._get_conflict_synthesis(conflicts)
                    
                    # Save conflicts to Neo4j
                    for conflict in conflicts:
                        await self.neo4j_kg.save_conflict(conflict)
                    
                    # Mark as synthesized
                    await self.neo4j_kg.synthesize_conflicts(conflicts_dict, synthesis)
                    result["synthesis_performed"] = True
                else:
                    print("⚠️  Pattern will be added despite conflicts.")
        
        # 3. Add pattern to Neo4j
        pattern_id = await self.neo4j_kg.add_pattern(pattern, scope, self.project_id)
        result["pattern_id"] = pattern_id
        
        # 4. Add pattern to ChromaDB for semantic search
        await self._add_pattern_to_chromadb(pattern, pattern_id)
        
        # 5. Check if should promote to global
        if scope == KnowledgeScope.PROJECT and self.project_id:
            promoted = await self.neo4j_kg.promote_to_global(pattern.name, self.project_id)
            result["added_to_global"] = promoted
        
        self.logger.info(f"Successfully added pattern '{pattern.name}' with ID {pattern_id}")
        return result
    
    async def _add_pattern_to_chromadb(self, pattern: ArchitecturalPattern, pattern_id: str):
        """Add pattern to ChromaDB for semantic search"""
        
        if "patterns" not in self.collections:
            self.logger.warning("Patterns collection not available in ChromaDB")
            return
        
        # Create searchable text combining all pattern information
        searchable_text = f"""
        {pattern.name} - {pattern.category}
        
        Description: {pattern.description}
        
        Benefits: {' '.join(pattern.benefits)}
        
        Drawbacks: {' '.join(pattern.drawbacks)}
        
        Implementation: {pattern.implementation_guidance}
        
        Author: {pattern.author}
        Source: {pattern.source}
        """
        
        # Create document ID
        doc_id = f"pattern_{pattern_id}"
        
        # Add to ChromaDB
        self.collections["patterns"].add(
            documents=[searchable_text.strip()],
            ids=[doc_id],
            metadatas=[{
                "pattern_name": pattern.name,
                "category": pattern.category,
                "author": pattern.author,
                "source": pattern.source,
                "confidence_score": pattern.confidence_score,
                "extraction_date": pattern.extraction_date.isoformat(),
                "neo4j_id": pattern_id,
                "scope": "project" if self.project_id else "global"
            }]
        )
        
        self.logger.debug(f"Added pattern '{pattern.name}' to ChromaDB")
    
    async def _get_conflict_synthesis(self, conflicts: List[ArchitecturalConflict]) -> str:
        """Get synthesis from user for conflicts"""
        
        print("\n🤝 Please provide a synthesis of these conflicts:")
        print("   (How should these different viewpoints be reconciled?)")
        print("-" * 60)
        
        synthesis = input("Your synthesis: ").strip()
        
        if not synthesis:
            synthesis = "Conflicts noted but no synthesis provided. Requires further analysis."
        
        return synthesis
    
    async def semantic_search_patterns(
        self,
        query: str,
        n_results: int = 5,
        scope: Optional[KnowledgeScope] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across architectural patterns.
        
        Args:
            query: Search query
            n_results: Number of results to return
            scope: Filter by scope (project/global)
            
        Returns:
            List of matching patterns with similarity scores
        """
        
        if "patterns" not in self.collections:
            self.logger.warning("Patterns collection not available")
            return []
        
        # Build where clause for scope filtering
        where_clause = {}
        if scope:
            where_clause["scope"] = scope.value
        elif self.project_id:
            # Default to project scope if we have a project ID
            where_clause["scope"] = "project"
        
        try:
            results = self.collections["patterns"].query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "pattern_name": results["metadatas"][0][i]["pattern_name"],
                    "category": results["metadatas"][0][i]["category"],
                    "author": results["metadatas"][0][i]["author"],
                    "source": results["metadatas"][0][i]["source"],
                    "confidence_score": results["metadatas"][0][i]["confidence_score"],
                    "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "content_preview": results["documents"][0][i][:200] + "..." if len(results["documents"][0][i]) > 200 else results["documents"][0][i]
                })
            
            self.logger.info(f"Found {len(formatted_results)} patterns matching '{query}'")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    async def get_pattern_graph(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get comprehensive pattern information including relationships from Neo4j
        and semantic search results from ChromaDB.
        """
        
        await self.ensure_neo4j_ready()
        
        # Get related patterns from Neo4j
        related_patterns = await self.neo4j_kg.get_related_patterns(pattern_name)
        
        # Get semantically similar patterns from ChromaDB
        similar_patterns = await self.semantic_search_patterns(pattern_name, n_results=3)
        
        # Get any conflicts involving this pattern
        conflicts = await self.neo4j_kg.get_conflicts_for_review()
        pattern_conflicts = [c for c in conflicts if pattern_name.lower() in c["topic"].lower()]
        
        return {
            "pattern_name": pattern_name,
            "related_patterns": related_patterns,
            "similar_patterns": similar_patterns,
            "conflicts": pattern_conflicts,
            "relationship_count": len(related_patterns),
            "similarity_count": len(similar_patterns),
            "conflict_count": len(pattern_conflicts)
        }
    
    async def export_knowledge_base(
        self, 
        output_path: Path,
        scope: Optional[KnowledgeScope] = None,
        format: str = "json"
    ) -> str:
        """
        Export knowledge base to file for backup or sharing.
        
        Returns:
            Path to exported file
        """
        
        await self.ensure_neo4j_ready()
        
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "project_id": self.project_id,
                "scope": scope.value if scope else "all",
                "format_version": "1.0"
            },
            "patterns": [],
            "conflicts": [],
            "relationships": []
        }
        
        # Export patterns from ChromaDB
        if "patterns" in self.collections:
            # Get all patterns
            all_patterns = self.collections["patterns"].get()
            
            for i, pattern_id in enumerate(all_patterns["ids"]):
                pattern_data = {
                    "id": pattern_id,
                    "metadata": all_patterns["metadatas"][i],
                    "content": all_patterns["documents"][i]
                }
                export_data["patterns"].append(pattern_data)
        
        # Export conflicts from Neo4j
        conflicts = await self.neo4j_kg.get_conflicts_for_review(limit=1000)
        export_data["conflicts"] = conflicts
        
        # Save to file
        output_file = output_path / f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Knowledge base exported to {output_file}")
        return str(output_file)
    
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        
        await self.ensure_neo4j_ready()
        
        stats = {
            "project_id": self.project_id,
            "chromadb_stats": {},
            "neo4j_stats": {},
            "integration_health": "unknown"
        }
        
        # ChromaDB statistics
        for collection_name, collection in self.collections.items():
            try:
                count = collection.count()
                stats["chromadb_stats"][collection_name] = {
                    "document_count": count,
                    "collection_name": collection.name
                }
            except Exception as e:
                stats["chromadb_stats"][collection_name] = {"error": str(e)}
        
        # Neo4j statistics (would require custom queries)
        try:
            # This would need implementation in Neo4jKnowledgeGraph
            stats["neo4j_stats"] = {
                "patterns": "Available",
                "conflicts": "Available", 
                "relationships": "Available"
            }
            stats["integration_health"] = "healthy"
        except Exception as e:
            stats["neo4j_stats"] = {"error": str(e)}
            stats["integration_health"] = "degraded"
        
        return stats
    
    async def shutdown(self):
        """Cleanup resources"""
        
        if self.neo4j_kg:
            self.neo4j_kg.close()
        
        # ChromaDB doesn't need explicit shutdown
        
        self.logger.info("Knowledge manager shutdown complete")


# Convenience functions

async def create_integrated_knowledge_manager(
    project_id: Optional[str] = None,
    chroma_path: str = "./data/architecture_knowledge"
) -> IntegratedKnowledgeManager:
    """Create and initialize integrated knowledge manager"""
    
    manager = IntegratedKnowledgeManager(
        project_id=project_id,
        chroma_path=chroma_path
    )
    
    # Wait for Neo4j to be ready
    await manager.ensure_neo4j_ready()
    
    return manager