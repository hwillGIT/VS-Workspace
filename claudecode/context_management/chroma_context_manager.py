"""
ChromaDB Context Manager for AI Agents

Provides semantic context storage and retrieval for Claude Code and other AI projects.
Supports multiple abstraction levels: conversation, session, project, and global contexts.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextLevel(Enum):
    """Context hierarchy levels"""
    IMMEDIATE = "immediate"    # Current conversation
    SESSION = "session"        # Current work session
    PROJECT = "project"        # Project-specific knowledge
    GLOBAL = "global"          # Universal patterns/knowledge


@dataclass
class ContextEntry:
    """Represents a single context entry"""
    content: str
    level: ContextLevel
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "content": self.content,
            "level": self.level.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ChromaContextManager:
    """
    Manages hierarchical context storage using ChromaDB.
    
    Features:
    - Multi-level context hierarchy
    - Semantic search with metadata filtering
    - Automatic embedding generation
    - Project isolation
    - Session management
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_context_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB context manager.
        
        Args:
            persist_directory: Where to store the database
            embedding_model: Which sentence transformer model to use
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                persist_directory=str(self.persist_directory),
                is_persistent=True
            )
        )
        
        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Initialize collections for each context level
        self._initialize_collections()
        
        # Track current session
        self.current_session_id = self._generate_session_id()
        self.current_project = "default"
        
        logger.info(f"ChromaDB Context Manager initialized at {persist_directory}")
    
    def _initialize_collections(self):
        """Initialize or get existing collections for each context level"""
        self.collections = {}
        
        for level in ContextLevel:
            collection_name = f"context_{level.value}"
            try:
                self.collections[level] = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"level": level.value}
                )
                logger.info(f"Initialized collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error initializing collection {collection_name}: {e}")
                import traceback
                traceback.print_exc()
                self.collections[level] = None # Explicitly set to None on failure
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def add_context(self,
                   content: str,
                   level: ContextLevel,
                   metadata: Optional[Dict[str, Any]] = None,
                   project: Optional[str] = None) -> str:
        """
        Add context entry to the appropriate collection.
        
        Args:
            content: The context content
            level: Context hierarchy level
            metadata: Additional metadata
            project: Project identifier (uses current if not specified)
            
        Returns:
            Document ID
        """
        if metadata is None:
            metadata = {}
        
        # Add standard metadata
        metadata.update({
            "session_id": self.current_session_id,
            "project": project or self.current_project,
            "timestamp": datetime.now().isoformat(),
            "char_count": len(content),
            "word_count": len(content.split())
        })
        
        # Generate unique ID
        doc_id = hashlib.md5(
            f"{content}{metadata}".encode()
        ).hexdigest()
        
        # Add to collection
        collection = self.collections[level]
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        logger.info(f"Added context to {level.value} level: {doc_id[:8]}...")
        return doc_id
    
    def search_context(self,
                      query: str,
                      level: Optional[ContextLevel] = None,
                      n_results: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant context.
        
        Args:
            query: Search query
            level: Specific level to search (None = all levels)
            n_results: Number of results to return
            filters: Metadata filters (e.g., {"project": "trading_system"})
            
        Returns:
            List of search results with content and metadata
        """
        results = []
        
        # Determine which collections to search
        if level:
            search_collections = [(level, self.collections[level])]
        else:
            search_collections = list(self.collections.items())
        
        # Search each collection
        for ctx_level, collection in search_collections:
            try:
                # Build where clause for filtering
                where_clause = {}
                if filters:
                    where_clause.update(filters)
                
                # Always filter by current project unless explicitly searching global
                if ctx_level != ContextLevel.GLOBAL and "project" not in where_clause:
                    where_clause["project"] = self.current_project
                
                # Query collection
                query_results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause if where_clause else None
                )
                
                # Process results
                for i in range(len(query_results['ids'][0])):
                    results.append({
                        "id": query_results['ids'][0][i],
                        "content": query_results['documents'][0][i],
                        "metadata": query_results['metadatas'][0][i],
                        "distance": query_results['distances'][0][i],
                        "level": ctx_level.value
                    })
                    
            except Exception as e:
                logger.error(f"Error searching {ctx_level.value}: {e}")
        
        # Sort by distance (relevance)
        results.sort(key=lambda x: x['distance'])
        
        return results[:n_results]
    
    def get_session_context(self, 
                           session_id: Optional[str] = None,
                           max_items: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve all context for a specific session.
        
        Args:
            session_id: Session to retrieve (current if None)
            max_items: Maximum items to return
            
        Returns:
            List of context entries
        """
        session_id = session_id or self.current_session_id
        
        results = []
        for level, collection in self.collections.items():
            try:
                # Get all documents for this session
                session_results = collection.get(
                    where={"session_id": session_id},
                    limit=max_items
                )
                
                # Process results
                for i in range(len(session_results['ids'])):
                    results.append({
                        "id": session_results['ids'][i],
                        "content": session_results['documents'][i],
                        "metadata": session_results['metadatas'][i],
                        "level": level.value
                    })
                    
            except Exception as e:
                logger.error(f"Error retrieving session context: {e}")
        
        # Sort by timestamp
        results.sort(key=lambda x: x['metadata'].get('timestamp', ''))
        
        return results
    
    def clear_immediate_context(self):
        """Clear immediate conversation context"""
        collection = self.collections[ContextLevel.IMMEDIATE]
        
        # Get IDs of immediate context for current session
        results = collection.get(
            where={
                "session_id": self.current_session_id,
                "project": self.current_project
            }
        )
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Cleared {len(results['ids'])} immediate context entries")
    
    def promote_context(self, 
                       doc_id: str, 
                       from_level: ContextLevel, 
                       to_level: ContextLevel):
        """
        Promote context from one level to another.
        
        Args:
            doc_id: Document ID to promote
            from_level: Source level
            to_level: Target level
        """
        from_collection = self.collections[from_level]
        to_collection = self.collections[to_level]
        
        # Get the document
        doc = from_collection.get(ids=[doc_id])
        
        if not doc['ids']:
            raise ValueError(f"Document {doc_id} not found in {from_level.value}")
        
        # Add to new level with updated metadata
        metadata = doc['metadatas'][0].copy()
        metadata['promoted_from'] = from_level.value
        metadata['promoted_at'] = datetime.now().isoformat()
        
        to_collection.add(
            documents=doc['documents'],
            metadatas=[metadata],
            ids=[f"{doc_id}_promoted"]
        )
        
        logger.info(f"Promoted {doc_id} from {from_level.value} to {to_level.value}")
    
    def set_project(self, project_name: str):
        """Set current project context"""
        self.current_project = project_name
        logger.info(f"Set current project to: {project_name}")
    
    def new_session(self) -> str:
        """Start a new session"""
        self.current_session_id = self._generate_session_id()
        logger.info(f"Started new session: {self.current_session_id}")
        return self.current_session_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            "current_session": self.current_session_id,
            "current_project": self.current_project,
            "collections": {}
        }
        
        for level, collection in self.collections.items():
            count = collection.count()
            stats["collections"][level.value] = {
                "document_count": count,
                "name": collection.name
            }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize context manager
    cm = ChromaContextManager()
    
    # Add some context at different levels
    cm.add_context(
        "User wants to implement a trading system with real-time data feeds",
        ContextLevel.IMMEDIATE
    )
    
    cm.add_context(
        "The project uses Python with asyncio for handling concurrent operations",
        ContextLevel.SESSION,
        metadata={"category": "technical_choice"}
    )
    
    cm.add_context(
        "Always use type hints in Python code for better maintainability",
        ContextLevel.PROJECT,
        metadata={"type": "coding_standard"}
    )
    
    cm.add_context(
        "Security by design: validate all inputs, use parameterized queries, implement rate limiting",
        ContextLevel.GLOBAL,
        metadata={"type": "best_practice", "domain": "security"}
    )
    
    # Search for relevant context
    results = cm.search_context("security implementation", n_results=5)
    
    print("\nSearch Results:")
    for result in results:
        print(f"- [{result['level']}] {result['content'][:100]}...")
        print(f"  Distance: {result['distance']:.4f}")
    
    # Get statistics
    stats = cm.get_statistics()
    print(f"\nDatabase Statistics: {json.dumps(stats, indent=2)}")