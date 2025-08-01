"""
Save Workspace State to Neo4j Knowledge Graph
Stores relationships between roles, principles, and modules
"""

from neo4j import GraphDatabase
from datetime import datetime
import json

class WorkspaceGraphManager:
    def __init__(self):
        # Neo4j connection details
        self.uri = "bolt://localhost:7687"
        self.username = "neo4j"
        self.password = "architecture123"
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            print("Connected to Neo4j successfully")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("\nMake sure Neo4j Desktop is running with:")
            print("- Database started")
            print("- Password set to 'architecture123'")
            raise
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def save_workspace_state(self):
        """Save current workspace configuration to graph"""
        
        with self.driver.session() as session:
            # Clear existing workspace data (optional)
            # session.run("MATCH (n:Workspace) DETACH DELETE n")
            
            # 1. Create Workspace node
            session.run("""
                MERGE (w:Workspace {name: 'claude_code_workspace'})
                SET w.updated_at = datetime(),
                    w.description = 'AI-enhanced development workspace with standardized modules'
            """)
            
            # 2. Create Role nodes
            roles = [
                {
                    "name": "Claude Code",
                    "focus": "Software Engineering",
                    "expertise": "Programming, architecture, testing, code review"
                },
                {
                    "name": "Claude Analyst",
                    "focus": "Data & Business Analysis",
                    "expertise": "Data analysis, business insights, reporting"
                },
                {
                    "name": "Claude Security",
                    "focus": "Security Architecture",
                    "expertise": "Security analysis, threat modeling, compliance"
                },
                {
                    "name": "Claude Teach",
                    "focus": "Educational Assistant",
                    "expertise": "Teaching, explanations, learning support"
                },
                {
                    "name": "Claude UI Designer",
                    "focus": "UI/UX Design",
                    "expertise": "User interface design, design systems, accessibility"
                }
            ]
            
            for role in roles:
                session.run("""
                    MERGE (r:Role {name: $name})
                    SET r.focus = $focus,
                        r.expertise = $expertise,
                        r.created_at = datetime()
                    WITH r
                    MATCH (w:Workspace {name: 'claude_code_workspace'})
                    MERGE (w)-[:HAS_ROLE]->(r)
                """, **role)
            
            # 3. Create Core Principles
            principles = [
                {
                    "name": "Plan Before Code",
                    "type": "workflow",
                    "description": "Always plan before implementation"
                },
                {
                    "name": "Automated Quality",
                    "type": "quality",
                    "description": "Tests and linters run automatically"
                },
                {
                    "name": "Security First",
                    "type": "security",
                    "description": "Never commit secrets, use sandboxed environments"
                },
                {
                    "name": "Standardized Modules Only",
                    "type": "architecture",
                    "description": "Use ONLY pre-built standardized modules from library"
                }
            ]
            
            for principle in principles:
                session.run("""
                    MERGE (p:Principle {name: $name})
                    SET p.type = $type,
                        p.description = $description,
                        p.created_at = datetime()
                    WITH p
                    MATCH (w:Workspace {name: 'claude_code_workspace'})
                    MERGE (w)-[:FOLLOWS_PRINCIPLE]->(p)
                """, **principle)
            
            # 4. Create Design Patterns
            design_patterns = [
                {
                    "name": "Gestalt Principles",
                    "category": "UI Design",
                    "elements": ["Proximity", "Similarity", "Continuation", "Closure", "Figure/Ground", "Common Fate"]
                },
                {
                    "name": "Modular Component Architecture",
                    "category": "System Design",
                    "elements": ["Atomic Modules", "Molecular Modules", "Organism Modules"]
                },
                {
                    "name": "Plan-Code-Review Loop",
                    "category": "Development Process",
                    "elements": ["Plan First", "Implement", "Test", "Review"]
                }
            ]
            
            for pattern in design_patterns:
                session.run("""
                    MERGE (dp:DesignPattern {name: $name})
                    SET dp.category = $category,
                        dp.elements = $elements,
                        dp.created_at = datetime()
                    WITH dp
                    MATCH (w:Workspace {name: 'claude_code_workspace'})
                    MERGE (w)-[:IMPLEMENTS_PATTERN]->(dp)
                """, **pattern)
            
            # 5. Create Module Types
            module_types = [
                {
                    "name": "Button Module",
                    "level": "Atomic",
                    "configurable": ["variant", "size", "theme", "effects"],
                    "immutable": ["behavior", "structure", "interaction"]
                },
                {
                    "name": "InputGroup Module", 
                    "level": "Molecular",
                    "configurable": ["type", "layout", "theme"],
                    "immutable": ["composition", "validation", "error handling"]
                },
                {
                    "name": "Form Module",
                    "level": "Organism", 
                    "configurable": ["template", "theme"],
                    "immutable": ["field arrangement", "submission logic"]
                }
            ]
            
            for module in module_types:
                session.run("""
                    MERGE (m:Module {name: $name})
                    SET m.level = $level,
                        m.configurable = $configurable,
                        m.immutable = $immutable,
                        m.created_at = datetime()
                    WITH m
                    MATCH (dp:DesignPattern {name: 'Modular Component Architecture'})
                    MERGE (dp)-[:CONTAINS_MODULE]->(m)
                """, **module)
            
            # 6. Create relationships between roles and principles
            role_principle_relationships = [
                ("Claude Code", "Plan Before Code"),
                ("Claude Code", "Automated Quality"),
                ("Claude UI Designer", "Standardized Modules Only"),
                ("Claude UI Designer", "Gestalt Principles"),
                ("Claude Security", "Security First")
            ]
            
            for role_name, principle_name in role_principle_relationships:
                session.run("""
                    MATCH (r:Role {name: $role_name})
                    MATCH (p:Principle {name: $principle_name})
                    MERGE (r)-[:APPLIES]->(p)
                """, role_name=role_name, principle_name=principle_name)
            
            # 7. Create AI Directives
            directives = [
                "Follow explicit plan-code-review loop",
                "Comment on changes noting defects",
                "Respect permission settings",
                "Update knowledge before proposing",
                "Be explicit and ask questions",
                "Refuse unsafe actions"
            ]
            
            for directive in directives:
                session.run("""
                    MERGE (d:Directive {text: $text})
                    SET d.created_at = datetime()
                    WITH d
                    MATCH (r:Role {name: 'Claude Code'})
                    MERGE (r)-[:MUST_FOLLOW]->(d)
                """, text=directive)
            
            print("Workspace state saved to Neo4j successfully!")
    
    def query_workspace(self):
        """Run some sample queries to verify the data"""
        
        with self.driver.session() as session:
            # Count nodes
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)
            
            print("\nNode counts:")
            for record in result:
                print(f"  {record['label']}: {record['count']}")
            
            # Show role relationships
            result = session.run("""
                MATCH (r:Role)-[:APPLIES]->(p:Principle)
                RETURN r.name as role, collect(p.name) as principles
                ORDER BY role
            """)
            
            print("\nRole-Principle relationships:")
            for record in result:
                print(f"  {record['role']}: {', '.join(record['principles'])}")
            
            # Show module hierarchy
            result = session.run("""
                MATCH (m:Module)
                RETURN m.level as level, collect(m.name) as modules
                ORDER BY level
            """)
            
            print("\nModule hierarchy:")
            for record in result:
                print(f"  {record['level']}: {', '.join(record['modules'])}")

def main():
    print("Saving Workspace State to Neo4j")
    print("=" * 40)
    
    try:
        manager = WorkspaceGraphManager()
        manager.save_workspace_state()
        manager.query_workspace()
        manager.close()
        
        print("\n" + "=" * 40)
        print("Success! Workspace relationships stored in Neo4j")
        print("\nYou can explore the graph in Neo4j Browser with queries like:")
        print("  MATCH (n) RETURN n LIMIT 50")
        print("  MATCH path = (w:Workspace)-[*]-(n) RETURN path")
        
    except Exception as e:
        print(f"\nError: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()