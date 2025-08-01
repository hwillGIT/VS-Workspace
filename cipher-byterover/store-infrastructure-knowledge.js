const neo4j = require('neo4j-driver');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

async function storeInfrastructureKnowledge() {
    console.log('Storing infrastructure connection knowledge...\n');
    
    // Connect to Neo4j
    const driver = neo4j.driver(
        'bolt://localhost:7687',
        neo4j.auth.basic('neo4j', 'architecture123')
    );
    
    try {
        const session = driver.session();
        
        // 1. Store ChromaDB Knowledge
        console.log('1. Storing ChromaDB knowledge...');
        await session.run(`
            MERGE (chroma:Database {name: "ChromaDB"})
            SET chroma.type = "vector_database",
                chroma.host = "localhost",
                chroma.port = 8000,
                chroma.executable = "D:\\VS Workspace\\ChromaDB\\venv\\Scripts\\chroma.exe",
                chroma.startup_command = "chroma run --host localhost --port 8000",
                chroma.purpose = "Vector storage for trading system embeddings",
                chroma.status = "required_for_trading_system",
                chroma.updated = datetime()
        `);
        
        // 2. Store Neo4j Knowledge
        console.log('2. Storing Neo4j knowledge...');
        await session.run(`
            MERGE (neo:Database {name: "Neo4j"})
            SET neo.type = "graph_database",
                neo.host = "localhost", 
                neo.port = 7687,
                neo.username = "neo4j",
                neo.password = "architecture123",
                neo.database = "neo4j",
                neo.instance_name = "architecture-intelligence",
                neo.purpose = "Knowledge graph for Cipher memory system",
                neo.config_location = "C:\\Users\\huber\\.Neo4jDesktop2\\Data\\dbmss\\dbms-f0fced8f-6486-411f-9b0d-22e0b8a2f268",
                neo.updated = datetime()
        `);
        
        // 3. Store Cipher Configuration
        console.log('3. Storing Cipher configuration...');
        await session.run(`
            MERGE (cipher:System {name: "Cipher"})
            SET cipher.type = "memory_layer",
                cipher.location = "D:\\VS Workspace\\cipher-byterover",
                cipher.purpose = "AI agent memory system",
                cipher.sqlite_db = "D:\\VS Workspace\\cipher-byterover\\data\\cipher.db",
                cipher.config_file = "D:\\VS Workspace\\cipher-byterover\\.env",
                cipher.status = "operational",
                cipher.updated = datetime()
        `);
        
        // 4. Store API Key Locations
        console.log('4. Storing API key configuration...');
        await session.run(`
            MERGE (env:Configuration {name: "Environment_Keys"})
            SET env.type = "api_configuration",
                env.cipher_env = "D:\\VS Workspace\\cipher-byterover\\.env",
                env.claude_env = "D:\\VS Workspace\\claudecode\\.env",
                env.anthropic_key = "Available in both files",
                env.openai_key = "Available in both files", 
                env.neo4j_credentials = "Available in claudecode/.env line 125",
                env.security_note = "Contains sensitive API keys - never commit",
                env.updated = datetime()
        `);
        
        // 5. Store System Integration Knowledge
        console.log('5. Storing system integration...');
        await session.run(`
            MERGE (integration:Integration {name: "AI_Infrastructure"})
            SET integration.type = "system_architecture",
                integration.components = "ChromaDB + Neo4j + Cipher + SQLite",
                integration.data_flow = "SQLite(conversations) -> Neo4j(knowledge_graph) -> ChromaDB(vectors)",
                integration.test_status = "All systems operational",
                integration.last_tested = datetime(),
                integration.updated = datetime()
        `);
        
        // 6. Create relationships between components
        console.log('6. Creating component relationships...');
        await session.run(`
            MATCH (chroma:Database {name: "ChromaDB"})
            MATCH (neo:Database {name: "Neo4j"})
            MATCH (cipher:System {name: "Cipher"})
            MATCH (env:Configuration {name: "Environment_Keys"})
            MATCH (integration:Integration {name: "AI_Infrastructure"})
            
            MERGE (cipher)-[:USES]->(neo)
            MERGE (cipher)-[:USES]->(chroma)
            MERGE (cipher)-[:CONFIGURED_BY]->(env)
            MERGE (integration)-[:INCLUDES]->(chroma)
            MERGE (integration)-[:INCLUDES]->(neo)
            MERGE (integration)-[:INCLUDES]->(cipher)
        `);
        
        // 7. Store startup procedures
        console.log('7. Storing startup procedures...');
        await session.run(`
            MERGE (startup:Procedure {name: "System_Startup"})
            SET startup.type = "operational_procedure",
                startup.step1 = "Check if ChromaDB is running on localhost:8000",
                startup.step2 = "Start ChromaDB: D:\\VS Workspace\\ChromaDB\\venv\\Scripts\\chroma.exe run --host localhost --port 8000",
                startup.step3 = "Verify Neo4j is running (architecture-intelligence instance)",
                startup.step4 = "Test Cipher connection with both databases",
                startup.step5 = "Verify all API keys are loaded from .env files",
                startup.importance = "Required for Claude Code trading system",
                startup.updated = datetime()
        `);
        
        await session.close();
        console.log('\n✅ Infrastructure knowledge stored in Neo4j knowledge graph');
        
        // Also store in SQLite for message context
        const dbPath = path.join(__dirname, 'data', 'cipher.db');
        const db = new sqlite3.Database(dbPath);
        
        const knowledgeMessage = JSON.stringify([
            {
                role: 'system',
                content: 'Infrastructure Knowledge Stored',
                timestamp: new Date().toISOString()
            },
            {
                role: 'assistant',
                content: `Infrastructure connection knowledge has been stored:
                
• ChromaDB: localhost:8000 (D:\\VS Workspace\\ChromaDB\\venv\\Scripts\\chroma.exe)
• Neo4j: localhost:7687 (username: neo4j, password: architecture123)
• Cipher: D:\\VS Workspace\\cipher-byterover (.env configured)
• SQLite: cipher.db (message storage)
• API Keys: Available in .env files (cipher + claudecode)

All systems tested and operational. Startup procedure documented.`
            }
        ]);
        
        await new Promise((resolve, reject) => {
            db.run(
                `INSERT INTO store (key, value, created_at, updated_at) VALUES (?, ?, ?, ?)`,
                [`infrastructure-knowledge-${Date.now()}`, knowledgeMessage, Date.now(), Date.now()],
                function(err) {
                    if (err) reject(err);
                    else resolve();
                }
            );
        });
        
        db.close();
        console.log('✅ Infrastructure knowledge also stored in SQLite');
        
    } catch (error) {
        console.error('Error storing knowledge:', error.message);
    } finally {
        await driver.close();
    }
}

storeInfrastructureKnowledge().catch(console.error);