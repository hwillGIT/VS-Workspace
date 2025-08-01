const neo4j = require('neo4j-driver');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

async function storeTargetConfigurations() {
    console.log('Storing Cipher and Neo4j configurations...');

    // Neo4j connection
    const driver = neo4j.driver(
        'bolt://localhost:7687',
        neo4j.auth.basic('neo4j', 'architecture123')
    );

    try {
        const session = driver.session({ database: 'neo4j' });

        // 1. Store Cipher Configuration
        console.log('1. Storing Cipher memory layer configuration...');
        const cipherSystemPrompt = `You are a MEMORY LAYER focused ONLY on these two tasks:

RETRIEVAL OPERATIONS:
- Primarily use cipher_search_memory to retrieve information, if user input contains reasoning steps, use cipher_search_reasoning_patterns
- Include comprehensive details of all retrieved information
- Organize information clearly with proper categorization

STORAGE OPERATIONS:
- Don\'t run any storage tool such as cipher_extract_and_operate_memory becaue these tools are run automatically in the background
- Respond as quickly as possible to optimize latency for external clients
- Confirm what will be stored in a concise manner or give a concise summary of the stored information`;

        await session.run(`
            MERGE (cipherConfig:Configuration {name: "Cipher_MemoryLayer_Config"})
            SET cipherConfig.type = "memory_layer_agent",
                cipherConfig.llm_provider = "openai",
                cipherConfig.llm_model = "gpt-4o-mini",
                cipherConfig.system_prompt = $prompt,
                cipherConfig.source_file = "D:\\VS Workspace\\cipher-byterover\\cipher\\examples\\03-strict-memory-layer\\cipher.yml",
                cipherConfig.updated = datetime()
        `, { prompt: cipherSystemPrompt });
        console.log('Stored Cipher configuration in Neo4j.');

        // 2. Store Neo4j Connection Configuration
        console.log('2. Storing Neo4j connection details...');
        await session.run(`
            MERGE (neo4jConfig:Configuration {name: "Neo4j_Connection_Config"})
            SET neo4jConfig.type = "graph_database_connection",
                neo4jConfig.uri = "bolt://localhost:7687",
                neo4jConfig.username = "neo4j",
                neo4jConfig.password = "architecture123",
                neo4jConfig.database = "neo4j",
                neo4jConfig.source_file = "D:\\VS Workspace\\ClaudeCode\\architecture_intelligence\\core\\neo4j_knowledge_graph.py",
                neo4jConfig.updated = datetime()
        `);
        console.log('Stored Neo4j configuration in Neo4j.');

        // 3. Create relationship
        console.log('3. Linking configurations...');
        await session.run(`
            MATCH (cipher:System {name: "Cipher"})
            MATCH (cipherConfig:Configuration {name: "Cipher_MemoryLayer_Config"})
            MATCH (neo4jDB:Database {name: "Neo4j"})
            MATCH (neo4jConfig:Configuration {name: "Neo4j_Connection_Config"})
            MERGE (cipher)-[:HAS_CONFIGURATION]->(cipherConfig)
            MERGE (neo4jDB)-[:HAS_CONFIGURATION]->(neo4jConfig)
        `);
        console.log('Created relationships in Neo4j.');


        await session.close();
        console.log('\n✅ Configurations stored successfully in Neo4j.');

        // Store summary in SQLite
        const dbPath = path.join(__dirname, 'data', 'cipher.db');
        const db = new sqlite3.Database(dbPath, (err) => {
            if (err) {
                console.error('Error opening SQLite database:', err.message);
                return;
            }
            console.log('Connected to the SQLite database.');
        });

        const summaryMessage = JSON.stringify([
            {
                role: 'system',
                content: 'Configuration Knowledge Stored',
                timestamp: new Date().toISOString()
            },
            {
                role: 'assistant',
                content: `Stored new configuration details in the knowledge graph:
                
• Cipher Config: Stored memory layer agent config (LLM, system prompt) from cipher.yml.
• Neo4j Config: Stored connection details (URI, user, pass) from neo4j_knowledge_graph.py.`
            }
        ]);

        await new Promise((resolve, reject) => {
            db.run(
                `INSERT INTO store (key, value, created_at, updated_at) VALUES (?, ?, ?, ?) `,
                [`config-knowledge-${Date.now()}`, summaryMessage, Date.now(), Date.now()],
                function(err) {
                    if (err) {
                        console.error('SQLite insert error:', err.message);
                        reject(err);
                    } else {
                        console.log('A new row has been inserted with rowid', this.lastID);
                        resolve();
                    }
                }
            );
        });

        db.close((err) => {
            if (err) {
                console.error('Error closing SQLite database:', err.message);
            } else {
                console.log('Closed the SQLite database connection.');
            }
        });
        console.log('✅ Configuration summary also stored in SQLite.');

    } catch (error) {
        console.error('Error storing configurations:', error.message);
    } finally {
        await driver.close();
    }
}

storeTargetConfigurations().catch(console.error);