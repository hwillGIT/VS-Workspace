const sqlite3 = require('sqlite3').verbose();
const neo4j = require('neo4j-driver');
const path = require('path');

// Test Neo4j connection with correct credentials
async function testNeo4j() {
    console.log('Testing Neo4j connection...');
    
    const driver = neo4j.driver(
        'bolt://localhost:7687',
        neo4j.auth.basic('neo4j', 'architecture123')
    );
    
    try {
        const session = driver.session();
        
        // Test connection
        const result = await session.run('RETURN "Connection successful" as message');
        console.log('✓ Neo4j connection:', result.records[0].get('message'));
        
        // Test creating a simple node
        await session.run(
            'MERGE (n:TestNode {id: $id, content: $content, timestamp: $timestamp})',
            { 
                id: 'test-' + Date.now(), 
                content: 'Cipher Neo4j integration test',
                timestamp: new Date().toISOString()
            }
        );
        console.log('✓ Created test node in Neo4j');
        
        // Query test nodes
        const queryResult = await session.run(
            'MATCH (n:TestNode) RETURN n.content as content, n.timestamp as timestamp LIMIT 5'
        );
        
        console.log('✓ Test nodes in Neo4j:');
        queryResult.records.forEach(record => {
            console.log(`  - ${record.get('content')} (${record.get('timestamp')})`);
        });
        
        await session.close();
        
    } catch (error) {
        console.error('✗ Neo4j error:', error.message);
        return false;
    } finally {
        await driver.close();
    }
    
    return true;
}

// Test Cipher memory storage
async function testCipherMemory() {
    console.log('\nTesting Cipher memory storage...');
    
    const dbPath = path.join(__dirname, 'data', 'cipher.db');
    const db = new sqlite3.Database(dbPath);
    
    return new Promise((resolve, reject) => {
        // Store a new memory
        const testMemory = {
            text: `Neo4j integration test - ${new Date().toISOString()}`,
            tags: JSON.stringify(['neo4j', 'integration', 'test']),
            embedding: JSON.stringify(Array(1536).fill(0).map(() => Math.random()))
        };
        
        db.run(
            `INSERT INTO memories (text, tags, embedding) VALUES (?, ?, ?)`,
            [testMemory.text, testMemory.tags, testMemory.embedding],
            function(err) {
                if (err) {
                    console.error('✗ Memory storage error:', err.message);
                    reject(err);
                    return;
                }
                
                console.log('✓ Stored memory with ID:', this.lastID);
                
                // Retrieve memories
                db.all(
                    `SELECT id, text, tags FROM memories WHERE text LIKE '%Neo4j%' ORDER BY id DESC LIMIT 3`,
                    [],
                    (err, rows) => {
                        if (err) {
                            console.error('✗ Memory retrieval error:', err.message);
                            reject(err);
                            return;
                        }
                        
                        console.log('✓ Retrieved Neo4j-related memories:');
                        rows.forEach(row => {
                            console.log(`  - ID ${row.id}: ${row.text}`);
                            console.log(`    Tags: ${row.tags}`);
                        });
                        
                        db.close();
                        resolve(true);
                    }
                );
            }
        );
    });
}

// Run tests
async function runTests() {
    console.log('=== Cipher Integration Tests ===\n');
    
    try {
        const neo4jSuccess = await testNeo4j();
        const memorySuccess = await testCipherMemory();
        
        console.log('\n=== Test Results ===');
        console.log(`Neo4j Connection: ${neo4jSuccess ? '✓ PASS' : '✗ FAIL'}`);
        console.log(`Memory Storage: ${memorySuccess ? '✓ PASS' : '✗ FAIL'}`);
        
        if (neo4jSuccess && memorySuccess) {
            console.log('\n🎉 All tests passed! Cipher is ready with Neo4j integration.');
        } else {
            console.log('\n❌ Some tests failed. Check configuration.');
        }
        
    } catch (error) {
        console.error('Test execution error:', error.message);
    }
}

runTests();