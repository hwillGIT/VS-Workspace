const sqlite3 = require('sqlite3').verbose();
const neo4j = require('neo4j-driver');
const path = require('path');

// Test complete Cipher functionality with proper database structure
async function testCipherSystem() {
    console.log('=== Testing Cipher System ===\n');
    
    // 1. Test Neo4j Connection
    console.log('1. Testing Neo4j Connection...');
    const driver = neo4j.driver(
        'bolt://localhost:7687',
        neo4j.auth.basic('neo4j', 'architecture123')
    );
    
    let neo4jWorking = false;
    try {
        const session = driver.session();
        const result = await session.run('RETURN "Connected to Neo4j" as message');
        console.log('   ✓', result.records[0].get('message'));
        
        // Add a test node
        await session.run(
            `CREATE (n:CipherTest {
                id: $id,
                content: $content,
                type: 'integration_test',
                timestamp: datetime()
            })`,
            { 
                id: 'cipher-test-' + Date.now(),
                content: 'Cipher system integration test'
            }
        );
        console.log('   ✓ Created test node in knowledge graph');
        
        // Query test nodes
        const queryResult = await session.run(
            'MATCH (n:CipherTest) RETURN n.content as content, n.timestamp as timestamp ORDER BY n.timestamp DESC LIMIT 3'
        );
        
        console.log('   ✓ Recent test nodes:');
        queryResult.records.forEach(record => {
            const timestamp = record.get('timestamp');
            console.log(`     - ${record.get('content')} (${timestamp ? timestamp.toString() : 'no timestamp'})`);
        });
        
        await session.close();
        neo4jWorking = true;
        
    } catch (error) {
        console.log('   ✗ Neo4j error:', error.message);
    } finally {
        await driver.close();
    }
    
    // 2. Test SQLite Database (Cipher's actual structure)
    console.log('\n2. Testing SQLite Message Storage...');
    const dbPath = path.join(__dirname, 'data', 'cipher.db');
    const db = new sqlite3.Database(dbPath);
    
    let sqliteWorking = false;
    try {
        // Store a message in Cipher format
        const sessionId = 'test-session-' + Date.now();
        const messageData = JSON.stringify([
            {
                role: 'user',
                content: 'Test message for Cipher integration'
            },
            {
                role: 'assistant', 
                content: 'Neo4j connection test successful with architecture123 password'
            }
        ]);
        
        await new Promise((resolve, reject) => {
            db.run(
                `INSERT INTO store (key, value, created_at, updated_at) VALUES (?, ?, ?, ?)`,
                [`messages:${sessionId}`, messageData, Date.now(), Date.now()],
                function(err) {
                    if (err) reject(err);
                    else {
                        console.log('   ✓ Stored message session:', sessionId);
                        resolve();
                    }
                }
            );
        });
        
        // Retrieve recent messages
        await new Promise((resolve, reject) => {
            db.all(
                `SELECT key, created_at FROM store WHERE key LIKE 'messages:%' ORDER BY created_at DESC LIMIT 3`,
                [],
                (err, rows) => {
                    if (err) reject(err);
                    else {
                        console.log('   ✓ Recent message sessions:');
                        rows.forEach(row => {
                            const date = new Date(row.created_at).toISOString();
                            console.log(`     - ${row.key} (${date})`);
                        });
                        resolve();
                    }
                }
            );
        });
        
        sqliteWorking = true;
        
    } catch (error) {
        console.log('   ✗ SQLite error:', error.message);
    } finally {
        db.close();
    }
    
    // 3. Summary
    console.log('\n=== Test Results ===');
    console.log(`Neo4j Knowledge Graph: ${neo4jWorking ? '✓ WORKING' : '✗ FAILED'}`);
    console.log(`SQLite Message Store: ${sqliteWorking ? '✓ WORKING' : '✗ FAILED'}`);
    
    if (neo4jWorking && sqliteWorking) {
        console.log('\n🎉 Cipher system is fully operational!');
        console.log('   - Neo4j knowledge graph connected with architecture123');
        console.log('   - SQLite message storage working');
        console.log('   - Ready for AI agent memory operations');
    } else {
        console.log('\n❌ System has issues that need attention');
    }
    
    return { neo4j: neo4jWorking, sqlite: sqliteWorking };
}

testCipherSystem().catch(console.error);