const neo4j = require('neo4j-driver');

async function retrieveTargetConfigurations() {
    console.log('Retrieving Cipher and Neo4j configurations...');

    const driver = neo4j.driver(
        'bolt://localhost:7687',
        neo4j.auth.basic('neo4j', 'architecture123')
    );

    try {
        const session = driver.session({ database: 'neo4j' });

        console.log('\n--- Retrieving Cipher Configuration ---');
        const cipherResult = await session.run(
            'MATCH (c:Configuration {name: "Cipher_MemoryLayer_Config"}) RETURN c'
        );

        if (cipherResult.records.length > 0) {
            const cipherConfig = cipherResult.records[0].get('c').properties;
            console.log(JSON.stringify(cipherConfig, null, 2));
        } else {
            console.log('Cipher configuration not found.');
        }

        console.log('\n--- Retrieving Neo4j Configuration ---');
        const neo4jResult = await session.run(
            'MATCH (n:Configuration {name: "Neo4j_Connection_Config"}) RETURN n'
        );

        if (neo4jResult.records.length > 0) {
            const neo4jConfig = neo4jResult.records[0].get('n').properties;
            console.log(JSON.stringify(neo4jConfig, null, 2));
        } else {
            console.log('Neo4j configuration not found.');
        }

        await session.close();

    } catch (error) {
        console.error('Error retrieving configurations:', error.message);
    } finally {
        await driver.close();
    }
}

retrieveTargetConfigurations().catch(console.error);
