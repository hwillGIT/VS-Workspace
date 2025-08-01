const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const dbPath = path.join(__dirname, 'data', 'cipher.db');

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Error opening database:', err.message);
    return;
  }
  console.log('Connected to the SQLite database.');
});

// Get all table names
db.all("SELECT name FROM sqlite_master WHERE type='table'", [], (err, tables) => {
  if (err) {
    console.error('Error getting tables:', err.message);
    return;
  }
  
  console.log('\n=== DATABASE TABLES ===');
  tables.forEach(table => {
    console.log(`- ${table.name}`);
  });
  
  // For each table, show schema and contents
  let tableCount = 0;
  tables.forEach(table => {
    const tableName = table.name;
    
    // Get table schema
    db.all(`PRAGMA table_info(${tableName})`, [], (err, columns) => {
      if (err) {
        console.error(`Error getting schema for ${tableName}:`, err.message);
        return;
      }
      
      console.log(`\n=== TABLE: ${tableName} ===`);
      console.log('Schema:');
      columns.forEach(col => {
        console.log(`  ${col.name}: ${col.type} ${col.pk ? '(PRIMARY KEY)' : ''}`);
      });
      
      // Get row count
      db.get(`SELECT COUNT(*) as count FROM ${tableName}`, [], (err, row) => {
        if (err) {
          console.error(`Error counting rows in ${tableName}:`, err.message);
          return;
        }
        
        console.log(`Rows: ${row.count}`);
        
        // Show some sample data if rows exist
        if (row.count > 0) {
          db.all(`SELECT * FROM ${tableName} LIMIT 5`, [], (err, rows) => {
            if (err) {
              console.error(`Error getting data from ${tableName}:`, err.message);
              return;
            }
            
            console.log('Sample data:');
            rows.forEach((row, index) => {
              console.log(`  Row ${index + 1}:`, JSON.stringify(row, null, 2));
            });
            
            tableCount++;
            if (tableCount === tables.length) {
              db.close();
            }
          });
        } else {
          tableCount++;
          if (tableCount === tables.length) {
            db.close();
          }
        }
      });
    });
  });
});