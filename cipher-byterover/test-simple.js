// Simple test to bypass the complex Cipher setup
console.log("Testing simple memory functionality...");

// Just test if we can store and retrieve basic data
const memories = [
  {
    id: 1,
    text: "ChromaDB runs on localhost:8000 for trading system",
    tags: ["database", "infrastructure"]
  }
];

console.log("Stored memories:", memories);

function searchMemory(query) {
  return memories.filter(m => 
    m.text.toLowerCase().includes(query.toLowerCase()) ||
    m.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
  );
}

console.log("Search for 'ChromaDB':", searchMemory("ChromaDB"));
console.log("Search for 'localhost':", searchMemory("localhost"));
console.log("Search for 'database':", searchMemory("database"));