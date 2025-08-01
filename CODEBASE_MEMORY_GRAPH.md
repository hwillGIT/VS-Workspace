# Codebase Memory Graph System

## Overview

A living knowledge system that captures, relates, and surfaces development insights from every commit, test, and rollback.

## Architecture

```
Git Hooks → Event Capture → Knowledge Extraction → Graph Storage → Intelligent Retrieval
    ↓            ↓                   ↓                    ↓                ↓
Pre-commit   Commit Meta      Pattern Mining      Neo4j + ChromaDB    AI Assistant
Post-commit  Test Results      Decision Capture    Embeddings         Developer IDE
```

## Core Components

### 1. Event Capture Layer

```python
# git_hooks/post-commit
#!/usr/bin/env python
import subprocess
import json
from datetime import datetime
from codebase_memory import MemoryCapture

def capture_commit_context():
    commit_data = {
        'sha': subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip(),
        'message': subprocess.check_output(['git', 'log', '-1', '--pretty=%B']).strip(),
        'author': subprocess.check_output(['git', 'log', '-1', '--pretty=%ae']).strip(),
        'timestamp': datetime.now().isoformat(),
        'diff_stats': subprocess.check_output(['git', 'diff', '--stat', 'HEAD~1']).strip(),
        'test_results': capture_test_results(),
        'lint_results': capture_lint_results()
    }
    
    # Extract learning if tagged
    if '[LEARNING]' in commit_data['message']:
        extract_and_store_learning(commit_data)
    
    # Detect patterns
    patterns = detect_commit_patterns(commit_data)
    
    # Store in graph
    MemoryCapture().store(commit_data, patterns)
```

### 2. Knowledge Extraction

```python
class KnowledgeExtractor:
    def extract_decision(self, commit_data):
        """Extract architectural decisions from commits"""
        decision_patterns = [
            r'(?i)decided to use (\w+) because (.*)',
            r'(?i)switched from (\w+) to (\w+)',
            r'(?i)chose (\w+) over (\w+)'
        ]
        
        decisions = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, commit_data['message'])
            if matches:
                decisions.append({
                    'type': 'architectural_decision',
                    'choice': matches[0][0],
                    'rationale': matches[0][1],
                    'commit': commit_data['sha']
                })
        return decisions
    
    def extract_gotcha(self, test_failure):
        """Extract edge cases from test failures"""
        return {
            'type': 'gotcha',
            'test_name': test_failure['name'],
            'failure_reason': test_failure['message'],
            'fix_commit': test_failure['fix_sha'],
            'prevention': self.suggest_prevention(test_failure)
        }
```

### 3. Graph Schema (Neo4j)

```cypher
// Nodes
(:Commit {sha, message, author, timestamp})
(:Test {name, file, type})
(:Decision {type, choice, rationale})
(:Learning {content, category, impact})
(:Pattern {name, description, frequency})
(:Rollback {from_sha, to_sha, reason})

// Relationships
(Commit)-[:CONTAINS_TEST]->(Test)
(Commit)-[:MADE_DECISION]->(Decision)
(Commit)-[:DISCOVERED]->(Learning)
(Commit)-[:EXHIBITS]->(Pattern)
(Rollback)-[:REVERTED]->(Commit)
(Learning)-[:PREVENTS]->(Rollback)
(Pattern)-[:SUGGESTS]->(Decision)
```

### 4. Embedding Layer (ChromaDB)

```python
class CodebaseEmbeddings:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("codebase_memory")
    
    def embed_learning(self, learning):
        """Store searchable learning with context"""
        self.collection.add(
            documents=[learning['content']],
            metadatas=[{
                'type': 'learning',
                'category': learning['category'],
                'commit': learning['commit_sha'],
                'impact': learning['impact'],
                'tags': learning['tags']
            }],
            ids=[f"learning_{learning['id']}"]
        )
    
    def search_similar_issues(self, error_message, k=5):
        """Find similar past issues and their solutions"""
        results = self.collection.query(
            query_texts=[error_message],
            n_results=k,
            where={"type": "gotcha"}
        )
        return results
```

### 5. Intelligent Retrieval

```python
class SmartIterativeAssistant:
    def suggest_next_steps(self, current_diff):
        """Suggest next micro-iteration based on patterns"""
        # Find similar past changes
        similar_changes = self.graph.find_similar_changes(current_diff)
        
        # Extract successful patterns
        patterns = []
        for change in similar_changes:
            next_commits = self.graph.get_following_commits(change)
            if self.were_successful(next_commits):
                patterns.append(self.extract_pattern(change, next_commits))
        
        return self.rank_suggestions(patterns)
    
    def warn_about_risks(self, proposed_change):
        """Warn about potential issues based on history"""
        # Search for similar changes that were rolled back
        risky_patterns = self.graph.query("""
            MATCH (c:Commit)-[:EXHIBITS]->(p:Pattern)
            WHERE p.name = $pattern_name
            AND EXISTS((c)<-[:REVERTED]-())
            RETURN c, p
        """, pattern_name=self.detect_pattern(proposed_change))
        
        warnings = []
        for pattern in risky_patterns:
            warnings.append({
                'pattern': pattern['p']['name'],
                'risk': pattern['c']['rollback_reason'],
                'prevention': self.get_prevention_strategy(pattern)
            })
        return warnings
```

## Usage Examples

### 1. Pre-commit Intelligence

```bash
$ git commit -m "refactor: extract payment validation"

🤖 Smart Iterative Assistant:
- Similar refactoring in auth module led to edge case (commit: abc123)
- Suggested test cases:
  * Empty payment method
  * Unicode characters in cardholder name
  * Concurrent validation calls
- Pattern match: "Extract validation" → Usually needs integration test update

Continue with commit? [y/n/e(dit)]
```

### 2. Learning Capture

```bash
$ git commit -m "fix: handle timezone edge case [LEARNING]

The DateTime parser was failing for timezones with fractional offsets
(e.g., India +5:30). Fixed by using pytz instead of manual offset calculation.

This affects all date inputs from external APIs."

✨ Learning captured and embedded!
- Category: datetime-handling
- Impact: high
- Linked to: 3 similar past issues
- Suggested pattern: "Always use pytz for timezone handling"
```

### 3. Rollback Intelligence

```bash
$ git revert HEAD

🔄 Rollback Analysis:
- Reverting: "feat: add async payment processing"
- Reason detected: Test failures in integration suite
- Similar rollbacks: 2 in payment module this quarter
- Root cause pattern: Missing timeout handling in async flows
- Suggested fix: Add timeout parameter to async decorator

Document this rollback? [y/n]
```

### 4. IDE Integration

```typescript
// VS Code: As you type...
function validateEmail(email: string) {
    // 💡 Codebase Memory: 
    // - Previous email validation had unicode domain issues (commit: def456)
    // - Team decided to use email-validator library after 3 custom attempts
    // - See: /docs/decisions/ADR-004-email-validation.md
    
    return emailRegex.test(email);
    //     ^^^^^^^^^^
    // ⚠️  Warning: Custom regex failed 5 edge cases in past
    // 💡 Suggestion: Use `email-validator` library (team standard)
}
```

## Query Examples

### Find Successful Patterns

```cypher
MATCH (c1:Commit)-[:EXHIBITS]->(p:Pattern)
WHERE NOT EXISTS((c1)<-[:REVERTED]-())
AND p.frequency > 5
RETURN p.name, p.description, COUNT(c1) as success_count
ORDER BY success_count DESC
```

### Trace Decision Evolution

```cypher
MATCH path = (d1:Decision)-[:INFLUENCED_BY*]->(d2:Decision)
WHERE d1.choice = 'Redis'
RETURN path
```

### Find Learning Gaps

```cypher
MATCH (r:Rollback)
WHERE NOT EXISTS((r)-[:PREVENTED_BY]->(:Learning))
RETURN r.reason, COUNT(r) as frequency
ORDER BY frequency DESC
```

## Implementation Phases

### Phase 1: Basic Capture (Week 1)
- Git hooks for commit/rollback events
- Simple file-based storage
- Manual learning tagging

### Phase 2: Graph Integration (Week 2-3)
- Neo4j schema setup
- Automated pattern detection
- Basic query interface

### Phase 3: Embeddings (Week 4)
- ChromaDB integration
- Semantic search capability
- Similar issue detection

### Phase 4: Intelligence (Week 5-6)
- IDE plugin development
- Real-time suggestions
- Rollback prediction

## Success Metrics

- **Learning Capture Rate**: Learnings documented per week
- **Pattern Reuse**: How often patterns inform new code
- **Rollback Prevention**: Reduction in similar rollbacks
- **Query Usage**: Developer queries per day
- **Time to Solution**: How fast developers find relevant past solutions

---

*The codebase memory graph transforms every commit into institutional knowledge. What we learn together, we remember forever.*