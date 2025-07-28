# Security Practices for Development

## 🚨 CRITICAL: Environment Files

**NEVER commit `.env` files to version control.**

### Always Include in .gitignore:
```gitignore
# Environment variables (CRITICAL - contains API keys and secrets)
.env
.env.*
!.env.example
!.env.template
.env.local
.env.development
.env.staging
.env.production

# API Keys and credentials
*.key
*.pem
secrets.json
config/secrets.*
```

### Instead Use:
- `.env.example` - Template with placeholder values
- `.env.template` - Template with descriptions
- Environment variable documentation in README

### If Accidentally Committed:
1. **Immediately revoke/regenerate all API keys**
2. Remove from git tracking: `git rm --cached .env`
3. Add to .gitignore if not already there
4. Consider BFG Repo-Cleaner to remove from history

## API Key Management

### Anthropic API Keys:
- Format: `sk-ant-api03-*`
- Revoke at: https://console.anthropic.com/settings/keys
- Store in environment variables, never hardcode

### Other Services:
- **Perplexity**: Revoke at dashboard
- **Google Cloud**: Revoke at console.cloud.google.com
- **OpenAI**: Revoke at platform.openai.com

## Best Practices:
1. Use environment variables for all secrets
2. Keep .env.example updated with required variables
3. Use least-privilege API keys when possible
4. Rotate API keys regularly
5. Monitor for leaked credentials using tools like git-secrets

---

*This document should be reviewed by all team members before contributing.*