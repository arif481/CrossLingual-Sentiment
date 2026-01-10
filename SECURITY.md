# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email your findings to: [your-email@example.com]
3. Include as much information as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability within 7 days
- **Resolution**: We aim to resolve critical issues within 30 days
- **Disclosure**: We will coordinate with you on public disclosure

### Scope

This security policy applies to:
- Source code in this repository
- Configuration files
- Scripts and automation

### Out of Scope

The following are NOT considered security vulnerabilities:
- Bugs in dependencies (report to upstream)
- Denial of service attacks requiring unusual circumstances
- Social engineering attacks

## Security Best Practices

When using this project, please follow these security practices:

### API Tokens and Credentials

```bash
# NEVER commit tokens to the repository
# Use environment variables
export HF_TOKEN=your_token_here

# Or use .env files (add to .gitignore)
echo "HF_TOKEN=your_token" > .env
echo ".env" >> .gitignore
```

### GitHub Secrets

For CI/CD workflows, use GitHub Secrets:

1. Go to Repository Settings → Secrets and Variables → Actions
2. Add secrets:
   - `HF_TOKEN`: Your Hugging Face API token
   - `HF_USERNAME`: Your Hugging Face username

### Model Security

When deploying models:

1. Validate input before inference
2. Set appropriate rate limits
3. Monitor for adversarial inputs
4. Keep dependencies updated

### Data Privacy

- Do not include PII in training data
- Review demo data for sensitive content
- Be cautious with user-submitted text in demos

## Known Security Considerations

### Model Vulnerabilities

Transformer models may be vulnerable to:
- Adversarial examples
- Prompt injection (if used with user input)
- Data poisoning during training

### Dependency Vulnerabilities

Regularly update dependencies:

```bash
# Check for known vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Security Updates

Security updates will be released as:
- Patch versions for minor issues
- Minor versions for significant issues
- Announcements in repository for critical issues

## Contact

For security-related inquiries:
- Email: [your-email@example.com]
- GPG Key: [if available]

Thank you for helping keep this project secure!
