# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly with details of the vulnerability
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-48 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

### Security Best Practices for Users

1. **Keep dependencies updated**: Run `pip install --upgrade -r requirements.txt` regularly
2. **Use virtual environments**: Isolate project dependencies
3. **Review file permissions**: Ensure output directories have appropriate access controls
4. **Validate input images**: Only process images from trusted sources

## Security Features

This project includes:

- **Dependabot**: Automated security updates for dependencies
- **CodeQL**: Static analysis for vulnerability detection
- **Pre-commit hooks**: Security checks before commits (detect-private-key)

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve this project's security.
