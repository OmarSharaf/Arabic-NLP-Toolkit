# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :x:                |

## Reporting a Vulnerability

**Please do not open public GitHub issues for security vulnerabilities.**

Send details to **omarsharaf@msn.com** with:

- Description of the issue
- Steps to reproduce
- Impact assessment (if known)
- Suggested fix (optional)

You should receive a response within **72 hours**. If the issue is confirmed, we will:

1. Develop a fix on a private branch
2. Release a patched version on PyPI
3. Publish a security advisory on GitHub

## Scope

This policy covers the `arabic-nlp-toolkit` Python package and the optional `webapp/` demo server.

The web demo binds to `127.0.0.1` by default and is intended for **local development only** — do not expose it to the public internet without authentication and hardening.
