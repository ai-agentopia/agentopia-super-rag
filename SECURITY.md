# Security Policy

## What this service owns

`agentopia-super-rag` is responsible for:
- Scope isolation enforcement (a bot cannot read outside its subscribed scopes)
- Ingest auth (internal token required for all write operations)
- Bot auth verification (bearer token + K8s Secret lookup)
- Embedding API key handling
- Document lifecycle integrity

Security issues in this repo are specifically about those boundaries.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email: security@agentopia.vn

Include:
- Description of the issue
- Steps to reproduce
- Impact assessment (which boundary is affected)
- Whether you have a proposed fix

You will receive an acknowledgement within 3 business days.

## Scope

In scope:
- Scope isolation bypass (bot accessing another tenant's knowledge)
- Auth bypass on any endpoint
- Token/key exposure via API response or logs
- Injection via document ingest path

Out of scope:
- Issues in external dependencies (Qdrant, Postgres, OpenRouter) — report to their upstream
- Agentopia platform issues outside retrieval (bot identity, LLM routing, UI)
- Issues that require physical or privileged K8s cluster access to exploit

## Disclosure policy

We follow responsible disclosure. We ask for 90 days to address confirmed vulnerabilities before public disclosure.

We do not currently offer a bug bounty program.
