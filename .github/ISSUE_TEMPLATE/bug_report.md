---
name: Bug report
about: Report a defect in the knowledge-api service
labels: bug
---

## What happened

<!-- Describe the bug clearly. What did you expect vs what actually occurred? -->

## Reproduction steps

1.
2.
3.

## Affected area

- [ ] Ingest path (document parsing, chunking, embedding, upsert)
- [ ] Search path (query, retrieval, citation)
- [ ] Auth (internal token, bot bearer)
- [ ] Scope isolation (wrong scope returned, cross-tenant leak)
- [ ] Binding cache (403 unexpectedly, stale subscriptions)
- [ ] Document lifecycle (document not superseded/deleted correctly)
- [ ] Evaluation / metrics
- [ ] Local dev / bootstrap
- [ ] Other

## Environment

- Service version (from `/health`):
- Qdrant version:
- Python version:
- Deployment: production cluster / local Podman / local native

## Logs / evidence

```
<!-- Paste relevant log lines here -->
```

## Is this a security issue?

If this involves scope isolation bypass, auth bypass, or credential exposure — do NOT open a public issue. See SECURITY.md.
