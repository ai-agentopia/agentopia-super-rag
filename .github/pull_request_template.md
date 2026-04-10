## What this changes

<!-- One paragraph: what problem does this solve and how -->

## Type

- [ ] Bug fix
- [ ] Retrieval pipeline change (requires evaluation evidence — see CONTRIBUTING.md)
- [ ] New feature
- [ ] Doc update
- [ ] Dependency update
- [ ] Other

## Test evidence

<!-- Paste exact fast-gate output -->

```
python -m pytest tests/ -m "not integration and not e2e" -x -q
→
```

## Evaluation evidence (retrieval changes only)

<!-- If this changes chunking, retrieval mode, or embedding model, paste evaluation results here.
     PRs that change retrieval behavior without evaluation results will not be reviewed. -->

Baseline nDCG@5:
New nDCG@5:
Delta:
Gate: PASS / FAIL

## Docs updated

- [ ] `README.md` (if local workflow, env vars, or feature roadmap changed)
- [ ] `docs/architecture.md` (if API surface or planned evolution changed)
- [ ] `docs/operations.md` (if env vars, health endpoints, or runbook changed)
- [ ] `docs/evaluation.md` (if retrieval outcome changed)
- [ ] N/A — no doc updates needed

## Checklist

- [ ] Fast gate passes
- [ ] No new hardcoded secrets, internal URLs, or credentials
- [ ] `.env.example` updated if new env vars added
