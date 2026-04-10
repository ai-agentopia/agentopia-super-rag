# Branch Protection Policy — agentopia-super-rag

This document defines the intended branch protection policy for `main`.

**Current status:** NOT APPLIED — requires public repo or GitHub Pro plan.
See issue #48 for the tracking task.

---

## Intended policy for `main`

| Rule | Setting |
|---|---|
| Require pull request before merging | Yes |
| Required approving reviews | 1 |
| Dismiss stale approvals on new push | Yes |
| Require review from Code Owners | Yes (CODEOWNERS: @thanhth2813) |
| Require status checks to pass | Yes |
| Required status check | `Fast Gate` (`.github/workflows/ci.yml`) |
| Require branches to be up to date | Yes (strict) |
| Require conversation resolution | Yes |
| Allow force pushes | No |
| Allow deletions | No |
| Bypass for admins | No (enforce for everyone) |

---

## How to apply

### When the repo is made public (free):

Via GitHub UI:
1. Settings → Branches → Add branch protection rule
2. Branch name pattern: `main`
3. Apply settings from the table above
4. Save

Via API (once public):
```bash
gh api repos/ai-agentopia/agentopia-super-rag/branches/main/protection \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  --field required_status_checks='{"strict":true,"contexts":["Fast Gate"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"dismiss_stale_reviews":true,"require_code_owner_reviews":true,"required_approving_review_count":1}' \
  --field restrictions=null \
  --field required_conversation_resolution=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

### With GitHub Pro (private repo):

Same as above — the API call will succeed on Pro.

---

## Release / tag policy

- Tags are created from `main` only
- Format: `v{MAJOR}.{MINOR}.{PATCH}` for stable releases
- Image tags follow `dev-{sha}` format (produced on every push to `main`)
- No release branches — tags point directly to `main` commits
