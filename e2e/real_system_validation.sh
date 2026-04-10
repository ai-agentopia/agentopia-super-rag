#!/usr/bin/env bash
# Real-system E2E validation for agentopia-super-rag
#
# Runs against the Agentopia cluster via SSH.
# Tests the real runtime path: bot auth → binding cache → scope resolution → Qdrant search.
#
# Prerequisites:
#   - SSH access to the cluster host (set via SSH_HOST env var)
#   - kubectl access via KUBECONFIG=/etc/rancher/k3s/k3s.yaml on the cluster host
#   - knowledge-api pod running in the target namespace
#   - At least one bot with knowledge bindings deployed
#
# Usage:
#   SSH_HOST=<your-cluster-host> bash e2e/real_system_validation.sh
#
# What this proves:
#   1. Service healthy with real Qdrant backend and binding cache
#   2. Bot with valid token retrieves from bound scopes (real embedding + Qdrant search)
#   3. Auth isolation: no token / wrong token / nonexistent bot → rejected
#   4. Scope isolation: bot bound to subset of scopes only sees its scopes
#   5. Internal token path works for operator access
#
# What is NOT tested:
#   - Gateway Telegram → plugin → knowledge-api chain (manual validation only)
#   - Dedicated test bot/scope (uses existing bots)
#   - Postgres document store (auth failure on dev — separate issue)

SSH_HOST="${SSH_HOST:?SSH_HOST must be set to your cluster host}"
NS="agentopia-dev"

PASSED=0
FAILED=0
TOTAL=0

pass() { ((PASSED++)); ((TOTAL++)); echo "  PASS"; }
fail() { ((FAILED++)); ((TOTAL++)); echo "  FAIL: $1"; }

# Run python3 inside the knowledge-api pod via SSH
run_in_pod() {
    ssh "$SSH_HOST" "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl exec -n $NS $KA_POD -- python3 -c \"$1\"" 2>/dev/null
}

get_secret() {
    ssh "$SSH_HOST" "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get secret $1 -n $NS -o jsonpath='{.data.$2}'" 2>/dev/null | base64 -d
}

echo "=== Agentopia Super RAG — Real System E2E Validation ==="
echo "=== Target: $SSH_HOST / namespace: $NS ==="
echo ""

# ── Discover pod and tokens ──────────────────────────────────────────────────

KA_POD=$(ssh "$SSH_HOST" "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get pod -n $NS -l app=knowledge-api -o jsonpath='{.items[0].metadata.name}'" 2>/dev/null)
echo "knowledge-api pod: $KA_POD"

INTERNAL_TOKEN=$(get_secret knowledge-api-env KNOWLEDGE_API_INTERNAL_TOKEN)
echo "internal token: ${#INTERNAL_TOKEN} chars"

SUCCESS_BOT="tim-joblogic-sa"
TOKEN_SUCCESS=$(get_secret "agentopia-gateway-env-$SUCCESS_BOT" AGENTOPIA_RELAY_TOKEN)
echo "success bot ($SUCCESS_BOT) token: ${#TOKEN_SUCCESS} chars"

ISOLATION_BOT="dan-joblogic-qa"
TOKEN_ISOLATION=$(get_secret "agentopia-gateway-env-$ISOLATION_BOT" AGENTOPIA_RELAY_TOKEN)
echo "isolation bot ($ISOLATION_BOT) token: ${#TOKEN_ISOLATION} chars"
echo ""

# ── Tests ────────────────────────────────────────────────────────────────────

echo "--- TEST 1: Service liveness ---"
RESULT=$(run_in_pod "
import urllib.request, json
d = json.loads(urllib.request.urlopen('http://localhost:8002/health').read())
print(d['status'], d['service'])
")
echo "  $RESULT"
[[ "$RESULT" == "ok knowledge-api" ]] && pass || fail "health not ok"

echo "--- TEST 2: Real Qdrant backend + binding cache ---"
RESULT=$(run_in_pod "
import urllib.request, json
r = urllib.request.Request('http://localhost:8002/internal/health', headers={'X-Internal-Token': '$INTERNAL_TOKEN'})
d = json.loads(urllib.request.urlopen(r).read())
print(d['qdrant'], d['binding_cache']['bot_count'])
")
echo "  qdrant=$RESULT"
[[ "$RESULT" == ok* ]] && pass || fail "qdrant not ok or binding cache empty"

echo "--- TEST 3: Bot auth SUCCESS — $SUCCESS_BOT (bound to api-docs + debate-docs) ---"
RESULT=$(run_in_pod "
import urllib.request, json
r = urllib.request.Request(
    'http://localhost:8002/api/v1/knowledge/search?query=architecture+deployment',
    headers={'Authorization': 'Bearer $TOKEN_SUCCESS', 'X-Bot-Name': '$SUCCESS_BOT'}
)
d = json.loads(urllib.request.urlopen(r).read())
scopes = sorted(set(x['scope'] for x in d['results']))
print(d['count'], ','.join(scopes))
")
echo "  results=$RESULT"
[[ "${RESULT%% *}" -gt 0 ]] && pass || fail "expected results, got $RESULT"

echo "--- TEST 4: Auth isolation — no token ---"
RESULT=$(run_in_pod "
import urllib.request, urllib.error
try:
    urllib.request.urlopen(urllib.request.Request('http://localhost:8002/api/v1/knowledge/search?query=test'))
    print('ALLOWED')
except urllib.error.HTTPError as e:
    print(e.code)
")
echo "  HTTP $RESULT"
[[ "$RESULT" == "401" ]] && pass || fail "expected 401, got $RESULT"

echo "--- TEST 5: Auth isolation — wrong token ---"
RESULT=$(run_in_pod "
import urllib.request, urllib.error
try:
    r = urllib.request.Request('http://localhost:8002/api/v1/knowledge/search?query=test',
        headers={'Authorization': 'Bearer FAKE-TOKEN', 'X-Bot-Name': '$SUCCESS_BOT'})
    urllib.request.urlopen(r)
    print('ALLOWED')
except urllib.error.HTTPError as e:
    print(e.code)
")
echo "  HTTP $RESULT"
[[ "$RESULT" == "401" ]] && pass || fail "expected 401, got $RESULT"

echo "--- TEST 6: Auth isolation — nonexistent bot ---"
RESULT=$(run_in_pod "
import urllib.request, urllib.error
try:
    r = urllib.request.Request('http://localhost:8002/api/v1/knowledge/search?query=test',
        headers={'Authorization': 'Bearer FAKE', 'X-Bot-Name': 'e2e-nonexistent-bot'})
    urllib.request.urlopen(r)
    print('ALLOWED')
except urllib.error.HTTPError as e:
    print(e.code)
")
echo "  HTTP $RESULT"
[[ "$RESULT" == "401" ]] && pass || fail "expected 401, got $RESULT"

echo "--- TEST 7: Scope isolation — $ISOLATION_BOT (api-docs ONLY, NOT debate-docs) ---"
RESULT=$(run_in_pod "
import urllib.request, json
r = urllib.request.Request(
    'http://localhost:8002/api/v1/knowledge/search?query=architecture+deployment',
    headers={'Authorization': 'Bearer $TOKEN_ISOLATION', 'X-Bot-Name': '$ISOLATION_BOT'}
)
d = json.loads(urllib.request.urlopen(r).read())
scopes = set(x['scope'] for x in d['results'])
if 'joblogic-kb/debate-docs' in scopes:
    print('ISOLATION_FAILURE')
else:
    print('ISOLATED count=' + str(d['count']))
")
echo "  $RESULT"
[[ "$RESULT" == ISOLATED* ]] && pass || fail "scope isolation failure"

echo "--- TEST 8: Internal token — operator explicit scope ---"
RESULT=$(run_in_pod "
import urllib.request, json
r = urllib.request.Request(
    'http://localhost:8002/api/v1/knowledge/search?query=bot+authentication&scopes=joblogic-kb/api-docs',
    headers={'X-Internal-Token': '$INTERNAL_TOKEN'}
)
d = json.loads(urllib.request.urlopen(r).read())
print(d['count'])
")
echo "  result count: $RESULT"
[[ "$RESULT" -gt 0 ]] && pass || fail "expected results"

echo "--- TEST 9: Scope isolation — nonexistent scope returns empty ---"
RESULT=$(run_in_pod "
import urllib.request, json
r = urllib.request.Request(
    'http://localhost:8002/api/v1/knowledge/search?query=test&scopes=fake-org/no-scope',
    headers={'X-Internal-Token': '$INTERNAL_TOKEN'}
)
d = json.loads(urllib.request.urlopen(r).read())
print(d['count'])
")
echo "  result count: $RESULT"
[[ "$RESULT" == "0" ]] && pass || fail "expected 0 results, got $RESULT"

echo ""
echo "=== SUMMARY: $PASSED/$TOTAL passed, $FAILED failed ==="
[[ "$FAILED" -eq 0 ]] && echo "ALL TESTS PASSED" || echo "FAILURES DETECTED"
exit "$FAILED"
