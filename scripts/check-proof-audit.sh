#!/usr/bin/env bash
# Proof-boundary audit (plan I-8; test-spec T6.6/T6.7). Run from the repo
# root. Fails (exit != 0) when:
#   1. a Kani harness has no limits.toml entry, or a limits.toml entry has
#      no matching harness (missing/orphan bounds);
#   2. a harness is missing from AUDIT.md, or an AUDIT.md production
#      call-site entry (file#fn) no longer exists (stale audit);
#   3. a production module bypasses an audited kernel (direct debit
#      arithmetic, out-of-crate ledger writes, unshared transfer sites).
# The CI `proof` job runs this before the Kani harnesses.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARNESSES="$ROOT/crates/proof-kernels/src/harnesses.rs"
LIMITS="$ROOT/crates/proof-kernels/limits.toml"
AUDIT="$ROOT/crates/proof-kernels/AUDIT.md"
README="$ROOT/crates/proof-kernels/README.md"

fail() { echo "proof-audit: FAIL: $1" >&2; exit 1; }

# --- 1. limits.toml <-> harnesses one-to-one mapping ------------------------
grep -q 'schema_version' "$LIMITS" || fail "limits.toml missing schema_version"

harness_names="$(grep -oE '^fn i8_t6[a-z0-9_]+' "$HARNESSES" | sed 's/^fn //' | sort)"
[ -n "$harness_names" ] || fail "no i8_t6* harnesses found in $HARNESSES"
limit_names="$(grep -oE '^\[harness\.[a-z0-9_]+\]' "$LIMITS" | sed -E 's/^\[harness\.//; s/\]$//' | sort)"
[ -n "$limit_names" ] || fail "no [harness.*] entries in limits.toml"

for h in $harness_names; do
  grep -qx "$h" <(echo "$limit_names") || fail "harness '$h' has no limits.toml entry (missing bounds)"
  grep -q "\`$h\`" "$AUDIT" || fail "harness '$h' is not mapped in AUDIT.md"
done
for l in $limit_names; do
  grep -qx "$l" <(echo "$harness_names") || fail "limits.toml entry '$l' has no matching harness (orphan bounds)"
done

# --- 2. AUDIT.md call-site entries must exist -------------------------------
# Rows look like: | `harness` | Spec | symbol (`file#fn`) | `file#fn`, `file#fn` |
audit_rows="$(grep -E '^\| `i8_t6' "$AUDIT")" || true
[ -n "$audit_rows" ] || fail "AUDIT.md has no harness rows"
for site in $(echo "$audit_rows" | grep -oE 'crates/[a-z0-9_-]+/src/[a-zA-Z0-9_./-]+\.rs#[a-zA-Z0-9_:]+' | sort -u); do
  file="${site%%#*}"; fn="${site##*#}"
  # A qualified call site (Type::method) is checked against its method name.
  case "$fn" in *::*) fn="${fn##*::}" ;; esac
  [ -f "$ROOT/$file" ] || fail "AUDIT.md call-site file does not exist: $file"
  grep -qE "fn $fn\b" "$ROOT/$file" || fail "AUDIT.md call-site is stale: $site (no 'fn $fn')"
done

# The audited kernel symbol itself must exist in production.
grep -q 'pub fn split_money_transfer' "$ROOT/crates/sim/src/lib.rs" \
  || fail "audited kernel split_money_transfer missing from rpg_sim"

# --- 3. bypass boundary checks ----------------------------------------------
# 3a. No direct inventory-money debit anywhere in production sources: every
# conserved money movement must go through the audited kernel.
if grep -rnE '\.inventory\.money -=|\.money -= [a-z]' "$ROOT"/crates/*/src --include='*.rs'; then
  fail "production bypass: direct money debit outside split_money_transfer"
fi
# 3b. Both audited transfer lowering sites route through the kernel.
call_sites="$(grep -c 'and_then(|(f, t)| split_money_transfer' "$ROOT/crates/sim/src/lib.rs" || true)"
[ "$call_sites" -eq 2 ] || fail "expected exactly 2 audited split_money_transfer call sites (transfer_money, do_pay), found $call_sites"
# 3c. The documented non-transfer credit exception is still the wage site.
grep -q 'do_repair' "$AUDIT" || fail "AUDIT.md must document the wage-credit exception (do_repair)"
# 3d. Ledger writes stay inside the behavior epistemics kernel.
if grep -rn 'record_evidence(' "$ROOT"/crates/sim/src "$ROOT"/crates/social/src "$ROOT"/crates/domain/src "$ROOT"/crates/agents/src 2>/dev/null; then
  fail "production bypass: record_evidence called outside crates/behavior/src/epistemics.rs"
fi

# --- 4. T6.6 inductive coverage checklist present ---------------------------
grep -q '## T6.6' "$README" || fail "README.md is missing the T6.6 coverage checklist"

echo "proof-audit: OK ($(( $(echo "$harness_names" | wc -l) )) harnesses, limits 1:1, call sites live, no bypasses)"
