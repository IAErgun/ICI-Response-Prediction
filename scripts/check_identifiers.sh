#!/bin/bash
# Refuse to let patient data or site-specific details into the repository.
#
#   bash scripts/check_identifiers.sh            # scan the working tree (what CI runs)
#   bash scripts/check_identifiers.sh --staged   # scan what is about to be committed
#                                                #   (what .githooks/pre-commit runs)
#
# Two kinds of check:
#   - files that should never be committed at all: tables, imaging, weights, documents.
#     .gitignore already excludes them, but `git add -f` bypasses that.
#   - text that looks like an identifier: home/cluster paths, email addresses, and runs
#     of 7+ digits (the shape of an MRN or accession number).
#
# Site-specific patterns (storage names, usernames, collaborators' names) must not be
# written here, since this file is public. Put them one regex per line in
# `.identifiers-local` at the repository root; it is git-ignored, and checked when present.

set -u
cd "$(dirname "$0")/.." || exit 2          # the repository root: this file lives in scripts/

FORBIDDEN_FILES='\.(csv|tsv|xlsx?|nii|nii\.gz|dcm|npy|npz|pth|pt|ckpt|docx|pdf)$'
GENERIC='/autofs/|/home/[a-z]|/Users/[A-Za-z]|/cluster/|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}|[0-9]{7,}'
LOCAL_FILE=.identifiers-local
SELF=scripts/check_identifiers.sh

staged=false
[ "${1:-}" = "--staged" ] && staged=true

if $staged; then
  files=$(git diff --cached --name-only --diff-filter=ACMR)
else
  files=$(git ls-files 2>/dev/null)
  [ -n "$files" ] || files=$(find . -type f -not -path './.git/*' -not -path '*/__pycache__/*' \
                                  -not -path '*/.ruff_cache/*' -not -path '*.egg-info/*' | sed 's|^\./||')
fi

content() {                  # the version being checked: staged blob, or the file on disk
  if $staged; then git show ":$1" 2>/dev/null; else cat "$1" 2>/dev/null; fi
}

patterns="$GENERIC"
if [ -f "$LOCAL_FILE" ]; then
  local_re=$(grep -v '^\s*\(#\|$\)' "$LOCAL_FILE" | paste -sd'|' -)
  [ -n "$local_re" ] && patterns="$patterns|$local_re"
fi

fail=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  if echo "$f" | grep -qiE "$FORBIDDEN_FILES"; then
    echo "  data/weights/document file: $f"
    fail=1
    continue
  fi
  case "$f" in "$SELF"|"$LOCAL_FILE"|LICENSE) continue ;; esac
  hits=$(content "$f" | grep -niE "$patterns" | grep -v 'noreply@' | cut -c1-120)
  if [ -n "$hits" ]; then
    echo "  $f"
    echo "$hits" | sed 's/^/      line /'
    fail=1
  fi
done <<< "$files"

if [ $fail -ne 0 ]; then
  echo "Possible identifying details found (above). Remove them before committing." >&2
  echo "If one is a false positive, rephrase it; do not weaken this check." >&2
  exit 1
fi
echo "identifier scan: clean ($(echo "$files" | grep -c .) files checked)"
