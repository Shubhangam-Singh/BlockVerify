#!/usr/bin/env bash
# Install the pre-commit secret scanner. Run once after cloning:
#   bash scripts/install-hooks.sh
set -e
ROOT="$(git rev-parse --show-toplevel)"
HOOK="$ROOT/.git/hooks/pre-commit"
cat > "$HOOK" <<'HOOKEOF'
#!/usr/bin/env bash
exec python3 "$(git rev-parse --show-toplevel)/scripts/check_secrets.py"
HOOKEOF
chmod +x "$HOOK"
echo "installed $HOOK"
