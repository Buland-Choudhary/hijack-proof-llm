#!/usr/bin/env bash
# Install the git pre-commit hook that runs the repo safety check before every commit.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
hook="$root/.git/hooks/pre-commit"
cat > "$hook" <<'EOF'
#!/usr/bin/env bash
exec python3 "$(git rev-parse --show-toplevel)/scripts/check_repo_safety.py"
EOF
chmod +x "$hook"
echo "Installed pre-commit hook: $hook"
