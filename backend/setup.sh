#!/bin/bash
# Run setup from project root instead.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../setup.sh" "$@"
