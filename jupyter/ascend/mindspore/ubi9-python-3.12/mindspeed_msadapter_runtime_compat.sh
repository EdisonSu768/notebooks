#!/bin/bash
set -Eeuo pipefail

main() {
    if [ "$#" -eq 0 ]; then
        echo "Usage: $0 <command> [args...]" >&2
        exit 1
    fi

    # Runtime compatibility now auto-loads from site-packages during Python
    # startup. Keep this wrapper as a no-op entrypoint for backward compatibility.
    exec "$@"
}

main "$@"
