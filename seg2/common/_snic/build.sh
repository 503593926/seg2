#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
gcc -O3 -march=native -ffast-math -fPIC -shared snic.c -I . -lm -o libsnic.so
