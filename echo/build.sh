#!/usr/bin/env bash
# Builds Pokémon Echo: clones pokeemerald-expansion at the pinned base commit,
# applies the Echo patches, and compiles pokeemerald.gba. No ROM is distributed.
set -euo pipefail
BASE=44bc3e0473b85d8e254c8dbc9004ac2d7b0a5fb3
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:-$HERE/pokeemerald-expansion}"
if [ ! -d "$DIR" ]; then
  git clone https://github.com/rh-hideout/pokeemerald-expansion.git "$DIR"
fi
cd "$DIR"
git checkout -q -B echo "$BASE"
git -c user.name=echo -c user.email=echo@localhost am -q "$HERE"/patches/*.patch
make -j"$(nproc)" ${TOOLCHAIN:+TOOLCHAIN="$TOOLCHAIN"}
echo "Built: $DIR/pokeemerald.gba"
