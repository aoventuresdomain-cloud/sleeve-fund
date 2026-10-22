#!/usr/bin/env bash
# Decrypt a sheet-export artifact (.github/workflows/sheet-export.yml).
# Usage: scripts/decrypt_sheet_export.sh <artifact dir> <private key .pem> <target dir>
# The artifact dir holds key.bin.enc and sheet.tar.gz.enc; the files are untarred into the target dir.
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <artifact dir> <private key .pem> <target dir>" >&2; exit 2; }
src=$1 key=$2 dst=$3
umask 077
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
openssl pkeyutl -decrypt -inkey "$key" -pkeyopt rsa_padding_mode:oaep -pkeyopt rsa_oaep_md:sha256 \
  -in "$src/key.bin.enc" -out "$tmp/key.hex"
openssl enc -d -aes-256-cbc -pbkdf2 -in "$src/sheet.tar.gz.enc" -out "$tmp/sheet.tar.gz" -pass "file:$tmp/key.hex"
mkdir -p "$dst"
tar -xzf "$tmp/sheet.tar.gz" -C "$dst" --no-same-owner --no-overwrite-dir
tar -tzf "$tmp/sheet.tar.gz"
