#!/usr/bin/env bash
set -euo pipefail
# Ubuntu 24.04 x86_64 user-local extraction; never changes system libraries.
: "${OSMESA_ROOT:?Set OSMESA_ROOT to a new user-local directory}"
mkdir -p "$OSMESA_ROOT"
cd "$OSMESA_ROOT"
package=libosmesa6_25.1.7-1ubuntu2~24.04.2_amd64.deb
apt-get download 'libosmesa6=25.1.7-1ubuntu2~24.04.2'
printf '%s  %s\n' '9c23e5994fa4c04be021401f16c6e88562af4397cb95e0edb67c222981c1544b' "$package" | sha256sum -c -
dpkg-deb -x "$package" root
lib_dir="$OSMESA_ROOT/root/usr/lib/x86_64-linux-gnu"
ln -sfn libOSMesa.so.8 "$lib_dir/libOSMesa.so"
ldd "$lib_dir/libOSMesa.so.8"
printf 'Set osmesa_lib to: %s\n' "$lib_dir"
