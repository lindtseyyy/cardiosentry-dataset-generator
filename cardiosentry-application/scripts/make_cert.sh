#!/usr/bin/env bash
# mkcert helper for phase-3 HTTPS (plan §15.3): give the phone a trusted cert
# for the laptop's LAN IP so the PWA service worker and getUserMedia work.
#
#   ./scripts/make_cert.sh
#   uvicorn backend.main:app --host 0.0.0.0 --port 8000 \
#       --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
#
# Requires mkcert (https://github.com/FiloSottile/mkcert) and trusting the
# local CA on the phone once. Never commit certs/.
set -euo pipefail
cd "$(dirname "$0")/.."

IP=""
for candidate in $(hostname -I 2>/dev/null); do
  case "$candidate" in
    127.*|::1*) continue ;;
    *) IP="$candidate"; break ;;
  esac
done
[ -n "$IP" ] || { echo "no LAN IP found" >&2; exit 1; }

mkdir -p certs
mkcert -install
mkcert -key-file certs/key.pem -cert-file certs/cert.pem "$IP" "$(hostname).local" localhost 127.0.0.1
echo "certs written to certs/ for $IP"
echo "install the mkcert root CA on the phone once, then start uvicorn with --ssl-keyfile/--ssl-certfile"
