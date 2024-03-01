#!/usr/bin/env bash
set -e

LKEY=$1
LEASE=$2

if [ -z "$LKEY" ] || [ "$LKEY" == "-h" ] || [ "$LKEY" == "--help" ] ; then
  echo "This script updates UPPAAL license lease automatically using curl."
  echo "The script expects a license key and lease-bidinging period in hours as arguments."
  echo "For example:"
  echo "$0 12345678-9abc-def0-1234-567890abcdef 24"
  echo
  if [ -z "$CURL_PROXY" ]; then
      CURL_PROXY="--proxy-anyauth"
      echo "CURL_PROXY is not set, using \"$CURL_PROXY\" instead."
      echo "The following can be used to setup NTLM proxy:"
      echo "export CURL_PROXY=--proxy-ntlm --proxy-user username:password --proxy server:port"
  fi
  exit 1
fi

if [ -z "$(command -v curl)" ]; then
    echo "Please install curl to connect to the license server over Internet."
    exit 1
fi

if [ -z "$CURL_PROXY" ]; then
    CURL_PROXY="--proxy-anyauth"
fi

if [ -z "$LEASE" ]; then
  LEASE=1
  echo "The lease duration is not provided, using \"$LEASE\" instead."
fi

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  HERE=$(cd -P $(dirname "$SOURCE") >/dev/null 2>&1 && pwd)
  SOURCE=$(readlink "$SOURCE")
  [[ "$SOURCE" != /* ]] && SOURCE="$HERE/$SOURCE"
done
HERE=$(cd -P $(dirname "$SOURCE") > /dev/null 2>&1 && pwd)

echo "Generating a lease request..."
request=$("$HERE/verifyta" --key "$LKEY" --lease "$LEASE" --lease-request)
echo "Requesting a lease..."
lease=$(curl --silent $CURL_PROXY -X POST --data-urlencode "license_key=$LKEY" --data-urlencode "request=$request" "https://uppaal.veriaal.dk/lease/lease.php")
grep "<title>" <<< $lease && exit 1
grep "^Error" <<< $lease && exit 1
echo "Installing the lease..."
echo $lease | "$HERE/verifyta" --lease "$LEASE" --lease-install > /dev/null
echo "Lease has been successfully installed in $HOME/.config/uppaal"
