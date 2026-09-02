#!/usr/bin/env bash
# Fetch the real pretrained checkpoints used by the evaluation (safetensors, ~65 MB).
set -e
cd "$(dirname "$0")/checkpoints"
base="https://huggingface.co"
dl() { [ -f "$2" ] || python3 -c "import urllib.request;urllib.request.urlretrieve('$base/$1/resolve/main/model.safetensors','$2')"; echo "ok $2"; }
dl "google/bert_uncased_L-2_H-128_A-2" "google_bert_uncased_L-2_H-128_A-2.safetensors"
dl "albert-base-v2"                    "albert-base-v2.safetensors"
