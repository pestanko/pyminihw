#! /bin/bash

TO=$1

echo "Copining to: $TO"

cp ./.gitlab-ci.yml "$TO"
cp ./run.sh "$TO"

echo "[DONE]"


