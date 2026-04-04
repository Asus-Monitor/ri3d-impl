#!/usr/bin/env bash
set -euo pipefail

ZIP="360_v2.zip"
OUTDIR="dataset"

if [ ! -f "$ZIP" ]; then
    echo "Error: $ZIP not found in current directory"
    exit 1
fi

# Get scene folders (top-level dirs that contain an images/ subfolder)
scenes=$(7z l "$ZIP" -slt | grep '^Path = ' | sed 's/^Path = //' \
    | grep -P '^[^/]+/images$' | awk -F/ '{print $1}' | sort -u)

mkdir -p "$OUTDIR"

for scene in $scenes; do
    echo "Extracting $scene/images ..."
    7z x "$ZIP" "${scene}/images/*" -o"$OUTDIR" -aoa > /dev/null
    # Move images up: dataset/scene/images/* -> dataset/scene/*
    mv "$OUTDIR/$scene/images/"* "$OUTDIR/$scene/"
    rmdir "$OUTDIR/$scene/images"
    echo "  -> $OUTDIR/$scene/ ($(ls "$OUTDIR/$scene/" | wc -l) images)"
done

echo "Done. Scenes extracted to $OUTDIR/"
