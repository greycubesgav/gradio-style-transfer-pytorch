#!/usr/bin/env bash

random='15'
cweight='0.004'
tvweight='3.0'
minscale='512'
endscale='512'
interations='800'
date=$(date +%Y%m%d%H%M%S)
lowerlim=3
upperlim=30

src="$1"
style="$2"

src='data/input/golden_gate.jpg'
style='data/styles/shipwreck.jpg'

if [ -z "$src" ] || [ -z "$style" ]; then
  echo "Error: both src and style must be provided as arguments." >&2
  exit 1
fi

src_filename=$(basename -- "$src")
src_filename_noext="${src_filename%.*}"

style_filename=$(basename -- "$style")
style_filename_noext="${style_filename%.*}"

python -m style_transfer.gradio-ui --output "$output_img" \
--random-seed "$random" \
--content-weight "$cweight" \
--end-scale "$endscale" \
--tv-weight "$tvweight" \
--iterations "$interations" \
--save-every "5000" \
--pooling max \
--content "$src" --styles "$style"