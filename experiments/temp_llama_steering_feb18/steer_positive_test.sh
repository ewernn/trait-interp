#!/bin/bash
# Steering test: 5 positive traits on Llama-3.1-8B-Instruct, +/-4 layers

python analysis/steering/evaluate.py \
  --experiment temp_llama_steering_feb18 \
  --traits emotion_set/acceptance,emotion_set/anger,emotion_set/guilt,emotion_set/spite,emotion_set/power_seeking \
  --direction positive \
  --trait-layers \
  "emotion_set/acceptance:7,8,9,10,11,12,13,14,15" \
  "emotion_set/anger:11,12,13,14,15,16,17,18,19" \
  "emotion_set/guilt:11,12,13,14,15,16,17,18,19" \
  "emotion_set/spite:13,14,15,16,17,18,19,20,21" \
  "emotion_set/power_seeking:16,17,18,19,20,21,22,23,24" \
  --search-steps 5 \
  --max-new-tokens 64
