#!/bin/bash
# Steering test: 4 negative traits on Llama-3.1-8B-Instruct, +/-4 layers

python analysis/steering/evaluate.py \
  --experiment temp_llama_steering_feb18 \
  --traits emotion_set/relief,emotion_set/gratitude,emotion_set/adaptability,emotion_set/helpfulness \
  --direction negative \
  --trait-layers \
  "emotion_set/relief:3,4,5,6,7,8,9,10,11" \
  "emotion_set/gratitude:7,8,9,10,11,12,13,14,15" \
  "emotion_set/adaptability:14,15,16,17,18,19,20,21,22" \
  "emotion_set/helpfulness:15,16,17,18,19,20,21,22,23" \
  --search-steps 5 \
  --max-new-tokens 64
