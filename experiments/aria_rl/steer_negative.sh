#!/bin/bash
# Steering eval: 30 negative-direction traits on Qwen3-4B-Instruct
# Generated from 14B best layers mapped via int(layer*36/48) +/-2

python analysis/steering/evaluate.py \
  --experiment aria_rl \
  --traits emotion_set/adaptability,emotion_set/analytical,emotion_set/authority_respect,emotion_set/brevity,emotion_set/confidence,emotion_set/fairness,emotion_set/gratitude,emotion_set/helpfulness,emotion_set/honesty,emotion_set/hope,emotion_set/humility,emotion_set/independence,emotion_set/moral_outrage,emotion_set/obedience,emotion_set/open_mindedness,emotion_set/optimism,emotion_set/patience,emotion_set/pride,emotion_set/refusal,emotion_set/relief,emotion_set/reverence_for_life,emotion_set/self_awareness,emotion_set/sincerity,emotion_set/skepticism,emotion_set/sympathy,emotion_set/triumph,emotion_set/trust,emotion_set/urgency,emotion_set/vigilance,emotion_set/warmth \
  --direction negative \
  --trait-layers \
  "emotion_set/adaptability:18,19,20,21,22" \
  "emotion_set/analytical:10,11,12,13,14" \
  "emotion_set/authority_respect:17,18,19,20,21" \
  "emotion_set/brevity:14,15,16,17,18" \
  "emotion_set/confidence:14,15,16,17,18" \
  "emotion_set/fairness:8,9,10,11,12" \
  "emotion_set/gratitude:10,11,12,13,14" \
  "emotion_set/helpfulness:19,20,21,22,23" \
  "emotion_set/honesty:13,14,15,16,17" \
  "emotion_set/hope:13,14,15,16,17" \
  "emotion_set/humility:13,14,15,16,17" \
  "emotion_set/independence:18,19,20,21,22" \
  "emotion_set/moral_outrage:10,11,12,13,14" \
  "emotion_set/obedience:10,11,12,13,14" \
  "emotion_set/open_mindedness:10,11,12,13,14" \
  "emotion_set/optimism:16,17,18,19,20" \
  "emotion_set/patience:11,12,13,14,15" \
  "emotion_set/pride:19,20,21,22,23" \
  "emotion_set/refusal:13,14,15,16,17" \
  "emotion_set/relief:5,6,7,8,9" \
  "emotion_set/reverence_for_life:11,12,13,14,15" \
  "emotion_set/self_awareness:14,15,16,17,18" \
  "emotion_set/sincerity:10,11,12,13,14" \
  "emotion_set/skepticism:15,16,17,18,19" \
  "emotion_set/sympathy:11,12,13,14,15" \
  "emotion_set/triumph:15,16,17,18,19" \
  "emotion_set/trust:9,10,11,12,13" \
  "emotion_set/urgency:8,9,10,11,12" \
  "emotion_set/vigilance:9,10,11,12,13" \
  "emotion_set/warmth:17,18,19,20,21" \
  --search-steps 5 \
  --max-new-tokens 64
