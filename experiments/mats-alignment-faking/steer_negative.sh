#!/bin/bash
# Steering eval: 30 negative-direction traits on Llama-3.3-70B-Instruct (INT4)
# Layers from CoM mapping ±3 (center from Llama-8B CoM where available, else Qwen-14B CoM)

python analysis/steering/evaluate.py \
  --experiment mats-alignment-faking \
  --traits emotion_set/adaptability emotion_set/analytical emotion_set/authority_respect emotion_set/brevity emotion_set/confidence emotion_set/fairness emotion_set/gratitude emotion_set/helpfulness emotion_set/honesty emotion_set/hope emotion_set/humility emotion_set/independence emotion_set/moral_outrage emotion_set/obedience emotion_set/open_mindedness emotion_set/optimism emotion_set/patience emotion_set/pride emotion_set/refusal emotion_set/relief emotion_set/reverence_for_life emotion_set/self_awareness emotion_set/sincerity emotion_set/skepticism emotion_set/sympathy emotion_set/triumph emotion_set/trust emotion_set/urgency emotion_set/vigilance emotion_set/warmth \
  --model-variant instruct \
  --extraction-variant base \
  --direction negative \
  --load-in-4bit \
  --search-steps 5 \
  --save-responses best \
  --trait-layers "emotion_set/adaptability:37,38,39,40,41,42,43" "emotion_set/analytical:27,28,29,30,31,32,33" "emotion_set/authority_respect:38,39,40,41,42,43,44" "emotion_set/brevity:37,38,39,40,41,42,43" "emotion_set/confidence:36,37,38,39,40,41,42" "emotion_set/fairness:27,28,29,30,31,32,33" "emotion_set/gratitude:24,25,26,27,28,29,30" "emotion_set/helpfulness:45,46,47,48,49,50,51" "emotion_set/honesty:31,32,33,34,35,36,37" "emotion_set/hope:28,29,30,31,32,33,34" "emotion_set/humility:36,37,38,39,40,41,42" "emotion_set/independence:37,38,39,40,41,42,43" "emotion_set/moral_outrage:30,31,32,33,34,35,36" "emotion_set/obedience:32,33,34,35,36,37,38" "emotion_set/open_mindedness:31,32,33,34,35,36,37" "emotion_set/optimism:36,37,38,39,40,41,42" "emotion_set/patience:26,27,28,29,30,31,32" "emotion_set/pride:36,37,38,39,40,41,42" "emotion_set/refusal:27,28,29,30,31,32,33" "emotion_set/relief:15,16,17,18,19,20,21" "emotion_set/reverence_for_life:29,30,31,32,33,34,35" "emotion_set/self_awareness:35,36,37,38,39,40,41" "emotion_set/sincerity:31,32,33,34,35,36,37" "emotion_set/skepticism:33,34,35,36,37,38,39" "emotion_set/sympathy:21,22,23,24,25,26,27" "emotion_set/triumph:36,37,38,39,40,41,42" "emotion_set/trust:22,23,24,25,26,27,28" "emotion_set/urgency:25,26,27,28,29,30,31" "emotion_set/vigilance:28,29,30,31,32,33,34" "emotion_set/warmth:35,36,37,38,39,40,41"
