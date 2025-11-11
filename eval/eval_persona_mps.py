"""
MPS-compatible version of eval_persona.py for Apple Silicon
Forces use of transformers backend instead of vllm
"""
import sys
import os

# Monkey-patch to force load_model instead of load_vllm_model
original_file = os.path.join(os.path.dirname(__file__), "eval_persona.py")

# Set environment variable to disable vllm
os.environ["USE_TRANSFORMERS"] = "1"

# Import the original module
exec(open(original_file).read(), globals())
