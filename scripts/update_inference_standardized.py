#!/usr/bin/env python3
"""
Update inference/monitor_dynamics.py to use standardized prompt structure

Changes:
- Save shared prompts to: experiments/{exp}/inference/prompts/prompt_N.json
- Save trait projections to: experiments/{exp}/inference/projections/{trait}/prompt_N.json
"""

import sys
from pathlib import Path

# Read current script
inference_script = Path("inference/monitor_dynamics.py")
if not inference_script.exists():
    print(f"❌ {inference_script} not found")
    sys.exit(1)

content = inference_script.read_text()

# Check if already updated
if "inference/prompts/" in content:
    print("✅ Script already uses standardized structure")
    sys.exit(0)

print("📝 Updating inference script for standardized prompts...")

# Add helper functions after imports
helper_functions = '''
def save_standardized_output(experiment, trait, prompt_idx, data):
    """Save inference data in standardized structure"""
    from pathlib import Path
    import json
    
    # Shared prompts directory
    prompts_dir = Path(f"experiments/{experiment}/inference/prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Trait projections directory
    projections_dir = Path(f"experiments/{experiment}/inference/projections/{trait}")
    projections_dir.mkdir(parents=True, exist_ok=True)
    
    # Save shared prompt data (only if doesn't exist)
    prompt_file = prompts_dir / f"prompt_{prompt_idx}.json"
    if not prompt_file.exists():
        shared_data = {
            "prompt": data["prompt"],
            "response": data["response"],
            "tokens": data["tokens"]
        }
        with open(prompt_file, 'w') as f:
            json.dump(shared_data, f, indent=2)
    
    # Save trait-specific projections
    projection_file = projections_dir / f"prompt_{prompt_idx}.json"
    projection_data = {
        "prompt_id": f"prompt_{prompt_idx}",
        "trait": trait,
        "projections": data.get("projections", {}),
        "trait_scores": data.get("trait_scores", {}),
        "dynamics": data.get("dynamics", {}),
        "metadata": data.get("metadata", {})
    }
    with open(projection_file, 'w') as f:
        json.dump(projection_data, f, indent=2)

'''

# Find where to insert (after imports, before main function)
import_end = content.rfind("import ")
import_end = content.find("\n\n", import_end)

updated = content[:import_end] + "\n\n" + helper_functions + content[import_end:]

# Find and replace save logic
# Look for: with open(output_path, 'w') as f:
#              json.dump(result, f, indent=2)

# This is a placeholder - actual implementation would need to match exact patterns
print("⚠️  Manual update required:")
print("  1. Add save_standardized_output() helper function")
print("  2. Replace output logic to call save_standardized_output()")
print("  3. Update data loader in visualizer")

# Save backup
backup_path = inference_script.with_suffix(".py.bak")
backup_path.write_text(content)
print(f"✅ Backup saved to: {backup_path}")

print("\n📖 Manual changes needed in inference/monitor_dynamics.py:")
print("  OLD: output_path = exp_dir / trait / 'inference' / 'residual_stream_activations' / f'prompt_{i}.json'")
print("  NEW: save_standardized_output(experiment, trait, i, result)")
