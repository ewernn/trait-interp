"""Validate annotation spans exist in response text, then merge into main annotations file."""
import json
import sys

RESPONSES_DIR = "experiments/audit-bench/inference/rm_lora/responses/rm_syco/train_100"
ANNOTATIONS_FILE = "experiments/audit-bench/inference/rm_lora/responses/rm_syco/train_100_annotations.json"

def validate_wave(wave_file):
    with open(wave_file) as f:
        wave = json.load(f)

    errors = []
    warnings = []

    for entry in wave:
        idx = entry["idx"]
        resp_file = f"{RESPONSES_DIR}/{idx}.json"
        try:
            with open(resp_file) as f:
                resp_data = json.load(f)
            response_text = resp_data["response"]
        except FileNotFoundError:
            errors.append(f"  [{idx}] Response file not found: {resp_file}")
            continue

        for span_entry in entry.get("spans", []):
            span_text = span_entry["span"]
            # Normalize newlines for matching
            normalized_response = response_text.replace("\\n", "\n")
            if span_text not in normalized_response:
                # Try with escaped quotes
                alt_span = span_text.replace('\\"', '"')
                if alt_span not in normalized_response:
                    errors.append(f"  [{idx}] SPAN NOT FOUND in response: {span_text[:80]}...")

        for span_entry in entry.get("borderline", []):
            span_text = span_entry["span"]
            normalized_response = response_text.replace("\\n", "\n")
            if span_text not in normalized_response:
                alt_span = span_text.replace('\\"', '"')
                if alt_span not in normalized_response:
                    warnings.append(f"  [{idx}] BORDERLINE SPAN NOT FOUND: {span_text[:80]}...")

    return wave, errors, warnings

def merge(annotations_file, wave):
    with open(annotations_file) as f:
        data = json.load(f)

    existing_ids = {e["idx"] for e in data["annotations"]}
    added = 0
    for entry in wave:
        if entry["idx"] not in existing_ids:
            data["annotations"].append(entry)
            added += 1
        else:
            print(f"  Skipping idx {entry['idx']} (already exists)")

    # Sort by idx
    data["annotations"].sort(key=lambda x: x["idx"])

    with open(annotations_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return added

if __name__ == "__main__":
    wave_file = sys.argv[1] if len(sys.argv) > 1 else "experiments/audit-bench/inference/rm_lora/responses/rm_syco/wave2_101_120.json"

    print(f"Validating {wave_file}...")
    wave, errors, warnings = validate_wave(wave_file)

    print(f"\n  Entries: {len(wave)}")
    print(f"  Span errors: {len(errors)}")
    print(f"  Borderline warnings: {len(warnings)}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(e)
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(w)

    if errors:
        print(f"\n{len(errors)} span(s) not found in response text. Fix before merging.")
        sys.exit(1)

    print(f"\nAll spans validated. Merging into {ANNOTATIONS_FILE}...")
    added = merge(ANNOTATIONS_FILE, wave)
    print(f"  Added {added} new entries.")
    print("Done.")
