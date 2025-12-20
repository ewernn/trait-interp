# Documentation Update Guidelines

## Core Principles
- **Delete first, add second** - remove outdated content before adding new
- **Present tense only** - document what IS, not what WAS
- **Concise and actionable** - every sentence should help the reader DO something
- **Zero history** - no changelogs, migration notes, or "previously" references
- **YAGNI for docs** - delete unused files and sections immediately; they create confusion
- **All docs in main.md** - every documentation file MUST be listed in docs/main.md Documentation Index
- **Verifiable claims** - every numeric claim should include a one-liner to verify it

## Verifiable Claims

Every numeric claim in documentation should be **immediately verifiable** with a one-line command.

### Examples

**Good:**
```markdown
**Gemma 2 2B IT**:
- Layers: 26 (transformer layers)
- Hidden dim: 2304

Verify:
```bash
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('google/gemma-2-2b-it'); print(f'Layers: {c.num_hidden_layers}, Hidden: {c.hidden_size}')"
```
```

**Bad:**
```markdown
❌ "Gemma 2B has 27 layers"  # No verification provided, wrong number
❌ "The model has many layers"  # Vague, unverifiable
❌ "Layer count varies by model"  # True but unhelpful
```

### What to Verify

Include verification commands for:
- **Model architecture** (layer counts, hidden dimensions)
- **Data shapes** (tensor dimensions in saved files)
- **File counts** (number of vectors, experiments, prompts)
- **Storage sizes** (actual disk usage)
- **Performance metrics** (speeds, accuracies from real runs)

### Format

Place verification commands immediately after the claim:

```markdown
Total vectors created: 4 methods × 26 layers = 104 files

Verify:
```bash
ls experiments/my_experiment/refusal/extraction/vectors/*.pt | wc -l
```
```

### Why This Matters

1. **Catches drift**: Code changes, docs stay stale
2. **Builds trust**: Reader can verify claims themselves
3. **Prevents bugs**: Off-by-one errors surface immediately
4. **Documents assumptions**: Makes implicit knowledge explicit

If you can't write a verification command, the claim might be too vague.

## Before Any Update

Ask yourself:
1. **Does this help someone DO something?** (If no, delete)
2. **Is this true RIGHT NOW?** (If no, update or delete)
3. **Why this doc?** - What specific problem are you solving?
4. **What else breaks?** - Which other docs reference this one?
5. **Is it salvageable?** - Should you rewrite instead of patch?

## Update Rules
1. **Delete anything that references:**
   - Old implementations
   - Fixed bugs
   - Previous versions
   - Historical context
   - Unused files or features

2. **Keep only:**
   - Current functionality
   - Active configuration
   - Working examples
   - Essential concepts

3. **File management:**
   - Delete entire unused documentation files
   - Remove orphaned sections that no longer apply
   - Clean up broken internal links to deleted content

4. **Test every:**
   - Code example
   - File path
   - Command
   - Link

## Creating New Documentation

**REQUIRED STEPS when adding a new .md file:**

1. **Create the file** in the appropriate location:
   - Core docs → `docs/`
   - traitlens docs → `docs/traitlens_reference.md`
   - Experiment-specific → `experiments/{name}/`
   - Pipeline docs → `extraction/`

2. **Add to docs/main.md Documentation Index** in the appropriate section:
   - See main.md for current section structure
   - Place it where similar docs live
   - Keep the index organized and logical

3. **Format the entry** consistently:
   ```markdown
   - **[Title](relative/path/to/file.md)** - Brief description of purpose
   ```

4. **Verify the link works** from docs/main.md

**If you don't add it to main.md, it doesn't exist.** Orphaned documentation creates confusion.

## Where Content Belongs

Check the Documentation Index in docs/main.md to find where new content fits:
- **Empirical findings** (accuracy, performance data) → research_findings.md or insights.md
- **Conceptual frameworks** (mental models, theory) → conceptual_framework.md or overview.md
- **Practical usage** (commands, examples) → main.md, extraction_pipeline.md, or README.md

## Related Documentation
When updating one doc, check these for consistency:
- Documentation Index in main.md (ALWAYS update if adding/removing docs)
- Other docs that reference the same concepts
- README if quick start or overview affected

## Red Flags for Full Rewrite
- More than 50% needs deletion
- Core purpose has changed
- Structure doesn't match current code
- Multiple broken examples

Remember: Good documentation is like good code - it's not done when there's nothing left to add, but when there's nothing left to remove.

## Anti-Patterns to Avoid

❌ **The Museum** - "Previously we used Redis...", "In version 1.0...", "Legacy users might still see..."

❌ **The Promise** - "Coming soon...", "TODO: Add feature...", "Will be deprecated in next version"

❌ **The Orphan** - Links to deleted documentation, references to removed features, examples using old APIs

## The Final Rule

**When in doubt, delete.**

Wrong documentation is worse than no documentation.
