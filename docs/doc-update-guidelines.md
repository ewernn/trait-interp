# Documentation Update Guidelines

## Core Principles
- **Delete first, add second** - remove outdated content before adding new
- **Present tense only** - document what IS, not what WAS
- **Concise and actionable** - every sentence should help the reader DO something
- **Zero history** - no changelogs, migration notes, or "previously" references
- **YAGNI for docs** - delete unused files and sections immediately; they create confusion
- **All docs in main.md** - every documentation file MUST be listed in docs/main.md Documentation Index
- **Verifiable claims** - every numeric claim should include a one-liner to verify it

## Before You Update
Ask yourself:
1. **Why this doc?** - What specific problem are you solving?
2. **What else breaks?** - Which other docs reference this one?
3. **Is it salvageable?** - Should you rewrite instead of patch?

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
   - Library docs → `traitlens/` or `traitlens/docs/`
   - Experiment-specific → `experiments/{name}/`
   - Pipeline docs → `extraction/`

2. **Add to docs/main.md Documentation Index** in the appropriate section:
   - Core Documentation (Start Here)
   - Trait Design & Creation
   - Pipeline & Extraction
   - Experiments & Analysis
   - Research & Methodology
   - Visualization & Monitoring
   - Infrastructure & Setup
   - traitlens Library
   - Meta-Documentation

3. **Format the entry** consistently:
   ```markdown
   - **[Title](relative/path/to/file.md)** - Brief description of purpose
   ```

4. **Verify the link works** from docs/main.md

**If you don't add it to main.md, it doesn't exist.** Orphaned documentation creates confusion.

## Related Documentation
When updating one doc, check these for consistency:
- `main.md` - if feature names or structure changed (ALWAYS update Documentation Index if adding/removing docs)
- `README.md` - if quick start or overview affected
- Experiment READMEs - if experiments or results changed

## Red Flags for Full Rewrite
- More than 50% needs deletion
- Core purpose has changed
- Structure doesn't match current code
- Multiple broken examples

Remember: Good documentation is like good code - it's not done when there's nothing left to add, but when there's nothing left to remove.

## Before Any Update, Ask:
1. **Does this help someone DO something?** (If no, delete)
2. **Is this true RIGHT NOW?** (If no, update or delete)
3. **Will this be true in a month?** (If no, reconsider adding)

## Anti-Patterns to Avoid

### The Museum
```markdown
❌ "Previously we used Redis, but now we use..."
❌ "In version 1.0, this worked differently..."
❌ "Legacy users might still see..."
```

### The Promise
```markdown
❌ "Coming soon: train-time steering"
❌ "TODO: Add visualization dashboard"
❌ "Will be deprecated in next version"
```

### The Orphan
```markdown
❌ Links to deleted documentation
❌ References to removed features
❌ Examples using old APIs
```

### The Session Status File
```markdown
❌ PHASE2_COMPLETE.md (session-specific status)
❌ MIGRATION_TODO.md (temporary checklist)
❌ REFACTOR_GUIDE.md (one-time instructions)
```

**Why this is bad:**
- Becomes stale immediately after session ends
- Duplicates info that should be in permanent docs
- Creates cleanup debt for later

**Instead:** Provide comprehensive summaries in conversation, integrate insights into permanent docs during work.

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
ls experiments/gemma_2b_cognitive_nov20/refusal/extraction/vectors/*.pt | wc -l
```
```

### Why This Matters

1. **Catches drift**: Code changes, docs stay stale
2. **Builds trust**: Reader can verify claims themselves
3. **Prevents bugs**: Off-by-one errors surface immediately
4. **Documents assumptions**: Makes implicit knowledge explicit

If you can't write a verification command, the claim might be too vague.

## The Final Rule

**When in doubt, delete.**

Wrong documentation is worse than no documentation.
