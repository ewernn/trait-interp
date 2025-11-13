# Documentation Update Guidelines

## Core Principles
- **Delete first, add second** - remove outdated content before adding new
- **Present tense only** - document what IS, not what WAS
- **Concise and actionable** - every sentence should help the reader DO something
- **Zero history** - no changelogs, migration notes, or "previously" references
- **YAGNI for docs** - delete unused files and sections immediately; they create confusion

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

## Related Documentation
When updating one doc, check these for consistency:
- `main.md` - if feature names or structure changed
- `README.md` - if quick start or overview affected
- Specialized docs (CONTAMINATION_RESULTS.md, PERTOKEN_EXPERIMENTS.md, etc.) - if experiments or results changed

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

## The Final Rule

**When in doubt, delete.**

Wrong documentation is worse than no documentation.
