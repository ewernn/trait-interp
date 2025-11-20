@docs/main.md

## Documentation Practice: Integrate, Don't Accumulate

### The Problem
Creating session-specific status files (PHASE2_COMPLETE.md, MIGRATION_GUIDE.md, TODO_SESSION.md) that become stale documentation debt.

### The Pattern

**During Work:**
- Track progress in conversation only
- Update permanent docs AS YOU GO when you learn something worth preserving
- Use `TEMP_` prefix only for essential session handoffs

**After Completion:**
- Provide comprehensive summary IN CONVERSATION
- Verify permanent docs already updated (did it during work)
- Delete any TEMP_ files immediately

**File Decision Tree:**
```
Creating a new .md file? Ask:

→ Is this permanent documentation that future users need?
   YES → Add to docs/ and list in docs/main.md Documentation Index
   NO → Keep in conversation

→ Does similar doc already exist?
   YES → Update that doc instead
   NO → Only create if truly needed for future users
```

### Examples

❌ **Anti-pattern:**
```
PHASE2_COMPLETE.md       (comprehensive status)
PHASE2_STATUS.md         (mid-progress)
REFACTOR_TODO.md         (checklist)
MIGRATION_GUIDE.md       (how to update)
→ Result: 4 stale files, duplicate info, confusion
```

✅ **Correct pattern:**
```
During: Update docs/architecture.md with structure changes
During: Update docs/main.md with new examples
End: Comprehensive summary in conversation
→ Result: Single source of truth, no cleanup needed
```

### Red Flags
- File names with dates, phases, versions (PHASE2_, NOV20_, V2_)
- "Status" or "Progress" in filename
- Files duplicating existing doc sections
- More than 2 TEMP_ files in repo root

### What You Still Get
- ✅ Detailed summaries of work (in conversation - copy if needed)
- ✅ Clear documentation of changes (in permanent docs)
- ✅ Testing results and verification (in conversation)
- ✅ Comprehensive overviews (in conversation)

### What You Avoid
- ❌ Temporary files cluttering repo
- ❌ Manual cleanup tasks
- ❌ Duplicate/conflicting documentation
- ❌ Stale "status" files

### Integration Happens During Work
Update relevant docs incrementally as you work:
- Fixed structure? → Update architecture.md right then
- New pattern discovered? → Add to main.md immediately
- Don't "save for later" → Do it as you go

Summary in conversation, insights in permanent docs.
