# Documentation Update Guidelines

## Core Principles

1. **Docs reflect reality** — Update docs when code changes, not after
2. **Single source of truth** — Each concept documented in one place, linked elsewhere
3. **Discoverable** — New users can find what they need from main.md
4. **Maintainable** — Prefer concise docs over comprehensive ones

## Code Change Principles

1. **Reuse core/ and utils/** — Always check for existing primitives before writing new code. `core/` has hooks, methods, math; `utils/` has paths, model loading, vectors.
2. **Read before writing** — Read all related files to understand existing patterns before implementing changes. Bias towards reading more, not less.
3. **No duplication** — If functionality exists, use it. If it's close, extend it. Don't create parallel implementations.

## When to Update

- **Code changes**: Update relevant docs in same commit
- **New features**: Add to appropriate existing doc or create new one
- **Bug fixes**: Update if docs described incorrect behavior
- **Refactors**: Update paths, module names, API signatures

## What NOT to Do

- Don't duplicate content across docs (link instead)
- Don't add "planned" or "TODO" sections that go stale
- Don't document internal implementation details that change frequently
- Don't create new docs when existing ones can be extended

## File Placement

| Content Type | Location |
|--------------|----------|
| Codebase navigation | `docs/` |
| How-to guides | `docs/` |
| Technical references | `docs/` |
| Research notes | `docs/other/` |
| Personal writing | `docs/other/` |
| Application materials | `docs/other/` |

## Style

- Start with one-line summary
- Use tables for structured data
- Use code blocks for commands/examples
- Keep sections short (prefer many small sections over few large ones)
- Link to other docs rather than duplicating content
