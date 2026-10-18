<!-- BEGIN DEV-MEMORY -->
## Project Memory

This repository uses an external Obsidian vault, The Vault, for persistent developer memory. Project slug: `cardiosentry-research`.

- **Vault path (`<vault>`):** `$THE_VAULT_PATH`, else the untracked `AGENTS.local.md` in this directory, else ask the user. Verify `<vault>/00-system/memory-policy.md` and `<vault>/01-projects/` exist; never guess.
- **Project memory:** `<vault>/01-projects/cardiosentry-research/`
- **Memory policy:** `<vault>/00-system/memory-policy.md` (follow its Quick card, §0)

### Session Startup
1. Read this project's `index.md`, `CURRENT.md`, and `NOW.md` from the project memory directory.
2. Read `decisions/index.md` before architecture/design changes, and relevant ADRs only as needed.
3. Read relevant `architecture.md`, `gotchas.md`, and `learnings/` notes selectively.
4. Verify material claims against the current repository code and canonical docs; do not treat vault notes as more authoritative than the implementation.

### Memory Updates
- Follow `00-system/memory-policy.md` in the vault.
- Update `NOW.md` after meaningful progress or handoff.
- Update `CURRENT.md` only when implementation changes have been verified against code; include the revision when available.
- Maintain `decisions/index.md` as the accepted-decision registry; preserve historical ADRs and correctly scoped partial supersession.
- Keep session history in this project's `sessions/`; record significant verified discoveries in `learnings/` or `gotchas.md`.
- Never store secrets. Do not silently change global preferences or rules.

### Project Onboarding
- If project memory is missing or needs reconciliation, follow `<vault>/03-runbooks/onboard-existing-project.md`.
- Do not assume the vault is writable merely because its path is listed. Request the necessary filesystem access when unavailable.
<!-- END DEV-MEMORY -->
