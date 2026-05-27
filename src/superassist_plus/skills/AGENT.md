# Skills Module Technical Documentation

IMPORTANT: Any change to skill discovery, prompt injection, virtual skill paths,
or loaded-skill persistence must update this document.

## Purpose

The `skills` module implements a lightweight DeerFlow lead-agent style skill
system. It exposes built-in skills from `skills/public` to the main agent as
metadata first, then allows the agent to load full `SKILL.md` instructions on
demand through the read-only `/mnt/skills` virtual path.

## Behavior

- Public skills live under `<project>/skills/public/<name>/SKILL.md`.
- The main prompt initially receives only each skill's name, description, and
  `/mnt/skills/.../SKILL.md` location.
- If the agent reads a skill's `SKILL.md`, that skill is marked loaded for the
  current thread and its full content is included in later system prompts for
  the same thread.
- Skills are read-only. File mutation tools must not write to `/mnt/skills`.

## Current Skills

- `deep-research`: systematic multi-angle web research workflow.
