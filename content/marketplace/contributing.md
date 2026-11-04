---
description: How to add or improve skills, agents and plugins in the CMS skills marketplace
---

# Contributing to the marketplace

<p class="cms-lead">New skills, agents and plugins are welcome, and so are fixes to the ones already there. This page summarizes how to contribute. The full guide is <a href="https://github.com/cms-ml/marketplace/blob/master/CONTRIBUTING.md">CONTRIBUTING.md</a> in the marketplace repository.</p>

<div class="cms-actions" markdown>
[:fontawesome-brands-github: Open the repository](https://github.com/cms-ml/marketplace){ .md-button .md-button--primary }
[:material-book-open-variant: Full contributing guide](https://github.com/cms-ml/marketplace/blob/master/CONTRIBUTING.md){ .md-button }
</div>

!!! info "Every pull request comes with a validation"
    A new or changed skill or agent comes with a **validation**: an analysis, or part of one, that uses it on real inputs, run by an agent, with its outputs committed under `validation/<name>/`. See [The validation](#the-validation) below.

## Ways to help

You don't have to write a new skill to contribute.

Report where a skill fell short
:   Run the [dimuon example prompt](https://github.com/cms-ml/marketplace/blob/master/examples/dimuon/PROMPT.md). It ends by writing `SKILLS_USED.md`, which notes where each skill helped or was wrong. Bring that file back as an [issue](https://github.com/cms-ml/marketplace/issues) or pull request.

Fix an existing skill
:   Add a gotcha you ran into (the error message plus the fix), correct a command, or update a version that has moved on.

Write a new skill
:   If you keep pasting the same instructions into a chat, that's a skill. Most contributions are a new skill in an existing plugin.

### Turn a page of these docs into a skill

A `cms-ml` plugin is planned for ML workflows: weaver training, and ONNX, TensorFlow and SONIC inference in CMSSW. Much of that knowledge is already written up on this site, for example [ONNX inference](../inference/onnx.md), [Sonic/Triton](../inference/sonic_triton.md) and the [integration checklist](../inference/checklist.md). If you maintain one of those pages, or have used those tools in an analysis, consider turning that knowledge into a tested skill.

## Which piece should I write?

| Piece | What it is | Write one when |
| --- | --- | --- |
| Skill | Reusable instructions and reference material, loaded only when a request matches | You keep explaining the same workflow or library to the agent |
| Subagent | A specialist with its own context window and instructions | A task needs a specialist role, or would flood the conversation with output |
| Hook | A script that runs on an event, such as after every edit | Something must happen every time, without asking |
| MCP server | A connection to an external service | The agent needs data from a system it can't see |
| Plugin | A package of the above | A distinct audience, such as one facility, needs its own set |

Create a new plugin only for a distinct audience. For example, `af-fnal-lpc` is only useful on the LPC. Otherwise, add to `cms`.

## Making a pull request

```bash
# 1. Clone and branch (or fork cms-ml/marketplace and branch in your fork)
git clone https://github.com/cms-ml/marketplace && cd marketplace
git switch -c feat/<name>

# 2. Write the skill or agent, and test every command on cmslpc or lxplus

# 3. Try it locally with your working copy of the plugin
claude --plugin-dir plugins/cms

# 4. Validate it with an agent, in validation/<name>/

# 5. Check, regenerate the catalogue, and bump the plugin version
python3 scripts/marketplace.py check
python3 scripts/marketplace.py sync
python3 scripts/marketplace.py bump cms <new-version>

# 6. One commit per skill or agent, then the validation (Conventional Commits)
git commit -m "feat(cms): add <name> skill"
git commit -m "test(cms): add <name> validation"
git push -u origin feat/<name>
gh pr create --repo cms-ml/marketplace --base master
```

The pull request template opens with a checklist.

## Writing a skill

```text
plugins/<plugin>/skills/<name>/
  SKILL.md              # required; aim for 300 lines or fewer (500 is the hard limit)
  references/*.md       # deeper content, loaded only when needed
  scripts/*             # tools the agent runs
  assets/*              # templates the agent copies
```

- **Frontmatter:** only `name`, which matches the directory name, and `description`, plus an optional `metadata: summary:` for the README. The description starts with **"Use when"** and lists the situations and error messages that should trigger the skill. The agent chooses skills by this text alone, so don't summarize the content.
- **Sections**, in this order: Overview, When to Use, Key Concepts, Canonical Patterns, **Gotchas**, Reference Files, **Docs** (last).
- **Test every command** before it goes in, and say what you tested it with (release, LCG view, versions). Gotchas come from real failures: the error message plus the fix.
- **Link every file** in `references/`, `scripts/` and `assets/` from `SKILL.md`. The checker fails on unlinked files.
- **Cite upstream docs** in `## Docs`. Credit adapted material in `CREDITS.md`.
- **Don't hard-code a CMSSW release as the answer.** Versions in examples are "tested with" versions; the user chooses the release.

Good examples to copy from are [`trigger-efficiency`](https://github.com/cms-ml/marketplace/tree/master/plugins/cms/skills/trigger-efficiency) (scripts plus a worked example) and [`hepdata`](https://github.com/cms-ml/marketplace/tree/master/plugins/cms/skills/hepdata) (a template that tests itself).

## Writing a subagent

A subagent is a Markdown file in `plugins/<plugin>/agents/<name>.md`. Its frontmatter has a `name`, a `description` that starts with "Use when" and says when to hand work to it, the `tools` it needs (include `Skill` so it can load the marketplace skills), and a `model`. The body is its system prompt: a role, how it works, what it never does (such as fabricating numbers), and what it hands back.

See [`cms-analysis-coder`](https://github.com/cms-ml/marketplace/blob/master/plugins/cms/agents/cms-analysis-coder.md) for an example.

## The validation

```text
validation/<name>/
  README.md          # goal, references, results, how to reproduce
  PROMPT.md          # the exact task given to the agent
  specification.md   # for analyses: written by the agent with analysis-spec-builder
  <code>             # re-runnable, relative paths, no usernames
  plots/ summary.json   # outputs (small files only; skims stay on EOS)
  SKILLS_USED.md     # which skill guided each step, and where it was wrong or missing
  check_results.py   # physics sanity checks on the outputs, run by CI
```

1. Write `PROMPT.md`: the data, the deliverables, and limits on files, jobs and memory.
2. Run it with an agent. If you can, run it once **before** writing the skill: whatever the agent had to work out alone is what the skill should say.
3. Read `SKILLS_USED.md`, fix the skill, and rerun the steps it covers. Record that the results agree.
4. Write `check_results.py` to test physics sense (efficiencies in [0, 1], fit quality, no NaN), not exact numbers.

[validation/dijet-2016](https://github.com/cms-ml/marketplace/tree/master/validation/dijet-2016) is the reference example, and [validation/lpc-condor](https://github.com/cms-ml/marketplace/tree/master/validation/lpc-condor) is a smaller one.

## Before you open the pull request

- [ ] `python3 scripts/marketplace.py check` passes, and you ran `sync` after any frontmatter or `plugin.json` change.
- [ ] The plugin version is bumped. Installed copies only update on a new version.
- [ ] Prettier and pre-commit pass.
- [ ] No personal paths or usernames (write `$USER` or `<user>`), and no CMS-internal content. Link internal twikis instead of copying them.
- [ ] Open Data results are labelled "CMS Open Data" or "Private work".
- [ ] Large inputs are not committed. The validation README says where they live and how to rebuild them.

## Learn more

- [Claude Code skills](https://code.claude.com/docs/en/skills), [subagents](https://code.claude.com/docs/en/sub-agents) and [plugins](https://code.claude.com/docs/en/plugins)
- [Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)
- [Agent Skills specification](https://agentskills.io/specification) and [skill best practices](https://agentskills.io/skill-creation/best-practices)
- [Marketplace authoring rules (CLAUDE.md)](https://github.com/cms-ml/marketplace/blob/master/CLAUDE.md), which the checker enforces
