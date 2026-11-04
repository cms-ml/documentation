---
description: Claude Code plugins for CMS physics analysis, maintained by the CMS ML group
---

# CMS skills marketplace

<p class="cms-lead">The <a href="https://github.com/cms-ml/marketplace">CMS marketplace</a> is a collection of plugins that teach AI coding agents how CMS analysis work is done. Each skill is grounded in commands tested on cmslpc against real CMS releases and data (CMS Open Data where possible), in both CMSSW and plain pip environments.</p>

<div class="cms-actions" markdown>
[:fontawesome-brands-github: Open the marketplace](https://github.com/cms-ml/marketplace){ .md-button .md-button--primary }
[:material-source-pull: How to contribute](contributing.md){ .md-button }
</div>

The marketplace is built for [Claude Code](https://code.claude.com/docs/en/overview). The skills follow the open [Agent Skills specification](https://agentskills.io/specification), so other agents can use them too.

## How it works

The marketplace is made of three kinds of pieces:

- **Skill**: a `SKILL.md` file, with optional reference files and scripts, on one topic, for example making CMS-style histograms with `hist` and `mplhep`. The agent only reads a skill's short description until a request matches it, and then loads the rest.
- **Subagent**: a specialist with its own instructions and context, for example `cms-analysis-coder`, which implements an analysis specification end to end.
- **Plugin**: an installable bundle of skills and subagents, such as `cms`. Installed skills are namespaced by plugin, for example `cms:hist`.

You don't call skills by name. Describe the physics task, and the agent picks up the skills whose descriptions match. Mentioning a skill ("use the analysis-spec-builder skill…") makes sure it is used.

## What's inside

!!! info "Current as of November 2026"
    New skills are added as pull requests are merged. The [marketplace README](https://github.com/cms-ml/marketplace#plugins) always has the current list.

=== "cms"

    Core CMS analysis: CMSSW releases and architectures, OS containers, EDM modules, `cmsRun` configuration, EDM file inspection, and the Scikit-HEP stack on NanoAOD, plus analysis specifications.

    | Skill | Use it for |
    | --- | --- |
    | `analysis-spec-builder` | Turning an analysis request into a structured `specification.md` (datasets, cuts, histograms, tools) |
    | `awkward-array` | Jagged arrays with Awkward 2 on CMS NanoAOD (records, selection, combinatorics, indices, I/O) |
    | `cmssw` | Release areas, architectures and containers, EDM modules, `cmsRun` configs, EDM file tools |
    | `hepdata` | HEPData records with `hepdata_lib`, validated offline, following CMS requirements |
    | `hist` | Histograms with `hist` and `mplhep` for CMS (weights, normalisation, UHI, CMS-style plots) |
    | `trigger-efficiency` | Trigger turn-ons and offline thresholds from NanoAOD (reference method, Clopper-Pearson, plateau rule) |
    | `vector-awkward` | Four-vector kinematics on NanoAOD with `vector` and Awkward (masses, ΔR, m~T~, boosts) |

    | Subagent | Role |
    | --- | --- |
    | `cms-analysis-coder` | Implements an analysis specification using the marketplace skills |
    | `cms-results-validator` | Reviews an analysis against published CMS results on HEPData, in the style of an ARC review |

=== "af-fnal-lpc"

    Skills for the CMS LPC analysis facility at Fermilab (cmslpc).

    | Skill | Use it for |
    | --- | --- |
    | `lpc-condor` | HTCondor batch jobs on cmslpc: inputs over XRootD, outputs to EOS, checking and resubmitting failed jobs |

=== "hep-python-tools"

    Experiment-neutral Python tooling for HEP work, with notes for cmslpc.

    | Skill | Use it for |
    | --- | --- |
    | `cli-creator` | Typer command-line tools wired into `pyproject.toml` console scripts |
    | `standalone-script` | Single-file PEP 723 scripts run by `uv`, which carry their own dependencies |

A `cms-ml` plugin for ML workflows, covering weaver training and ONNX, TensorFlow and SONIC inference in CMSSW, is planned. See [Contributing](contributing.md#turn-a-page-of-these-docs-into-a-skill) if you'd like to help write it.

## Install

=== "In Claude Code"

    Type these in the Claude Code prompt. They are slash commands, not shell commands.

    ```text
    /plugin marketplace add cms-ml/marketplace
    /plugin install cms@cms-marketplace
    /plugin install hep-python-tools@cms-marketplace
    /plugin install af-fnal-lpc@cms-marketplace
    ```

=== "From a terminal"

    ```bash
    claude plugin marketplace add cms-ml/marketplace
    claude plugin install cms@cms-marketplace
    claude plugin install hep-python-tools@cms-marketplace
    claude plugin install af-fnal-lpc@cms-marketplace   # LPC users: condor batch jobs
    claude plugin list                                  # each should show "enabled"
    ```

=== "Other agents"

    The skills follow the [Agent Skills specification](https://agentskills.io/specification), so you can link a plugin's `skills/` directory into another agent's skill path. From a clone of the marketplace:

    ```bash
    git clone https://github.com/cms-ml/marketplace && cd marketplace
    ln -s $PWD/plugins/cms/skills ~/.agents/skills/cms
    ```

After installing or updating, **start a new conversation**: skills are loaded when a conversation starts. `/skills` lists what is loaded. In VS Code, run "Developer: Reload Window" if a new chat still doesn't see them.

### Updating

When new skills or plugins are merged, refresh the marketplace and install any new plugins:

```bash
claude plugin marketplace update cms-marketplace
claude plugin install <new-plugin>@cms-marketplace
```

### One-time setup on cmslpc

??? tip "Keep pip packages from shadowing CMSSW"
    Packages in `~/.local` from `pip install --user` break CMSSW's Python. Add this to your shell setup:

    ```bash
    export PYTHONNOUSERSITE=1
    ```

??? tip "Install uv in nobackup, for standalone scripts"
    This keeps `uv` and its cache out of the 3 GB home quota:

    ```bash
    export UV_INSTALL_DIR=~/nobackup/.local/bin
    curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
    cat >> ~/.bashrc <<'EOF'
    export PATH=~/nobackup/.local/bin:$PATH
    export UV_CACHE_DIR=~/nobackup/.cache/uv
    export UV_PYTHON_INSTALL_DIR=~/nobackup/.local/share/uv/python
    EOF
    ```

??? tip "Using Claude Code from VS Code on cmslpc"
    The VS Code extension ships its own `claude` binary, which isn't on `PATH`. Add an alias for it:

    ```bash
    alias claude="$(ls -d ~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude | tail -1)"
    ```

## Writing good requests

Good requests name:

- the **data**: DAS name or Open Data path, era, and the number of files to start with
- the **output**: working directory, plot directory and format
- the **deliverables**: specification, script, plots, notes

| Example request | Skills it uses |
| --- | --- |
| "Plot the dimuon mass in this NanoAOD file with CMS style." | `awkward-array`, `vector-awkward`, `hist` |
| "Write a spec for a Z→μμ data/MC comparison in 2016H." | `analysis-spec-builder` |
| "Set up CMSSW_14_0_15 in ~/nobackup/work and write an EDAnalyzer that histograms muon p~T~ from MiniAOD." | `cmssw` |
| "Split these 142 Open Data files into 10 condor jobs, write the outputs to my EOS area, and resubmit any that fail." | `lpc-condor` |
| "Turn the spectra and efficiencies in summary.json into a HEPData record in hepdata/, with the plots attached, and validate it." | `hepdata` |
| "Make this check a single-file script I can send to a colleague." | `standalone-script` |
| "Use the cms-analysis-coder agent to implement specification.md: start on 2 files, then run everything." | `cms-analysis-coder` agent |

For a complete, ready-to-paste example, try the [dimuon resonances prompt](https://github.com/cms-ml/marketplace/blob/master/examples/dimuon/PROMPT.md). It exercises five skills at once on 2016H DoubleMuon NanoAOD and needs no grid proxy. It ends by writing `SKILLS_USED.md`, a record of which skill guided each step and where a skill fell short.

## How skills are validated

Every skill and agent in the marketplace is validated by an analysis that uses it, run by an agent, with the outputs committed to the repository. Each validation records the prompt given to the agent and a `SKILLS_USED.md` noting where the skills helped and where they fell short. The reference example is [validation/dijet-2016](https://github.com/cms-ml/marketplace/tree/master/validation/dijet-2016), a b-tagged dijet search on 2016 Open Data.

## Get involved

The marketplace grows through contributions from CMS members. If a skill gave you a wrong answer, open an [issue](https://github.com/cms-ml/marketplace/issues). If you have expertise worth sharing, [write a skill](contributing.md).

<div class="cms-actions" markdown>
[:material-source-pull: Contributing guide](contributing.md){ .md-button .md-button--primary }
[:material-alert-circle-outline: Report a problem](https://github.com/cms-ml/marketplace/issues){ .md-button }
</div>
