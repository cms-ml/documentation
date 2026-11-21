---
description: How the CMS marketplace skills are validated, with results and plots from each validation analysis
---

# Validation

Every skill and agent in the [CMS marketplace](https://github.com/cms-ml/marketplace) is validated by an analysis that uses it, run by an agent, with the outputs committed to the repository under [`validation/`](https://github.com/cms-ml/marketplace/tree/master/validation). Each validation records:

- `PROMPT.md`: the task given to the agent
- `SKILLS_USED.md`: where the skills helped and where they fell short
- `README.md`: goal, results and how to reproduce
- `check_results.py`: checks on the committed `summary.json`, which CI reruns

What the agent finds goes back into the skill before the pull request is merged. The sections below show the results of each validation on `master`.

!!! warning "Open Data tests of tools"
    These analyses use CMS Open Data to test the skills. They are not CMS results and make no physics claim. Plots are labelled "CMS Open Data".

| Validation | Skills and agents tested | Result |
| --- | --- | --- |
| [b-tagged dijet search](#b-tagged-dijet-search-on-2016-open-data) | `cms-analysis-coder`, `trigger-efficiency`, `hepdata`, `analysis-spec-builder`, `awkward-array`, `vector-awkward`, `hist`, `lpc-condor` | Full analysis on 142 files: turn-on, fits, signal injection, HEPData record |
| [Comparison with CMS results](#comparison-with-published-cms-results) | `cms-results-validator` | 21 tests against 6 CMS HEPData records |
| [Muon tag and probe](#muon-tag-and-probe-scale-factors) | `muon-tag-and-probe` | Tight ID scale factors agree with the Muon POG within 0.1% |
| [NanoAOD skimming](#nanoaod-skimming-with-bookkeeping) | `nanoaod-skimming` | 33/33 checks, including a cutflow with N-1; Z peak data/MC 0.945 |
| [Batch jobs on cmslpc and lxplus](#batch-jobs-on-cmslpc-and-lxplus) | `lpc-condor`, `lxplus-access`, `lxplus-htcondor`, `eos-xrootd` | All jobs exit 0; event counts identical at both sites |

## b-tagged dijet search on 2016 Open Data

[:fontawesome-brands-github: validation/dijet-2016](https://github.com/cms-ml/marketplace/tree/master/validation/dijet-2016)

The `cms-analysis-coder` agent implemented a b-tagged dijet resonance search from a [specification](https://github.com/cms-ml/marketplace/blob/master/validation/dijet-2016/specification.md) on 2016G+H JetHT NanoAOD (16.4 fb^-1^, 142 files, run on cmslpc condor). It follows CMS-EXO-20-008 and the Run 3 analysis by A. Mankel (Texas Tech University) with the CMS B2G group. The agent's notes from this run were used to write the `trigger-efficiency` and `hepdata` skills. A second run redid those steps following the new skills and reproduced every number.

| Quantity | Value |
| --- | --- |
| Trigger plateau (reference `HLT_PFJet260`) | 0.99928 (+0.00012 −0.00015) above m~jj~ = 1058 GeV |
| Events with ≥1 / 2 b tags | 878 757 / 72 688 |
| Background fit, 2 b tags | 4 parameters, χ²/ndf = 0.83 |
| Z′ (2 TeV) injection, 200 toys | bias −1.5%, pull mean −0.04, width 1.12 |

<figure markdown>
![Trigger efficiency versus dijet mass](../images/marketplace/validation/dijet_trigger_efficiency.png){ width="560" }
<figcaption markdown>**`trigger-efficiency`**: turn-on of the signal-trigger OR versus m~jj~ with the orthogonal reference method and Clopper-Pearson intervals. Two other reference triggers cross-check the result, and the offline threshold is placed where the efficiency reaches its plateau.</figcaption>
</figure>

<figure markdown>
![Dijet mass fit in the 2 b-tag category](../images/marketplace/validation/dijet_fit_2b.png){ width="560" }
<figcaption markdown>**`cms-analysis-coder`, `hist`**: m~jj~ spectrum with 2 b tags, fitted with the 4-parameter dijet function. The order was chosen with an F-test. Z′ → bb̄ signal shapes are overlaid. This spectrum and the fit results are in the validated [HEPData record](https://github.com/cms-ml/marketplace/tree/master/validation/dijet-2016/hepdata) written with the **`hepdata`** skill.</figcaption>
</figure>

<figure markdown>
![Signal injection test](../images/marketplace/validation/dijet_injection_2b.png)
<figcaption markdown>Signal injection into real data for a 2 TeV Z′ with 2 b tags. Left: recovered versus injected signal yield. Right: pulls from 200 toys.</figcaption>
</figure>

## Comparison with published CMS results

[:fontawesome-brands-github: validation/dijet-2016/cms-comparison](https://github.com/cms-ml/marketplace/tree/master/validation/dijet-2016/cms-comparison)

The `cms-results-validator` agent reviewed the dijet analysis above and the [dimuon example](https://github.com/cms-ml/marketplace/tree/master/examples/dimuon) against published CMS results on HEPData, in the style of an ARC review. It ran 21 tests, each with a comparability statement, a number and a verdict ([COMPARISON.md](https://github.com/cms-ml/marketplace/blob/master/validation/dijet-2016/cms-comparison/COMPARISON.md)).

| Test | CMS counterpart | Result | Verdict |
| --- | --- | --- | --- |
| DeepJet working points, 2016G+H luminosity | BTV POG JSON, Open Data record | exact match; 16.3927 vs 16.3934 fb^-1^ | consistent |
| m~jj~ shape, 2 b tags | CMS-EXO-20-008 | χ²/ndf = 22.3/23 | consistent |
| dσ/dm, 2 b tags | CMS-EXO-20-008 | ours/CMS = 0.968 ± 0.009 | consistent |
| dσ/dm, ≥1 b tag | CMS-EXO-20-008 | ratio 0.72–0.84 (category definitions differ) | inconsistent |
| Inclusive m~jj~ shape | CMS-EXO-16-056 | χ²/ndf = 43.2/34 | consistent |
| Υ(2S)/Υ(1S), Υ(3S)/Υ(1S) | CMS-BPH-15-005 | ours/CMS = 1.017 ± 0.045, 1.021 ± 0.053 | consistent |
| Z p~T~, \|y\| and mass shapes | CMS-SMP-17-010, SMP-17-001 | detector level vs unfolded | inconsistent, as expected |

The agent also found three errors in the metadata of published HEPData records. One example: the 2016 tables of the CMS-EXO-20-008 record say `LUMINOSITY: 138` but are normalised to 36.3 fb^-1^.

## Muon tag and probe scale factors

[:fontawesome-brands-github: validation/muon-tnp](https://github.com/cms-ml/marketplace/tree/master/validation/muon-tnp)

An agent used the `muon-tag-and-probe` skill to measure tight ID, tight PF isolation and IsoMu24 efficiencies and data/MC scale factors with Z → μμ tag and probe. It used 2016G+H SingleMuon data (3.55 fb^-1^, every 5th file) and DY MC. It compared bin by bin with the official Muon POG scale factors (`2016postVFP_UL/muon_Z.json.gz`).

| Working point | Weighted SF / POG SF, 30–60 GeV | Outcome |
| --- | --- | --- |
| Tight ID | 1.0008 | agrees with the POG; no bin above 3σ |
| Tight PF isolation | 1.0028 | 0.3% above the POG; confirmed by a fit-free counting check |
| IsoMu24 or IsoTkMu24 | 1.013 | 1% above the POG. Matching both muons to filter bit 2 only closes this for \|η\| > 0.9; the barrel offset is still open. |

<figure markdown>
![Scale factor ratio to CMS per eta bin](../images/marketplace/validation/tnp_sf_vs_cms_overview.png){ width="560" }
<figcaption markdown>Ratio of the measured scale factors to the CMS Muon POG values at 30–60 GeV, per probe |η| bin, for the three working points. Green shows IsoMu24 with the bit-2-only trigger matching.</figcaption>
</figure>

<figure markdown>
![Tight ID scale factors versus CMS](../images/marketplace/validation/tnp_sf_vs_cms_tightid.png)
<figcaption markdown>Tight ID scale factor versus probe p~T~ in four |η| bins, with the Muon POG result (red) and the ratio to it.</figcaption>
</figure>

<figure markdown>
![Tight ID scale factor ratio map](../images/marketplace/validation/tnp_sf_ratio_map_tightid.png)
<figcaption markdown>Tight ID, every (p~T~, |η|) bin: our scale factor divided by the POG one, with the pull in brackets. All bins are within 2.1σ. The largest deviations are in the statistically limited p~T~ > 120 GeV bins.</figcaption>
</figure>

<figure markdown>
![Tight isolation scale factor ratio map](../images/marketplace/validation/tnp_sf_ratio_map_tightiso.png)
<figcaption markdown>Tight PF isolation, same layout. Our scale factors are 2–4% above the POG below 30 GeV, rising with |η|. At 30–50 GeV the offset is only 0.2–1%, but the pulls reach 3–5σ because both measurements are very precise there.</figcaption>
</figure>

<figure markdown>
![Pass and fail fits](../images/marketplace/validation/tnp_fit_tightid.png)
<figcaption markdown>Pass and fail m~μμ~ fits (signal plus exponential background) for the tight ID in one bin, |η| < 0.9 and 40 < p~T~ < 50 GeV.</figcaption>
</figure>

The validation also corrected the skill. For example, it did not reproduce the skill's quoted IsoMu24 difference between 2016G and H, and the skill now says so.

## NanoAOD skimming with bookkeeping

[:fontawesome-brands-github: validation/nanoaod-skimming](https://github.com/cms-ml/marketplace/tree/master/validation/nanoaod-skimming)

A `cms-analysis-coder` agent running on Claude Haiku used the `nanoaod-skimming` scripts to:

- skim 3 SingleMuon 2016G files with the golden JSON, and 3 DY files
- merge the outputs
- compute Σ genEventSumw and the luminosity
- build the Z → μμ peak

The Z peak is the end-to-end test: the DY prediction is only right if the luminosity, the sum of weights and both skims are right. `check_results.py` passes 33 of 33 checks.

| Quantity | Value |
| --- | --- |
| Data skim | 645 k of 9.46 M events, 2.0% of the input size, 490.4 pb^-1^ |
| DY skim | 386 k of 2.86 M events, 2.7% of the input size, 15.8% negative weights |
| Σ genEventSumw | identical in the original files and the merged skim |
| Full 2016 golden luminosity | 36.31 fb^-1^ |
| Z peak (81–101 GeV), data/DY | 0.945, with no scale factors applied |

<figure markdown>
![Dimuon mass data versus DY](../images/marketplace/validation/skim_mmumu.png){ width="480" }
<figcaption markdown>m~μμ~ from the skims, with DY normalised from the computed luminosity and Σ genEventSumw. The 5% deficit is the expected size of the muon and pileup corrections that are not applied.</figcaption>
</figure>

Every skim also records its own cutflow: a `Cutflow` tree with the sequential and N-1 counts of each preselection step, and Σ genWeight and Σ genWeight² for MC. It survives `hadd` like the Runs tree. The validation checks that the cutflow is identical in the per-file and merged skims, then continues it through the analysis cuts to the Z peak.

| Step | Data | Data N-1 | DY (normalised) | Data/DY |
| --- | --- | --- | --- | --- |
| all | 9 457 915 | | 2 980 197 | |
| trigger (skim) | 6 245 012 | 952 534 | 645 441 | 9.68 |
| two muons p~T~ > 20 GeV (skim) | 645 024 | 6 245 012 | 408 893 | 1.58 |
| muon tight ID | 375 867 | 290 315 | 378 594 | 0.993 |
| muon isolation | 332 107 | 307 560 | 339 920 | 0.977 |
| leading muon p~T~ > 26 GeV | 327 114 | 283 999 | 335 725 | 0.974 |
| 81 < m~μμ~ < 101 GeV | 281 527 | 327 114 | 297 829 | 0.945 |

<figure markdown>
![Cutflow data versus DY](../images/marketplace/validation/skim_cutflow.png){ width="520" }
<figcaption markdown>Cutflow from the skims' `Cutflow` trees and the analysis steps, data vs normalised DY. Data/DY is far above 1 until the tight ID because the SingleMuon data also contain W, top and QCD events with non-prompt muons, while the MC is DY only. The full table, with every step's N-1, is in the [validation README](https://github.com/cms-ml/marketplace/tree/master/validation/nanoaod-skimming#cutflow-run-2).</figcaption>
</figure>

## Batch jobs on cmslpc and lxplus

[:fontawesome-brands-github: validation/lpc-condor](https://github.com/cms-ml/marketplace/tree/master/validation/lpc-condor) · [:fontawesome-brands-github: validation/lxplus-condor](https://github.com/cms-ml/marketplace/tree/master/validation/lxplus-condor)

Both validations run the same task on 2016G JetHT NanoAOD Open Data:

1. split the input files into jobs
2. run one test job
3. submit the rest
4. stage the outputs to EOS
5. check the outputs and resubmit any failures

Each agent had no context except the skill files. The task is the same at both sites, so the event counts can be compared directly.

| Site | Skills | Agent | Jobs | Totals (160 k events) |
| --- | --- | --- | --- | --- |
| cmslpc (LPC condor) | `lpc-condor` | Claude Sonnet, two runs | all exit 0, no resubmissions | `HLT_PFHT900` 21 383; ≥2 jets with p~T~ > 500 GeV: 2 987 |
| lxplus (lxbatch) | `lxplus-access`, `lxplus-htcondor`, `eos-xrootd` | Claude Haiku, driving lxplus over ssh from cmslpc | all exit 0; outputs checked by size and adler32 | identical to cmslpc |

Each run found problems that were fixed in the skills. For example, a `$(VAR:default)` in a `.jdl` is not expanded when the default contains `-`, so jobs exit 0 but write nothing. On lxbatch, small memory requests are raised to 3000 MB.

## Add a validation

A pull request that adds a skill or agent includes its validation in `validation/<name>/`. See the [contributing guide](contributing.md) and [CONTRIBUTING.md](https://github.com/cms-ml/marketplace/blob/master/CONTRIBUTING.md#7-the-validation) in the marketplace.
