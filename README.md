# Speech Level × Interpersonal Attitude in Korean Requests — experiment code (HW2)

Inference-only evaluation of `Qwen/Qwen2.5-1.5B-Instruct` on KLUE YNAT and KLUE NLI under a
2 (sentence-final ending: *haeyo* / *hae*) × 2 (attitude phrase: affiliative / rude) design.

## Environment
- Google Colab, T4 GPU (16 GB), FP16 inference, no weight updates.
- Python packages: `requirements.txt` (torch, transformers, accelerate, datasets, numpy, pandas, openpyxl).
- Model and dataset revisions are pinned in `config.json`. Exact library versions, GPU name, runtime and peak
  memory of each run are written to `results/meta_*.jsonl`.
- Random seed: `20261020` (item sampling, option rotation, rating-sheet order, bootstrap).
  Decoding is greedy, so inference itself is deterministic.

## Layout
| Path | Content |
|---|---|
| `config.json` | Protocol constants (sample sizes, gates, token limits, bootstrap settings) |
| `stimuli/candidates.json` | 12 candidate quartets and 6 control phrases before human validation |
| `stimuli/selected.json` | Stimuli used by the runner (4 quartets, 4 control phrases) |
| `ratings/` | Rating sheets, sentence key, selection report |
| `data/` | Fixed item samples (guid, text, gold label, option rotation). KLUE, CC BY-SA 4.0 |
| `kspeech/` | `data` sampling, `prompts`, `run` inference, `parse` answer parser, `analyze` statistics, `extra` post hoc analyses, `tables` LaTeX tables, `ratings` |
| `results/` | Saved outputs of the reported run (Colab T4, 2026-11-22): raw rows `*_score_*.jsonl`, `*_gen_*.jsonl`, `meta_*.jsonl`, `summary_*.json`, `extra_main.json`, `tables/` |
| `notebooks/colab_vscode.ipynb` | Self-contained notebook used for the reported run (VS Code + Colab kernel) |
| `notebooks/colab_vscode_executed.ipynb` | The same notebook as executed, with its outputs |
| `notebooks/colab_run.ipynb` | Variant for the Colab web UI |

## Reproducing the results
```bash
pip install -r requirements.txt
python -m kspeech.data                      # optional: rebuild data/*.jsonl from KLUE (same seed -> same items)
python -m kspeech.ratings select            # stimuli/selected.json from the three filled rating sheets
                                            # (ranked by naturalness, then contrast; at most 2 quartets per rude-phrase type)
python -m kspeech.run --phase pilot         # pilot on training-split items
python -m kspeech.analyze --phase pilot     # operational gates and throughput
python -m kspeech.run --phase main          # all main evaluations (resumable)
python -m kspeech.analyze --phase main      # results/summary_main.json and results/tables/*.tex
python -m kspeech.extra                     # post hoc analyses -> results/extra_main.json (standard library only)
python -m kspeech.tables                    # optional: re-render the tables from summary_main.json without recomputing
```

The archive also contains the saved outputs of the reported run in `results/` (raw evaluation rows, run metadata,
summaries, generated tables), so every analysis command below can be re-run without a GPU and reproduces the
reported numbers exactly. Only `kspeech.run` needs a GPU.

| Paper item | Command | Where the numbers are |
|---|---|---|
| Table 1 (selected quartets and ratings), Table 9 (all candidates) | `python -m kspeech.ratings select` | `ratings/report.json`, `stimuli/candidates.json`, `stimuli/selected.json` |
| Table 2 (accuracy by condition, interaction D, Holm p) | `python -m kspeech.analyze --phase main` | `results/tables/main.tex`, `results/summary_main.json` (`acc`, `acc_neutral`, `D`) |
| Table 3 (ending, attitude, neutral and single-scale contrasts) | `analyze --phase main`, then `python -m kspeech.extra` | `results/summary_main.json` (`ending_effect`, `attitude_effect`, `attitude_main_effect`), `results/extra_main.json` (`vs_neutral`, `single_scale`) |
| Table 4 (per-quartet accuracy and D) | `python -m kspeech.analyze --phase main` | `results/tables/quartets.tex` |
| Figure 2 (per-quartet attitude effect) | `python -m kspeech.extra` | `results/extra_main.json` (`attitude_by_quartet`) |
| Figure 3 (predicted-class shares on the control subset) | `python -m kspeech.extra` | `results/extra_main.json` (`pred_share_control_subset`) |
| Table 5 (per-quartet prediction shift) | `python -m kspeech.extra` | `results/extra_main.json` (`pred_share_by_quartet`, `recall_by_quartet`, `pred_share_all_items`, `recall_all_items`) |
| Table 6 (prediction transitions) | `python -m kspeech.analyze --phase main` | `results/tables/transitions.tex` |
| Table 7 (generation audit) | `python -m kspeech.analyze --phase main` | `results/tables/generation.tex` |
| Table 8 (seeded random examples) | `python -m kspeech.extra` | `results/extra_main.json` (`sampled_examples`) |
| Table 10 (planned control comparison) | `python -m kspeech.analyze --phase main` | `results/tables/controls.tex` |
| Table 11 (prediction-share differences with CIs) | `python -m kspeech.extra` | `results/extra_main.json` (`share_shift_R_minus_P_all_items`, `share_shift_control_subset`, `accuracy_control_subset_quartet_mean`) |
| Table 12 (runs, time, memory; pilot gates) | `python -m kspeech.analyze --phase pilot` | `results/meta_*.jsonl`, `results/summary_pilot.json` |
| Section 6.4 and Appendix B (strict-format contrast, non-strict outputs by option letter, scoring/generation disagreements) | `python -m kspeech.extra` | `results/extra_main.json` (`strict_format_R_minus_P`, `non_strict_outputs`, `score_vs_generation`) |

The manuscript typesets these numbers in its own table layout; `results/tables/*.tex` hold the same values.
`kspeech.analyze` uses NumPy for the bootstrap and `kspeech.extra` uses Python's `random`, both with the seed above.

`kspeech.run` appends one JSON line per evaluation and skips finished keys, so an interrupted session can be
resumed with the same command. No model checkpoints are stored in this archive.
