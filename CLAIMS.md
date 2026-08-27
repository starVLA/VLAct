# VLAct project-page claim audit

This file records the source and scope of every headline claim used by the project page. The active manuscript is `StarVLA.tex`, which includes files under `RAW/sections/`.

| Page claim | Scope | Manuscript source |
|---|---|---|
| 82.6% on LIBERO-Plus | Official `Total`; not the arithmetic mean of the seven rounded axis values | `RAW/sections/4_experiments.tex`, LIBERO-Plus paragraph and table |
| +7.6 percentage points over Qwen3VL-OFT | 82.6 vs 75.0; same Qwen3-VL backbone family and downstream OFT head | `RAW/sections/4_experiments.tex`, LIBERO-Plus paragraph |
| LIBERO-Plus comparison bars | Official `Total`: VLAct 82.6, ABot-M0 80.5, Qwen3VL-OFT 75.0, OpenVLA-OFT 69.6 | `RAW/sections/4_experiments.tex`, LIBERO-Plus table |
| Caption co-training reaches 82.6% | Highest LIBERO-Plus result among the auxiliary-data variants shown | `RAW/figures/vlm_data_ablation_trend_bar_clean_v2.pdf` and `RAW/sections/X_appendix.tex`, VLM co-training section |
| 53.4% on VLA-Arena | Official suite-weighted average across 11 task suites; category values average L0/L1/L2 | `RAW/sections/X_appendix.tex`, VLA-Arena table |
| VLA-Arena comparison bars | Official average: VLAct 53.4, π0.5 44.3, π0 42.3, OpenVLA-OFT 39.9, Qwen3VL-OFT 33.4 | `RAW/sections/X_appendix.tex`, VLA-Arena table |
| 92.5% / 90.8% on RoboTwin 2.0 | Data Scaling setting, OFT head, Clean / Random evaluation; 50 clean plus 500 randomized trajectories per task | `RAW/sections/4_experiments.tex`, RoboTwin table |
| RoboTwin comparison bars | Data Scaling Clean: VLAct-OFT 92.5, LingBot-VLA 88.6, Qwen3VL-OFT 88.2, ABot-M0 86.1, π0.5 82.7 | `RAW/sections/4_experiments.tex`, RoboTwin table |
| DOMINO 18.50% SR / 34.20 MS | Best reported result on both metrics; one multi-task policy across all 35 tasks in the clean dynamic setting | `RAW/sections/X_appendix.tex`, DOMINO paragraph and table |
| DOMINO comparison bars | Success Rate: VLAct-OFT 18.50, Qwen3VL-OFT 10.86, π0.5 9.63, OpenVLA-OFT 9.06, π0 8.17 | `RAW/sections/X_appendix.tex`, DOMINO table |
| RoboCasa-GR1 curve: 41.42 / 49.5 / 51.0 / 54.0 | Task success rate at 10% / 20% / 50% / 100% of downstream fine-tuning trajectories | `RAW/sections/5_analysis.tex` and panel (d) of `RAW/figures/real_world_robocasa_vx_v2.pdf` |
| 20%-data VLAct exceeds full-data baselines | 49.5 exceeds Qwen3VL-OFT 48.8, GR00T-N1.6 47.6, and π0.5 37.0 in the reported comparison | `RAW/sections/5_analysis.tex` |
| RoboDojo 10.66 score / 7.60% success | Eighth of 35 by partial-progress score and sixth by success rate; official snapshot dated August 24, 2026; 42 tasks, 50 episodes each | `RAW/sections/5_analysis.tex`, RoboDojo paragraph and table |
| RoboDojo comparison bars | Average success rate for selected entries: VLAct 7.60, InternVLA-A1.5 7.14, π0.5 6.91, X-WAM 3.83; DM0.5 is intentionally omitted from the project-page chart | `RAW/sections/5_analysis.tex`, RoboDojo table |
| Real-world 92.5% vs 77.5% | Mean binary success over four in-domain short-horizon tasks; 10 rollouts per task; comparator is Qwen3VL-4B-OFT without VLA pre-training | `RAW/sections/4_experiments.tex` and `RAW/sections/X_appendix.tex`, real-world section |
| Novel-object 90.0% / 90.0% | Separate averages for object-from-pot and object-in-cup task families; comparator values are 73.3% / 65.0% | `RAW/sections/X_appendix.tex`, short-horizon OOD section |
| Long-horizon 86.6% / 80.0% | Stage-weighted completion rates for table cleaning / scooping beans, not binary task success | `RAW/sections/X_appendix.tex`, success-rate criterion and long-horizon section |
| Dual-arm 72.0% vs 44.0% | Mean binary success over five real-world dual-arm Franka tasks; 10 rollouts per task | `RAW/sections/X_appendix.tex`, dual-arm section |
| Real-world fine-tuning protocol | Separate single-arm and dual-arm models; each trained for 50k steps on 8 H800 GPUs; 10 rollouts per task | `RAW/sections/4_experiments.tex`, real-world setup |
| Qwen3-VL-4B, four open robot datasets, 16 GPUs | Robot datasets are DROID, InternData-A1, RoboCoin, and MolmoAct; caption supervision is additional | `RAW/sections/4_experiments.tex` and `RAW/sections/X_appendix.tex`, data details |
| “What makes a VLM backbone a strong foundation for action in the physical world?” | Research framing, not a claim that VLAct identifies one universally best VLM; the paper studies how pre-training shapes transferable representations | `RAW/sections/1_introduction.tex`, representation-centric motivation and method overview |

## Known manuscript discrepancy

The active prose and abstract report GR00T-N1.6 at **47.6%** on RoboCasa-GR1, while the embedded `real_world_robocasa_vx_v2.pdf` figure labels it **47.5**. The project page follows the repeated prose value, 47.6%, and intentionally does not display that stale figure. The paper figure should be reconciled before release.

The auxiliary-data plot shows `+ Pretrain` at **79.6** and image-caption co-training at **82.6**, while the surrounding appendix prose calls **75.0** the robot-only baseline. The public page therefore states only the unambiguous 82.6 caption result and does not publish an incremental gain for this ablation.

The main experiments section calls one pre-training dataset **InternA1**, while the appendix and the cited dataset name use **InternData-A1**. The project page uses `InternData-A1`; the manuscript should standardize the spelling before release.

## Publication-state notes

- The BibTeX entry is provisional because no arXiv identifier or venue is currently present in the manuscript.
- The checkpoint section links the public VLAct Hugging Face collection. RoboDojo OFT (100k steps) and RoboTwin GR00T (50k steps) contained downloadable weights at the August 27, 2026 audit.
- The downloadable RoboTwin card is explicitly the GR00T-head checkpoint; it is not presented as the artifact behind the page's separate 92.5% OFT result.
- Per the release owner's instruction, the continued-pretraining backbone, VLA-Arena, and LIBERO-Plus cards are labeled `Weights available`; their repositories must be populated before public deployment.
- Canonical and social URLs point to the manuscript's intended project URL, `https://starvla.github.io/VLAct/`, which must be deployed before public sharing.
- The manuscript marks Shu Liu as correspondence but lists `yangsenqiao.ai@gmail.com`; confirm the intended corresponding contact before adding named correspondence metadata.
