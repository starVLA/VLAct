# VLAct project-page claim audit

This file records the source and scope of every headline claim used by the project page. The authoritative release checked on August 28, 2026 is `Beyond_Data__Scaling_VLAct_Arxiv_3.pdf` (37 pages; SHA-256 `ca6ddbdad2c8475de9d152e13ad5b9c101f15d0cc8b71ae2758f3d41199ffec3`).

| Page claim | Scope | PDF source |
|---|---|---|
| 82.6% on LIBERO-Plus | Reported `Total`; not the arithmetic mean of the seven rounded axis values | pp. 8–9, Table 1 |
| +7.6 percentage points over Qwen3VL-OFT | 82.6 vs 75.0; same Qwen3-VL backbone family and downstream OFT head | p. 8 |
| LIBERO-Plus comparison bars | Reported `Total`: VLAct 82.6, ABot-M0 80.5, Qwen3VL-OFT 75.0, OpenVLA-OFT 69.6 | p. 9, Table 1 |
| Caption co-training reaches 82.6% | Highest LIBERO-Plus result among the variants in Figure 8's auxiliary co-training ablation | p. 25, Figure 8 |
| 54.8% on VLA-Arena | Official suite-weighted average across 11 task suites; category values average L0/L1/L2 | pp. 20–21, Table 4 |
| VLA-Arena comparison bars | Official average: VLAct 54.8, π0.5 44.3, π0 42.3, OpenVLA-OFT 39.9, Qwen3-VL-OFT 33.4 | p. 21, Table 4 |
| 92.5% / 90.8% on RoboTwin 2.0 | Data Scaling setting, default OFT head, Clean / Random evaluation; 50 clean plus 500 randomized trajectories per task | pp. 9–10, Table 2 |
| RoboTwin comparison bars | Selected Data Scaling Clean results: VLAct-PI 93.0, VLAct-OFT 92.5, HoloBrain-0-QW 91.9, Fast-WAM 91.9, Being-H0.7 90.2 | p. 10, Table 2 |
| DOMINO 18.50% SR / 34.20 MS | Best result in the paper's comparison on both metrics; one multi-task policy across all 35 tasks in the clean dynamic setting | p. 21, Table 5 |
| DOMINO comparison bars | Success Rate: VLAct-OFT 18.50, Qwen3VL-OFT 10.86, π0.5 9.63, OpenVLA-OFT 9.06, π0 8.17 | p. 21, Table 5 |
| RoboCasa-GR1 curve: 41.42 / 49.5 / 51.0 / 54.0 | Task success rate at 10% / 20% / 50% / 100% of downstream fine-tuning trajectories | p. 10 and Figure 5(d), p. 12 |
| 20%-data VLAct exceeds full-data baselines | 49.5 exceeds Qwen3VL-OFT 48.8, GR00T-N1.6 47.6, and π0.5 37.0 in the reported comparison | p. 10 |
| RoboDojo 10.66 score / 7.60% success | Eighth of 35 by partial-progress score and sixth by success rate; official snapshot dated August 24, 2026; 42 tasks, 50 episodes each | pp. 10–11, Table 3 |
| RoboDojo comparison bars | Average success rate for selected entries: VLAct 7.60, InternVLA-A1.5 7.14, π0.5 6.91, X-WAM 3.83; DM0.5 is intentionally omitted from the project-page chart | p. 11, Table 3 |
| Real-world 92.5% vs 77.5% | Mean binary success over four in-domain short-horizon tasks; 10 rollouts per task; comparator is Qwen3VL-4B-OFT without VLA continued pre-training | pp. 9, 33 |
| Novel-object 90.0% / 90.0% | Separate averages for object-from-pot and object-in-cup task families; comparator values are 73.3% / 65.0% | p. 33 |
| Long-horizon 86.6% / 80.0% | Stage-weighted completion rates for table cleaning / scooping beans, not binary task success | p. 33 |
| Dual-arm 72.0% vs 44.0% | Mean binary success over five real-world dual-arm Franka tasks; 10 rollouts per task | p. 33 |
| Real-world fine-tuning protocol | 50 demonstrations per single-arm task and 100 per dual-arm task; separate single-arm and dual-arm models; each trained for 50k steps on 8 H800 GPUs; 10 fixed evaluation rollouts per task | pp. 9, 32–33 |
| Qwen3-VL-4B, four open robot datasets, 16 GPUs | Robot datasets are DROID, InternData-A1, RoboCoin, and MolmoAct; caption supervision is additional | pp. 8, 28 |
| “What makes a VLM backbone a strong foundation for action in the physical world?” | Research framing, not a claim that VLAct identifies one universally best VLM; the paper studies how continued pre-training shapes transferable representations | pp. 1, 3–4 |

## Source-text cautions

- The main experiments section calls one pre-training dataset **InternA1**, while Appendix H and the cited dataset name use **InternData-A1**. The project page uses the canonical `InternData-A1` name.
- The paper describes the default RoboTwin result as VLAct-OFT at **92.5% Clean / 90.8% Random**, while VLAct-PI records the highest Clean value at **93.0%**. The page names the head and setting explicitly and does not claim absolute state of the art.
- Figure 1 combines unlike benchmark metrics and settings, so the page does not reproduce it as a uniform success-rate chart.
- The paper does not state a total trajectory count for the main four-dataset continued-pretraining mixture; the page makes no such claim.

## Publication-state notes

- The BibTeX entry is provisional because no arXiv identifier or venue is currently present in the manuscript.
- The checkpoint section links the public VLAct Hugging Face collection. RoboDojo OFT (100k steps) and RoboTwin GR00T (50k steps) contained downloadable weights at the August 28, 2026 audit.
- The downloadable RoboTwin card is explicitly the GR00T-head checkpoint; it is not presented as the artifact behind the page's separate 92.5% OFT result.
- The RoboDojo repository reports a separate 30-episode-per-task scaled local evaluation (8.17% success / 11.28 score); the page's 7.60% success / 10.66 score is the paper's official 50-episode-per-task leaderboard result.
- Per the release owner's instruction, the continued-pretraining backbone, VLA-Arena, and LIBERO-Plus cards remain labeled `Weights available`; their repositories were still public placeholders awaiting model files at the audit time.
- Canonical and social URLs point to the deployed project URL, `https://starvla.github.io/VLAct/`.
- The page mirrors the manuscript's correspondence mark on Shu Liu and uses the listed project contact, `yangsenqiao.ai@gmail.com`.
