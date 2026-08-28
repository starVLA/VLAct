# VLAct project page

This is a dependency-free project page built with HTML, CSS, and JavaScript. It can be deployed directly with GitHub Pages.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/website/`.

## Deployment

The directory is self-contained and can be published as a GitHub Pages artifact or copied into a dedicated `VLAct` Pages repository.

`assets/VLAct.pdf` is a snapshot of the authoritative `Beyond_Data__Scaling_VLAct_Arxiv_3.pdf`; refresh it whenever the manuscript changes. The real-robot gallery uses 13 H.264 rollout sources under `assets/videos/`; three derived portrait crops remove baked-in sidebars while retaining the original MP4s. Gallery posters live under `assets/video-posters/`.

The checkpoint section follows the release owner's requested public labels. At the August 28, 2026 audit, RoboDojo and RoboTwin GR00T contained downloadable weights; the continued-pretraining, VLA-Arena, and LIBERO-Plus repositories were public placeholders awaiting their model files. The page explicitly distinguishes each released artifact's action head from the head used for the corresponding headline result in the paper.

See `CLAIMS.md` for the source and scope of every headline number used on the page.

Canonical and social-preview URLs target the deployed project URL, `https://starvla.github.io/VLAct/`.
