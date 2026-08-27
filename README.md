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

`assets/VLAct.pdf` is a snapshot of `out/StarVLA.pdf`; refresh it whenever the manuscript changes. The real-robot gallery uses 13 H.264 rollout sources under `assets/videos/`; three derived portrait crops remove baked-in sidebars while retaining the original MP4s. Gallery posters live under `assets/video-posters/`.

The checkpoint section uses the verified public VLAct collection and presents the backbone and four downstream repositories as released checkpoints. At the time of the August 27, 2026 audit, RoboDojo and RoboTwin GR00T contained downloadable weights; populate the continued-pretraining, VLA-Arena, and LIBERO-Plus repositories before public deployment.

See `CLAIMS.md` for the source and scope of every headline number used on the page.

Canonical and social-preview URLs target `https://starvla.github.io/VLAct/`; that address must be deployed before the page is publicly shared.
