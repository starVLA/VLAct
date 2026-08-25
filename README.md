# VLAct — project page (`gh-pages`)

This branch serves <https://starvla.github.io/VLAct>. It contains **only** the static site;
the VLAct code lives on the `main` branch.

```
index.html                 the page
static/css/styles.css      styles
static/js/script.js        demo carousels, comparison players, nav scroll-spy
static/figures/            figures rendered from the paper source
static/videos/             real-robot rollout clips
static/VLAct_paper.pdf     the paper
.nojekyll                  serve files verbatim (no Jekyll processing)
```

Preview locally:

```bash
git switch gh-pages
python3 -m http.server 8000    # then open http://localhost:8000
```

GitHub Pages is configured as: **Settings → Pages → Deploy from a branch → `gh-pages` / `(root)`**.
