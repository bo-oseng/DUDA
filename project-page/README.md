# DUDA project page

Static research project page inspired by the CURE page, with a palette drawn
from the DUDA figures. No build step or runtime dependencies are required.

## Preview

From the repository root:

```bash
python -m http.server 8000 --directory project-page
```

Open http://localhost:8000. CSS and JavaScript use relative URLs so the page can
be hosted at https://bo-oseng.github.io/DUDA/.

## Content

- Figure assets are copies of the provided problem-setting and method figures.
- Tables and component studies use manuscript Tables 1, 2, and 4. Values retain
  the paper's precision; they are not rounded anew from reference metrics.
- Dataset selectors update the benchmark table. The four headline cards always
  show overall results. Bold marks maxima among the six displayed methods.
- Paper, models, demo, and citation remain unavailable until release details exist.
- Author metadata, qualitative comparisons, and the incomplete dynamic-stream
  table are intentionally omitted pending final material.

## Deployment

GitHub Actions deploys `project-page/` to https://bo-oseng.github.io/DUDA/
when page files or `.github/workflows/pages.yml` change on `main`.
The workflow can also be run manually from the Actions tab.
GitHub Pages uses GitHub Actions as its publishing source.
