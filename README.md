# Wh-Question Polarity

Code and data for *A Utility-Based Account of the Choice of Constituent
Question Polarity* (Lee & Hawkins, CogSci 2026).

## Citation

```bibtex
@inproceedings{lee2026utility,
  title     = {A utility-based account of the choice of constituent question polarity},
  author    = {Lee, Seo-young and Hawkins, Robert D.},
  booktitle = {Proceedings of the 48th Annual Meeting of the Cognitive Science Society},
  year      = {2026}
}
```

## Reproducing Results

- **Model fits + AIC table:** `cd model && python fit.py`
- **Model predictions figure:** `quarto render model/simulations.qmd`
- **Empirical analyses + figures:** `quarto render analysis/cogsci.qmd`
- **Test suite:** `cd model && python test_model.py`

## Layout

- `model/` — RSA model (Python) and parameter fitting
- `analysis/` — empirical analysis (Quarto/R)
- `data/` — per-participant CSVs (anonymized; Prolific IDs replaced with `subj_NNN`)
- `experiment/` — jsPsych source for the two experiments

## Dependencies

- Model: `pip install -r model/requirements.txt`
- Analysis: R + Quarto + `tidyverse`, `tidyboot`, `ggthemes`, `here`
