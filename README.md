# Wh-Question Polarity

Code and data for reproducing the analyses in *A Utility-Based Account of the Choice of Constituent Question Polarity*.

## Reproducing Results

- **Model simulations**: `model/simulations.qmd`
- **Empirical analyses**: `analysis/cogsci_whQ_analysis.qmd`
- **Empirical data**: `data/`

## Model

The computational model is in `model/`. Install dependencies with:

```bash
pip install -r model/requirements.txt
```

Run the test suite:

```bash
cd model && python test_model.py
```
