# solarcell-ml

ML regression benchmark for solar-cell device parameters.

Predicts **Voc, Jsc, FF, eta** from 5 inputs (ETL / absorber / HTL thickness,
defect density, acceptor density) using 6 regressors.

## Branches
- `main` — project scaffold
- `independent-training` — independently written training script + results

## Run
```bash
pip install scikit-learn xgboost pandas matplotlib openpyxl
python train.py --data /path/to/dataset.xlsx --out ./results
```
