# Student Performance Factors — EDA and Machine Learning

This project analyzes a tabular dataset of student performance factors and trains models to (1) predict **exam scores** (regression) and (2) predict **pass vs. fail** (classification). It is written for Python using pandas, scikit-learn, and matplotlib/seaborn.

The analysis pipeline has three entry points:

| Script | Purpose |
|--------|---------|
| `explore.py` | Load the CSV, print shape, dtypes, missing values, summaries, and show plots (exam score distribution and numeric correlation heatmap). |
| `regression.py` | Predict `Exam_Score` with linear regression and a random forest regressor; report MAE, RMSE, R²; plot feature importances and actual vs. predicted for the forest. |
| `classification.py` | Binarize outcomes into **Pass** (exam score ≥ 60) vs **Fail** after clipping scores at 100; train logistic regression and a random forest classifier with stratified split; report accuracy, classification report, confusion matrix, and feature importances. |

**Dataset:** `data/StudentPerformanceFactors.csv` — one row per student with mixed numeric and categorical columns (e.g. hours studied, attendance, parental involvement, school type, gender) and a numeric target `Exam_Score`.

> **Repository note:** `.gitignore` lists `data/*.csv`, so the CSV may not be present after a fresh clone. Place `StudentPerformanceFactors.csv` in the `data/` folder before running the scripts, or adjust the path in each script.

Run all commands from the **project root** (`student_performance/`), since paths are relative to `data/StudentPerformanceFactors.csv`.

## Instructions for Build and Use

**1. Environment**

- Install [Python](https://www.python.org/downloads/) 3.9+ (3.10 or 3.11 recommended).
- Create a virtual environment (optional but recommended):

  ```bash
  python -m venv venv
  ```

  On Windows, activate with `venv\Scripts\activate`; on macOS/Linux, `source venv/bin/activate`.

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the scripts**

```bash
python explore.py
python regression.py
python classification.py
```

Each script that uses matplotlib will open interactive plot windows (`plt.show()`). Use a local environment with a display, or adapt the scripts to save figures (e.g. `plt.savefig(...)`) if you run headless.

**What the models do:**

- Missing values: numeric columns filled with the **median**; categorical/object columns filled with the **mode**.
- Categorical columns are label-encoded with a single shared `LabelEncoder` per column (as in the current code).
- Features are standardized with `StandardScaler` after an 80/20 train-test split (`random_state=42`; classification uses stratification on the grade label).

## Development Environment

Software and libraries used (versions are not pinned in `requirements.txt`; install the latest compatible releases with `pip`):

| Package | Role |
|---------|------|
| Python | Runtime |
| pandas | Data loading and manipulation |
| numpy | Numerics |
| scikit-learn | Preprocessing, models, metrics |
| matplotlib | Plotting |
| seaborn | EDA plots (e.g. histograms, heatmaps) |

To match a clean environment, use only `requirements.txt` and upgrade pip if installs fail: `python -m pip install --upgrade pip`.

## Useful Websites to Learn More

- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) — preprocessing, regression, classification, metrics
- [pandas documentation](https://pandas.pydata.org/docs/) — reading CSV, dtypes, missing data
- [matplotlib](https://matplotlib.org/stable/contents.html) and [seaborn](https://seaborn.pydata.org/tutorial.html) — visualization

## Future Work

Possible improvements:

- [ ] Pin dependency versions in `requirements.txt` for reproducibility
- [ ] Add k-fold cross-validation and hyperparameter search (e.g. for forest depth and `n_estimators`)
- [ ] Use separate label encoders per column (or one-hot / target encoding) to avoid accidental reuse of a single encoder across columns
- [ ] Save figures to disk and add CLI flags (e.g. `--no-show`) for servers/CI
- [ ] Evaluate regression with residual plots; evaluate classification with ROC-AUC and calibration, especially if class balance shifts
