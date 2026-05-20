# Push Checklist

## Ready

- Dependencies are listed in `requirements.txt`, including `optuna`.
- High-score XGBoost reproduction script runs from the repository root.
- Demo notebook runs from `notebooks/` with automatic project-root detection.
- Final report PDF/TEX, figures, and tables are included under `report/`.
- Experiment evidence used by the paper is included under `experiments/`.
- Selected representative submissions are included under `submissions/`.
- Secrets and bulky scratch artifacts are excluded by `.gitignore`.

## Verify Before Final Submission

- Replace the GitHub URL placeholder in `report/final_ieee_paper.tex`.
- Replace the official student-name/ID contribution placeholders in `report/final_ieee_paper.tex`.
- Rebuild/export `report/final_ieee_paper.pdf` after those administrative edits, if the final PDF must contain the live GitHub URL.

## Suggested Git Commands

```powershell
git init
git add .
git commit -m "Prepare reproducible Spaceship Titanic project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
