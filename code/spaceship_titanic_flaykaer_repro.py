
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from catboost import CatBoostClassifier

from project_paths import DATA_DIR, EXPERIMENTS_TABLE_DIR, SUBMISSIONS_DIR

ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
y = train['Transported'].astype(int).values
X = train.drop(columns=['Transported']).copy()
X_test = test.copy()
SPEND = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = df['PassengerId'].astype(str).str.split('_').str[0]
    df['PassengerNo'] = pd.to_numeric(df['PassengerId'].astype(str).str.split('_').str[1], errors='coerce')
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    parts = df['Cabin'].astype('string').str.split('/', expand=True)
    df['Deck'] = parts[0]
    df['CabinNum'] = pd.to_numeric(parts[1], errors='coerce')
    df['Side'] = parts[2]
    for c in SPEND:
        df[c] = df[c].fillna(0.0)
    df['TotalSpend'] = df[SPEND].sum(axis=1)
    df['NoSpend'] = (df['TotalSpend'] == 0).astype(int)
    df['SpendPerPerson'] = df['TotalSpend'] / df['GroupSize'].replace(0, 1)
    df['LuxurySpend'] = df[['RoomService','Spa','VRDeck']].sum(axis=1)
    df['BasicSpend'] = df[['FoodCourt','ShoppingMall']].sum(axis=1)
    cs_str = df['CryoSleep'].astype('string').str.lower()
    cs_num = cs_str.map({'true': 1.0, 'false': 0.0})
    df['GroupCryoRate'] = cs_num.groupby(df['Group']).transform('mean').fillna(0.5)
    mismatch_bool = (cs_str == 'true') & (df['TotalSpend'] > 0)
    df['SpendMismatch'] = mismatch_bool.fillna(False).astype(int)
    df['CryoNoSpendMatch'] = ((cs_str == 'true') & (df['TotalSpend'] == 0)).fillna(False).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

X = build_features(X)
X_test = build_features(X_test)
num_cols = [
    'Age','CabinNum','PassengerNo','GroupSize','IsAlone','TotalSpend','NoSpend','SpendPerPerson',
    'LuxurySpend','BasicSpend','GroupCryoRate','SpendMismatch','CryoNoSpendMatch'
] + SPEND
cat_cols = ['HomePlanet','Destination','Deck','Side','CryoSleep','VIP','Group']
used_cols = num_cols + cat_cols
for c in num_cols:
    med = float(X[c].median())
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)
for c in cat_cols:
    X[c] = X[c].astype('object').fillna('Missing')
    X_test[c] = X_test[c].astype('object').fillna('Missing')

def fit_cb(seed=42, iterations=1500, lr=0.06, depth=6, l2=3.0, subsample=0.8, use_group=True, folds=5):
    if use_group:
        splitter = GroupKFold(n_splits=folds)
        split_iter = splitter.split(X, y, X['Group'])
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    scores=[]
    losses=[]
    oof=np.zeros(len(X))
    best_iters=[]
    cat_idx = [used_cols.index(c) for c in cat_cols]
    for fold,(tr_idx,va_idx) in enumerate(split_iter,1):
        X_tr, X_va = X.iloc[tr_idx][used_cols], X.iloc[va_idx][used_cols]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=depth, l2_leaf_reg=l2,
                                   subsample=subsample, loss_function='Logloss', random_seed=seed,
                                   verbose=False, early_stopping_rounds=100)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_idx, use_best_model=True, verbose=False)
        proba = model.predict_proba(X_va)[:,1]
        oof[va_idx]=proba
        pred=(proba>=0.5).astype(int)
        scores.append(accuracy_score(y_va,pred))
        losses.append(log_loss(y_va,proba))
        best_iters.append(model.get_best_iteration())
        print(f'fold {fold}: acc={scores[-1]:.5f} logloss={losses[-1]:.5f} best_iter={best_iters[-1]}')
    print('cv mean', np.mean(scores), 'std', np.std(scores), 'logloss', np.mean(losses), 'oof_acc', accuracy_score(y,(oof>=0.5).astype(int)))
    final = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=depth, l2_leaf_reg=l2,
                               subsample=subsample, loss_function='Logloss', random_seed=seed, verbose=False)
    final.fit(X[used_cols], y, cat_features=cat_idx, verbose=False)
    test_proba = final.predict_proba(X_test[used_cols])[:,1]
    return oof, test_proba, {'scores':scores,'losses':losses,'best_iters':best_iters,
                             'cv_mean':float(np.mean(scores)), 'cv_std':float(np.std(scores)),
                             'oof_acc':float(accuracy_score(y,(oof>=0.5).astype(int)))}

oof, proba, metrics = fit_cb()
out = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': proba >= 0.5})
out_path = SUBMISSIONS_DIR / 'submission_flaykaer_catboost_repro_plus.csv'
out.to_csv(out_path, index=False)
np.save(EXPERIMENTS_TABLE_DIR / 'flaykaer_catboost_repro_plus_test_probs.npy', proba)
np.save(EXPERIMENTS_TABLE_DIR / 'flaykaer_catboost_repro_plus_oof_probs.npy', oof)
base = pd.read_csv(SUBMISSIONS_DIR / 'submission_catboost_pycaret_exact.csv')
metrics.update({'true_count': int(out.Transported.sum()), 'diff_vs_exact': int(out.Transported.ne(base.Transported).sum())})
(EXPERIMENTS_TABLE_DIR / 'flaykaer_catboost_repro_plus_results.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('saved', out_path)
print(metrics)
