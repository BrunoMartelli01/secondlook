"""Cross-fitted word-trust features for cost-aware whole-image escalation.

All counterfactual labels use native spotting counts. Calibration-score sums are
features, not unbiased estimates or formal risk guarantees. No target tuning."""

def new_word_model(name,cfg):
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s=cfg['word_trust']
    if name.startswith('logistic_'):return make_pipeline(StandardScaler(),LogisticRegression(C=s['logistic_C'],max_iter=1000,random_state=s['head_seed']))
    if name.startswith('hgb_'):return HistGradientBoostingClassifier(max_iter=s['hgb']['iterations'],max_leaf_nodes=s['hgb']['leaves'],l2_regularization=s['hgb']['l2'],min_samples_leaf=s['hgb']['minimum_leaf'],random_state=s['head_seed'],early_stopping=False)
    raise ValueError('Unsupported selected word classifier')
