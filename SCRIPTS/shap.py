def plot_shap(
    clf: BaseEstimator,
    X_test: pd.DataFram | np.ndarray,
    y_test: pd.Series | np.ndarray,
    subsample: int | None,
) -> tuple[np.ndarray, Figure]:
    """Plot shap summary for a fitted estimator and a set of test with its labels."""
    if subsample:
        assert subsample > 0
        _, X, _, _ = train_test_split(X_test, y_test, test_size=subsample)
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception:
        explainer = shap.Explainer(clf)
    shap_values = explainer.shap_values(X)
    shap_summary = shap.summary_plot(shap_values=shap_values[1], features=X, plot_type="violin")
    return explainer, shap_summary



# QUESTIONS :

# why do I scale? -