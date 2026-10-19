"""Quick end-to-end pipeline test script."""
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
print('=== END-TO-END PIPELINE TEST ===')

# 1. Generate synthetic data
from src.sample_data import generate_sample_train
print('\n[1] Generating synthetic data...')
df = generate_sample_train()
print(f'    Shape: {df.shape}, Stores: {df.store_nbr.nunique()}, Families: {df.family.nunique()}')

# 2. Validate
from src.data_pipeline import validate_sales, dataset_summary
df = validate_sales(df)
summary = dataset_summary(df)
print(f'    Summary: {summary}')

# 3. Build features
print('\n[2] Building features...')
from src.features import build_features, FEATURE_COLS
feat_df = build_features(df, is_training=True)
print(f'    Feature shape: {feat_df.shape}, Num features: {len(FEATURE_COLS)}')

# 4. Train models
print('\n[3] Training quantile models...')
from src.train_model import train_quantile_models, save_models
q10, q50, q90, val_df, meta = train_quantile_models(feat_df)
save_models(q10, q50, q90, meta)
print(f'    Models trained. Val rows: {len(val_df)}, Split: {meta["split_date"]}')

# 5. Metrics
print('\n[4] Evaluation metrics...')
from src.evaluation import compute_all_metrics
metrics = compute_all_metrics(val_df)
for k, v in metrics.items():
    print(f'    {k}: {round(v,4) if isinstance(v, float) else v}')

# 6. Forecast
print('\n[5] 7-day forecast...')
from src.train_model import load_models
from src.forecasting import generate_forecast
q10m, q50m, q90m, mmeta = load_models()
fc = generate_forecast(df, q10m, q50m, q90m, mmeta, store_nbr=1, family='GROCERY I', horizon=7)
print(fc[['date','pred_q10','pred_q50','pred_q90']].to_string(index=False))

# Check crossing
assert (fc['pred_q10'] <= fc['pred_q50']).all(), 'CROSSING ERROR: q10 > q50!'
assert (fc['pred_q50'] <= fc['pred_q90']).all(), 'CROSSING ERROR: q50 > q90!'
print('    Quantile ordering: OK')

# 7. Inventory
print('\n[6] Inventory calculation...')
from src.inventory import InventoryInputs, compute_inventory
inputs = InventoryInputs(
    on_hand=200, on_order=50, backorders=10,
    review_period=7, lead_time=3, service_level=0.95
)
result = compute_inventory(inputs, fc)
print(f'    D_H={result.median_demand_H}, SS={result.safety_stock}, Q={result.order_quantity}')

# 8. SHAP
print('\n[7] SHAP feature importance...')
from src.explainability import is_shap_available, global_feature_importance
if is_shap_available():
    imp = global_feature_importance(q50m, mmeta['feature_cols'], top_n=5)
    print(imp.to_string(index=False))
else:
    print('SHAP not available')

print('\n=== ALL CHECKS PASSED ===')
