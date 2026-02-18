from stoichml.utils import save_featurized

df_feat = save_featurized(
    input_path="data/data.pkl",
    output_path="data/data_feat.pkl"
)
print(f"Featurized data saved to 'data/data_feat.pkl' with shape: {df_feat.shape}")