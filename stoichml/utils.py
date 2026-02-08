import pandas as pd
import os
from .featurizer import featurize

def load_and_featurize(
    path="data/data.pkl",
    elements_col="elements",
    composition_col="composition",
    verbose=True
):
    """
    Loads a pickled DataFrame and featurizes it.

    Returns the featurized DataFrame.
    """
    if verbose:
        print(f"Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found at {path}")

    df = pd.read_pickle(path)

    if verbose:
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Featurizing dataset...")

    df_feat = featurize(df, elements_col=elements_col, composition_col=composition_col)

    if verbose:
        print(f"Featurization complete: {df_feat.shape[0]} rows, {df_feat.shape[1]} columns")


    # Assign half-metal classification
    def hm_class(egap_type):
        if egap_type == 'metal':
            return 0
        elif egap_type == 'half-metal':
            return 2
        else:
            return 1

    df_feat['hm_class'] = df_feat['Egap_type'].apply(hm_class)

    return df_feat


def save_featurized(
    input_path="data/data.pkl",
    output_path="data/data_feat.pkl",
    elements_col="elements",
    composition_col="composition",
    verbose=True
):
    """
    Loads a dataset, featurizes it, assigns numeric labels, and saves as a pickle.

    Returns
    -------
    pd.DataFrame
        The featurized DataFrame with numeric labels.
    """
    df_feat = load_and_featurize(
        path=input_path,
        elements_col=elements_col,
        composition_col=composition_col,
        verbose=verbose
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if verbose:
        print(f"Saving featurized dataset to: {output_path}")

    df_feat.to_pickle(output_path)

    if verbose:
        print("Save complete.")

    return df_feat
