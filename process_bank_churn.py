from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and validation sets using stratification.
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )


def separate_inputs_targets(
    df: pd.DataFrame, input_cols: List[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features and target column.
    """
    return df[input_cols].copy(), df[target_col].copy()


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    return numeric_cols, categorical_cols


def drop_unused_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop unnecessary columns from dataframe.
    """
    return df.drop(columns=columns_to_drop)


def scale_numeric_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_df, val_df


def encode_categorical_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    One-hot encode categorical features.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    train_encoded = encoder.transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])

    train_df[encoded_cols] = train_encoded
    val_df[encoded_cols] = val_encoded

    train_df = train_df.drop(columns=categorical_cols)
    val_df = val_df.drop(columns=categorical_cols)

    return train_df, val_df, encoded_cols


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    drop_cols: List[str] = ["Surname", "CustomerId"],
    scale_numeric: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Full preprocessing pipeline:
    - split data
    - drop unused columns
    - scale numeric features (optional)
    - encode categorical features
    """
    train_df, val_df = split_data(raw_df, target_col)

    input_cols = [col for col in raw_df.columns if col != target_col]
    train_inputs, train_targets = separate_inputs_targets(train_df, input_cols, target_col)
    val_inputs, val_targets = separate_inputs_targets(val_df, input_cols, target_col)

    train_inputs = drop_unused_columns(train_inputs, drop_cols)
    val_inputs = drop_unused_columns(val_inputs, drop_cols)

    numeric_cols, categorical_cols = get_column_types(train_inputs)

    if scale_numeric:
        train_inputs, val_inputs = scale_numeric_features(
            train_inputs, val_inputs, numeric_cols
        )

    train_inputs, val_inputs, encoded_cols = encode_categorical_features(
        train_inputs, val_inputs, categorical_cols
    )

    final_input_cols = numeric_cols + encoded_cols

    return {
        "train_inputs": train_inputs[final_input_cols],
        "train_targets": train_targets,
        "val_inputs": val_inputs[final_input_cols],
        "val_targets": val_targets,
    }


def prepare_training_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Public function to call instead of preprocess_data().
    """
    return preprocess_data(raw_df, scale_numeric=scale_numeric)


def preprocess_new_data(
    new_df: pd.DataFrame,
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
    target_col: str = "Exited",
    drop_cols: List[str] = ["Surname", "CustomerId"]
) -> Dict[str, pd.DataFrame]:
    """
    Preprocess new data using pre-trained scaler and encoder.
    
    This function is useful for processing test data or new predictions
    before feeding into a trained model.
    """
    # Drop unused columns
    new_df = drop_unused_columns(new_df, drop_cols)
    
    # Separate inputs and targets if target column exists
    if target_col in new_df.columns:
        input_cols = [col for col in new_df.columns if col != target_col]
        inputs, targets = separate_inputs_targets(new_df, input_cols, target_col)
    else:
        inputs = new_df
        targets = None
    
    # Identify numeric and categorical columns
    numeric_cols, categorical_cols = get_column_types(inputs)
    
    # Scale numeric features using the pre-trained scaler
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
    
    # Encode categorical features using the pre-trained encoder
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    encoded_data = encoder.transform(inputs[categorical_cols])
    inputs[encoded_cols] = encoded_data
    inputs = inputs.drop(columns=categorical_cols)
    
    # Final input columns
    final_input_cols = numeric_cols + encoded_cols
    
    # Prepare result dictionary
    result = {"inputs": inputs[final_input_cols]}
    if targets is not None:
        result["targets"] = targets
    
    return result