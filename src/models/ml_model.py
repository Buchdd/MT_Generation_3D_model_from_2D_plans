from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger("MLModel")

NON_FEATURE_COLUMNS = [
    'label', 'file_path', 'file_name', 'entity_type',
    'layer', 'entity_id', 'class', 'predicted_class'
]

class MLClassifier:
    def __init__(self, model_type='random_forest', class_weight=None, **kwargs):
        self.model_type = model_type
        self.class_weight = class_weight  # 🔥 Сохраняем веса
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = self._init_model(model_type, class_weight=class_weight, **kwargs)
        self.feature_columns = None
        self.class_names = None

    def _init_model(self, model_type, class_weight=None, **kwargs):
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 15),
                class_weight=class_weight,  # 🔥 Передаём веса в sklearn
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            )
        else:
            return RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=42, n_jobs=-1)

    def prepare_data(self, df: pd.DataFrame, label_col='label'):
        logger.info(f"Preparing data from {len(df)} samples")
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found")
        
        y = df[label_col].values
        self.class_names = np.unique(y)
        logger.info(f"Found {len(self.class_names)} classes")
        
        X_df = df.drop(columns=[label_col], errors='ignore')
        cols_to_drop = [c for c in NON_FEATURE_COLUMNS if c in X_df.columns]
        if cols_to_drop:
            X_df = X_df.drop(columns=cols_to_drop, errors='ignore')
        
        X_df = X_df.select_dtypes(include=[np.number])
        if X_df.empty:
            raise ValueError("No numeric features found!")
        
        self.feature_columns = X_df.columns.tolist()
        logger.info(f"Using {len(self.feature_columns)} numeric features")
        
        X = X_df.values
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded

    def train(self, X, y, val_size=0.2):
        logger.info(f"Training {self.model_type} on {len(X)} samples")

        if val_size is None or val_size <= 0.0:
            # Внешний split уже сделан — обучаем на всём что пришло
            self.model.fit(X, y)
            train_score = self.model.score(X, y)
            logger.info(f"Train accuracy (full data, no internal val): {train_score:.4f}")
            return {'train_acc': train_score, 'val_acc': None}

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        logger.info(f"Train accuracy: {train_score:.4f}")
        logger.info(f"Val accuracy:   {val_score:.4f}")
        return {'train_acc': train_score, 'val_acc': val_score}

    def predict(self, X):
        encoded_preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(encoded_preds)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'class_names': self.class_names,
            'model_type': self.model_type,
            'class_weight': self.class_weight  # 🔥 Сохраняем веса
        }, path)
        logger.info(f"✅ ML model saved to {path}")

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        data = joblib.load(path)
        instance = cls(model_type=data['model_type'], class_weight=data.get('class_weight'))
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.label_encoder = data['label_encoder']
        instance.feature_columns = data['feature_columns']
        instance.class_names = data.get('class_names', None)
        logger.info(f"✅ ML model loaded from {path}")
        return instance