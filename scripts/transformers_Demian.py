import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotCodificador(BaseEstimator, TransformerMixin):
    """
    Transformer personalizado para aplicar one-hot encoding
    a una lista de columnas categóricas, replicando la función
    `codificador` del notebook original, pero de forma escalable
    y compatible con pipelines de sklearn.
    """

    def __init__(self, columns=None, drop_original=True, dtype=int, prefix_sep="_"):
        self.columns = columns
        self.drop_original = drop_original
        self.dtype = dtype
        self.prefix_sep = prefix_sep

    def fit(self, X, y=None):
        # Asegurar DataFrame
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Detectar columnas si no se especifican
        if self.columns is None:
            self.columns_ = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.columns_ = list(self.columns)

        # Obtener dummies para aprender columnas resultantes
        dummies = pd.get_dummies(
            X_df[self.columns_],
            prefix=self.columns_,
            prefix_sep=self.prefix_sep
        )

        # Guardar todas las columnas dummy que existen en fit
        self.dummy_columns_ = dummies.columns.tolist()
        return self

    def transform(self, X):
        # Copia del DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Generar dummies en transform
        dummies = pd.get_dummies(
            X_df[self.columns_],
            prefix=self.columns_,
            prefix_sep=self.prefix_sep
        )

        # Alinear columnas con lo visto en fit
        for col in self.dummy_columns_:
            if col not in dummies:
                dummies[col] = 0  # columna que existía antes pero ahora no aparece

        # Ignorar dummies nuevas que no existían en entrenamiento
        dummies = dummies[self.dummy_columns_]

        # Eliminar columnas originales si corresponde
        if self.drop_original:
            X_df = X_df.drop(columns=self.columns_, errors="ignore")

        # Agregar las dummies al DataFrame
        X_df[dummies.columns] = dummies.astype(self.dtype)

        return X_df