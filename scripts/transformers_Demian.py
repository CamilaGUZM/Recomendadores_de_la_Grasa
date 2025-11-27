import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotCodificador(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, drop_original=True, dtype=int, prefix_sep="_"):
        self.columns = columns
        self.drop_original = drop_original
        self.dtype = dtype
        self.prefix_sep = prefix_sep

    def fit(self, X, y=None):
        # Aseguramos DataFrame
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Ubicamos columnas si no se especifican
        if self.columns is None:
            self.columns_ = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.columns_ = list(self.columns)

        # Obtenemos dummies para aprender columnas resultantes
        dummies = pd.get_dummies(
            X_df[self.columns_],
            prefix=self.columns_,
            prefix_sep=self.prefix_sep
        )

        # Guardarmos todas las columnas dummy que existen en fit
        self.dummy_columns_ = dummies.columns.tolist()
        return self

    def transform(self, X):
        # Copiamos el DataFrame
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Generamos dummies en transform
        dummies = pd.get_dummies(
            X_df[self.columns_],
            prefix=self.columns_,
            prefix_sep=self.prefix_sep
        )

        # Alineamos columnas con lo que salió en fit
        for col in self.dummy_columns_:
            if col not in dummies:
                dummies[col] = 0 

        # Ignoramos dummies nuevas que no existían en entrenamiento
        dummies = dummies[self.dummy_columns_]

        # Eliminamos columnas originales si corresponde
        if self.drop_original:
            X_df = X_df.drop(columns=self.columns_, errors="ignore")

        # Agregamos las dummies al DataFrame
        X_df[dummies.columns] = dummies.astype(self.dtype)

        return X_df