import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

STOPWORDS_ES = [
    "a", "acá", "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", 
    "allá", "allí", "ante", "antes", "aquí", "arriba", "así", "aun", "aunque",
    "bajo", "bastante", "bien", "cada", "casi", "como", "con", "cual",
    "cuales", "cuando", "cuanto", "cuantos", "de", "del", "demasiado",
    "demás", "dentro", "desde", "donde", "dos", "el", "él", "ella", "ellas", 
    "ellos", "en", "encima", "entonces", "entre", "era", "eran", "eres", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado", "estados",
    "estamos", "están", "este", "estos", "estoy", "fue", "fueron", "fui",
    "fuimos", "ha", "haber", "había", "han", "hasta", "hay", "la", "las", "le",
    "les", "lo", "los", "más", "me", "mi", "mis", "mientras", "muy", "nada",
    "ni", "no", "nos", "nosotros", "nuestra", "nuestros", "o", "os", "otra",
    "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "que",
    "quien", "quienes", "se", "sea", "ser", "si", "sido", "siempre", "sin",
    "sobre", "solo", "su", "sus", "tal", "también", "tanto", "te", "tener",
    "ti", "tiene", "tienen", "todo", "todos", "tu", "tus", "un", "una", "unas",
    "uno", "unos", "usted", "ustedes", "va", "vamos", "van", "varios", "vaya",
    "veces", "voy", "ya", "yo"
]

class VectorizarTexto(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column
        self.count_vec = None
        self.tfidf_vec = None

    def fit(self, X, y=None):
        textos = X[self.column].fillna("").astype(str).tolist()

        self.count_vec = CountVectorizer(stop_words=STOPWORDS_ES)
        self.count_vec.fit(textos)

        self.tfidf_vec = TfidfVectorizer(stop_words=STOPWORDS_ES)
        self.tfidf_vec.fit(textos)

        return self

    def transform(self, X):
        X = X.copy()

        textos = X[self.column].fillna("").astype(str).tolist()

        count_matrix = self.count_vec.transform(textos)
        count_df = pd.DataFrame(
            count_matrix.toarray(),
            columns=[f"{self.column}_count_{w}" for w in self.count_vec.get_feature_names_out()],
            index=X.index
        )

        tfidf_matrix = self.tfidf_vec.transform(textos)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"{self.column}_tfidf_{w}" for w in self.tfidf_vec.get_feature_names_out()],
            index=X.index
        )

        X = X.drop(columns=[self.column])
        X = pd.concat([X, count_df, tfidf_df], axis=1)

        return X