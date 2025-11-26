def reccoseno(new_lubricant_processed, df, preprocessor, X_processed, top_k=5):
    """
    Calcula las grasas más similares usando similitud coseno.

    Inputs:
    ----------
    new_lubricant_data : pd.DataFrame (1 fila)
        Datos de la grasa introducida por el usuario.
    df : pd.DataFrame
        Catálogo completo de grasas lubricantes.
    preprocessor : ColumnTransformer
        Objeto que estandariza y codifica los datos (de load_and_preprocess_data()).
    X_processed : np.ndarray
        Datos del catálogo ya procesados con el preprocessor.
    top_k : int
        Número de recomendaciones a devolver (default = 5).

    Output:
    ----------
    results : pd.DataFrame
        Subconjunto de df con las top_k grasas más similares y su puntaje de similitud.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    # Calcular similitud coseno entre la grasa nueva y todas las del catálogo
    similarities = cosine_similarity(new_lubricant_processed, X_processed)[0]

    # Crear copia del catálogo con columna de similitud
    df_with_similarity = df.copy()
    df_with_similarity['Similitud'] = similarities

    # Ordenar por similitud descendente
    results = df_with_similarity.sort_values('Similitud', ascending=False).head(top_k)

    return results