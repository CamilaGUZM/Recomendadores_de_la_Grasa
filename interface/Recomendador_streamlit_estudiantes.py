
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import streamlit as st
import plotly.express as px

from scripts.Recomendador_coseno import reccoseno
from scripts.transformers_JuanPablo import CheckColumnNames,UnknownToZero,FixRanges
from scripts.transformer_Alfredo import FillNaNsWithCeros
from scripts.transformers_Demian import OneHotCodificador
from scripts.transformer_Gonzalo import VectorizarTexto

# Configurar la página
st.set_page_config(
    page_title="Sistema de Recomendación de Grasas",
    page_icon="🛢️",
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data():
    """Cargar y preprocesar datos"""
    try:
        # Cargar datos
        df = pd.read_csv("data/datos_grasas_Tec.csv", encoding="latin1")

        categorical_columns = ["Aceite Base","Espesante","Clasificacion ISO 6743-9","color","textura"]
        preprocessor = Pipeline(steps=[
            ("To have columns names needed", CheckColumnNames()),
            ("To change unkown data to zeros", UnknownToZero("Grado NLGI Consistencia")),
            ("To fix ranges and single values", FixRanges("Penetración de Cono a 25°C, 0.1mm")),
            ...,
            ("OneHot_categoricals", OneHotCodificador(columns=categorical_cols,drop_original=True,dtype=int)),
            ("To fill NaNs with zeros", FillNaNsWithCeros()),
            ("Vectorizar subtitulo", VectorizarTexto("subtitulo")),
            ("Vectorizar descripcion", VectorizarTexto("descripcion")),
            ("Vectorizar beneficios", VectorizarTexto("beneficios")),
            ("Vectorizar aplicaciones", VectorizarTexto("aplicaciones")),
            ('MinMax', ColumnTransformer(transformers=[('MinMax', MinMaxScaler(), slice(1,None))]))
        ])
        X_processed=preprocessor.fit_transform(df)

        return df, preprocessor, X_processed, numeric_cols, categorical_cols
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None, None, None, None

def create_lubricant_from_input(input_data, df_template):
    """Crear una grasa a partir de los datos de entrada"""
    new_lubricant = df_template.iloc[[0]].copy().reset_index(drop=True)
    
    # Actualizar con los valores de entrada
    for key, value in input_data.items():
        if value is not None and value != '':
            if key in df_template.columns:
                new_lubricant[key] = float(value) if isinstance(value, (int, float)) or value.replace('.', '').isdigit() else value
    
    return new_lubricant

def recommend_similar_lubricant(new_lubricant_data, df, preprocessor, X_processed, top_k=5):
    """Recomendar grasas similares"""
    # Here new_lubricant_data will be transform
    results = reccoseno(new_lubricant_processed, df, X_processed, top_k)
    
    return results

# Cargar datos
df, preprocessor, X_processed, numeric_cols, categorical_cols = load_and_preprocess_data()

if df is not None:
    # Título principal
    st.title("🛢️ Sistema de Recomendación de Grasas Lubricantes")
    st.markdown("---")
    
    # Sidebar para entrada de datos
    with st.sidebar:
        st.header("🔍 Características Deseadas")
        
        # Campos de entrada principales
        aceite_base = st.selectbox(
            "Aceite Base",
            options=['0.1', '0.0'], #Modifica los rangos
            index=0
        )
        
        espesante = st.selectbox(
            "Espesante",
            options=['0.1', '0.0'], #Modifica los rangos
            index=0
        )
        
        grado_nlgi = st.selectbox(
            "Grado NLGI Consistencia",
            options=['0.1', '0.0'], #Modifica los rangos
            index=0
        )
        
        viscosidad = st.number_input(
            "Viscosidad a 40°C (cSt)",
            min_value=0.0,
            value=None,
            placeholder="Ej: 150.0"
        )
        
        # Campos adicionales en expansor
        with st.expander("Características Adicionales"):
            penetracion = st.number_input(
                "Penetración de Cono (0.1mm)",
                min_value=0.0,
                value=None
            )
            
            punto_gota = st.number_input(
                "Punto de Gota (°C)",
                min_value=0.0,
                value=None
            )
            
            temp_min = st.number_input(
                "Temperatura Mínima de Servicio (°C)",
                value=None
            )
            
            temp_max = st.number_input(
                "Temperatura Máxima de Servicio (°C)",
                value=None
            )
        
        # Botón de búsqueda
        buscar_btn = st.button(
            "🚀 Buscar Recomendaciones",
            use_container_width=True,
            type="primary"
        )
        
        # Estadísticas en el sidebar
        st.markdown("---")
        st.header("📊 Estadísticas")
        st.metric("Grasas en catálogo", len(df))
        st.metric("Características", len(df.columns))

    # Contenido principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📋 Resumen de Búsqueda")
        if any([aceite_base, espesante, grado_nlgi, viscosidad]):
            input_data = {
                'Aceite Base': aceite_base,
                'Espesante': espesante,
                'Grado NLGI Consistencia': grado_nlgi,
                'Viscosidad del Aceite Base a 40°C. cSt': viscosidad,
                'Penetración de Cono a 25°C, 0.1mm': penetracion,
                'Punto de Gota, °C': punto_gota,
                'Temperatura de Servicio °C, min': temp_min,
                'Temperatura de Servicio °C, max': temp_max
            }
            
            st.json({k: v for k, v in input_data.items() if v not in [None, '']})
        else:
            st.info("Complete las características en el panel lateral")
    
    with col2:
        if buscar_btn and any([aceite_base, espesante, grado_nlgi, viscosidad]):
            try:
                # Crear grasa de entrada
                input_data_dict = {
                    'Aceite Base': aceite_base if aceite_base else None,
                    'Espesante': espesante if espesante else None,
                    'Grado NLGI Consistencia': grado_nlgi if grado_nlgi else None,
                    'Viscosidad del Aceite Base a 40°C. cSt': viscosidad,
                    'Penetración de Cono a 25°C, 0.1mm': penetracion,
                    'Punto de Gota, °C': punto_gota,
                    'Temperatura de Servicio °C, min': temp_min,
                    'Temperatura de Servicio °C, max': temp_max
                }
                
                nueva_grasa = create_lubricant_from_input(input_data_dict, df)
                recomendaciones = recommend_similar_lubricant(nueva_grasa, df, preprocessor, X_processed, top_k=5)
                
                # Mostrar resultados
                st.header("🎯 Grasas Recomendadas")
                
                # Gráfico de similitudes
                fig = px.bar(
                    recomendaciones.reset_index(),
                    x='Similitud',
                    y='index',
                    orientation='h',
                    title='Nivel de Similitud de las Grasas Recomendadas',
                    labels={'index': 'ID Grasa', 'Similitud': 'Similitud'}
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de resultados
                st.subheader("📋 Detalles de las Recomendaciones")
                
                columnas_mostrar = [
                    'Aceite Base', 'Espesante', 'Grado NLGI Consistencia',
                    'Viscosidad del Aceite Base a 40°C. cSt', 'Punto de Gota, °C',
                    'Temperatura de Servicio °C, min', 'Temperatura de Servicio °C, max', 'Similitud'
                ]
                
                # Formatear resultados para mostrar
                display_df = recomendaciones[columnas_mostrar].copy()
                display_df['Similitud'] = display_df['Similitud'].round(3)
                
                st.dataframe(
                    display_df.style.format({
                        'Viscosidad del Aceite Base a 40°C. cSt': '{:.1f}',
                        'Punto de Gota, °C': '{:.1f}',
                        'Similitud': '{:.3f}'
                    }).background_gradient(subset=['Similitud'], cmap='Blues'),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error al buscar recomendaciones: {str(e)}")
        
        elif buscar_btn:
            st.warning("⚠️ Por favor, ingrese al menos algunas características principales")
        
        else:
            # Pantalla de bienvenida
            st.header("👋 ¡Bienvenido!")
            st.markdown("""
            Este sistema te ayuda a encontrar las grasas lubricantes más similares 
            según las características que necesites.
            
            **Cómo usar:**
            1. 📝 Completa las características deseadas en el panel lateral
            2. 🚀 Haz clic en 'Buscar Recomendaciones'
            3. 📊 Revisa los resultados y gráficos
            
            **Características principales a considerar:**
            - **Aceite Base**: Tipo de base lubricante
            - **Espesante**: Agente espesante utilizado  
            - **Grado NLGI**: Consistencia de la grasa
            - **Viscosidad**: Viscosidad del aceite base a 40°C
            """)
            
            # Mostrar algunas estadísticas
            st.subheader("📈 Distribución del Catálogo")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1 = px.histogram(df, x='Aceite Base', title='Distribución por Aceite Base')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.histogram(df, x='Espesante', title='Distribución por Espesante')
                st.plotly_chart(fig2, use_container_width=True)
            
            with col3:
                fig3 = px.histogram(df, x='Grado NLGI Consistencia', title='Distribución por Grado NLGI')
                st.plotly_chart(fig3, use_container_width=True)

else:
    st.error("No se pudieron cargar los datos. Verifica que el archivo 'expanded_data.csv' esté en la misma carpeta.")