import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def eda_page():
    st.title("Análisis Exploratorio de Datos (EDA)")
    
    try:
        # Cargar el conjunto de datos
        datos = pd.read_csv("data/airlines_delay.csv")
        st.write("Vista previa del conjunto de datos:")
        st.dataframe(datos.head())  # Muestra las primeras filas del conjunto de datos
        
        # Información básica del conjunto de datos
        st.write("Información del conjunto de datos:")
        st.write(datos.info())

        # Resumen estadístico de las columnas numéricas y categóricas
        st.write("Resumen estadístico de las columnas numéricas y categóricas:")
        summary = datos.describe(include='all')
        st.write(summary)
        
        # Tipos de las columnas
        st.write("Tipos de columnas:")
        column_types = datos.dtypes
        st.write(column_types)

        # Tamaño del dataset
        st.write(f"Tamaño del dataset: {datos.shape[0]} filas, {datos.shape[1]} columnas")
        
        # Convertir columnas categóricas a tipo 'category'
        categorical_columns = ['Airline', 'AirportFrom', 'AirportTo', 'Class']
        datos[categorical_columns] = datos[categorical_columns].astype('category')
        
        # Visualización de la distribución de la hora del día
        time_data = datos['Time']
        fig_time = go.Figure(data=[go.Histogram(
            x=time_data,
            nbinsx=50,
            marker_color='blue',
            opacity=0.7,
            name='Time'
        )])
        fig_time.update_layout(
            title="Distribución de Retrasos (Time)",
            xaxis_title="Minutos de Retraso",
            yaxis_title="Frecuencia",
            bargap=0.1,
            legend=dict(title="Leyenda"),
            template="plotly_white"
        )
        fig_time.update_xaxes(showgrid=True)
        fig_time.update_yaxes(showgrid=True)
        st.plotly_chart(fig_time)

        # Visualización de la duración de los vuelos
        length_data = datos['Length']
        fig_length = go.Figure(data=[go.Histogram(
            x=length_data,
            nbinsx=50,
            marker_color='green',
            opacity=0.7,
            name='Length'
        )])
        fig_length.update_layout(
            title="Distribución de Duración de Vuelos (Length)",
            xaxis_title="Duración (minutos)",
            yaxis_title="Frecuencia",
            bargap=0.1,
            legend=dict(title="Leyenda"),
            template="plotly_white"
        )
        fig_length.update_xaxes(showgrid=True)
        fig_length.update_yaxes(showgrid=True)
        st.plotly_chart(fig_length)

        # Visualización de las aerolíneas más frecuentes
        airline_counts = datos['Airline'].value_counts()
        airline_data = airline_counts.head(10)
        fig_airlines = go.Figure(data=[go.Bar(
            x=airline_data.index,
            y=airline_data.values,
            marker_color='orange',
            opacity=0.8,
            name='Frecuencia'
        )])
        fig_airlines.update_layout(
            title="Top 10 Aerolíneas Más Frecuentes",
            xaxis_title="Aerolínea",
            yaxis_title="Frecuencia",
            template="plotly_white",
            bargap=0.2
        )
        fig_airlines.update_yaxes(showgrid=True)
        fig_airlines.update_xaxes(showgrid=False)
        st.plotly_chart(fig_airlines)

        # Visualización de los aeropuertos de origen más frecuentes
        airport_from_counts = datos['AirportFrom'].value_counts()
        airport_data = airport_from_counts.head(10)
        fig_airports = go.Figure(data=[go.Bar(
            x=airport_data.index,
            y=airport_data.values,
            marker_color='purple',
            opacity=0.8,
            name='Frecuencia'
        )])
        fig_airports.update_layout(
            title="Top 10 Aeropuertos de Origen Más Frecuentes",
            xaxis_title="Aeropuerto de Origen",
            yaxis_title="Frecuencia",
            template="plotly_white",
            bargap=0.2
        )
        fig_airports.update_xaxes(showgrid=False)
        fig_airports.update_yaxes(showgrid=True)
        st.plotly_chart(fig_airports)

    except FileNotFoundError:
        st.error("No se pudo cargar el archivo 'airlines_delay.csv'. Asegúrese de que el archivo esté en la carpeta correcta.")
        return

# Llamar la función para mostrar la página
eda_page()
