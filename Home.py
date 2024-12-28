import streamlit as st
from PIL import Image

def main():
    # Introduction page
    st.title("Bienvenido Viajero🧳")
    
    # Display PNG image
    image = Image.open("utils/plane.png")  # Path to your image file
    st.image(image, caption="Airline_delay", use_container_width=True)

    st.write("""
    **Este es un proyecto de análisis de datos sobre vuelos y retrasos.**
    
    En esta aplicación, exploraremos los datos relacionados con los vuelos, analizaremos posibles hipótesis sobre los retrasos y desarrollaremos modelos predictivos para mejorar la gestión de los vuelos. El dataset contiene información detallada sobre vuelos, como la aerolínea, los aeropuertos de origen y destino, la duración del vuelo, y si hubo un retraso o no.
    
    A continuación, podrás explorar diferentes secciones, como:
    - **Análisis Exploratorio de Datos (EDA)**: Análisis y visualización inicial de los datos.
    - **Hipótesis**: Exploración de posibles hipótesis relacionadas con los retrasos de vuelos.
    - **Modelos Predictivos**: Implementación de modelos para predecir los retrasos.
    
    ¡Esperamos que disfrutes explorando los datos y los modelos!
    """)
    
if __name__ == "__main__":
    main()
