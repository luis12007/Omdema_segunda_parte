import streamlit as st
import pandas as pd
import plotly.express as px

def hipotesis_page():
    st.title("Análisis de Hipótesis")
    st.write("En esta sección, exploraremos posibles hipótesis relacionadas con los retrasos de vuelos.")
    st.markdown("### Ejemplo de hipótesis:")
    st.write("- Los retrasos son más frecuentes en ciertas aerolíneas específicas.")
    st.write("- Los vuelos por la tarde tienen mayor probabilidad de retraso que los de la mañana.")
    st.write("- La aerolínea con el código 'DL' tiene un promedio de retraso menor que las demás aerolíneas.")
    st.write("- Los vuelos que salen de aeropuertos principales como 'ATL' tienen menores retrasos en promedio que los vuelos que parten de otros aeropuertos.")
    st.write("- Los vuelos realizados en días laborales (días 1-5) tienen menos retrasos que los vuelos de fin de semana (días 6-7).")
    st.write("- Los vuelos realizados en días específicos de la semana tienen un promedio de retraso significativamente mayor.")

    # H1: Relación entre duración y retrasos
    st.markdown("### H1: Relación entre duración y retrasos")
    st.write("""
    **Hipótesis:** Los vuelos más largos (en minutos) tienen más probabilidad de presentar retrasos mayores en "Time".
    
    Este gráfico explora la relación entre la duración del vuelo ("Length") y el retraso ("Time").
    """)

    # Gráfico de líneas para representar el retraso promedio en función de intervalos de duración
    dataset = pd.read_csv("data/airlines_delay.csv")  # Load dataset
    dataset['Length_bins'] = pd.cut(dataset['Length'], bins=10)
    promedio_por_bin = dataset.groupby('Length_bins', observed=False)['Time'].mean().reset_index()
    
    promedio_por_bin['Length_bins'] = promedio_por_bin['Length_bins'].astype(str)

    # Create the Plotly line plot
    fig = px.line(
        promedio_por_bin,
        x='Length_bins',
        y='Time',
        title="Promedio de Retrasos según Duración del Vuelo (Agrupado)",
        labels={"Length_bins": "Duración del Vuelo (rangos en minutos)", "Time": "Promedio de Retraso (minutos)"},
        markers=True
    )

    # Customize the plot (optional)
    fig.update_traces(line_color='blue', line_width=2)
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    st.plotly_chart(fig)

    st.write("""
    **Conclusión de la Hipótesis:**
    Aunque existe una tendencia general donde vuelos más largos tienen mayores retrasos, no es una relación completamente lineal. 
    Esto sugiere que otros factores además de la duración podrían influir en los retrasos.
    """)

    # H2: La aerolínea 'DL' tiene un promedio de retraso menor que las demás
    st.markdown("### H2: La aerolínea 'DL' tiene un promedio de retraso menor que las demás")
    st.write("""
    **Hipótesis:** La aerolínea con el código 'DL' tiene un promedio de retraso menor que las demás aerolíneas.
    
    Este gráfico compara el desempeño de las aerolíneas usando la columna "Airline".
    """)

    promedio_retrasos = dataset.groupby('Airline', observed=False)['Time'].mean().sort_values().reset_index()

    fig = px.bar(
        promedio_retrasos,
        x='Airline',
        y='Time',
        title="Promedio de Retrasos por Aerolínea",
        labels={"Airline": "Aerolínea", "Time": "Promedio de Retraso (minutos)"},
        color_discrete_sequence=['orange']  # Set bar color to orange
    )

    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    fig.update_yaxes(showgrid=True)  # Add gridlines to y-axis
    fig.update_xaxes(showgrid=False) # Remove gridlines from x-axis
    st.plotly_chart(fig)

    st.write("""
    **Conclusión de la Hipótesis:**
    La aerolínea 'DL' tiene un desempeño destacable en términos de retraso promedio, validando parcialmente la hipótesis. 
    Sin embargo, no es la mejor aerolínea en este aspecto.
    """)

    # H3: Los aeropuertos principales tienen menores retrasos promedio
    st.markdown("### H3: Los aeropuertos principales tienen menores retrasos promedio")
    st.write("""
    **Hipótesis:** Los vuelos que salen de aeropuertos principales como 'ATL' tienen menores retrasos en promedio que los vuelos que parten de otros aeropuertos.
    
    Este gráfico explora la relación entre los aeropuertos de origen y el retraso promedio.
    """)

    # Calculate average delays by airport
    retrasos_por_aeropuerto = dataset.groupby('AirportFrom', observed=False)['Time'].mean().sort_values(ascending=False)

    # Select the top 10 airports for the plot
    top_10_aeropuertos = retrasos_por_aeropuerto.head(10).reset_index()

    # Create the Plotly bar plot
    fig = px.bar(
        top_10_aeropuertos,
        x='AirportFrom',
        y='Time',
        title="Top 10 Aeropuertos con Mayores Retrasos Promedio",
        labels={"AirportFrom": "Aeropuerto de Origen", "Time": "Promedio de Retraso (minutos)"},
        color_discrete_sequence=['green']  # Set bar color to green
    )

    # Customize the plot layout
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    fig.update_yaxes(showgrid=True)  # Add gridlines to y-axis
    fig.update_xaxes(showgrid=False) # Remove gridlines to x-axis
    st.plotly_chart(fig)

    st.write("""
    **Conclusión de la Hipótesis:**
    La hipótesis se mantiene, ya que los aeropuertos principales como 'ATL' y 'ORD' no figuran entre los aeropuertos con mayores retrasos promedio, confirmando que son más eficientes en comparación con otros aeropuertos.
    """)

    # H4: Los vuelos en días laborales tienen menos retrasos que los de fin de semana
    st.markdown("### H4: Los vuelos en días laborales tienen menos retrasos que los de fin de semana")
    st.write("""
    **Hipótesis:** Los vuelos realizados en días laborales (días 1-5 en "DayOfWeek") tienen menos retrasos que los vuelos de fin de semana (días 6-7).
    
    Este gráfico muestra cómo los retrasos varían según el día de la semana.
    """)

    retrasos_por_dia = dataset.groupby('DayOfWeek', observed=False)['Time'].mean().reset_index()

    # Create the Plotly bar plot
    fig = px.bar(
        retrasos_por_dia,
        x='DayOfWeek',
        y='Time',
        title="Promedio de Retrasos por Día de la Semana",
        labels={"DayOfWeek": "Día de la Semana (1=Lunes, 7=Domingo)", "Time": "Promedio de Retraso (minutos)"},
        color_discrete_sequence=['purple']  # Set bar color to purple
    )

    # Customize the plot layout (optional)
    fig.update_layout(template="plotly_white")
    fig.update_yaxes(showgrid=True)  # Add gridlines to y-axis
    fig.update_xaxes(showgrid=False, type='category')  # Format x-axis as category
    st.plotly_chart(fig)

    st.write("""
    **Conclusión de la Hipótesis:**
    La hipótesis se valida. Los vuelos en días laborales (1-5) tienden a tener menos retrasos en promedio, posiblemente debido a menor tráfico aéreo o una mejor gestión operativa en comparación con los fines de semana (6-7), donde los retrasos tienden a ser mayores, especialmente el domingo.
    """)

    # H5: Algunos días específicos tienen retrasos mayores
    st.markdown("### H5: Algunos días específicos tienen retrasos mayores")
    st.write("""
    **Hipótesis:** Los vuelos realizados en días específicos de la semana tienen un promedio de retraso significativamente mayor.
    
    Este gráfico analiza los retrasos promedio para cada día de la semana, mostrando las tendencias y picos de retrasos en ciertos días.
    """)

    # Create the Plotly line plot to show the trend
    fig = px.line(
        retrasos_por_dia,
        x='DayOfWeek',
        y='Time',
        title="Promedio de Retrasos por Día de la Semana (Tendencia)",
        labels={"DayOfWeek": "Día de la Semana (1=Lunes, 7=Domingo)", "Time": "Promedio de Retraso (minutos)"},
        markers=True  # Adds markers to data points
    )

    # Customize the plot (optional)
    fig.update_traces(line_color='red', line_width=2)  # Set line color and width
    fig.update_layout(template="plotly_white")  # Use a clean template
    fig.update_xaxes(type='category')  # Treat DayOfWeek as categorical for proper spacing
    st.plotly_chart(fig)

    st.write("""
    **Conclusión de la Hipótesis:**
    La hipótesis se confirma. Los domingos destacan como días con mayores retrasos promedio, lo que podría estar relacionado con el incremento de vuelos por razones comerciales o turísticas. Esto muestra que algunos días específicos, como el domingo, tienen condiciones que favorecen los retrasos.
    """)

# Call the function to run the page
hipotesis_page()
