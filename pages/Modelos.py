import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.express as px
from lightgbm import LGBMClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image


# Assuming dataset is already loaded
dataset = pd.read_csv("data/airlines_delay.csv")  # Load dataset

def modelos_page():
    st.title("Modelos Predictivos")
    st.write("En esta sección, implementaremos modelos predictivos para analizar y predecir retrasos de vuelos.")

    # Introducción
    st.markdown("### ¿Supervisado o No Supervisado?")
    st.write("""
    **Modelo Supervisado:**
    El dataset ya incluye una columna llamada **Class**, que es una variable binaria indicando si un vuelo tuvo retraso (1) o no (0). 
    Esto implica que ya contamos con datos etiquetados, lo cual es el principal requisito para aplicar **aprendizaje supervisado**.
    
    **Problema de clasificación:**
    El objetivo sería predecir si un vuelo tendrá retraso basado en las características disponibles (hora, duración, aerolínea, aeropuertos, etc.). Este es un caso típico de **clasificación binaria**.
    
    **Datos estructurados:**
    Todas las columnas en el dataset están bien definidas y estructuradas, lo que facilita entrenar un modelo supervisado como un **árbol de decisión**, **regresión logística**, o incluso modelos más avanzados como **Random Forest** o **SVM** (Máquinas de Vectores de Soporte).
    
    **¿Por qué no aprendizaje no supervisado?**
    El aprendizaje no supervisado se utiliza cuando no hay una variable objetivo o etiqueta que predetermine las categorías. En este caso, la columna **Class** ya proporciona esa guía, por lo que el uso de métodos como clustering (por ejemplo, K-Means) no sería ideal.
    """)

    # Features and target variable
    Features = dataset.drop('Class', axis=1)
    target = dataset['Class']

    # Train-Test Split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Features, target, random_state=202, test_size=0.3)

    # Preprocessing (scaling numerical features, one-hot encoding for categorical features)
    numerical_features = ['Flight', 'Time', 'Length', 'DayOfWeek']
    categorical_features = ['Airline', 'AirportFrom', 'AirportTo']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),  # Scaling for numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-Hot encoding for categorical features
        ]
    )

    # Transform the training and test datasets
    Xtrain_processed = preprocessor.fit_transform(Xtrain)  # Fit and transform the training data
    Xtest_processed = preprocessor.transform(Xtest)  # Transform the test data

    # Modelo 1: Regresión Logística
    st.markdown("### Primer Modelo Supervisado: Regresión Logística")
    st.write("""
    La **Regresión Logística** es un modelo estadístico utilizado para problemas de clasificación binaria, lo que significa que clasifica los datos en una de dos categorías posibles:
    
    - 0: **sin retraso**
    - 1: **con retraso**
    
    La regresión logística utiliza la función logística o sigmoide para transformar una combinación lineal de las características en un rango de probabilidad entre 0 y 1.
    
    Si P ≥ 0.5, se clasifica como **1** (retraso). Si P < 0.5, se clasifica como **0** (sin retraso).
    """)

        # Display PNG image
    image = Image.open("utils/regression.png")  # Path to your image file
    st.image(image, caption="regression", use_container_width=True)


    # Entrenamiento del modelo de Regresión Logística
    logistic_model = LogisticRegression(max_iter=5000)
    logistic_model.fit(Xtrain_processed, Ytrain)

    # Predicciones
    predictions_logistic = logistic_model.predict(Xtest_processed)

    # Reporte de Clasificación
    st.write("**Reporte de Clasificación (Regresión Logística):**")
    st.text(classification_report(Ytest, predictions_logistic))

    # Matriz de Confusión
    st.write("**Matriz de Confusión (Regresión Logística):**")
    # Create the confusion matrix
    conf_matrix = confusion_matrix(Ytest, predictions_logistic)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logistic_model.classes_)
    disp.plot(cmap="Blues", ax=ax)
    # Set the title
    ax.set_title("Matriz de Confusión - Regresión Logística")

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Métricas Adicionales
    st.write("**Métricas Adicionales (Regresión Logística):**")
    accuracy = accuracy_score(Ytest, predictions_logistic)
    precision = precision_score(Ytest, predictions_logistic)
    recall = recall_score(Ytest, predictions_logistic)
    f1 = f1_score(Ytest, predictions_logistic)

    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-Score:** {f1:.4f}")

    # Modelo 2: LightGBM
    st.markdown("### Segundo Modelo Supervisado: LightGBM")
    st.write("""
    **LightGBM** (Light Gradient Boosting Machine) es un algoritmo basado en el método de boosting de gradiente. Es muy eficiente y está optimizado para trabajar con grandes conjuntos de datos y características complejas. LightGBM utiliza árboles de decisión como modelos base, pero con varias mejoras que lo hacen rápido y preciso.

    El boosting de gradiente es una técnica que combina múltiples modelos débiles (generalmente árboles de decisión) para crear un modelo fuerte. Esto permite entrenar más rápido, incluso en grandes conjuntos de datos.

    En lugar de crecer nivel por nivel (como en otros algoritmos), LightGBM expande primero las ramas más prometedoras. Esto se llama crecimiento basado en hojas (leaf-wise growth), lo que mejora la precisión.

    **¿Por qué es adecuado para este caso?**

    - **Conjunto de datos con múltiples variables:** El dataset tiene varias características relevantes (Time, Length, Airline, etc.) que LightGBM puede procesar eficientemente para identificar patrones complejos.
    - **Precisión para clasificación binaria:** Este caso es una tarea de clasificación binaria (Class: retraso/no retraso), y LightGBM es ideal para este tipo de problemas.
    - **Eficiencia en grandes volúmenes de datos:** Tenemos un dataset con más de 300,000 datos, lo que hace perfecta la aplicación de LightGBM ya que se especializa en la optimización de grandes datasets.
    """)

            # Display PNG image
    image = Image.open("utils/download.png")  # Path to your image file
    st.image(image, caption="GBM", use_container_width=True)


    # Entrenamiento del modelo LightGBM
    lgbm_model = LGBMClassifier(random_state=202, n_estimators=100, learning_rate=0.1)
    lgbm_model.fit(Xtrain_processed, Ytrain)

    # Predicciones
    predictions_lgbm = lgbm_model.predict(Xtest_processed)

    # Reporte de Clasificación
    st.write("**Reporte de Clasificación (LightGBM):**")
    st.text(classification_report(Ytest, predictions_lgbm))

    # Matriz de Confusión
    st.write("**Matriz de Confusión (LightGBM):**")
    conf_matrix_lgbm = confusion_matrix(Ytest, predictions_lgbm)

    # Create a figure and axis
    fig_lgbm, ax_lgbm = plt.subplots(figsize=(8, 6))

    # Display the confusion matrix
    disp_lgbm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lgbm, display_labels=lgbm_model.classes_)
    disp_lgbm.plot(cmap="Blues", ax=ax_lgbm)
    # Set the title
    ax_lgbm.set_title("Matriz de Confusión - LightGBM")

    # Display the plot in Streamlit
    st.pyplot(fig_lgbm)

    # Métricas Adicionales
    st.write("**Métricas Adicionales (LightGBM):**")
    accuracy_lgbm = accuracy_score(Ytest, predictions_lgbm)
    precision_lgbm = precision_score(Ytest, predictions_lgbm)
    recall_lgbm = recall_score(Ytest, predictions_lgbm)
    f1_lgbm = f1_score(Ytest, predictions_lgbm)

    st.write(f"**Accuracy (LightGBM):** {accuracy_lgbm:.4f}")
    st.write(f"**Precision (LightGBM):** {precision_lgbm:.4f}")
    st.write(f"**Recall (LightGBM):** {recall_lgbm:.4f}")
    st.write(f"**F1-Score (LightGBM):** {f1_lgbm:.4f}")

# Call the function to run the page
modelos_page()
