import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from sklearn import metrics
from streamlit_option_menu import option_menu

 
#Extraer los archivos pkl
with open('modelo_rf.pkl', 'rb') as rf:
    modelo_rf = pickle.load(rf)

data = [
        "- age: age of the patient (years)",
        "- anaemia: decrease of red blood cells or hemoglobin (boolean)",
        "- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)",
        "- diabetes: if the patient has diabetes (boolean)",
        "- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)",
        "- high blood pressure: if the patient has hypertension (boolean)",
        "- platelets: platelets in the blood (kiloplatelets/mL)",
        "- serum creatinine: level of serum creatinine in the blood (mg/dL)",
        "- serum sodium: level of serum sodium in the blood (mEq/L)",
        "- sex: woman or man (binary)",
        "- smoking: if the patient smokes or not (boolean)",
        "- time: follow-up period (days)"
    ]

#Funcion para realizar las predicciones del modelo
def classifier(df):
    X=df.iloc[:, :].values
    prediction = modelo_rf.predict(X)
    df['Prediction'] = prediction
    df_k = df
    return df_k

#Funcion para realizar las predicciones del modelo cuando tiene Y
def classifier2(df):
    X=df.iloc[:, :-1].values
    Y=df.iloc[:, -1].values
    prediction = modelo_rf.predict(X)
    df['Prediction'] = prediction
    df_k = df
    report = metrics.classification_report(Y, df_k['Prediction'], output_dict=True)
    report_df = pd.DataFrame(report)
    return df_k, report_df

#Funcion para mostrar la grafica del peso de variables en el modelo
def plot():
    importances = modelo_rf.feature_importances_
    features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    importances_df = pd.DataFrame({'Variable': features, 'Importancia': importances})
    importances_df = importances_df.sort_values('Importancia', ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x='Variable', y='Importancia', data=importances_df, ax= ax)
    plt.xticks(rotation=90)
    plt.xlabel('Variables')
    plt.ylabel('Importancia')
    st.pyplot(fig)

def plot_p(df):
    fig, ax = plt.subplots(3,2, figsize = (10,10))
    sns.boxplot(y = 'time', x = 'Prediction', data = df, ax= ax[0, 0])
    sns.boxplot(y = 'serum_creatinine', x = 'Prediction', data = df, ax= ax[0, 1])
    sns.boxplot(y = 'ejection_fraction', x = 'Prediction', data = df, ax= ax[1, 0])
    sns.boxplot(y = 'age', x = 'Prediction', data = df, ax= ax[1, 1])
    sns.boxplot(y = 'creatinine_phosphokinase', x = 'Prediction', data = df, ax= ax[2, 0])
    sns.boxplot(y = 'serum_sodium', x = 'Prediction', data = df, ax= ax[2, 1])
    st.pyplot(fig)

def main():
    st.title('Clasificación de supervivencia de pacientes con insuficiencia cardiaca.')
    
    selected = option_menu(
        menu_title= 'Bienvenidos',
        options= ['Inicio', 'Prueba el Modelo'],
        icons= ['house', 'calculator'],
        menu_icon='bar-chart',
        default_index=0,
        orientation='horizontal'
    )

    if selected == 'Inicio':
        st.header('Acerca del Modelo')
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>El modelo de clasificación sobre la supervivencia de los pacientes con insuficiencia cardíaca, ha sido entrenado con datos de pacientes con insuficiencia cardíaca que fueron admitidos en el Instituto de Cardiología y el hospital Aliado de Faisalabad, Pakistán, entre abril y diciembre de 2015. </span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>Se construyó el modelo a partir de los datos disponibles en: \nhttps://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records</span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>Todos los pacientes eran mayores de 40 años, tenían disfunción sistólica ventricular izquierda y pertenecían a la clase III y IV de la NYHA. </span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>Se analizaron 12 atributos clínicos, corporales y de estilo de vida de 299 pacientes con insuficiencia cardiaca, de los cuales 203 sobrevivieron (death event = 0) en el tiempo de seguimiento y lastimosamente 96 fallecieron (death event = 1). Se utilizaron diversos factores potenciales, como la edad, la fracción de eyección, la creatinina sérica, el sodio sérico, la anemia, las plaquetas, la creatina-fosfocinasa, la presión arterial, el género, la diabetes, el hábito de fumar y el tiempo que lleva el paciente en observación.</span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>A continuación se presenta una matriz de correlación entre las variables involucradas, en donde los colores que sean más claros representaran mayor relación comparada con los más oscuros.</span>", unsafe_allow_html=True)
        st.image('correlacion.jpeg', width=600)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>Se realizó un estudio detallado sobre la base de datos, comprendiendo que se encontraba desbalanceada, por lo que se procedió a realizar la comparación de diferentes técnicas de balanceo con distintos modelos de clasificación. Obteniendo como resultado el modelo Random Forest con datos balanceados por la técnica SMOTE presentado en esta página. </span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>El modelo de clasificación de Random Forest disponible en esta página tiene una precisión (accuracy) del 85%, significa que el modelo ha clasificado correctamente el 85% de las muestras de datos que se utilizaron para evaluarlo. En otras palabras, de cada 100 muestras de datos que el modelo ha evaluado, se espera que 85 de ellas sean clasificadas correctamente.</span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>Ademas, en el estudio al aplicar K-Fold Cross Validation se identifica que para el modelo de random forest se tiene una metrida de F1 de 0.871072 indicando que el modelo tiene un buen equilibrio entre la precisión y la exhaustividad.</span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>La importancia de cada característica se mide como un valor entre 0 y 1, donde los valores más altos indican una mayor importancia. En el siguiente grafico se presenta la relevancia de las variables involucradas en los datos clínicos de los pacientes para lograr realizar la clasificación.</span>", unsafe_allow_html=True)
        
        plot()
    if selected == 'Prueba el Modelo':
        st.header('Prueba del Modelo')
        st.subheader('Adjunte el archivo .csv')
        st.write("Lista de datos:")
        st.write('\n'.join(data), sep='\n\n')
        # for item in data:
            #st.write(item)
        uploaded_file = st.file_uploader('A continuación cargue el archivo .csv:')
        if uploaded_file is not None:
          df = pd.read_csv(uploaded_file)

        opciones = ['Sin variable respuesta', 'Con variable respuesta']
        opcion_seleccionada = st.radio('Seleccione una opción:', opciones)
        if st.button('RUN'):

            if opcion_seleccionada == 'Sin variable respuesta':
                rf_pred = classifier(df)
                st.subheader('Clasificaciones del modelo')
                st.dataframe(rf_pred)  
                st.subheader('Gráficas')
                st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>A continuación se pueden observar diagramas de cajas y bigotes que demuestran la relación entre las 6 variables más importantes con respecto a la respuesta final de clasificación, (0) si sobrevive y (1) si fallece el paciente.</span>", unsafe_allow_html=True)
                plot_p(rf_pred)

            elif opcion_seleccionada == 'Con variable respuesta':
                rf_pred, report_df = classifier2(df)
                st.subheader('Clasificaciones del modelo')
                st.dataframe(rf_pred) 
                st.subheader('Reporte del modelo')
                st.dataframe(report_df) 
                st.subheader('Gráficas')
                st.write("<span style='font-size: 18px; font-family: Times New Roman; text-align: justify;'>A continuación se pueden observar diagramas de cajas y bigotes que demuestran la relación entre las 6 variables más importantes con respecto a la respuesta final de clasificación, (0) si sobrevive y (1) si fallece el paciente.</span>", unsafe_allow_html=True)
                plot_p(rf_pred)

            

if __name__ == '__main__':
    main()