import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
from chat_module import get_response  # Import your local chat module

# Load ML models
diabetes_model = pickle.load(open(r'C:\Users\dell\Downloads\third year units\training_models\diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(r'C:\Users\dell\Downloads\third year units\training_models\heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open(r'C:\Users\dell\Downloads\third year units\training_models\parkinson_model.sav', 'rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu('Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           "Parkinson’s Prediction",
                           "Patient Chat",
                           "Treatment Plan",
                           "Health Analytics"],
                          menu_icon='hospital-fill',
                          icons=['activity', 'heart', 'person', 'chat', 'clipboard-pulse', 'bar-chart-line'],
                          default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    pregnancies = st.number_input('Number of Pregnancies', min_value=0)
    glucose = st.number_input('Glucose Level', min_value=0)
    blood_pressure = st.number_input('Blood Pressure value', min_value=0)
    skin_thickness = st.number_input('Skin Thickness value', min_value=0)
    insulin = st.number_input('Insulin Level', min_value=0)
    bmi = st.number_input('BMI value')
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function value')
    age = st.number_input('Age of the Person', min_value=0)

    if st.button('Diabetes Test Result'):
        prediction = diabetes_model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        st.success('The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic')

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    age = st.number_input('Age', min_value=0)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.number_input('Chest Pain types (0-3)', min_value=0, max_value=3)
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Serum Cholestoral in mg/dl')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    restecg = st.number_input('Resting Electrocardiographic results (0-2)', min_value=0, max_value=2)
    thalach = st.number_input('Maximum Heart Rate achieved')
    exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('ST depression induced')
    slope = st.number_input('Slope of the peak exercise ST segment', min_value=0, max_value=2)
    ca = st.number_input('Major vessels colored by fluoroscopy (0-3)', min_value=0, max_value=3)
    thal = st.number_input('Thal (0-3)', min_value=0, max_value=3)

    if st.button('Heart Disease Test Result'):
        data = [age, 1 if sex == 'Male' else 0, cp, trestbps, chol, 1 if fbs == 'Yes' else 0,
                restecg, thalach, 1 if exang == 'Yes' else 0, oldpeak, slope, ca, thal]
        prediction = heart_disease_model.predict([data])
        st.success('The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease')

# Parkinson's Prediction
elif selected == "Parkinson’s Prediction":
    st.title("Parkinson’s Disease Prediction using ML")

    fo = st.number_input('MDVP:Fo(Hz)')
    fhi = st.number_input('MDVP:Fhi(Hz)')
    flo = st.number_input('MDVP:Flo(Hz)')
    jitter_percent = st.number_input('MDVP:Jitter(%)')
    jitter_abs = st.number_input('MDVP:Jitter(Abs)')
    rap = st.number_input('MDVP:RAP')
    ppq = st.number_input('MDVP:PPQ')
    ddp = st.number_input('Jitter:DDP')
    shimmer = st.number_input('MDVP:Shimmer')
    shimmer_db = st.number_input('MDVP:Shimmer(dB)')
    apq3 = st.number_input('Shimmer:APQ3')
    apq5 = st.number_input('Shimmer:APQ5')
    apq = st.number_input('MDVP:APQ')
    dda = st.number_input('Shimmer:DDA')
    nhr = st.number_input('NHR')
    hnr = st.number_input('HNR')
    rpde = st.number_input('RPDE')
    dfa = st.number_input('DFA')
    spread1 = st.number_input('spread1')
    spread2 = st.number_input('spread2')
    d2 = st.number_input('D2')
    ppe = st.number_input('PPE')

    if st.button("Parkinson's Test Result"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinsons_model.predict(input_data)
        st.success("The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease")

# Patient Chat (local module)
elif selected == "Patient Chat":
    st.title("💬 AI Health Assistant")
    question = st.text_area("Ask your health-related question:")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            reply = get_response(question)
            st.success("🧠 Assistant's Answer:")
            st.markdown(reply)

# Treatment Plan
elif selected == 'Treatment Plan':
    st.title("🦥 Personalized Treatment Plan Generator")
    st.write("Enter your diagnosed condition to receive a recommended treatment plan.")

    condition = st.text_input("Diagnosed Condition (e.g., Diabetes, Hypertension, etc.)")

    if st.button("Generate Plan"):
        if condition:
            treatment_suggestions = {
                "diabetes": {
                    "medications": "Metformin, Insulin",
                    "lifestyle": "Low-carb diet, Regular exercise",
                    "tests": "HbA1c every 3 months"
                },
                "hypertension": {
                    "medications": "ACE inhibitors, Beta-blockers",
                    "lifestyle": "Reduce salt, Stress management",
                    "tests": "BP check weekly, ECG yearly"
                }
            }
            data = treatment_suggestions.get(condition.lower())

            if data:
                st.subheader("📋 Recommended Treatment Plan")
                st.markdown(f"**Medications:** {data['medications']}")
                st.markdown(f"**Lifestyle Advice:** {data['lifestyle']}")
                st.markdown(f"**Follow-up Tests:** {data['tests']}")
            else:
                st.info("No standard treatment found. Please consult a doctor.")
        else:
            st.warning("Please enter a condition to generate treatment plan.")

# Health Analytics
elif selected == 'Health Analytics':
    st.title("📊 Health Analytics Dashboard")
    st.write("Visualize your vital signs over the past week.")

    days = pd.date_range(end=pd.Timestamp.today(), periods=7)
    heart_rate = np.random.randint(70, 100, size=7)
    blood_pressure = np.random.randint(110, 140, size=7)
    glucose = np.random.randint(90, 140, size=7)

    df = pd.DataFrame({
        "Date": days,
        "Heart Rate (bpm)": heart_rate,
        "Blood Pressure (mmHg)": blood_pressure,
        "Blood Glucose (mg/dL)": glucose
    })

    st.line_chart(df.set_index("Date"))
    st.success("✅ AI Insight: Your vitals are mostly stable. Maintain a balanced lifestyle.")
