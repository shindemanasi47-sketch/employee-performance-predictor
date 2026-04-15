import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Employee Performance Predictor", page_icon="🏢", layout="wide")

# --- Load Models ---
model = pickle.load(open('model.pkl', 'rb'))
le_target = pickle.load(open('le_target.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("🏢 Employee Performance")
st.sidebar.markdown("---")
st.sidebar.info("This app predicts employee performance category using an XGBoost model trained on HR data.")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗂️ Modes")
mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Bulk CSV Prediction", "📊 EDA"])

# --- Column Order ---
feature_cols = ['Age', 'Gender', 'City', 'Education_Level', 'Department',
                'Experience_Years', 'Monthly_Salary', 'Projects_Completed',
                'Training_Hours', 'Certifications', 'Work_Life_Balance',
                'Job_Satisfaction', 'Manager_Rating', 'Overtime_Hours',
                'Commute_Time_Min', 'Laptop_Issue_Count', 'Cafeteria_Rating',
                'Internet_Stability', 'Last_Promotion_Years', 'Absenteeism_Days']

num_cols = ['Age', 'Experience_Years', 'Monthly_Salary', 'Projects_Completed',
            'Manager_Rating', 'Overtime_Hours', 'Commute_Time_Min',
            'Laptop_Issue_Count', 'Cafeteria_Rating', 'Last_Promotion_Years',
            'Absenteeism_Days', 'Training_Hours', 'Certifications',
            'Work_Life_Balance', 'Job_Satisfaction']

cat_cols = ['Gender', 'City', 'Department', 'Education_Level', 'Internet_Stability']

# ================================
# MODE 1 — SINGLE PREDICTION
# ================================
if mode == "Single Prediction":

    st.title("🏢 Employee Performance Predictor")
    st.markdown("Fill in the employee details below to predict their performance category.")
    st.markdown("---")

    st.subheader("📊 Employee Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 65, 30)
        experience = st.number_input("Experience Years", 0, 40, 5)
        salary = st.number_input("Monthly Salary", 10000, 200000, 50000)
        projects = st.number_input("Projects Completed", 0, 50, 10)
        manager_rating = st.slider("Manager Rating", 1.0, 5.0, 3.0)

    with col2:
        overtime = st.number_input("Overtime Hours", 0, 100, 10)
        commute = st.number_input("Commute Time (min)", 0, 180, 30)
        laptop_issues = st.number_input("Laptop Issue Count", 0, 20, 1)
        cafeteria = st.slider("Cafeteria Rating", 1.0, 5.0, 3.0)
        last_promotion = st.number_input("Last Promotion (Years ago)", 0, 20, 2)

    with col3:
        absenteeism = st.number_input("Absenteeism Days", 0, 60, 5)
        training = st.number_input("Training Hours", 0, 100, 20)
        certs = st.number_input("Certifications", 0, 10, 1)
        wlb = st.slider("Work Life Balance", 1.0, 5.0, 3.0)
        job_sat = st.slider("Job Satisfaction", 1.0, 5.0, 3.0)

    st.markdown("---")
    st.subheader("👤 Personal Info")
    col4, col5 = st.columns(2)

    with col4:
        gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
        city = st.selectbox("City", label_encoders['City'].classes_)
        department = st.selectbox("Department", label_encoders['Department'].classes_)

    with col5:
        education = st.selectbox("Education Level", label_encoders['Education_Level'].classes_)
        internet = st.selectbox("Internet Stability", label_encoders['Internet_Stability'].classes_)

    st.markdown("---")

    if st.button("🔍 Predict Performance", use_container_width=True):
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'City': label_encoders['City'].transform([city])[0],
            'Education_Level': label_encoders['Education_Level'].transform([education])[0],
            'Department': label_encoders['Department'].transform([department])[0],
            'Experience_Years': experience,
            'Monthly_Salary': salary,
            'Projects_Completed': projects,
            'Training_Hours': training,
            'Certifications': certs,
            'Work_Life_Balance': wlb,
            'Job_Satisfaction': job_sat,
            'Manager_Rating': manager_rating,
            'Overtime_Hours': overtime,
            'Commute_Time_Min': commute,
            'Laptop_Issue_Count': laptop_issues,
            'Cafeteria_Rating': cafeteria,
            'Internet_Stability': label_encoders['Internet_Stability'].transform([internet])[0],
            'Last_Promotion_Years': last_promotion,
            'Absenteeism_Days': absenteeism,
        }])

        prediction = model.predict(input_data)
        result = le_target.inverse_transform(prediction)[0]
        proba = model.predict_proba(input_data)[0]
        classes = le_target.classes_

        color_map = {"High Performer": "🟢", "Average Performer": "🟡", "Low Performer": "🔴"}
        bg_map = {"High Performer": "#d4edda", "Average Performer": "#fff3cd", "Low Performer": "#f8d7da"}

        st.markdown(f"""
        <div style='background-color:{bg_map[result]}; padding:20px; border-radius:10px; text-align:center;'>
            <h2>{color_map[result]} {result}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 Prediction Confidence")
        proba_df = pd.DataFrame({
            'Category': classes,
            'Confidence (%)': [round(p * 100, 2) for p in proba]
        })

        fig = px.bar(
            proba_df, x='Category', y='Confidence (%)',
            color='Category',
            color_discrete_map={
                'High Performer': 'green',
                'Average Performer': 'gold',
                'Low Performer': 'red'
            },
            text='Confidence (%)',
            title='Prediction Confidence per Category'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_range=[0, 110])
        st.plotly_chart(fig, use_container_width=True)

# ================================
# MODE 2 — BULK CSV PREDICTION
# ================================
elif mode == "Bulk CSV Prediction":

    st.title("📁 Bulk Employee Performance Prediction")
    st.markdown("Upload a CSV file with employee data to predict performance for multiple employees at once.")
    st.markdown("---")

    st.info("📌 Your CSV should have these columns: Age, Gender, City, Education_Level, Department, Experience_Years, Monthly_Salary, Projects_Completed, Training_Hours, Certifications, Work_Life_Balance, Job_Satisfaction, Manager_Rating, Overtime_Hours, Commute_Time_Min, Laptop_Issue_Count, Cafeteria_Rating, Internet_Stability, Last_Promotion_Years, Absenteeism_Days")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("🔍 Predict for All Employees", use_container_width=True):
            try:
                df_encoded = df.copy()

                # Fill missing numeric values with median
                for col in num_cols:
                    if col in df_encoded.columns:
                        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

                # Fill missing categorical values with mode
                for col in cat_cols:
                    if col in df_encoded.columns:
                        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])

                # Encode categorical columns
                for col in cat_cols:
                    df_encoded[col] = label_encoders[col].transform(df_encoded[col])

                # Predict
                X = df_encoded[feature_cols]
                predictions = model.predict(X)
                df['Predicted_Performance'] = le_target.inverse_transform(predictions)

                st.markdown("---")
                st.subheader("✅ Prediction Results")
                show_cols = ['Employee_ID', 'Predicted_Performance'] if 'Employee_ID' in df.columns else df.columns
                st.dataframe(df[show_cols], use_container_width=True)

                # Distribution Chart
                st.subheader("📊 Performance Distribution")
                fig2 = px.pie(df, names='Predicted_Performance',
                              color='Predicted_Performance',
                              color_discrete_map={
                                  'High Performer': 'green',
                                  'Average Performer': 'gold',
                                  'Low Performer': 'red'
                              },
                              title='Predicted Performance Distribution')
                st.plotly_chart(fig2, use_container_width=True)

                # Download
                st.markdown("---")
                st.subheader("⬇️ Download Results")
                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_out,
                    file_name='predicted_performance.csv',
                    mime='text/csv',
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")
# ================================
# MODE 3 — EDA
# ================================
elif mode == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")
    st.markdown("Visual analysis of the training data used to build the model.")
    st.markdown("---")

    # Load training data
    try:
        train = pd.read_csv('perf_train (1).csv')

        # Fill missing values for display
        num_cols_eda = ['Age', 'Experience_Years', 'Monthly_Salary', 'Projects_Completed',
                        'Manager_Rating', 'Overtime_Hours', 'Commute_Time_Min',
                        'Laptop_Issue_Count', 'Cafeteria_Rating', 'Last_Promotion_Years',
                        'Absenteeism_Days', 'Training_Hours', 'Certifications',
                        'Work_Life_Balance', 'Job_Satisfaction']

        for col in num_cols_eda:
            train[col] = pd.to_numeric(train[col], errors='coerce')
            train[col] = train[col].fillna(train[col].median())

        # ---- Chart 1: Pie Chart ----
        st.subheader("🎯 Performance Category Distribution")
        fig1 = px.pie(
            train, names='Performance_Category',
            color='Performance_Category',
            color_discrete_map={
                'High Performer': 'green',
                'Average Performer': 'gold',
                'Low Performer': 'red'
            },
            title='Target Class Distribution'
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("---")

        # ---- Chart 2: Feature Distribution ----
        st.subheader("📈 Feature Distribution by Performance Category")
        selected_feature = st.selectbox("Select a feature to explore:", num_cols_eda)

        fig2 = px.histogram(
            train, x=selected_feature,
            color='Performance_Category',
            barmode='overlay',
            color_discrete_map={
                'High Performer': 'green',
                'Average Performer': 'gold',
                'Low Performer': 'red'
            },
            title=f'{selected_feature} Distribution by Performance Category',
            opacity=0.7
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")

        # ---- Chart 3: Boxplot ----
        st.subheader("📦 Boxplot by Performance Category")
        selected_box = st.selectbox("Select a feature for boxplot:", num_cols_eda, key='box')

        fig3 = px.box(
            train, x='Performance_Category', y=selected_box,
            color='Performance_Category',
            color_discrete_map={
                'High Performer': 'green',
                'Average Performer': 'gold',
                'Low Performer': 'red'
            },
            title=f'{selected_box} by Performance Category'
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("---")

        # ---- Chart 4: Correlation Heatmap ----
        st.subheader("🔥 Correlation Heatmap")
        import plotly.figure_factory as ff
        import numpy as np

        corr = train[num_cols_eda].corr().round(2)
        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Heatmap',
            aspect='auto'
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ---- Summary Stats ----
        st.markdown("---")
        st.subheader("📋 Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Employees", len(train))
        col2.metric("Total Features", len(num_cols_eda))
        col3.metric("Performance Classes", train['Performance_Category'].nunique())

        st.dataframe(train[num_cols_eda].describe().round(2), use_container_width=True)

    except FileNotFoundError:
        st.error("❌ perf_train (1).csv not found! Please copy it into your project folder.")
# ================================
# MODE 3 — EDA
# ================================
elif mode == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")
    st.markdown("Visual analysis of the training data used to build the model.")
    st.markdown("---")

    # Load training data
    try:
        train = pd.read_csv('perf_train (1).csv')

        # Fill missing values for display
        num_cols_eda = ['Age', 'Experience_Years', 'Monthly_Salary', 'Projects_Completed',
                        'Manager_Rating', 'Overtime_Hours', 'Commute_Time_Min',
                        'Laptop_Issue_Count', 'Cafeteria_Rating', 'Last_Promotion_Years',
                        'Absenteeism_Days', 'Training_Hours', 'Certifications',
                        'Work_Life_Balance', 'Job_Satisfaction']

        for col in num_cols_eda:
            train[col] = pd.to_numeric(train[col], errors='coerce')
            train[col] = train[col].fillna(train[col].median())

        # ---- Chart 1: Pie Chart ----
        st.subheader("🎯 Performance Category Distribution")
        fig1 = px.pie(
            train, names='Performance_Category',
            color='Performance_Category',
            color_discrete_map={
                'High Performer': 'green',
                'Average Performer': 'gold',
                'Low Performer': 'red'
            },
            title='Target Class Distribution'
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("---")

        # ---- Chart 2: Feature Distribution ----
        st.subheader("📈 Feature Distribution by Performance Category")
        selected_feature = st.selectbox("Select a feature to explore:", num_cols_eda)

        fig2 = px.histogram(
            train, x=selected_feature,
            color='Performance_Category',
            barmode='overlay',
            color_discrete_map={
                'High Performer': 'green',
                'Average Performer': 'gold',
                'Low Performer': 'red'
            },
            title=f'{selected_feature} Distribution by Performance Category',
            opacity=0.7
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")

        # ---- Chart 3: Boxplot ----
        st.subheader("📦 Boxplot by Performance Category")
        selected_box = st.selectbox("Select a feature for boxplot:", num_cols_eda, key='box')

        fig3 = px.box(
            train, x='Performance_Category', y=selected_box,
            color='Performance_Category',
            color_discrete_map={
                'High Performer': 'green',
                'Average Performer': 'gold',
                'Low Performer': 'red'
            },
            title=f'{selected_box} by Performance Category'
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("---")

        # ---- Chart 4: Correlation Heatmap ----
        st.subheader("🔥 Correlation Heatmap")
        import plotly.figure_factory as ff
        import numpy as np

        corr = train[num_cols_eda].corr().round(2)
        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Heatmap',
            aspect='auto'
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ---- Summary Stats ----
        st.markdown("---")
        st.subheader("📋 Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Employees", len(train))
        col2.metric("Total Features", len(num_cols_eda))
        col3.metric("Performance Classes", train['Performance_Category'].nunique())

        st.dataframe(train[num_cols_eda].describe().round(2), use_container_width=True)

    except FileNotFoundError:
        st.error("❌ perf_train (1).csv not found! Please copy it into your project folder.")