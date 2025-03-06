import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
import seaborn as sns
import matplotlib.pyplot as plt
import time


st.title("Machine Learning App") 

st.sidebar.header("Upload File")  # رفع الملف
uploaded_file = st.sidebar.file_uploader("Choose a file (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    if uploaded_file.name.endswith("csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        data = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith("json"):
        data = pd.read_json(uploaded_file)

    st.write("Data Overview")  # نظرة عامة على البيانات
    st.dataframe(data.head())
    st.write(f"Data Shape: {data.shape}")  # حجم البيانات
    st.write(f"Missing Values: {data.isnull().sum()}")  # القيم المفقودة

    target_column = st.sidebar.selectbox("Select Target Column", options=data.columns.tolist())  # اختيار العمود الهدف

    st.sidebar.header("Preprocessing Options")  # خيارات المعالجة المبدئية
    drop_columns = st.sidebar.multiselect("Columns to Drop", options=data.columns.tolist())  # الأعمدة التي سيتم حذفها
    if drop_columns:
        data = data.drop(columns=drop_columns)
        st.write(f"Dropped Columns: {drop_columns}")  # تم حذف الأعمدة

    missing_value_method = st.sidebar.selectbox("Handling Missing Values", ["Drop", "Fill with Mean/Median/Mode"])  # كيفية التعامل مع القيم المفقودة
    if missing_value_method == "Drop":
        data = data.dropna()
        st.write("Dropped rows with missing values.")  # تم حذف الصفوف التي تحتوي على قيم مفقودة
    elif missing_value_method == "Fill with Mean/Median/Mode":
        fill_method = st.sidebar.selectbox("Fill with", ["Mean", "Median", "Mode"])  # الملء بـ
        if fill_method == "Mean":
            data = data.fillna(data.mean())
        elif fill_method == "Median":
            data = data.fillna(data.median())
        elif fill_method == "Mode":
            data = data.fillna(data.mode().iloc[0])
        st.write(f"Filled missing values using {fill_method}.")  # تم ملء القيم المفقودة

    encoding_method = st.sidebar.selectbox("Categorical Encoding", ["None", "Label Encoding", "One-Hot Encoding"])  # ترميز البيانات التصنيفية
    if encoding_method == "Label Encoding":
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col] = data[col].astype('category').cat.codes
        st.write("Applied Label Encoding to categorical columns.")  # تم تطبيق ترميز التسمية
    elif encoding_method == "One-Hot Encoding":
        data = pd.get_dummies(data)
        st.write("Applied One-Hot Encoding.")  # تم تطبيق الترميز الواحد

    st.write("Data Overview after Processing")  # نظرة عامة على البيانات بعد المعالجة
    st.dataframe(data.head())

    # Display Correlation Heatmap
    st.subheader("Correlation Heatmap")  # خريطة الارتباط
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.sidebar.header("Train Model")  # تدريب النموذج

    if target_column and target_column in data.columns:
        if data[target_column].dtype == 'object' or len(data[target_column].unique()) < 20:
            task_type = "Classification"  # تصنيف
        else:
            task_type = "Regression"  # تنبؤ

        st.write(f"Detected Task Type: {task_type}")  # تم اكتشاف نوع المهمة

        if task_type == "Classification":
            class_counts = data[target_column].value_counts()
            st.write("Class Distribution:")  # توزيع الفئات
            st.write(class_counts)

            if class_counts.min() < 2:
                st.warning("Some classes have fewer than 2 samples. They will be removed.")  # حذف الفئات القليلة
                data = data[data[target_column].map(class_counts) > 1]

        st.subheader("Compare Models")  # مقارنة النماذج
        
        if task_type == "Classification":
            setup_class = setup(data, target=target_column, session_id=123, data_split_stratify=False)
            best_model = compare_models()
        else:
            setup_reg = setup(data, target=target_column, session_id=123)
            best_model = compare_models()

        st.write(best_model)

        if isinstance(best_model, list):
            best_model_to_save = best_model[0]
        else:
            best_model_to_save = best_model

        if st.sidebar.button("Save Model"):  # حفظ النموذج
            save_model(best_model_to_save, f"best_model_{task_type.lower()}")
            st.write("Model saved successfully!")  # تم حفظ النموذج بنجاح

        if st.sidebar.button("Tune Parameters"):  # تعديل المعلمات
            tuned_model = tune_model(best_model_to_save)
            st.write("Tuned Model:", tuned_model)  # النموذج المعدل

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)

        st.subheader("Make Predictions")  # إجراء التنبؤ
        user_input = {}
        for col in data.columns:
            if col != target_column:
                user_input[col] = st.number_input(f"Enter value for {col}")

        if st.button("Predict"):  # تنبأ
            input_df = pd.DataFrame([user_input])
            prediction = predict_model(best_model_to_save, data=input_df)
            st.write("Prediction:", prediction)  # التنبؤ

    else:
        if not target_column:
            st.error("Please select a valid target column.")  # الرجاء اختيار عمود هدف صالح
        else:
            st.error(f"The target column '{target_column}' is not in the dataset.")  # العمود الهدف غير موجود

else:
    st.error("Please upload a file.")  # الرجاء رفع ملف
