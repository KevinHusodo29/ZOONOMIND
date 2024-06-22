import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from opencage.geocoder import OpenCageGeocode

# Your OpenCage API key
OPENCAGE_API_KEY = 'f17ebdec0307410aa095e516d6799252'

# Function to get latitude and longitude from a country name

def get_lat_long(country):
    geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
    result = geocoder.geocode(country)
    if result and len(result):
        return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    else:
        return None, None
# Set page configuration
st.set_page_config(page_title="Prediksi Dengue")

# Load and preprocess the dataset
file_path = 'Dengue-worldwide-dataset-modified.xlsx'
df = pd.read_excel(file_path)

# Remove unwanted leading and trailing spaces from column names
df.columns = df.columns.str.strip()

def convert_days(days_str):
    if isinstance(days_str, str):
        if 'days' in days_str:
            return float(days_str.replace(' days', ''))
        elif 'weeks' in days_str:
            return float(days_str.replace(' weeks', '')) * 7
    return float(days_str)

df['dengue.days'] = df['dengue.days'].apply(convert_days)
df['current_temp'] = df['current_temp'].astype(float)
df['dengue.wbc'] = df['dengue.wbc'].astype(float)
df['dengue.hemoglobin'] = df['dengue.hemoglobin'].astype(float)
df['dengue._hematocri'] = df['dengue._hematocri'].astype(float)

label_encoders = {}
for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

features = df.drop(columns=['dengue_or_not', 'dengue.date_of_fever'])
target = df['dengue_or_not']
target_encoder = LabelEncoder()
target = target_encoder.fit_transform(target)

imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features)
feature_names = features.columns

X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.3, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_smote, y_train_smote)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_encoder.classes_.astype(str), output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
feature_importances = rf_model.feature_importances_

# Preprocess new data
def preprocess_new_data(new_data):
    # Ensure new data has the necessary columns
    for column in feature_names:
        if column not in new_data.columns:
            new_data[column] = ''  # Add missing columns with default values

    # Convert 'dengue.days' in new data to float
    if 'dengue.days' in new_data.columns:
        new_data['dengue.days'] = new_data['dengue.days'].astype(float)

    # Ensure new data is in the same format as the training data
    new_data_encoded = new_data.copy()
    for column in new_data.columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                new_data_encoded[column] = le.transform(new_data[column])
            except ValueError:
                new_data_encoded[column] = -1  # Assign -1 for unseen labels

    # Ensure the order of columns matches the training data
    new_data_encoded = new_data_encoded[feature_names]

    # Handle missing values in new data
    new_data_encoded = imputer.transform(new_data_encoded)

    return new_data_encoded
def predict_dengue(new_data):
    prediction = rf_model.predict(new_data)
    prediction_proba = rf_model.predict_proba(new_data)
    prediction_decoded = target_encoder.inverse_transform(prediction)
    prediction_proba = prediction_proba.max(axis=1)
    return prediction_decoded, prediction_proba
def get_lat_long(country):
    geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
    result = geocoder.geocode(country)
    if result and len(result):
        return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    else:
        return None, None
# Add Font Awesome icons and custom CSS for background color and text color
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
<style>
        .main {
            background-color: #ffffff;
            color: #28a745; /* Green color for text */
        }
        .sidebar .sidebar-content {
            background-color: #d4edda;
        }
        h1, h2, h3, .stButton>button {
            color: #28a745;
        }
        .css-1d391kg input {
            color: #28a745;
        }
        .stTable {
            color: #28a745;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #28a745;
        }
        .custom-table th, .custom-table td {
            color: #28a745;
            border: 1px solid #28a745; /* Green border for table */
        }
        .stButton>button {
            background-color: #28a745; /* Green background */
            color: white; /* White text */
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #218838; /* Darker green on hover */
        }
        # .stTextInput, .stNumberInput{
        #     background-color: #28a745 !important; /* White background */
        #     color: #28a745 !important; /* Green text */
        # }
        .stTextInput input, .stNumberInput input{
            background-color: #28a745 !important; /* White background */
            color: #ffffff!important; /* Green text */
        }
        # .stNumberInput div{
        #     background-color: #ffffff !important; /* White background */
        # }
        .stNumberInput button{
            background-color: #28a745 !important; /* White background */
            color: #ffffff !important; /* Green text */
        }
        .stSelectbox label, .stTextInput label, .stNumberInput label {
            color: #28a745 !important; /* Green label text */
        }
        .stSelectbox:first-of-type > div[data-baseweb="select"] > div {
            background-color: #28a745;
	    }
    }
    </style>
    """, unsafe_allow_html=True)

# Application title
st.title("Prediksi Dengue Menggunakan Random Forest")

with st.container() as container:
  st.markdown(""" """, unsafe_allow_html=True)

# Use st.columns to create equal-sized buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    menu_home = st.button("🏠Home", key="home")
with col2:
    menu_input = st.button("📝Predict Patient", key="input")
with col3:
    menu_report = st.button("📊Accuration Report", key="report")
with col4:
    menu_info = st.button("💁infomation", key="info")

# Initialize state
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = 'home'

# Update state based on button click
if menu_home:
    st.session_state.current_menu = 'home'
elif menu_input:
    st.session_state.current_menu = 'input'
elif menu_report:
    st.session_state.current_menu = 'report'
elif menu_info:
    st.session_state.current_menu = 'info'


# Navigation logic
if st.session_state.current_menu == 'home':
    # st.markdown("<h3>Selamat Datang di Aplikasi Prediksi Penyakit Dengue</h3>", unsafe_allow_html=True)
    # Convert the date column to datetime
    data = pd.read_excel(file_path)
    # st.write("Nama kolom dalam dataset:")
    # st.write(data.columns)
    # Convert the date column to datetime
    try:
        data['dengue.date_of_fever'] = pd.to_datetime(data['dengue.date_of_fever'], format='%d-%b', errors='coerce')
        # st.write("Data tanggal setelah konversi:")
        # st.write(data[['dengue.date_of_fever']].dropna().sort_values(by='dengue.date_of_fever'))
    except Exception as e:
        st.error(f"Error converting date: {e}")

    st.title('Analisis Data Kasus Dengue')

    # Analisis Deret Waktu menggunakan Bar Chart interaktif
    # st.header('Analisis Kasus Dengue dari Waktu ke Waktu')
    time_series_data = data.groupby('dengue.date_of_fever').size().reset_index(name='jumlah_kasus')
    location_data = data.groupby('dengue.dwelling_place').size().reset_index(name='jumlah_kasus_lokasi')
    # st.write("Jumlah kasus per tanggal:")
    # st.write(time_series_data)
    # if 'dengue.dwelling_place' in data.columns:
    #     st.write("Data dari kolom 'dengue.dwelling_place':")
    #     st.write(data['dengue.dwelling_place'].unique())
    # else:
    #     st.error("Kolom 'dengue.dwelling_place' tidak ditemukan dalam dataset.")
    if time_series_data.empty:
        st.warning("Tidak ada data yang cukup untuk membuat grafik deret waktu.")
    else:
        fig_bar = px.bar(time_series_data, x='dengue.date_of_fever', y='jumlah_kasus',
                        labels={'dengue.date_of_fever': 'Tanggal', 'jumlah_kasus': 'Jumlah Kasus'},
                        title='Jumlah Kasus Dengue per Tanggal')
        fig_bar.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
        # st.plotly_chart(fig_bar)

    # Line Chart
    st.header('Line Chart Kasus Dengue dari Waktu ke Waktu')
    fig_line = px.line(time_series_data, x='dengue.date_of_fever', y='jumlah_kasus',
                    labels={'dengue.date_of_fever': 'Tanggal', 'jumlah_kasus': 'Jumlah Kasus'})
    # fig_line.update_layout(plot_bgcolor='white', paper_bgcolor='white', )
    st.plotly_chart(fig_line)

    fig_bar = px.bar(location_data, x='dengue.dwelling_place', y='jumlah_kasus_lokasi',
                        labels={'dengue.dwelling_place': 'Location', 'jumlah_kasus_lokasi': 'Jumlah Kasus'},
                        title='Jumlah Kasus Dengue per Lokasi')
    # fig_bar.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
    st.plotly_chart(fig_bar)
    # Scatter Plot
    st.header('Scatter Plot Hubungan antara Suhu Tubuh dan Jumlah Kasus Dengue')
    fig_scatter = px.scatter(data, x='current_temp', y=data.index,
                            labels={'current_temp': 'Suhu Tubuh', 'index': 'Jumlah Kasus'})
    # fig_scatter.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
    st.plotly_chart(fig_scatter)

    # Box Plot Tingkat Keparahan
    st.header('Distribusi Jumlah Hari Sakit Berdasarkan Gejala Keparahan')
    data['dengue.days'] = data['dengue.days'].str.extract('(\d+)').astype(float)
    fig_box = px.box(data, x='dengue.servere_headche', y='dengue.days',
                    labels={'dengue.servere_headche': 'Severe Headache', 'dengue.days': 'Jumlah Hari Sakit'})
    # fig_box.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
    st.plotly_chart(fig_box)

    st.header('Total Kasus Dengue per Lokasi')
     # Group data by 'dengue.dwelling_place' and count the number of cases
    grouped_data = data.groupby('dengue.dwelling_place').size().reset_index(name='Jumlah_Kasus')
    
    # Get lat and long for each unique place
    lat_lng_dict = {}
    for place in grouped_data['dengue.dwelling_place']:
        lat, lng = get_lat_long(place)
        lat_lng_dict[place] = (lat, lng)

    # Map the lat/long back to the grouped dataframe
    grouped_data['Latitude'] = grouped_data['dengue.dwelling_place'].map(lambda x: lat_lng_dict[x][0])
    grouped_data['Longitude'] = grouped_data['dengue.dwelling_place'].map(lambda x: lat_lng_dict[x][1])
    
    st.write("Data dengan koordinat setelah digrouping:")
    st.write(grouped_data)

    # Filter out rows without coordinates
    grouped_data = grouped_data.dropna(subset=['Latitude', 'Longitude'])

    # Create a choropleth map using Plotly
    st.header('Choropleth Map of Dengue Cases by Country')
    fig_map = px.choropleth(
        data_frame=grouped_data,
        locations='dengue.dwelling_place',  # column with country or state names
        locationmode='country names',  # if the column contains country names
        color='Jumlah_Kasus',  # column to be used for coloring
        hover_name='dengue.dwelling_place',  # column to be displayed on hover
        title='Choropleth Map of Dengue Cases',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig_map.update_layout(
        geo=dict(
            bgcolor='white'
        ),
        font=dict(color='green'),plot_bgcolor='white', paper_bgcolor='white'
    )
    fig_map.update_coloraxes(colorbar_bgcolor = '#28a745')
    # fig_map.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
    st.plotly_chart(fig_map)
    # if 'dengue.dwelling_place' in data.columns:
    # location_data = data['dengue.dwelling_place'].value_counts().reset_index(name='jumlah_kasus')
    # fig_location = px.bar(location_data, x='index', y='jumlah_kasus',
    #                     labels={'index': 'Lokasi', 'jumlah_kasus': 'Jumlah Kasus'},
    #                     title='Total Kasus Dengue per Lokasi')
    # fig_location.update_layout(
    #     plot_bgcolor='white',
    #     paper_bgcolor='white',
    #     font=dict(color='green'),
    #     xaxis=dict(color='black'),
    #     yaxis=dict(color='black')
    # )
    # fig_location.update_traces(texttemplate='%{y}', textposition='outside', textfont=dict(color='black'))
    # st.plotly_chart(fig_location)
    # else:
    #     st.error("Kolom 'dengue.dwelling_place' tidak ditemukan dalam dataset.")

elif st.session_state.current_menu == 'input':
    st.header("Kirim Informasi Pasien untuk Prediksi Dengue")

    input_data = {}
    input_data['dengue.p_i_d'] = st.text_input("Patient ID")
    input_data['dengue.dwelling_place'] = st.text_input("Dwelling Place (e.g., Jakarta)")
    input_data['dengue.days'] = st.number_input("Days of Fever (days)", value=5)
    input_data['current_temp'] = st.number_input("Current Temperature", min_value=90.0, max_value=110.0, value=98.6)
    input_data['dengue.wbc'] = st.number_input("WBC Count", min_value=0.0, max_value=20.0, value=5.0)
    input_data['dengue.hemoglobin'] = st.number_input("Hemoglobin Level", min_value=0.0, max_value=20.0, value=14.0)
    input_data['dengue._hematocri'] = st.number_input("Hematocrit Level", min_value=0, max_value=100, value=40)
    input_data['dengue.servere_headche'] = st.selectbox("Severe Headache", options=['yes', 'no'], index=0)
    input_data['dengue.pain_behind_the_eyes'] = st.selectbox("Pain Behind the Eyes", options=['yes', 'no'], index=0)
    input_data['dengue.joint_muscle_aches'] = st.selectbox("Joint and Muscle Aches", options=['yes', 'no'], index=0)
    input_data['dengue.metallic_taste_in_the_mouth'] = st.selectbox("Metallic Taste in the Mouth", options=['yes', 'no'], index=0)
    input_data['dengue.appetite_loss'] = st.selectbox("Appetite Loss", options=['yes', 'no'], index=0)
    input_data['dengue.addominal_pain'] = st.selectbox("Abdominal Pain", options=['yes', 'no'], index=0)
    input_data['dengue.nausea_vomiting'] = st.selectbox("Nausea/Vomiting", options=['yes', 'no'], index=0)
    input_data['dengue.diarrhoea'] = st.selectbox("Diarrhoea", options=['yes', 'no'], index=0)
    
    if st.button("Prediksi", key="predict"):
        new_input = pd.DataFrame([input_data])
        new_input_encoded = preprocess_new_data(new_input)
        prediction, prediction_proba = predict_dengue(new_input_encoded)
        
        # Menentukan label prediksi
        prediction_label = "Dengue" if prediction[0] == 1 else "Not Dengue"
        prediction_percent = prediction_proba[0] * 100
        
        st.markdown(f"<h3>Prediction: {prediction_label}, Probability: {prediction_percent:.2f}%</h3>", unsafe_allow_html=True)

elif st.session_state.current_menu == 'report':
    st.header("Laporan dan Visualisasi")

    st.markdown(f"<h3>Akurasi Model: {accuracy:.3f}</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3>Classification Report</h3>", unsafe_allow_html=True)
    report_df = pd.DataFrame(report).transpose()
    report_df_html = report_df.to_html(classes='custom-table', escape=False)
    st.markdown(report_df_html, unsafe_allow_html=True)

    st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=target_encoder.classes_.astype(str), yticklabels=target_encoder.classes_.astype(str))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    st.markdown("<h3>Feature Importances</h3>", unsafe_allow_html=True)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    plt.title('Feature Importances')
    st.pyplot(fig)

    st.markdown("<h3>Distribusi Kelas</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.countplot(x=target_encoder.inverse_transform(target), ax=ax)
    plt.xlabel('Class')
    plt.ylabel('Count')
    st.pyplot(fig)

elif st.session_state.current_menu == 'info':
    st.markdown("<h3>Informasi Tentang Demam Berdarah Dengue (DBD)</h3>", unsafe_allow_html=True)
    st.write("""
    **Apa itu Demam Berdarah Dengue (DBD)?**
    
    Demam Berdarah Dengue (DBD) adalah penyakit yang disebabkan oleh virus dengue yang ditularkan melalui gigitan nyamuk Aedes aegypti. Penyakit ini dapat menyebabkan gejala ringan hingga berat dan dapat berakibat fatal jika tidak ditangani dengan cepat dan tepat.
    
    **Gejala DBD:**
    - Demam tinggi mendadak
    - Sakit kepala parah
    - Nyeri belakang mata
    - Nyeri sendi dan otot
    - Mual dan muntah
    - Ruam kulit
    
    **Pencegahan:**
    - Menghindari gigitan nyamuk dengan menggunakan obat nyamuk atau kelambu.
    - Menguras, menutup, dan mendaur ulang wadah yang dapat menampung air untuk mencegah berkembangnya nyamuk.
    - Memelihara kebersihan lingkungan.

    **Pengobatan:**
    - Istirahat yang cukup
    - Minum banyak cairan untuk mencegah dehidrasi
    - Mengonsumsi obat penurun demam dan pereda nyeri
    - Segera mencari perawatan medis jika gejala memburuk
    
    **Faktor Risiko:**
    - Tinggal atau bepergian ke daerah tropis dan subtropis yang sering mengalami wabah dengue.
    - Sistem kekebalan tubuh yang lemah, seperti pada anak-anak, orang tua, atau individu dengan kondisi kesehatan tertentu.
    - Infeksi sebelumnya dengan salah satu dari empat jenis virus dengue dapat meningkatkan risiko terkena bentuk penyakit yang lebih parah jika terinfeksi ulang dengan jenis yang berbeda.

    **Diagnosis:**
    - Tes darah untuk mendeteksi virus dengue atau antibodi terhadap virus tersebut.
    - Pemeriksaan medis dan pengamatan gejala klinis oleh tenaga kesehatan profesional.

    **Komplikasi:**
    - Dengue parah (dengue hemorrhagic fever atau dengue shock syndrome) yang dapat menyebabkan pendarahan hebat, kerusakan pada pembuluh darah, dan penurunan tekanan darah yang berbahaya.
    - Perawatan intensif di rumah sakit sering diperlukan untuk kasus yang parah.

    **Vaksinasi:**
    - Vaksin dengue tersedia di beberapa negara, tetapi penggunaannya biasanya terbatas pada orang yang telah terinfeksi virus dengue sebelumnya.
    - Konsultasikan dengan tenaga medis untuk informasi lebih lanjut mengenai vaksinasi dengue.
    """)