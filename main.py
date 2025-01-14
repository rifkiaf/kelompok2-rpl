import streamlit as st
from scipy.stats import boxcox
import joblib
import pandas as pd

#judul web
st.title("Website Pendeteksi Gagal Jantung")
st.image("images/header-Jantung.jpg")
st.subheader("Apakah Anda bertanya-tanya tentang kondisi jantung Anda? Aplikasi ini akan membantu Anda mendiagnosisnya!")

col1, col2 = st.columns([1, 3])

with col1:
    st.image("images/doctor.png",
            caption="Saya akan membantu Anda mendiagnosis kesehatan jantung Anda! - Dr. Support Vector Machine",
            width=150)
with col2:
        st.markdown("""
        Tahukah Anda bahwa model pembelajaran mesin dapat membantu Anda memprediksi penyakit jantung dengan cukup akurat? 
        Dalam aplikasi ini, Anda dapat memperkirakan kemungkinan Anda terkena penyakit jantung (ya/tidak) dalam hitungan detik!
        
        Di sini, model dibangun menggunakan arsitektur support vector machine dengan data yang diambil dari kaggle dan berjumlah 745. 
        Aplikasi ini didasarkan pada model ini karena mencapai akurasi sekitar 86%, cukup baik.
        
        Untuk memprediksi status penyakit jantung Anda, cukup ikuti langkah-langkah di bawah ini:
        1. Masukkan parameter yang paling menggambarkan Anda;
        2. Tekan tombol "Prediksi" dan tunggu hasilnya.
            
        **Perlu diingat bahwa hasil ini hanya membantu anda memberikan keputusan untuk memeriksa lebih lanjut ke dokter spesialis setelah menerima hasil dari aplikasi ini.** 
        """)

#bagian prediksi
st.write("---")
st.header('Informasi Parameter:')
st.markdown('''
1. **Umur**: Usia pasien
2. **Jenis kelamin**: Untuk pria = M dan untuk wanita = F
3. **Jenis nyeri dada**:
   - Angina Tipikal (TA) = nyeri dada substernal yang dipicu oleh aktivitas fisik atau stres emosional dan dapat diredakan dengan istirahat atau nitrogliserin,
   - Angina Atypikal (ATA) = keluhan yang berkaitan dengan serangan jantung, disebabkan oleh berkurangnya aliran darah ke jantung,
   - Nyeri Non-Angina (NAP) = Nyeri dada kronis yang terasa seperti di jantung, tetapi sebenarnya tidak,
   - Asymptomatic (ASY) = istilah yang digunakan untuk menggambarkan ketika seseorang menderita suatu penyakit namun tidak menunjukkan gejala klinis apa pun.
4. **Tekanan Darah Saat istirahat**: tekanan darah yang diukur ketika seseorang sedang dalam keadaan tenang dan santai (mmHg)
5. **Kolesterol**: Kolesterol serum [mm/dl]
6. **Gula Darah Puasa**: ula darah puasa merujuk pada kadar glukosa (gula) dalam darah setelah seseorang tidak makan atau minum apapun kecuali air selama periode waktu tertentu, biasanya 8 hingga 12 jam. [1: jika Gula darah puasa > 120 mg/dl, 0: sebaliknya]
7. **Elektrodiagram Saat Istirahat**: Hasil elektrokardiogram istirahat [Normal: Normal, ST: memiliki kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0,05 mV), LVH: menunjukkan hipertrofi ventrikel kiri yang mungkin atau pasti menurut kriteria Estes]
8. **Detak Jantung Maksimum**: Detak jantung maksimum yang dicapai [Nilai numerik antara 60 dan 202]
9. **Angina Saat Olahraga**: Angina (jenis nyeri dada yang diakibatkan oleh berkurangnya aliran darah ke jantung.) yang disebabkan oleh olahraga [Y: Ya, N: Tidak]
10. **Oldpeak**: oldpeak = ST [Nilai numerik yang diukur dalam depresi]
11. **Kemiringan Segmen ST**: Kemiringan segmen ST latihan puncak [Up: menanjak, Flat: datar, Down: menurun]
''')

st.write("---")
st.header('Ayo, Cek Kondisi Kesehatan Jantung Anda!')
# Simulasi data medis
is_medical_professional = st.checkbox("Saya seorang profesional medis")
has_medical_data = st.checkbox("Saya memiliki data medis")

if not is_medical_professional:
    st.warning("Peringatan: Anda bukan seorang profesional medis. Harap berhati-hati dalam menggunakan aplikasi ini.")

if not has_medical_data:
    st.warning("Peringatan: Anda tidak memiliki data medis. Hasil prediksi mungkin tidak akurat.")

if is_medical_professional and has_medical_data:
    st.write("Silakan masukkan data medis Anda untuk melakukan prediksi.")
    # Tambahkan kode untuk form input data medis dan prediksi di sini
else:
    st.write("Mohon lengkapi kriteria di atas.")

st.subheader('Isi parameter di bawah ini untuk mengetahui kondisi jantung Anda.')
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Umur', min_value=0, max_value=120, value=25)
    restbp = st.number_input('Tekanan Darah Saat Istirahat (mmHg)', min_value=0, max_value=300, value=120)
    restecg = st.selectbox('Hasil Elektrokardiogram Istirahat', ['Normal', 'ST', 'LVH'])
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
with col2:
    gender = st.selectbox('Jenis Kelamin', ['M', 'F'])
    chol = st.number_input('Kolesterol Serum (mm/dl)', min_value=0, max_value=600, value=200)
    maxhr = st.number_input('Detak Jantung Maksimum', min_value=60, max_value=202, value=150)
    slope = st.selectbox('Kemiringan Segmen ST', ['Up', 'Flat', 'Down'])
with col3:
    cp = st.selectbox('Jenis Nyeri Dada', ['TA', 'ATA', 'NAP', 'ASY'])
    fbs = st.selectbox('Gula darah puasa [1: Jika GDP > 120 mg/dl, 0: sebaliknya]', [0, 1])
    exang = st.selectbox('Angina akibat olahraga', ['Ya', 'Tidak'])

#gula darah puasa
gdp = ""
if(fbs == 1):
    gdp = "> 120 mg/dl"
else:
    gdp = "< 120 mg/dl"

submit = st.button("Prediksi")

# Mapping untuk representasi biner
gender_map = {'F': 0, 'M': 1}
exang_map = {'Tidak': 0, 'Ya': 1}
slope_map = {'Down': 0, 'Flat': 1, 'Up': 2}

 # Mengubah nilai menggunakan mapping
gender_bn = gender_map[gender]
exang_bn = exang_map[exang]
slope_bn = slope_map[slope]

# Memuat model yang disimpan
model_path = 'model/model_sklearn.pkl'
model = joblib.load(model_path)

# Fungsi untuk preprocessing data input
def preprocess_input(input_data):
    # Membuat DataFrame dari data input
    input_df = pd.DataFrame([input_data])

    #mapping ChestPainType dan RestingECG
    chest_pain_mapping = {
    'ASY': [False, False, False],
    'ATA': [True, False, False],
    'NAP': [False, True, False],
    'TA': [False, False, True]
    }

    resting_ecg_mapping = {
    'LVH': [False, False],
    'Normal': [True, False],
    'ST': [False, True]
    }

    # Ubah nilai RestingECG berdasarkan mapping
    input_df['RestingECG'] = input_df['RestingECG'].map(resting_ecg_mapping)
    input_df[['RestingECG_1', 'RestingECG_2']] = pd.DataFrame(input_df['RestingECG'].tolist(), index=input_df.index)

    # Ubah nilai ChestPainType berdasarkan mapping
    input_df['ChestPainType'] = input_df['ChestPainType'].map(chest_pain_mapping)
    input_df[['ChestPainType_1', 'ChestPainType_2', 'ChestPainType_3']] = pd.DataFrame(input_df['ChestPainType'].tolist(), index=input_df.index)

    # Menghapus fitur kategorikal asli
    input_df.drop(['ChestPainType', 'RestingECG'], axis=1, inplace=True)

    # Daftar kolom yang diharapkan oleh model
    expected_columns = [
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
        'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'ChestPainType_1',
        'ChestPainType_2', 'ChestPainType_3', 'RestingECG_1', 'RestingECG_2'
    ]

    # Menyesuaikan urutan kolom agar sesuai dengan data pelatihan
    input_df = input_df[expected_columns]

    lambdas = {
        'Age': 1.1665823290742556,
        'RestingBP': -0.6283126636670258,
        'Cholesterol': 0.11065427247095147,
        'MaxHR': 1.2782320122806756,
        'Oldpeak': 0.084524462201563
    }

    continuous_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    for i, col in enumerate(continuous_features):
    #Menerapkan transformasi box-cox
      if input_df[col].min() > 0:
        input_df[col] = boxcox(input_df[col], lmbda=lambdas[col])
      else:
        input_df[col] = input_df[col] + 0.001
        input_df[col] = boxcox(input_df[col], lmbda=lambdas[col])

    return input_df

#fungsi untuk memeriksa rekaman medis
def rk_checker():
    # Inisialisasi status
    st_chol = 0
    st_restbp = 0
    st_gdp = 0
    st_maxhr = 0
    
    # Check kolesterol
    if chol > 239:
        st_chol = 1
    
    # Check tekanan darah saat istirahat
    if restbp > 80:
        st_restbp = 1
    
    # Check gula darah
    if fbs == 1:
        st_gdp = 1
    
    # Check detak jantung
    if maxhr > 100:
        st_maxhr = 1
    
    # Kumpulkan status yang perlu diperhatikan
    notes = []
    if st_chol == 1:
        notes.append('Kolesterol tinggi (Kolesterol dikatakan tinggi apabila melebihi 239 mg/dL')
    if st_restbp == 1:
        notes.append('Tekanan darah tinggi (Tekanan darah normal saat istirahat (diastolik) itu 80 mmHg)')
    if st_gdp == 1:
        notes.append('Gula darah tinggi (Gula darah puasa normal itu dibawah 100mg/DL, prediabetes 100-125 mg/dL, dan Diabetes Melitus tingkat 2 diatas 126 mg/dL')
    if st_maxhr == 1:
        notes.append('Detak jantung tinggi (Detak jantung normal itu 60 - 100 bpm')
    
    return {
        "cholesterol_status": st_chol,
        "restbp_status": st_restbp,
        "gdp_status": st_gdp,
        "maxhr_status": st_maxhr,
        "notes": notes
    }

# Contoh data input sesuai dengan fitur yang diharapkan model
input_data = {
    'Age': age,
    'Sex': gender_bn,
    'RestingBP': restbp,
    'Cholesterol': chol,
    'FastingBS': fbs,
    'MaxHR': maxhr,
    'ExerciseAngina': exang_bn,
    'Oldpeak': oldpeak,
    'ST_Slope': slope_bn,
    'ChestPainType': cp,
    'RestingECG': restecg
}

def evaluate_record(prediction, notes):
    if prediction == 0:
        if notes:
            return {
                "status": "Anda sehat dengan catatan",
                "notes": notes
            }
        else:
            return {
                "status": "Anda Sehatf",
                "notes": []
            }
    else:
        return {
            "status": "Jantung anda tidak sehat",
            "notes": []
        }

# Preprocessing user input
input_processed = preprocess_input(input_data)

#memeriksa rekaman medis
result = rk_checker()

if submit:
        # Membuat prediksi
        prediction = model.predict(input_processed)
        predict_prob = model.predict_proba(input_processed)
        hasil = evaluate_record(prediction, result["notes"])
        status = hasil["status"]
        catatan = hasil["notes"]
        st.markdown(f"""
            **Rekaman Medis Anda**:

            1. Umur: {age}
            2. Jenis Kelamin: {gender}
            3. Jenis Nyeri Dada: {cp}
            4. Tekanan Darah Saat Istirahat: {restbp} mmHg
            5. Kolesterol Serum: {chol} mm/dl
            6. Gula Darah Puasa: {gdp}
            7. Hasil Elektrokardiogram Istirahat: {restecg}
            8. Detak Jantung Maksimum: {maxhr}
            9. Angina Akibat Olahraga: {exang}
            10. Oldpeak: {oldpeak}
            11. Kemiringan Segmen ST: {slope}
        """)


        if prediction == 0:
                st.markdown(f"**Kemungkinan Anda menderita"
                        f" penyakit gagal jantung adalah {round(predict_prob[0][1] * 100, 2)}%."
                        f" {status}!**")
                st.image("images/jantung-sehat.png",
                     caption="Jantung Anda tampaknya baik-baik saja! - Dr. Support Vector Machine")
                if catatan:
                    st.write("Catatan: ")
                    for i in range(len(catatan)):
                        st.write(f"{i+1}. {catatan[i]}")
        else:
            st.markdown(f"**Kemungkinan Anda menderita"
                        f" penyakit gagal jantung adalah {round(predict_prob[0][1] * 100, 2)}%."
                        f" {status}.**")
            st.image("images/penyakit-jantung.png",
                     caption="Kesehatan jantung adalah investasi terbaik. Jantung Anda terdeteksi tidak sehat, segera temui dokter! - Dr. Support Vector Machine")
