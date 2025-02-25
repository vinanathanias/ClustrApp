import streamlit as st

st.header("abc")

import zipfile
import pandas as pd

# Path ke file ZIP
zip_path = 'filtered_data.zip'

# Membuka ZIP dan membaca daftar file di dalamnya
with zipfile.ZipFile(zip_path, 'r') as zfile:
    # Menampilkan daftar file di dalam ZIP
    file_list = zfile.namelist()
    csv_files = [f for f in file_list if f.endswith('.csv')]
    
    # Jika tidak ada file CSV
    if not csv_files:
        st.error("Tidak ada file CSV di dalam ZIP.")
    else:
        # Pilih file CSV yang ingin dibaca
        selected_file = st.selectbox("Pilih CSV file:", csv_files)
        
        # Membaca CSV ke dalam DataFrame
        with zfile.open(selected_file) as csvfile:
            df = pd.read_csv(csvfile)
            st.dataframe(df)
