import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px  # untuk membuat grafik scatter 3D
import numpy as np
from sklearn.metrics import pairwise_distances  # untuk menghitung jarak antar data
import plotly.graph_objects as go  # untuk membuat grafik

# Fungsi untuk membuat scatter plot 3D dan menentukan cluster dari titik baru
def scatter(model, model_name, data, new_point, features, color_scale, title):
    clusters = model.fit_predict(data[features])  # Prediksi cluster untuk setiap titik
    data[f"{model_name}_Cluster"] = clusters

    # Menentukan cluster untuk titik baru
    if model_name == "KMeans_model":
        # Pada K-Means, prediksi langsung
        new_cluster = model.predict(new_point[features])[0]
    else:
        # Pada Agglomerative dan DBSCAN, dihitung berdasarkan jarak terdekat
        distances = pairwise_distances(new_point[features], data[features])
        nearest_index = distances.argmin()
        new_cluster = clusters[nearest_index]

    # Membuat grafik 3D menggunakan Plotly Express
    fig = px.scatter_3d(
        data,
        x='Avg_Credit_Limit',
        y='Total_Credit_Cards',
        z='Total_visits_online',
        color=f"{model_name}_Cluster",
        title=title,
        color_continuous_scale=color_scale
    )

    # Menambahkan titik baru pada grafik
    fig.add_trace(
        go.Scatter3d(
            x=new_point['Avg_Credit_Limit'],
            y=new_point['Total_Credit_Cards'],
            z=new_point['Total_visits_online'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='New Point'
        )
    )

    return fig, new_cluster

st.set_page_config(
    page_title="11679 - Unsupervised Learning",  # 12345 diisi dengan 5 digit NPM
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.markdown("<h1 style='text-align: center;'>Unsupervised Learning - Listya</h1>", unsafe_allow_html=True)  # Nama diganti dengan nama panggilan
    st.dataframe(input_data)

    # Direktori tempat penyimpanan ketiga model yang telah di-dump sebelumnya
    # model_directory = r"D:/coolyeah/semester5/ml/Unsupervised Learning (Praktek)/Unsupervised Learning (Praktek)/Tugas4_B_11679"
    model_path = {
        "AGG_model": "AGG_model.pkl",
        "KMeans_model": "KMeans_model.pkl",
        "DBSCAN_model": "DBSCAN_model.pkl"
    }

    # Load ketiga model ke dalam dictionary
    models = {}
    for model_name, path in model_path.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
        else:
            st.write(f"Model {model_name} tidak ditemukan di path: {path}")

    # Sidebar untuk memasukkan nilai untuk titik baru yang akan diprediksi clusternya
    avg_CL = st.sidebar.number_input("Average Credit Limit", 0, 100000)
    sum_CC = st.sidebar.number_input("Total Credit Cards", 0, 10)
    sum_VO = st.sidebar.number_input("Total Visits Online", 0, 16)

    if st.sidebar.button("Prediksi !"):
        # Fitur yang digunakan untuk prediksi
        features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_online']
        
        # Memasukkan data titik baru ke dalam DataFrame
        new_point = pd.DataFrame({
            'Avg_Credit_Limit': [avg_CL],
            'Total_Credit_Cards': [sum_CC],
            'Total_visits_online': [sum_VO]
        })

        # Model clustering yang digunakan dan warna grafik scatternya
        cluster_method = [
            ("KMeans_model", models["KMeans_model"], "KMeans Clustering", px.colors.sequential.Cividis),
            ("AGG_model", models["AGG_model"], "Agglomerative Clustering", px.colors.sequential.Mint),
            ("DBSCAN_model", models["DBSCAN_model"], "DBSCAN Clustering", px.colors.sequential.Plasma)
        ]

        # Membuat tiga kolom untuk menampilkan grafik
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, (model_name, model, title, color_scale) in enumerate(cluster_method):
            fig, new_cluster = scatter(model, model_name, input_data, new_point, features, color_scale, title)
            with cols[i]:
                st.plotly_chart(fig)
                st.markdown(f"<p style='text-align: center;'>Titik baru masuk ke dalam cluster: {new_cluster}</p>", unsafe_allow_html=True)
