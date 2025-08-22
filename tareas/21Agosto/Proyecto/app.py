import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Título de la app ---
st.title("Análisis de Datos Musicales de Spotify 🎶")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.title("Cargar Datos")
st.sidebar.markdown("Sube tu archivo CSV con los datos de Spotify.")
file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

# --- LÓGICA PRINCIPAL DE LA APP ---
if file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(file)

    # --- KPIs ---
    st.subheader("Métricas Clave de Tu Colección")
    
    # Asegurarse de que la columna 'liked' existe y es numérica
    if 'liked' in df.columns:
        total_songs = len(df)
        liked_songs_count = df['liked'].sum()
        disliked_songs_count = total_songs - liked_songs_count
        
        # Filtrar canciones que el usuario ha marcado como "me gusta"
        liked_songs = df[df['liked'] == 1]
        
        # Calcular promedios solo si hay canciones con "me gusta"
        if not liked_songs.empty:
            avg_danceability = liked_songs['danceability'].mean()
            avg_valence = liked_songs['valence'].mean()
            
            # Mostrar métricas en columnas
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Total de Canciones", total_songs)
            with c2:
                st.metric("Canciones que te gustan", liked_songs_count)
            with c3:
                st.metric("Canciones que no te gustan", disliked_songs_count)
            with c4:
                st.metric("Bailabilidad Promedio", f"{avg_danceability:.2f}")
            with c5:
                st.metric("Felicidad Promedio", f"{avg_valence:.2f}")
        else:
            st.warning("No se encontraron canciones marcadas como 'me gusta' en tus datos. Algunas secciones no se mostrarán.")
            
    else:
        st.error("El archivo CSV debe contener una columna llamada 'liked' (0 para no, 1 para sí).")
        st.stop()

    st.markdown("---")
    
    # --- Análisis de Sensibilidad Musical ---
    st.subheader("Sensibilidad Musical del Usuario")
    
    # Seleccionamos columnas musicales clave
    features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
    
    # Calculamos la desviación estándar de cada columna
    if not liked_songs.empty and all(f in liked_songs.columns for f in features):
        sensitivity = liked_songs[features].std()
        overall_sensitivity = np.mean(sensitivity)
        
        st.write("Esta sección explora qué tan variada es tu preferencia musical. Valores más altos indican una mayor variedad en esa característica.")
        
        # Mostrar desviación estándar por característica
        st.write("Sensibilidad por atributo:")
        st.dataframe(sensitivity.to_frame(name="Desviación Estándar").T)
        
        st.write(f"Sensibilidad global estimada: **{overall_sensitivity:.2f}**")
    else:
        st.warning("No se puede calcular la sensibilidad musical. Verifica que tu archivo tenga las columnas necesarias y canciones que te gusten.")

    st.markdown("---")

    # --- Predicción y Recomendaciones ---
    st.subheader("Predicción de Canciones para Ti")
    
    if not liked_songs.empty:
        # Perfil promedio del usuario
        user_profile = liked_songs[features].mean()

        # Calcular similitud con todas las canciones
        def similarity(row):
            return np.linalg.norm(user_profile - row[features])

        df['similarity'] = df.apply(similarity, axis=1)

        # Sugerir top 10 canciones más parecidas que aún no ha marcado como liked
        recommendations = df[df['liked'] == 0].sort_values('similarity').head(10)
        
        st.write("Basado en el perfil musical de tus canciones favoritas, estas son 10 canciones que podrías disfrutar en el futuro.")
        st.dataframe(recommendations[['danceability', 'energy', 'valence', 'tempo', 'acousticness']])
    else:
        st.warning("No se pueden generar recomendaciones. Asegúrate de tener canciones marcadas como 'me gusta'.")

    st.markdown("---")
    
    # --- Análisis de Felicidad (Valence) ---
    st.subheader("Análisis de Felicidad en Tus Canciones Favoritas")
    
    if 'valence' in liked_songs.columns and not liked_songs.empty:
        # Top 5 canciones más felices según valence
        top_happy = liked_songs.sort_values('valence', ascending=False).head(5)
        top_happy['song_id'] = top_happy.index
        
        st.write("Estas son las 5 canciones que, según la métrica 'valence', más felicidad te provocan actualmente.")
        st.dataframe(top_happy[['song_id', 'danceability', 'energy', 'valence']])
        
        # Gráfica de dispersión
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(liked_songs['danceability'], liked_songs['valence'], 
                    color='lightgray', alpha=0.5, s=50, label='Otras canciones')
        
        colors = ['gold','orange','green','blue','red']
        for idx, row in enumerate(top_happy.itertuples()):
            ax.scatter(row.danceability, row.valence, 
                        color=colors[idx], s=row.energy*300, alpha=0.9, 
                        label=f"Top {idx+1}: ID {row.song_id}")
            ax.text(row.danceability + 0.01, row.valence + 0.01, f"{row.song_id}", color=colors[idx], fontsize=10)

        ax.set_title('Top 5 canciones más felices actualmente', fontsize=16)
        ax.set_xlabel('Bailabilidad (Danceability)', fontsize=12)
        ax.set_ylabel('Valencia (Felicidad)', fontsize=12)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    st.markdown("---")

    # --- Clustering de Canciones Favoritas ---
    st.subheader("Agrupación (Clustering) de Canciones Favoritas")
    st.markdown("Aquí agrupamos tus canciones favoritas en 3 grupos según su **bailabilidad**, **energía** y **felicidad** para ver si tienes patrones de gusto definidos.")
    
    if 'danceability' in liked_songs.columns and 'energy' in liked_songs.columns and 'valence' in liked_songs.columns and not liked_songs.empty:
        # Seleccionamos las características para el clustering
        X = liked_songs[['danceability', 'energy', 'valence']]
        
        # Normalizamos los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Definimos número de clusters (por ejemplo 3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        liked_songs['cluster'] = kmeans.fit_predict(X_scaled)

        # Gráfica del clustering
        fig, ax = plt.subplots(figsize=(10,6))
        colors = ['red', 'blue', 'green']
        
        for cluster in range(3):
            cluster_data = liked_songs[liked_songs['cluster'] == cluster]
            ax.scatter(cluster_data['danceability'], cluster_data['valence'], 
                        color=colors[cluster], label=f'Grupo {cluster}', alpha=0.6, s=100)
            
        ax.set_title('Clustering de canciones favoritas', fontsize=14)
        ax.set_xlabel('Bailabilidad (Danceability)')
        ax.set_ylabel('Valencia (Felicidad)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
        st.write("Ejemplo de canciones con su grupo asignado:")
        st.dataframe(liked_songs[['danceability','energy','valence','cluster']].head(10))
    else:
        st.warning("No se puede realizar el clustering. Verifica que tu archivo tenga las columnas 'danceability', 'energy' y 'valence' y que haya canciones que te gusten.")

else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis musical.")
