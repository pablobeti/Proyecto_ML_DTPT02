from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_clean = pd.read_csv("movies_dataset_clean.csv")

# Entrenamiento del modelo de recomendación
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_clean['title'])
matriz_simil = cosine_similarity(tfidf_matrix)

app = FastAPI()

# http://127.0.0.1:8000/docs
 
@app.get("/peliculas_idioma/{Idioma}")
def peliculas_idioma(Idioma: str):
    count = df_clean[df_clean['original_language'] == Idioma].shape[0]
    return f"{count} películas fueron estrenadas en {Idioma}"

@app.get("/peliculas_duracion/{Pelicula}")
def peliculas_duracion(Pelicula: str):
    movie = df_clean[df_clean['title'] == Pelicula].iloc[0]
    duracion = movie['runtime']
    anio = movie['release_year']
    return f"Duración: {duracion}. Año: {anio}"
@app.get("/franquicia/{Franquicia}")
def franquicia(Franquicia: str):
    franquicia_movies = df_clean[df_clean['belongs_to_collection'].str.contains(Franquicia, na=False)]
    peliculas = franquicia_movies.shape[0]
    ganancia_total = franquicia_movies['revenue'].sum()
    ganancia_promedio = franquicia_movies['revenue'].mean()
    return f"La franquicia {Franquicia} posee {peliculas} película(s), una ganancia total de {ganancia_total} y una ganancia promedio de {ganancia_promedio}"
@app.get("/peliculas_pais/{Pais}")
def peliculas_pais(Pais: str):
    count = df_clean[df_clean['production_countries'].str.contains(Pais)].shape[0]
    return f"Se produjeron {count} películas en el país {Pais}"
@app.get("/productoras_exitosas/{Productora}")
def productoras_exitosas(Productora: str):
    productora_movies = df_clean[df_clean['production_companies'].str.contains(Productora)]
    peliculas = productora_movies.shape[0]
    revenue_total = productora_movies['revenue'].sum()
    return f"La productora {Productora} ha tenido un revenue de {revenue_total} en {peliculas} películas"
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    peliculas_director = df_clean[df_clean['director'].str.contains(nombre_director)]
    retorno_total_director = peliculas_director['return'].sum()
    peliculas = []
    for _, pelicula in peliculas_director.iterrows():
        pelicula_info = {
            'nombre': pelicula['title'],
            'fecha_lanzamiento': pelicula['release_date'],
            'return': pelicula['return'],
            'costo': pelicula['budget'],
            'ganancia': pelicula['revenue']
        }
        peliculas.append(pelicula_info)
    return {
        'director': nombre_director,
        'retorno_total_director': retorno_total_director,
        'peliculas': peliculas
    }
@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    indice_peli = df_clean[df_clean['title'] == titulo].index[0]
    sim_scores = list(enumerate(matriz_simil[indice_peli]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_pelis_simil = [df_clean.iloc[sim_score[0]]['title'] for sim_score in sim_scores[1:6]]
    return {
        'titulo': titulo,
        'recomendaciones': top_pelis_simil
    }
