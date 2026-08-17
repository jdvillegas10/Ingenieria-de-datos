# Análisis Dimensional y Desempeño Comercial en Spotify (2017 - 2021)

##  Descripción del Proyecto
Este proyecto de Ingeniería de Datos tiene como objetivo procesar, analizar y modelar analíticamente más de **3.5 millones de registros** del ecosistema de Spotify. A través de una arquitectura por capas (*RAW, Staging, Analítica*), evaluamos la relación entre los atributos técnicos de audio (`danceability`, `energy`, `tempo`, etc.), la popularidad global de los artistas y su permanencia semanal en el Top 200 por país.

---

## 👥 Equipo de Trabajo
* **Julián Villegas** 
* **Ana Mican**  
* **Juan Torres** 

---

## 📊 Arquitectura de Datos y Fuentes (Capa RAW)
Los datos brutos provienen de Kaggle y se almacenan inalterados en la carpeta `data/RAW/`:

| Archivo | Descripción | Registros | Fuente |
| :--- | :--- | :--- | :--- |
| `tracks.csv` | Catálogo de canciones y variables de audio | 586,672 | Kaggle (Yamac Eren Ay)[cite: 1] |
| `artists.csv` | Perfiles de artistas, seguidores y géneros | 1,162,095 | Kaggle (Yamac Eren Ay)[cite: 1] |
| `final.csv` | Ranking histórico de Charts Top 200 por país | 1,787,999 | Kaggle (Yelexa/Spotify200)[cite: 1] |
| **TOTAL** | **Volumen Consolidado en RAW** | **~3,536,766** | **Cumple meta de ~3M**[cite: 1] |

---

## 📂 Estructura del Repositorio

```text
├── README.md              # Documentación principal del proyecto
├── .gitignore             
├── activity_report.md     # Matriz de asignación de responsabilidades del equipo
├── data/
│   ├── RAW/               # CSVs originales sin modificar (tracks, artists, final)
│   └── STAGING/           # Datasets limpios y procesados
├── docs/
│   ├── informe_corte_1.pdf # Entregable final del Primer Corte
│   └── diagramas/         # Modelo Dimensional (Estrella / Copo de Nieve)
└── src/
    ├── __init__.py
    ├── diagnostico_raw.py # Script de perfilamiento de la capa RAW
    └── etl_limpieza.py    # Scripts de transformación y validación de datos
