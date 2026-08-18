import os
import sys

import pandas as pd

try:
    import kagglehub
except ImportError:
    print("Falta la librería 'kagglehub'.")
    print("Instálala con: pip install kagglehub[pandas-datasets]")
    sys.exit(1)


# Configuración de pandas para que se vea mejor en consola
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)


# =============================================================================
# DATASETS DISPONIBLES
# =============================================================================

DATASETS = {
    "1": {
        "nombre": "Spotify Dataset 1921-2020 (600k Tracks)",
        "handle": "yamaerenay/spotify-dataset-19212020-600k-tracks",
    },
    "2": {
        "nombre": "Spotify200 (Top 200 semanal)",
        "handle": "yelexa/spotify200",
    },
}

# Caché en memoria para no volver a descargar / releer en cada operación
_cache_rutas = {}
_cache_dataframes = {}


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def descargar_dataset(handle):
    """Descarga (o reutiliza la copia local en caché de kagglehub) el dataset completo."""
    if handle in _cache_rutas:
        return _cache_rutas[handle]

    print(f"\nDescargando / verificando dataset '{handle}' desde Kaggle...")
    ruta = kagglehub.dataset_download(handle)
    _cache_rutas[handle] = ruta
    print(f"Dataset disponible en: {ruta}")
    return ruta


def listar_archivos(ruta):
    """Lista los archivos de datos (csv/json/parquet) dentro del dataset descargado."""
    archivos = []
    for root, _, files in os.walk(ruta):
        for f in files:
            if f.lower().endswith((".csv", ".json", ".parquet")):
                archivos.append(os.path.join(root, f))
    return sorted(archivos)


def elegir_archivo(archivos):
    print("\nArchivos disponibles en el dataset:")
    for i, a in enumerate(archivos, start=1):
        print(f"  {i}. {os.path.relpath(a)}")

    while True:
        sel = input("Selecciona el número del archivo a cargar (o 'c' para cancelar): ").strip()
        if sel.lower() == "c":
            return None
        if sel.isdigit() and 1 <= int(sel) <= len(archivos):
            return archivos[int(sel) - 1]
        print("Opción inválida, intenta de nuevo.")


def leer_archivo(ruta_archivo):
    print(f"\nCargando '{ruta_archivo}' con pandas...")
    if ruta_archivo.endswith(".csv"):
        return pd.read_csv(ruta_archivo, low_memory=False)
    if ruta_archivo.endswith(".json"):
        return pd.read_json(ruta_archivo)
    if ruta_archivo.endswith(".parquet"):
        return pd.read_parquet(ruta_archivo)
    raise ValueError("Formato de archivo no soportado.")


def cargar_dataframe(clave_dataset):
    """Carga (con caché) el DataFrame elegido por el usuario para un dataset de Kaggle."""
    info = DATASETS[clave_dataset]

    if clave_dataset in _cache_dataframes:
        return _cache_dataframes[clave_dataset], info["nombre"]

    ruta = descargar_dataset(info["handle"])
    archivos = listar_archivos(ruta)

    if not archivos:
        print("No se encontraron archivos csv/json/parquet en el dataset.")
        return None, info["nombre"]

    archivo = archivos[0] if len(archivos) == 1 else elegir_archivo(archivos)
    if archivo is None:
        return None, info["nombre"]

    try:
        df = leer_archivo(archivo)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None, info["nombre"]

    _cache_dataframes[clave_dataset] = df
    return df, info["nombre"]


# =============================================================================
# EXPLORACIÓN GENERAL DE LOS DATOS (por dataset individual)
# =============================================================================

def mostrar_info_general(df, nombre):
    print(f"\n===== {nombre} =====")
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    print("\nColumnas y tipos de datos:")
    print(df.dtypes)
    print("\nPrimeros 5 registros:")
    print(df.head())


def mostrar_estadisticas(df):
    print("\nEstadísticas descriptivas (columnas numéricas):")
    print(df.describe())


def mostrar_nulos(df):
    print("\nValores nulos por columna:")
    print(df.isnull().sum())


# =============================================================================
# UTILIDADES PARA COMBINAR / DEDUPLICAR DATOS DE VARIAS FUENTES
# =============================================================================

def buscar_columna(df, candidatos):
    """Busca la primera columna existente (sin importar mayúsculas/minúsculas)
    dentro de una lista de nombres candidatos."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidatos:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def limpiar_artista(valor):
    """Limpia valores de artista que a veces vienen como texto de lista,
    p. ej. "['Nombre del Artista']" -> "Nombre del Artista"."""
    if pd.isna(valor):
        return valor
    texto = str(valor).strip()
    if texto.startswith("[") and texto.endswith("]"):
        texto = texto.strip("[]")
        texto = texto.split(",")[0]  # si hay varios artistas, toma el primero
        texto = texto.strip().strip("'").strip('"')
    return texto


def normalizar(serie):
    """Normaliza texto (minúsculas, sin espacios extra) para poder comparar /
    deduplicar entre datasets distintos sin importar mayúsculas o espacios."""
    return serie.astype(str).str.strip().str.lower()


def _mostrar_tabla(df_tabla, nombre_tabla):
    if df_tabla is None:
        return
    print(f"\n{nombre_tabla} ({len(df_tabla):,} filas):")
    print(df_tabla.head(10))


# =============================================================================
# ESQUEMA ESTRELLA POR DATASET INDIVIDUAL
# =============================================================================

def construir_dim_tiempo(df):
    col_fecha = buscar_columna(df, ["date", "release_date", "week", "chart_week", "fecha"])
    if not col_fecha:
        print("No se encontró columna de fecha en este dataset; se omite dim_tiempo.")
        return None

    fechas = pd.to_datetime(df[col_fecha], errors="coerce").dropna().dt.date
    fechas_unicas = pd.Series(fechas.unique(), name="fecha").sort_values().reset_index(drop=True)

    dim = pd.DataFrame({"fecha": fechas_unicas})
    fecha_dt = pd.to_datetime(dim["fecha"])

    dim.insert(0, "tiempo_key", range(1, len(dim) + 1))
    dim["anio"] = fecha_dt.dt.year
    dim["mes"] = fecha_dt.dt.month
    dim["nombre_mes"] = fecha_dt.dt.month_name()
    dim["dia_mes"] = fecha_dt.dt.day
    dim["dia_semana"] = fecha_dt.dt.dayofweek
    dim["nombre_dia_semana"] = fecha_dt.dt.day_name()
    dim["trimestre"] = fecha_dt.dt.quarter
    dim["es_fin_de_semana"] = dim["dia_semana"].isin([5, 6])
    # 'hora' no existe en el dataset de origen (solo hay fecha) -> se omite
    return dim


def construir_dim_artista(df):
    col_artista = buscar_columna(df, ["artists", "artist", "artist_name", "nombre_artista"])
    if not col_artista:
        print("No se encontró columna de artista en este dataset; se omite dim_artista.")
        return None

    artistas = df[[col_artista]].dropna().copy()
    artistas.columns = ["nombre_artista"]
    artistas["nombre_artista"] = artistas["nombre_artista"].apply(limpiar_artista)
    artistas = artistas.drop_duplicates().reset_index(drop=True)
    artistas.insert(0, "artista_key", range(1, len(artistas) + 1))
    artistas["artista_id"] = artistas["artista_key"].apply(lambda x: f"ART-{x}")
    # 'sello_discografico' y 'pais_artista' no existen en el dataset de origen -> se omiten
    return artistas


def construir_dim_track(df):
    col_titulo = buscar_columna(df, ["name", "track_name", "title", "titulo_track"])
    if not col_titulo:
        print("No se encontró columna de título de canción; se omite dim_track.")
        return None

    col_dur = buscar_columna(df, ["duration_ms", "duracion_total_ms"])
    col_genero = buscar_columna(df, ["genre", "track_genre", "genero_musical"])
    col_pop = buscar_columna(df, ["popularity", "popularidad_actual"])

    seleccion = [col_titulo]
    rename = {col_titulo: "titulo_track"}
    for col, nuevo_nombre in ((col_dur, "duracion_total_ms"),
                               (col_genero, "genero_musical"),
                               (col_pop, "popularidad_actual")):
        if col:
            seleccion.append(col)
            rename[col] = nuevo_nombre

    tracks = df[seleccion].dropna(subset=[col_titulo]).drop_duplicates().reset_index(drop=True)
    tracks = tracks.rename(columns=rename)
    tracks.insert(0, "track_key", range(1, len(tracks) + 1))
    tracks["track_id"] = tracks["track_key"].apply(lambda x: f"TRK-{x}")
    # 'idioma' no existe en el dataset de origen -> se omite
    return tracks


def construir_hecho_streaming(df, dim_track, dim_tiempo):
    if dim_track is None:
        print("No hay dim_track disponible; se omite hecho_streaming.")
        return None

    col_titulo = buscar_columna(df, ["name", "track_name", "title"])
    col_fecha = buscar_columna(df, ["date", "release_date", "week", "chart_week"])
    col_streams = buscar_columna(df, ["streams", "playcount", "cantidad_reproducciones"])

    if not col_titulo:
        print("No se encontró columna de título para relacionar con dim_track; se omite hecho_streaming.")
        return None

    hecho = pd.DataFrame({"titulo_track": df[col_titulo]})
    hecho = hecho.merge(dim_track[["track_key", "titulo_track"]], on="titulo_track", how="left")

    if col_fecha and dim_tiempo is not None:
        hecho["fecha"] = pd.to_datetime(df[col_fecha], errors="coerce").dt.date
        hecho = hecho.merge(dim_tiempo[["tiempo_key", "fecha"]], on="fecha", how="left")
        hecho = hecho.drop(columns=["fecha"])

    hecho["cantidad_reproducciones"] = df[col_streams] if col_streams else 1

    hecho = hecho.drop(columns=["titulo_track"]).dropna(subset=["track_key"])
    hecho.insert(0, "streaming_id", range(1, len(hecho) + 1))
    # 'tiempo_reproducido_ms', 'reproduccion_completa_flag', 'usuario_key',
    # 'dispositivo_key' y 'contexto_key' no existen en el dataset de origen -> se omiten
    return hecho


# =============================================================================
# ESQUEMA ESTRELLA UNIFICADO (los dos datasets combinados en una sola tabla
# por dimensión / hecho, deduplicando registros repetidos entre ambos)
# =============================================================================

def construir_dim_artista_combinado(df1, df2):
    piezas = []
    for df in (df1, df2):
        if df is None:
            continue
        col = buscar_columna(df, ["artists", "artist", "artist_name", "nombre_artista"])
        if col:
            piezas.append(df[col].dropna().apply(limpiar_artista))

    if not piezas:
        print("Ninguno de los datasets tiene columna de artista; se omite dim_artista.")
        return None

    todos = pd.concat(piezas, ignore_index=True)
    todos = todos[todos.astype(str).str.strip() != ""]

    dim = pd.DataFrame({"nombre_artista": todos})
    dim["_clave"] = normalizar(dim["nombre_artista"])
    dim = dim.drop_duplicates(subset="_clave").drop(columns="_clave").reset_index(drop=True)
    dim.insert(0, "artista_key", range(1, len(dim) + 1))
    dim["artista_id"] = dim["artista_key"].apply(lambda x: f"ART-{x}")
    # 'sello_discografico' y 'pais_artista' no existen en ninguno de los dos datasets -> se omiten
    return dim


def construir_dim_track_combinado(df1, df2, dim_artista):
    piezas = []
    for df in (df1, df2):
        if df is None:
            continue
        col_titulo = buscar_columna(df, ["name", "track_name", "title", "titulo_track"])
        if not col_titulo:
            continue

        col_artista = buscar_columna(df, ["artists", "artist", "artist_name"])
        col_dur = buscar_columna(df, ["duration_ms", "duracion_total_ms"])
        col_genero = buscar_columna(df, ["genre", "track_genre", "genero_musical"])
        col_pop = buscar_columna(df, ["popularity", "popularidad_actual"])

        pieza = pd.DataFrame()
        pieza["titulo_track"] = df[col_titulo]
        pieza["nombre_artista"] = df[col_artista].apply(limpiar_artista) if col_artista else pd.NA
        pieza["duracion_total_ms"] = df[col_dur] if col_dur else pd.NA
        pieza["genero_musical"] = df[col_genero] if col_genero else pd.NA
        pieza["popularidad_actual"] = df[col_pop] if col_pop else pd.NA
        piezas.append(pieza)

    if not piezas:
        print("Ninguno de los datasets tiene columna de título de canción; se omite dim_track.")
        return None

    combinado = pd.concat(piezas, ignore_index=True).dropna(subset=["titulo_track"])
    combinado["_clave_track"] = normalizar(combinado["titulo_track"])
    combinado["_clave_artista"] = normalizar(combinado["nombre_artista"].fillna(""))

    # Se agrupa por título + artista (incluso si viene de datasets distintos) y se
    # toma el primer valor no nulo de cada columna para completar la información.
    agrupado = (
        combinado.groupby(["_clave_track", "_clave_artista"], as_index=False)
        .agg({
            "titulo_track": "first",
            "nombre_artista": "first",
            "duracion_total_ms": "first",
            "genero_musical": "first",
            "popularidad_actual": "first",
        })
    )

    if dim_artista is not None:
        dim_artista_tmp = dim_artista.copy()
        dim_artista_tmp["_clave_artista"] = normalizar(dim_artista_tmp["nombre_artista"])
        agrupado = agrupado.merge(
            dim_artista_tmp[["_clave_artista", "artista_key"]], on="_clave_artista", how="left"
        )
    else:
        agrupado["artista_key"] = pd.NA

    agrupado = agrupado.drop(columns=["_clave_track", "_clave_artista", "nombre_artista"])
    # Se quitan columnas que quedaron completamente vacías (ej. si ningún dataset trajo duración)
    agrupado = agrupado.dropna(axis=1, how="all")
    agrupado.insert(0, "track_key", range(1, len(agrupado) + 1))
    agrupado["track_id"] = agrupado["track_key"].apply(lambda x: f"TRK-{x}")
    # 'idioma' no existe en ninguno de los dos datasets -> se omite
    return agrupado


def construir_dim_tiempo_combinado(df1, df2):
    piezas = []
    for df in (df1, df2):
        if df is None:
            continue
        col_fecha = buscar_columna(df, ["date", "release_date", "week", "chart_week", "fecha"])
        if col_fecha:
            piezas.append(pd.to_datetime(df[col_fecha], errors="coerce").dropna().dt.date)

    if not piezas:
        print("Ninguno de los datasets tiene columna de fecha; se omite dim_tiempo.")
        return None

    fechas_unicas = pd.Series(pd.concat(piezas, ignore_index=True).unique(), name="fecha")
    fechas_unicas = fechas_unicas.sort_values().reset_index(drop=True)

    dim = pd.DataFrame({"fecha": fechas_unicas})
    fecha_dt = pd.to_datetime(dim["fecha"])
    dim.insert(0, "tiempo_key", range(1, len(dim) + 1))
    dim["anio"] = fecha_dt.dt.year
    dim["mes"] = fecha_dt.dt.month
    dim["nombre_mes"] = fecha_dt.dt.month_name()
    dim["dia_mes"] = fecha_dt.dt.day
    dim["dia_semana"] = fecha_dt.dt.dayofweek
    dim["nombre_dia_semana"] = fecha_dt.dt.day_name()
    dim["trimestre"] = fecha_dt.dt.quarter
    dim["es_fin_de_semana"] = dim["dia_semana"].isin([5, 6])
    # 'hora' no existe en ninguno de los dos datasets -> se omite
    return dim


def construir_hecho_streaming_combinado(fuentes, dim_track, dim_tiempo):
    """fuentes: lista de tuplas (nombre_dataset, dataframe)."""
    if dim_track is None:
        print("No hay dim_track disponible; se omite hecho_streaming.")
        return None

    partes = []
    for nombre_fuente, df in fuentes:
        if df is None:
            continue
        col_streams = buscar_columna(df, ["streams", "playcount", "cantidad_reproducciones"])
        col_titulo = buscar_columna(df, ["name", "track_name", "title"])
        col_fecha = buscar_columna(df, ["date", "release_date", "week", "chart_week"])

        if not col_streams or not col_titulo:
            print(f"'{nombre_fuente}' no tiene una métrica real de reproducciones; no aporta filas a hecho_streaming.")
            continue

        parte = pd.DataFrame()
        parte["_clave_track"] = normalizar(df[col_titulo])
        parte["cantidad_reproducciones"] = df[col_streams]
        parte["dataset_origen"] = nombre_fuente  # columna extra informativa (no está en el esquema original)
        if col_fecha:
            parte["fecha"] = pd.to_datetime(df[col_fecha], errors="coerce").dt.date
        partes.append(parte)

    if not partes:
        print("Ningún dataset tiene una métrica real de reproducciones; se omite hecho_streaming.")
        return None

    hechos = pd.concat(partes, ignore_index=True)

    dim_track_tmp = dim_track.copy()
    dim_track_tmp["_clave_track"] = normalizar(dim_track_tmp["titulo_track"])
    hechos = hechos.merge(dim_track_tmp[["_clave_track", "track_key"]], on="_clave_track", how="left")
    hechos = hechos.drop(columns=["_clave_track"]).dropna(subset=["track_key"])

    if "fecha" in hechos.columns and dim_tiempo is not None:
        hechos = hechos.merge(dim_tiempo[["tiempo_key", "fecha"]], on="fecha", how="left")
        hechos = hechos.drop(columns=["fecha"])

    hechos.insert(0, "streaming_id", range(1, len(hechos) + 1))
    # 'tiempo_reproducido_ms', 'reproduccion_completa_flag', 'usuario_key',
    # 'dispositivo_key' y 'contexto_key' no existen en ninguno de los datasets -> se omiten
    return hechos


# =============================================================================
# MENÚS
# =============================================================================

def menu_estructura(df, nombre):
    """Menú para ver la estructura tipo Data Warehouse de UN solo dataset."""
    contexto = {}

    while True:
        print(f"\n--- Esquema estrella (solo tablas disponibles) - {nombre} ---")
        print("1. Ver dim_tiempo")
        print("2. Ver dim_artista")
        print("3. Ver dim_track")
        print("4. Ver hecho_streaming")
        print("5. Volver")
        op = input("Selecciona una opción: ").strip()

        if op == "1":
            contexto.setdefault("tiempo", construir_dim_tiempo(df))
            _mostrar_tabla(contexto["tiempo"], "dim_tiempo")
        elif op == "2":
            contexto.setdefault("artista", construir_dim_artista(df))
            _mostrar_tabla(contexto["artista"], "dim_artista")
        elif op == "3":
            contexto.setdefault("track", construir_dim_track(df))
            _mostrar_tabla(contexto["track"], "dim_track")
        elif op == "4":
            contexto.setdefault("tiempo", construir_dim_tiempo(df))
            contexto.setdefault("track", construir_dim_track(df))
            contexto.setdefault(
                "hecho",
                construir_hecho_streaming(df, contexto.get("track"), contexto.get("tiempo")),
            )
            _mostrar_tabla(contexto["hecho"], "hecho_streaming")
        elif op == "5":
            break
        else:
            print("Opción inválida.")


def menu_dataset(clave):
    df, nombre = cargar_dataframe(clave)
    if df is None:
        return

    while True:
        print(f"\n=== {nombre} ===")
        print("1. Ver información general (shape, tipos, primeros registros)")
        print("2. Ver estadísticas descriptivas")
        print("3. Ver valores nulos")
        print("4. Ver estructura tipo Data Warehouse (dimensiones y hechos)")
        print("5. Volver al menú principal")
        op = input("Selecciona una opción: ").strip()

        if op == "1":
            mostrar_info_general(df, nombre)
        elif op == "2":
            mostrar_estadisticas(df)
        elif op == "3":
            mostrar_nulos(df)
        elif op == "4":
            menu_estructura(df, nombre)
        elif op == "5":
            break
        else:
            print("Opción inválida.")


def menu_combinado():
    """Menú para ver AMBOS datasets juntos en un solo esquema estrella unificado."""
    print("\nCargando ambos datasets para construir el esquema unificado...")
    df1, nombre1 = cargar_dataframe("1")
    df2, nombre2 = cargar_dataframe("2")

    if df1 is None and df2 is None:
        print("No se pudo cargar ningún dataset.")
        return

    contexto = {}
    while True:
        print("\n--- Esquema estrella UNIFICADO (datos reales de ambos datasets) ---")
        print("1. Ver dim_tiempo (combinada)")
        print("2. Ver dim_artista (combinada)")
        print("3. Ver dim_track (combinada)")
        print("4. Ver hecho_streaming (combinado)")
        print("5. Volver al menú principal")
        op = input("Selecciona una opción: ").strip()

        if op == "1":
            contexto.setdefault("tiempo", construir_dim_tiempo_combinado(df1, df2))
            _mostrar_tabla(contexto["tiempo"], "dim_tiempo (unificada)")
        elif op == "2":
            contexto.setdefault("artista", construir_dim_artista_combinado(df1, df2))
            _mostrar_tabla(contexto["artista"], "dim_artista (unificada)")
        elif op == "3":
            contexto.setdefault("artista", construir_dim_artista_combinado(df1, df2))
            contexto.setdefault("track", construir_dim_track_combinado(df1, df2, contexto["artista"]))
            _mostrar_tabla(contexto["track"], "dim_track (unificada)")
        elif op == "4":
            contexto.setdefault("tiempo", construir_dim_tiempo_combinado(df1, df2))
            contexto.setdefault("artista", construir_dim_artista_combinado(df1, df2))
            contexto.setdefault("track", construir_dim_track_combinado(df1, df2, contexto["artista"]))
            fuentes = [(nombre1, df1), (nombre2, df2)]
            contexto.setdefault(
                "hecho",
                construir_hecho_streaming_combinado(fuentes, contexto["track"], contexto["tiempo"]),
            )
            _mostrar_tabla(contexto["hecho"], "hecho_streaming (unificado)")
        elif op == "5":
            break
        else:
            print("Opción inválida.")


def menu_principal():
    while True:
        print("\n============================")
        print(" MENÚ DE DATASETS DE SPOTIFY")
        print("============================")
        for clave, info in DATASETS.items():
            print(f"{clave}. {info['nombre']}")
        print("3. Ver todos los datasets juntos (esquema unificado)")
        print("0. Salir")
        op = input("Selecciona una opción: ").strip()

        if op == "0":
            print("¡Hasta luego!")
            break
        elif op in DATASETS:
            menu_dataset(op)
        elif op == "3":
            menu_combinado()
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    menu_principal()