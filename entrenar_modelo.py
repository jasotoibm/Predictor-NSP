from pathlib import Path
import argparse
import io
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TIMEZONE = "America/Santiago"
LATITUD_DEFAULT = -36.97594
LONGITUD_DEFAULT = -72.93839


# ============================================================
# LECTURA Y NORMALIZACIÓN
# ============================================================
def quitar_acentos(texto):
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def normalizar_columna(col):
    col = quitar_acentos(col).strip().upper()
    col = re.sub(r"[^A-Z0-9]+", "_", col)
    return col.strip("_")


def normalizar_columnas(df):
    df = df.copy()
    nuevas = []
    usados = {}

    for col in df.columns:
        base = normalizar_columna(col)
        if base in usados:
            usados[base] += 1
            nuevas.append(f"{base}_{usados[base]}")
        else:
            usados[base] = 0
            nuevas.append(base)

    df.columns = nuevas
    return df


def leer_archivo(ruta):
    ruta = Path(ruta)
    contenido = ruta.read_bytes()
    inicio = contenido[:1000].lower()

    if ruta.suffix.lower() == ".csv":
        try:
            return pd.read_csv(io.BytesIO(contenido))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(contenido), encoding="latin1")

    if b"<table" in inicio or b"<html" in inicio:
        tablas = pd.read_html(io.BytesIO(contenido), header=0)
        if not tablas:
            raise ValueError("No se encontró una tabla en el archivo.")
        return tablas[0]

    return pd.read_excel(io.BytesIO(contenido))


def columna(df, candidatos, requerida=False):
    for candidato in candidatos:
        if candidato in df.columns:
            return candidato

    if requerida:
        raise ValueError(f"Falta una columna requerida: {candidatos}")

    return None


def texto_normalizado(serie):
    return (
        serie.fillna("SIN_DATO")
        .astype(str)
        .map(quitar_acentos)
        .str.strip()
        .str.upper()
    )


def parsear_hora(valor):
    if pd.isna(valor):
        return np.nan

    txt = str(valor).strip()
    match = re.match(r"^(\d{1,2}):(\d{2})", txt)

    if not match:
        return np.nan

    hora = int(match.group(1))
    minuto = int(match.group(2))

    if not (0 <= hora <= 23 and 0 <= minuto <= 59):
        return np.nan

    return hora + minuto / 60


# ============================================================
# CLIMA EXTERNO
# ============================================================
def descargar_clima_historico(fecha_inicio, fecha_fin, latitud, longitud):
    """
    Descarga datos horarios desde Open-Meteo y los transforma a:
    - precipitación acumulada diaria
    - humedad relativa media diaria
    """
    endpoint = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitud,
        "longitude": longitud,
        "start_date": pd.Timestamp(fecha_inicio).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(fecha_fin).strftime("%Y-%m-%d"),
        "hourly": "relative_humidity_2m,precipitation",
        "timezone": TIMEZONE,
    }

    respuesta = requests.get(endpoint, params=params, timeout=60)
    respuesta.raise_for_status()
    data = respuesta.json()

    horario = pd.DataFrame({
        "FECHA_HORA": pd.to_datetime(data["hourly"]["time"]),
        "HUMEDAD_RELATIVA_PCT": data["hourly"]["relative_humidity_2m"],
        "PRECIPITACION_MM": data["hourly"]["precipitation"],
    })

    horario["FECHA_CLIMA"] = horario["FECHA_HORA"].dt.normalize()

    diario = (
        horario.groupby("FECHA_CLIMA", as_index=False)
        .agg(
            PRECIPITACION_MM=("PRECIPITACION_MM", "sum"),
            HUMEDAD_RELATIVA_PCT=("HUMEDAD_RELATIVA_PCT", "mean"),
        )
    )

    return diario


def crear_temporalidad(df):
    lluvia = df["PRECIPITACION_MM"].fillna(0) > 0.1
    humedad_alta = df["HUMEDAD_RELATIVA_PCT"].fillna(0) >= 80

    condiciones = [
        lluvia & humedad_alta,
        lluvia,
        humedad_alta,
    ]

    categorias = [
        "LLUVIA_Y_HUMEDAD_ALTA",
        "LLUVIA",
        "HUMEDAD_ALTA",
    ]

    df["TEMPORALIDAD"] = np.select(
        condiciones,
        categorias,
        default="SECO_NORMAL",
    )

    return df


# ============================================================
# PREPARACIÓN DE CITACIONES
# ============================================================
def preparar_citaciones(df):
    df = normalizar_columnas(df)

    col_estado = columna(df, ["ESTADO"], True)
    col_fecha_creacion = columna(df, ["FECHA_CREACION"], True)
    col_fecha_cita = columna(df, ["FECHA_CITA"], True)
    col_hora = columna(df, ["HORA_CITA"], True)

    salida = pd.DataFrame(index=df.index)

    edad_col = columna(df, ["EDAD_PACIENTE", "EDAD"])
    salida["EDAD_PACIENTE"] = (
        pd.to_numeric(df[edad_col], errors="coerce")
        if edad_col
        else np.nan
    )

    fecha_creacion = pd.to_datetime(
        df[col_fecha_creacion], errors="coerce", dayfirst=True
    )
    fecha_cita = pd.to_datetime(
        df[col_fecha_cita], errors="coerce", dayfirst=True
    )

    salida["FECHA_CITA_ORDEN"] = fecha_cita
    salida["FECHA_CLIMA"] = fecha_cita.dt.normalize()

    salida["DIAS_ANTICIPACION"] = (
        fecha_cita.dt.normalize() - fecha_creacion.dt.normalize()
    ).dt.days
    salida.loc[
        salida["DIAS_ANTICIPACION"] < 0, "DIAS_ANTICIPACION"
    ] = np.nan

    salida["HORA_CITA_NUM"] = df[col_hora].map(parsear_hora)
    salida["DIA_SEMANA"] = fecha_cita.dt.dayofweek.astype("Int64").astype(str)
    salida["MES"] = fecha_cita.dt.month.astype("Int64").astype(str)

    mapa = {
        "SEXO_PACIENTE": ["SEXO_PACIENTE", "SEXO"],
        "SECTOR": ["SECTOR"],
        "PREVISION": ["PREVISION"],
        "PROFESION": ["PROFESION"],
        "ACTIVIDAD": ["ACTIVIDAD"],
        "JORNADA": ["JORNADA"],
        "TIPO_SOLICITUD": ["TIPO_SOLICITUD"],
        "ESTAB_ORIGEN": ["ESTAB_ORIGEN", "ESTABLECIMIENTO_ORIGEN"],
        "GES": ["GES"],
        "TIPO_ATENCION": ["TIPO_ATENCION"],
        "CATEGORIA_ATENCION": ["CATEGORIA_ATENCION"],
        "SOBRE_CUPO": ["SOBRE_CUPO"],
        "ESPECIALIDAD": ["ESPECIALIDAD"],
    }

    for nombre, candidatos in mapa.items():
        c = columna(df, candidatos)
        salida[nombre] = (
            texto_normalizado(df[c]) if c else "SIN_DATO"
        )

    estado = texto_normalizado(df[col_estado])

    target = pd.Series(np.nan, index=df.index, dtype=float)
    target[
        estado.str.contains(
            r"NO ASISTIO|NSP|INASISTENTE",
            regex=True,
        )
    ] = 1
    target[
        estado.str.contains(
            r"EJECUTADA|ATENDIDA|REALIZADA|REALIZADO",
            regex=True,
        )
    ] = 0

    salida["TARGET_NSP"] = target

    return salida


# ============================================================
# MODELOS
# ============================================================
NUMERICAS_BASE = [
    "EDAD_PACIENTE",
    "DIAS_ANTICIPACION",
    "HORA_CITA_NUM",
]

NUMERICAS_CLIMA = NUMERICAS_BASE + [
    "PRECIPITACION_MM",
    "HUMEDAD_RELATIVA_PCT",
]

CATEGORICAS_BASE = [
    "SEXO_PACIENTE",
    "SECTOR",
    "PREVISION",
    "PROFESION",
    "ACTIVIDAD",
    "JORNADA",
    "TIPO_SOLICITUD",
    "ESTAB_ORIGEN",
    "GES",
    "TIPO_ATENCION",
    "CATEGORIA_ATENCION",
    "SOBRE_CUPO",
    "ESPECIALIDAD",
    "DIA_SEMANA",
    "MES",
]

CATEGORICAS_CLIMA = CATEGORICAS_BASE + ["TEMPORALIDAD"]

FEATURES_BASE = NUMERICAS_BASE + CATEGORICAS_BASE
FEATURES_CLIMA = NUMERICAS_CLIMA + CATEGORICAS_CLIMA


def crear_pipeline(numericas, categoricas):
    pre_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    pre_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5,
            ),
        ),
    ])

    preprocesamiento = ColumnTransformer([
        ("num", pre_num, numericas),
        ("cat", pre_cat, categoricas),
    ])

    modelo = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    return Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", modelo),
    ])


def metricas(y_true, prob, umbral=0.50):
    pred = (prob >= umbral).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, prob)
        ),
        "pr_auc": float(
            average_precision_score(y_true, prob)
        ),
    }


def obtener_opciones(df):
    columnas = CATEGORICAS_CLIMA
    resultado = {}

    for c in columnas:
        resultado[c] = (
            df[c]
            .fillna("SIN_DATO")
            .astype(str)
            .value_counts()
            .index
            .tolist()[:150]
        )

    return resultado


# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar(ruta, salida, latitud, longitud):
    print("1/5 Leyendo archivo...")
    bruto = leer_archivo(ruta)

    print("2/5 Preparando citas...")
    datos = preparar_citaciones(bruto)
    datos = datos[datos["TARGET_NSP"].notna()].copy()

    if len(datos) < 200:
        raise ValueError("No hay suficientes registros observados.")

    if datos["TARGET_NSP"].nunique() < 2:
        raise ValueError("Se requieren asistencias y NSP.")

    fecha_min = datos["FECHA_CITA_ORDEN"].min()
    fecha_max = datos["FECHA_CITA_ORDEN"].max()

    print(
        f"3/5 Descargando clima histórico "
        f"{fecha_min.date()} a {fecha_max.date()}..."
    )
    clima = descargar_clima_historico(
        fecha_min,
        fecha_max,
        latitud,
        longitud,
    )

    datos = datos.merge(
        clima,
        on="FECHA_CLIMA",
        how="left",
    )
    datos = crear_temporalidad(datos)

    datos = datos.sort_values(
        "FECHA_CITA_ORDEN",
        na_position="last",
    ).reset_index(drop=True)

    corte = int(len(datos) * 0.80)
    train = datos.iloc[:corte].copy()
    val = datos.iloc[corte:].copy()

    if (
        train["TARGET_NSP"].nunique() < 2
        or val["TARGET_NSP"].nunique() < 2
    ):
        train, val = train_test_split(
            datos,
            test_size=0.20,
            random_state=42,
            stratify=datos["TARGET_NSP"].astype(int),
        )
        metodo = "Validación estratificada 80/20"
    else:
        metodo = "Validación temporal 80/20"

    y_train = train["TARGET_NSP"].astype(int)
    y_val = val["TARGET_NSP"].astype(int)

    print("4/5 Entrenando modelo base y modelo con clima...")

    base = crear_pipeline(
        NUMERICAS_BASE,
        CATEGORICAS_BASE,
    )
    base.fit(train[FEATURES_BASE], y_train)
    prob_base = base.predict_proba(
        val[FEATURES_BASE]
    )[:, 1]
    m_base = metricas(y_val, prob_base)

    clima_modelo = crear_pipeline(
        NUMERICAS_CLIMA,
        CATEGORICAS_CLIMA,
    )
    clima_modelo.fit(
        train[FEATURES_CLIMA],
        y_train,
    )
    prob_clima = clima_modelo.predict_proba(
        val[FEATURES_CLIMA]
    )[:, 1]
    m_clima = metricas(y_val, prob_clima)

    print("Métricas modelo base:", m_base)
    print("Métricas modelo + clima:", m_clima)

    print("5/5 Reentrenando modelo final con todos los datos...")
    final = crear_pipeline(
        NUMERICAS_CLIMA,
        CATEGORICAS_CLIMA,
    )
    final.fit(
        datos[FEATURES_CLIMA],
        datos["TARGET_NSP"].astype(int),
    )

    metadata = {
        "modelo": "RandomForestClassifier + clima",
        "metodo_validacion": metodo,
        "registros_observados": int(len(datos)),
        "n_nsp": int((datos["TARGET_NSP"] == 1).sum()),
        "n_asistencias": int(
            (datos["TARGET_NSP"] == 0).sum()
        ),
        "tasa_nsp": float(datos["TARGET_NSP"].mean()),
        "metricas_base": m_base,
        "metricas_clima": m_clima,
        "features": FEATURES_CLIMA,
        "threshold": 0.50,
        "latitud": float(latitud),
        "longitud": float(longitud),
        "timezone": TIMEZONE,
        "fuente_clima": "Open-Meteo Historical Weather API",
        "descripcion_temporalidad": {
            "SECO_NORMAL": "Sin lluvia y humedad < 80%",
            "LLUVIA": "Precipitación > 0.1 mm",
            "HUMEDAD_ALTA": "Humedad >= 80%",
            "LLUVIA_Y_HUMEDAD_ALTA": (
                "Precipitación > 0.1 mm y humedad >= 80%"
            ),
        },
    }

    bundle = {
        "modelo": final,
        "metadata": metadata,
        "opciones": obtener_opciones(datos),
    }

    joblib.dump(bundle, salida)

    print()
    print(f"Modelo guardado en: {salida}")
    print(
        "Ahora copie modelo_nsp.joblib junto a streamlit_app.py "
        "y ejecute Streamlit."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena el predictor NSP con variables climáticas."
    )
    parser.add_argument(
        "archivo",
        nargs="?",
        default="GestionCitaciones(1).xls",
        help="Ruta al Excel/XLS/CSV histórico.",
    )
    parser.add_argument(
        "--salida",
        default="modelo_nsp.joblib",
        help="Nombre del archivo de modelo.",
    )
    parser.add_argument(
        "--latitud",
        type=float,
        default=LATITUD_DEFAULT,
    )
    parser.add_argument(
        "--longitud",
        type=float,
        default=LONGITUD_DEFAULT,
    )

    args = parser.parse_args()

    entrenar(
        args.archivo,
        args.salida,
        args.latitud,
        args.longitud,
    )
