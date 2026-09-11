from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Predictor NSP",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "modelo_nsp.joblib"

# Centro urbano de Hualqui; se puede modificar si se desea.
LATITUD_DEFAULT = -36.97594
LONGITUD_DEFAULT = -72.93839
TIMEZONE = "America/Santiago"

st.title("📊 Predictor de No Se Presenta (NSP)")
st.caption(
    "Visualización del clasificador y prueba interactiva de un nuevo registro. "
    "El modelo incorpora variables administrativas, temporales y climáticas."
)


# ============================================================
# CLIMA EXTERNO
# ============================================================
@st.cache_data(ttl=3600)
def obtener_clima_fecha(fecha_cita, latitud, longitud):
    """
    Obtiene precipitación diaria y humedad media diaria desde Open-Meteo.
    Para fechas futuras usa Forecast API; para fechas pasadas usa Historical API.
    """
    fecha_txt = pd.Timestamp(fecha_cita).strftime("%Y-%m-%d")
    hoy = pd.Timestamp.now(tz=TIMEZONE).date()
    fecha_obj = pd.Timestamp(fecha_cita).date()

    if fecha_obj >= hoy:
        endpoint = "https://api.open-meteo.com/v1/forecast"
    else:
        endpoint = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitud,
        "longitude": longitud,
        "start_date": fecha_txt,
        "end_date": fecha_txt,
        "hourly": "relative_humidity_2m,precipitation",
        "timezone": TIMEZONE,
    }

    respuesta = requests.get(endpoint, params=params, timeout=20)
    respuesta.raise_for_status()
    data = respuesta.json()

    if "hourly" not in data:
        raise ValueError("Open-Meteo no devolvió información horaria.")

    clima = pd.DataFrame({
        "fecha_hora": pd.to_datetime(data["hourly"]["time"]),
        "humedad": data["hourly"]["relative_humidity_2m"],
        "precipitacion": data["hourly"]["precipitation"],
    })

    precipitacion_mm = float(
        pd.to_numeric(clima["precipitacion"], errors="coerce").fillna(0).sum()
    )
    humedad_pct = float(
        pd.to_numeric(clima["humedad"], errors="coerce").mean()
    )

    return precipitacion_mm, humedad_pct


def temporalidad_clima(precipitacion_mm, humedad_pct):
    lluvia = precipitacion_mm > 0.1
    humedad_alta = humedad_pct >= 80

    if lluvia and humedad_alta:
        return "LLUVIA_Y_HUMEDAD_ALTA"
    if lluvia:
        return "LLUVIA"
    if humedad_alta:
        return "HUMEDAD_ALTA"
    return "SECO_NORMAL"


# ============================================================
# CARGA DEL MODELO
# ============================================================
@st.cache_resource
def cargar_modelo():
    if not MODELO_PATH.exists():
        return None
    bundle = joblib.load(MODELO_PATH)

    if isinstance(bundle, dict) and "modelo" in bundle:
        return bundle

    # Compatibilidad con un pipeline guardado directamente.
    return {
        "modelo": bundle,
        "metadata": {},
        "opciones": {},
    }


bundle = cargar_modelo()

if bundle is None:
    st.error(
        "No se encontró `modelo_nsp.joblib` en la carpeta del proyecto. "
        "Ejecute primero `entrenar_modelo.py` y luego coloque el archivo generado "
        "junto a `streamlit_app.py`."
    )
    st.stop()

modelo = bundle["modelo"]
metadata = bundle.get("metadata", {})
opciones_modelo = bundle.get("opciones", {})
features = metadata.get("features", [])


# ============================================================
# AYUDANTES DE INTERFAZ
# ============================================================
def obtener_opciones_ui(columna, por_defecto):
    vals = opciones_modelo.get(columna, [])
    vals = [str(x) for x in vals if str(x).strip()]
    return vals if vals else por_defecto


def crear_registro(
    edad,
    dias_anticipacion,
    hora_decimal,
    sexo,
    sector,
    prevision,
    profesion,
    actividad,
    jornada,
    tipo_solicitud,
    estab_origen,
    ges,
    tipo_atencion,
    categoria_atencion,
    sobre_cupo,
    especialidad,
    fecha_cita,
    precipitacion_mm,
    humedad_pct,
):
    fecha = pd.Timestamp(fecha_cita)

    fila = {
        "EDAD_PACIENTE": float(edad),
        "DIAS_ANTICIPACION": int(dias_anticipacion),
        "HORA_CITA_NUM": float(hora_decimal),
        "PRECIPITACION_MM": float(precipitacion_mm),
        "HUMEDAD_RELATIVA_PCT": float(humedad_pct),
        "SEXO_PACIENTE": sexo,
        "SECTOR": sector,
        "PREVISION": prevision,
        "PROFESION": profesion,
        "ACTIVIDAD": actividad,
        "JORNADA": jornada,
        "TIPO_SOLICITUD": tipo_solicitud,
        "ESTAB_ORIGEN": estab_origen,
        "GES": ges,
        "TIPO_ATENCION": tipo_atencion,
        "CATEGORIA_ATENCION": categoria_atencion,
        "SOBRE_CUPO": sobre_cupo,
        "ESPECIALIDAD": especialidad,
        "DIA_SEMANA": str(fecha.dayofweek),
        "MES": str(fecha.month),
        "TEMPORALIDAD": temporalidad_clima(
            float(precipitacion_mm), float(humedad_pct)
        ),
    }

    # Mantiene exactamente el orden esperado por el modelo.
    return pd.DataFrame([fila], columns=features)


# ============================================================
# PESTAÑAS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Resultados del modelo",
    "🧪 Probar clasificador",
    "ℹ️ Metodología",
])


# ============================================================
# RESULTADOS
# ============================================================
with tab1:
    st.subheader("Rendimiento del clasificador")

    metricas = metadata.get("metricas_clima", metadata.get("metricas", {}))
    metricas_base = metadata.get("metricas_base", {})

    if not metricas:
        st.warning("El archivo del modelo no contiene métricas guardadas.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metricas.get('accuracy', 0):.4f}")
        c2.metric("Precision", f"{metricas.get('precision', 0):.4f}")
        c3.metric("Recall", f"{metricas.get('recall', 0):.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("F1", f"{metricas.get('f1', 0):.4f}")
        c5.metric("ROC-AUC", f"{metricas.get('roc_auc', 0):.4f}")
        c6.metric("PR-AUC", f"{metricas.get('pr_auc', 0):.4f}")

        df_metricas = pd.DataFrame({
            "Métrica": [
                "Accuracy", "Precision", "Recall",
                "F1", "ROC-AUC", "PR-AUC"
            ],
            "Modelo + clima": [
                metricas.get("accuracy", np.nan),
                metricas.get("precision", np.nan),
                metricas.get("recall", np.nan),
                metricas.get("f1", np.nan),
                metricas.get("roc_auc", np.nan),
                metricas.get("pr_auc", np.nan),
            ],
        }).set_index("Métrica")

        if metricas_base:
            df_metricas["Modelo base"] = [
                metricas_base.get("accuracy", np.nan),
                metricas_base.get("precision", np.nan),
                metricas_base.get("recall", np.nan),
                metricas_base.get("f1", np.nan),
                metricas_base.get("roc_auc", np.nan),
                metricas_base.get("pr_auc", np.nan),
            ]

        st.markdown("#### Comparación de métricas")
        st.bar_chart(df_metricas, height=390)

        if metricas_base:
            delta_f1 = metricas.get("f1", 0) - metricas_base.get("f1", 0)
            if delta_f1 > 0:
                st.success(
                    f"Al incorporar lluvia y humedad, el F1 mejora "
                    f"{delta_f1:+.4f} respecto del modelo base."
                )
            elif delta_f1 < 0:
                st.info(
                    f"Las variables climáticas no mejoraron el F1 en esta prueba "
                    f"({delta_f1:+.4f}). Se mantienen como experimento comparativo."
                )
            else:
                st.info("El F1 no cambió al incorporar las variables climáticas.")

    st.markdown("#### Información del entrenamiento")
    st.write(
        f"**Modelo:** {metadata.get('modelo', 'Random Forest')}  \n"
        f"**Validación:** {metadata.get('metodo_validacion', 'No informada')}  \n"
        f"**Registros:** {metadata.get('registros_observados', 'No informado')}  \n"
        f"**Tasa NSP:** {metadata.get('tasa_nsp', 0):.1%}"
    )


# ============================================================
# PRUEBA INTERACTIVA
# ============================================================
with tab2:
    st.subheader("Clasificar un dato ingresado por el usuario")

    st.write(
        "Ingrese las características de una cita. La lluvia y la humedad pueden "
        "obtenerse automáticamente desde una fuente meteorológica externa o "
        "ingresarse manualmente."
    )

    with st.sidebar:
        st.markdown("### Ubicación para clima")
        latitud = st.number_input(
            "Latitud",
            value=float(metadata.get("latitud", LATITUD_DEFAULT)),
            format="%.5f",
        )
        longitud = st.number_input(
            "Longitud",
            value=float(metadata.get("longitud", LONGITUD_DEFAULT)),
            format="%.5f",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        fecha_cita = st.date_input("Fecha de la cita", value=date.today())
    with col_b:
        hora_cita = st.time_input("Hora de la cita")

    hora_decimal = hora_cita.hour + hora_cita.minute / 60

    if "precipitacion_input" not in st.session_state:
        st.session_state.precipitacion_input = 0.0
    if "humedad_input" not in st.session_state:
        st.session_state.humedad_input = 70.0

    if st.button("🌧️ Obtener lluvia y humedad automáticamente"):
        try:
            lluvia_mm, humedad_pct = obtener_clima_fecha(
                fecha_cita, latitud, longitud
            )
            st.session_state.precipitacion_input = round(lluvia_mm, 2)
            st.session_state.humedad_input = round(humedad_pct, 1)
            st.success(
                f"Clima obtenido: {lluvia_mm:.2f} mm de precipitación y "
                f"{humedad_pct:.1f}% de humedad media."
            )
        except Exception as e:
            st.warning(
                "No fue posible obtener el clima automáticamente. "
                "Puede ingresar los valores manualmente."
            )
            st.caption(str(e))

    with st.form("form_prediccion"):
        c1, c2, c3 = st.columns(3)

        with c1:
            edad = st.number_input(
                "Edad",
                min_value=0,
                max_value=120,
                value=45,
            )
            dias_anticipacion = st.number_input(
                "Días de anticipación de la cita",
                min_value=0,
                max_value=1000,
                value=7,
            )
            sexo = st.selectbox(
                "Sexo",
                obtener_opciones_ui("SEXO_PACIENTE", ["FEMENINO", "MASCULINO"]),
            )
            sector = st.selectbox(
                "Sector",
                obtener_opciones_ui("SECTOR", ["AMARILLO", "ROJO", "RURAL", "CECOSF"]),
            )
            prevision = st.selectbox(
                "Previsión",
                obtener_opciones_ui("PREVISION", ["FONASA - A", "FONASA - B", "FONASA - C", "FONASA - D"]),
            )

        with c2:
            profesion = st.selectbox(
                "Profesión",
                obtener_opciones_ui("PROFESION", ["MEDICO", "ENFERMERO", "PSICOLOGO"]),
            )
            actividad = st.selectbox(
                "Actividad",
                obtener_opciones_ui("ACTIVIDAD", ["MORBILIDAD", "CONTROL"]),
            )
            jornada = st.selectbox(
                "Jornada",
                obtener_opciones_ui("JORNADA", ["MAÑANA", "TARDE", "EXTENSION HORARIA"]),
            )
            tipo_solicitud = st.selectbox(
                "Tipo de solicitud",
                obtener_opciones_ui("TIPO_SOLICITUD", ["P. VENTANILLA", "TELEFONICO"]),
            )
            estab_origen = st.selectbox(
                "Establecimiento de origen",
                obtener_opciones_ui("ESTAB_ORIGEN", ["CESFAM HUALQUI"]),
            )

        with c3:
            ges = st.selectbox(
                "GES",
                obtener_opciones_ui("GES", ["SIN_DATO"]),
            )
            tipo_atencion = st.selectbox(
                "Tipo de atención",
                obtener_opciones_ui("TIPO_ATENCION", ["CONSULTA", "CONTROL"]),
            )
            categoria = st.selectbox(
                "Categoría de atención",
                obtener_opciones_ui("CATEGORIA_ATENCION", ["CONSULTA", "CONTROL"]),
            )
            sobre_cupo = st.selectbox(
                "Sobre cupo",
                obtener_opciones_ui("SOBRE_CUPO", ["N", "S"]),
            )
            especialidad = st.selectbox(
                "Especialidad",
                obtener_opciones_ui("ESPECIALIDAD", ["CONSULTA / CONTROL"]),
            )

        st.markdown("#### Variables climáticas externas")
        cc1, cc2, cc3 = st.columns(3)

        with cc1:
            precipitacion_mm = st.number_input(
                "Precipitación del día (mm)",
                min_value=0.0,
                max_value=500.0,
                value=float(st.session_state.precipitacion_input),
                step=0.1,
            )

        with cc2:
            humedad_pct = st.number_input(
                "Humedad relativa media (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.humedad_input),
                step=1.0,
            )

        with cc3:
            st.text_input(
                "Temporalidad",
                value=temporalidad_clima(
                    precipitacion_mm, humedad_pct
                ),
                disabled=True,
            )

        clasificar = st.form_submit_button(
            "🔎 Clasificar dato",
            type="primary",
            width="stretch",
        )

    if clasificar:
        X = crear_registro(
            edad,
            dias_anticipacion,
            hora_decimal,
            sexo,
            sector,
            prevision,
            profesion,
            actividad,
            jornada,
            tipo_solicitud,
            estab_origen,
            ges,
            tipo_atencion,
            categoria,
            sobre_cupo,
            especialidad,
            fecha_cita,
            precipitacion_mm,
            humedad_pct,
        )

        prob = float(modelo.predict_proba(X)[0, 1])
        umbral = float(metadata.get("threshold", 0.50))
        pred = int(prob >= umbral)

        st.markdown("---")
        st.subheader("Resultado")

        if pred == 1:
            st.error(f"⚠️ RIESGO DE NSP · Probabilidad: {prob:.1%}")
        else:
            st.success(f"✅ MENOR RIESGO DE NSP · Probabilidad: {prob:.1%}")

        st.progress(min(max(prob, 0.0), 1.0))
        st.write(
            f"**Condición climática:** "
            f"{temporalidad_clima(precipitacion_mm, humedad_pct)}"
        )

        st.caption(
            "La predicción corresponde a una estimación estadística y debe "
            "utilizarse como apoyo para priorizar acciones preventivas."
        )


# ============================================================
# METODOLOGÍA
# ============================================================
with tab3:
    st.subheader("Variables utilizadas")

    st.markdown(
        """
**Variables administrativas y temporales**
- Edad del paciente.
- Días de anticipación entre creación y cita.
- Hora de la cita.
- Día de la semana y mes.
- Sexo, sector y previsión.
- Profesión, actividad y jornada.
- Tipo de solicitud y establecimiento.
- Tipo/categoría de atención, sobrecupo y especialidad.

**Variables externas de clima**
- `PRECIPITACION_MM`: precipitación acumulada del día.
- `HUMEDAD_RELATIVA_PCT`: humedad relativa media del día.
- `TEMPORALIDAD`: variable derivada con cuatro categorías:
  `SECO_NORMAL`, `LLUVIA`, `HUMEDAD_ALTA` y
  `LLUVIA_Y_HUMEDAD_ALTA`.

Para el entrenamiento se utilizan datos meteorológicos históricos. Para una
cita futura, la aplicación consulta información de pronóstico cuando está
disponible.
"""
    )

    st.info(
        "Para evaluar si el clima aporta información predictiva, el entrenamiento "
        "guarda las métricas del modelo base y del modelo con clima utilizando "
        "la misma partición de validación."
    )
