# Predictor NSP con variables climáticas

Aplicación Streamlit para visualizar y probar un clasificador de
**No Se Presenta (NSP)**.

## Estructura final para GitHub

```text
Predictor-NSP/
├── streamlit_app.py
├── requirements.txt
├── modelo_nsp.joblib
└── README.md
```

Para generar `modelo_nsp.joblib` se incluye además, durante el desarrollo:

```text
entrenar_modelo.py
```

Después de entrenar puede mantenerlo en el repositorio o eliminarlo si desea
dejar únicamente los cuatro archivos de entrega.

## Variables climáticas externas

El modelo incorpora:

- `PRECIPITACION_MM`: precipitación acumulada del día.
- `HUMEDAD_RELATIVA_PCT`: humedad relativa media del día.
- `TEMPORALIDAD`, derivada en:
  - `SECO_NORMAL`
  - `LLUVIA`
  - `HUMEDAD_ALTA`
  - `LLUVIA_Y_HUMEDAD_ALTA`

El clima se obtiene mediante Open-Meteo.

## 1. Instalar dependencias

En PowerShell, dentro de la carpeta del proyecto:

```powershell
py -m pip install -r requirements.txt
```

Si usa Python 3.13:

```powershell
py -3.13 -m pip install -r requirements.txt
```

## 2. Generar el modelo

Coloque la base histórica junto a `entrenar_modelo.py`.

Ejemplo:

```text
Predictor-NSP/
├── entrenar_modelo.py
├── GestionCitaciones(1).xls
├── streamlit_app.py
├── requirements.txt
└── README.md
```

Ejecute:

```powershell
py -3.13 .\entrenar_modelo.py "GestionCitaciones(1).xls"
```

El entrenamiento:

1. identifica `EJECUTADA` como clase 0;
2. identifica `NO ASISTIO (NSP)` como clase 1;
3. obtiene lluvia y humedad históricas;
4. entrena un Random Forest base;
5. entrena un Random Forest con clima usando la misma validación;
6. guarda las métricas comparativas;
7. genera `modelo_nsp.joblib`.

Se requiere conexión a Internet durante este paso para consultar los datos
meteorológicos externos.

## 3. Ejecutar la aplicación

Cuando `modelo_nsp.joblib` esté en la misma carpeta:

```powershell
py -3.13 -m streamlit run .\streamlit_app.py
```

## 4. Publicar en Streamlit Community Cloud

Suba a GitHub:

```text
streamlit_app.py
requirements.txt
modelo_nsp.joblib
README.md
```

Seleccione `streamlit_app.py` como archivo principal.

## Nota metodológica

Las variables que reflejan lo ocurrido después de la cita, como el estado final,
hora de recepción o atención, no se utilizan como predictores.

Para entrenamiento se utiliza clima histórico. Para una cita futura, la
aplicación intenta obtener información meteorológica de pronóstico; si no está
disponible, el usuario puede ingresar lluvia y humedad manualmente.

La salida del modelo es una estimación de riesgo y no una certeza de inasistencia.
