"""
Estas funciones las vamos a usar para limpias y modificar el dataset para luego trabajar en el, por ejemplos separandolos por decada.


"""

import numpy as np
import pandas as pd


def asignar_decada(año: int) -> str:
    """
    separamos por decada

    """
    inicio_decada = (año // 10) * 10
    return f"{inicio_decada}s"


def limpiar_dataframe(df: pd.DataFrame, *columnas_numericas: str, **opciones) -> pd.DataFrame:
    """
    nos fijamos si tenemos que lipiar el dataset por ejemplo si tiene valores nulos.
    """
    df_limpio = df.copy()

    for columna in columnas_numericas:
        df_limpio[columna] = pd.to_numeric(df_limpio[columna], errors="coerce")

    if opciones.get("eliminar_nulos", True):
        df_limpio = df_limpio.dropna(subset=list(columnas_numericas))

    año_min = opciones.get("año_min")
    año_max = opciones.get("año_max")
    if año_min is not None:
        df_limpio = df_limpio[df_limpio["Year"] >= año_min]
    if año_max is not None:
        df_limpio = df_limpio[df_limpio["Year"] <= año_max]

    return df_limpio.reset_index(drop=True)


def agregar_columna_decada(df: pd.DataFrame, columna_año: str = "Year") -> pd.DataFrame:
    """
    Agregamos la columna decade calculada basandonos en los años de cada fila.
    Agrega al DataFrame una columna 'Decade' calculada a partir del anio.
    por ejemplo el año 2020 lo ponemos en la decada de 2010 para no tener una decada formada por un solo año.

    """
    df_resultado = df.copy()
    df_resultado["Decade"] = df_resultado[columna_año].apply(
        lambda año: asignar_decada(año) if año != 2020 else "2010s"
    )
    return df_resultado


def resumir_por_grupo(df: pd.DataFrame, columnas_grupo: list, columna_valor: str,
                       funciones_resumen, kwargs) -> pd.DataFrame:
    """
    Agrupamos las columnas indicadas para sacar conclusiones sobre ellas.
    """
    if not funciones_resumen:
        funciones_resumen = ("mean",)

    resumen = (
        df.groupby(columnas_grupo)[columna_valor]
        .agg(list(funciones_resumen))
        .reset_index()
    )

    if kwargs.get("como_lista", False):
        resumen = resumen.rename(
            columns={f: f"{columna_valor}_{f}" for f in funciones_resumen}
        )

    return resumen


def filtrar_por_continentes(df: pd.DataFrame, continentes) -> pd.DataFrame:
    """
    Dejamos solo los continentes que queremos analizar.
    """
    if not continentes:
        return df.copy()
    return df[df["Continent"].isin(continentes)].reset_index(drop=True)


def calcular_variacion_entre_decadas(df_resumen: pd.DataFrame, columna_valor: str,
                                      columna_decada: str = "Decade") -> pd.DataFrame:
    """
    Calcula por continente, la variacion entre la primera y la ultima decada.

    """
    decadas_ordenadas = sorted(df_resumen[columna_decada].unique())
    primera, ultima = decadas_ordenadas[0], decadas_ordenadas[-1]

    pivote = df_resumen.pivot(index="Continent", columns=columna_decada, values=columna_valor)
    variacion = (pivote[ultima] - pivote[primera]).reset_index()
    variacion.columns = ["Continent", f"Variacion_{primera}_a_{ultima}"]
    return variacion
