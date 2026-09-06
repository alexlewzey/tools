"""Geo tool-kit convenience functions for geopandas."""

import logging
import webbrowser

import folium
import geopandas as gpd
import pandas as pd
from dstk import slibtk
from path import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", "{:.5f}".format)

CRS_BNG = "EPSG:27700"
CRS_LATLONG = "EPSG:4326"
UK_COORDINATES = (53.1098932107962, -1.6436354006598177)

ICON_COLORS = [
    "red",
    "blue",
    "gray",
    "darkred",
    "lightred",
    "orange",
    "beige",
    "green",
    "darkgreen",
    "lightgreen",
    "darkblue",
    "lightblue",
    "purple",
    "darkpurple",
    "pink",
    "cadetblue",
    "lightgray",
    "black",
]
ICON_COLORS = list(reversed(ICON_COLORS))

# geopandas ############################################################################


def to_geodataframe_with_points(
    df: pd.DataFrame, longitude: str = "longitude", latitude: str = "latitude"
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[longitude], df[latitude])
    )


def batch_distance(batch, polygon):
    """Batch=df['geometry'].values polygon=shape.boundary."""
    return [polygon.distance(point).iloc[0] for point in tqdm(batch)]


def dissolve_all(shapes):
    """Dissolve whole dataframe into one polygon."""
    shapes["dissolve"] = 1
    return shapes.dissolve(by="dissolve")


# folium ###############################################################################


def folium_points(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    popups: str | list[str],
    color: str = "green",
) -> folium.Map:
    popups = slibtk.listify(popups)
    logger.info("creating point map")
    points = [[point.xy[1][0], point.xy[0][0]] for point in gdf["geometry"]]
    fg = folium.FeatureGroup(name=layer_name)
    m.add_child(fg)
    for i, coord in enumerate(points):
        popup_text = "\n".join([f"{popup}={gdf[popup].iloc[i]}" for popup in popups])
        folium.Marker(
            location=coord, popup=popup_text, icon=folium.Icon(color=color)
        ).add_to(fg)
    return m


def folium_polygon(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    popups: str | list[str],
    color: str = "hex",
    opacity: float = 0.5,
) -> folium.Map:
    popups = slibtk.listify(popups)
    fg = folium.FeatureGroup(name=layer_name)
    m.add_child(fg)
    layer = folium.GeoJson(
        gdf,
        style_function=lambda feature: {
            "fillColor": feature["properties"][color],
            "color": feature["properties"][color],
            "fillOpacity": opacity,
        },
    ).add_to(fg)
    folium.GeoJsonTooltip(popups).add_to(layer)
    return m


def folium_choropleth(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    geo_code: str,
    color: str,
    popups: str | list[str],
    layer_name: str | None = None,
    legend_name: str | None = None,
    color_scheme: str = "YlOrBr",
    **kwargs,
) -> folium.Map:
    popups = slibtk.listify(popups)
    legend_name = legend_name if legend_name else color
    # layer_name = layer_name if layer_name else color
    # fg = folium.FeatureGroup(name=layer_name)
    # m.add_child(fg)
    layer = folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=[geo_code, color],
        fill_color=color_scheme,
        key_on=f"feature.properties.{geo_code}",
        legend_name=legend_name,
        **kwargs,
    ).add_to(m)
    layer.geojson.add_child(folium.features.GeoJsonTooltip(popups))
    return m


def show_map(m: folium.Map, path: Path) -> None:
    folium.LayerControl().add_to(m)
    m.save(path.as_posix())
    webbrowser.open("file://" + path.as_posix())
