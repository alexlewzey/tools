"""Geo tool-kit convenience functions for geopandas.

reading tabular points
----------------------

stores = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude'])).dropna(subset=['latitude'])


reading in shapes
-----------------

regions: gpd.GeoDataFrame = gpd.read_file('path/fname.shp')
regions.set_crs(epsg='27700', inplace=True)


changing crs
------------

stores.crs = {"init":"epsg:4326"}
stores.to_crs(uk_boundary.crs, inplace=True)


plot color map
--------------

stores.plot(ax=ax, cmap='jet', column='distance_to_coast', legend=True)


interactive geopandas plot with mplleaflet
------------------------------------------

ax = df.plot(column='col', alpha=0.8)
mplleaflet.show(fig=ax.figure, path=(Paths.DESKTOP / 'map.html').as_posix(), crs=df.crs)


mplleaflet requirements
-----------------------
matplotlib==3.1.1
pyproj==1.9.6
geopandas==0.6.3

You have to recursively clone the mplleaflet repo (otherwise alpha param in geopandas does not work):
    1. git clone --recursive <url>
    2. in .git/config replace git@github.com: with https://github.com/
    3. run git submodule update --init
    4. pip install -e .


mplleaftlet choropleth & scatter map
------------------------------------

fig, ax = plt.subplots()
df['geometry'] = df.geometry.simplify(tolerance=500)
df.plot(column='col', alpha=0.8, ax=ax)

coord = gpd.GeoDataFrame(coord, geometry=gpd.points_from_xy(coord['longitude'], coord['latitude']))
coord.crs = {"init": "epsg:4326"}
coord.to_crs(t.crs, inplace=True)
coord.plot(color='magenta', ax=ax)
mplleaflet.show(fig=ax.figure, path=(Paths.DESKTOP / 'map.html').as_posix(), crs=df.crs)

coord.to_crs({'init': 'epsg:27700'}, inplace=True)

# british national grid = EPSG:27700
# latlong = EPSG:4326

# pandas columns as added to the geojson here
# js = json.loads(shape.to_json())
# js['features'][0]['properties'].keys()


folium how to make heatmap
--------------------------


heat_data = list(zip(df['latitude'].tolist(), df['longitude'].tolist()))
m = folium.Map(location=geotk.UK_COORDINATES, zoom_start=8)
plugins.HeatMap(heat_data).add_to(m)
geotk.show_map(m, Paths.OUTPUT / 'heatmap.html')
"""
from __future__ import annotations

import logging
import webbrowser

import folium
import geopandas as gpd
from tqdm import tqdm

from dstk import slibtk

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

# geopandas ############################################################################################################


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


# folium ###############################################################################################################


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
