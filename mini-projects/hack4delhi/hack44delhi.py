# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from bs4 import BeautifulSoup
import requests
soup = BeautifulSoup(requests.get("https://traffic.delhipolice.gov.in/water-logging-area").text)

# %%
soup.find("tbody").find_all("tr")[79].find_all("td")[3].text.strip()

# %%
import csv
file = open("delhi_water_logging_areas.csv", "w", newline='', encoding='utf-8')
writer = csv.writer(file)
colhead = soup.find("tbody").find_all("tr")[0].find_all("td")
writer.writerow([col.text.strip() for col in colhead])

# %%
for row in soup.find("tbody").find_all("tr")[1::]:
    cols = row.find_all("td")
    writer.writerow([cols[0].text.strip(), cols[1].text.strip(), cols[2].text.strip(), cols[3].text.strip().split("\n"), cols[4].text.strip()])

# %%
file.close()

# %%
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm


# -----------------------------
# Create geocoder
# -----------------------------
geolocator = Nominatim(user_agent="monsoon-readiness-hackathon")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


# %%

# -----------------------------
# Load your CSV
# -----------------------------
df = pd.read_csv("delhi_water_logging_areas.csv")

# -----------------------------
# Build full query per row
# -----------------------------
def build_query(row):
    parts = []

    if pd.notna(row.get("Specific location")) and str(row["Specific location"]).strip() not in ["--", "-"]:
        parts.append(str(row["Specific location"]))

    if pd.notna(row.get("Name of the Road")):
        parts.append(str(row["Name of the Road"]))

    parts.append("Delhi, India")

    return ", ".join(parts)

latitudes = []
longitudes = []

print("Starting geocoding...\n")

# Add tqdm progress bar
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding locations"):
    query = build_query(row)

    try:
        # verbose print
        print(f"🔍 Querying: {query}")

        location = geocode(query)

        if location:
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)
            print(f"✅ Found: ({location.latitude}, {location.longitude})\n")
        else:
            latitudes.append(None)
            longitudes.append(None)
            print("⚠️ No result found\n")

    except Exception as e:
        latitudes.append(None)
        longitudes.append(None)
        print(f"❌ Error: {e}\n")

# -----------------------------
# Save results
# -----------------------------
df["latitude"] = latitudes
df["longitude"] = longitudes

df.to_csv("waterlogging_with_coordinates.csv", index=False)

print("\n🎉 Done! Saved as waterlogging_with_coordinates.csv")


# %%
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import re

df = pd.read_csv("D:\me\CODING\Github\DeepRed\delhi_water_logging_areas.csv")

geolocator = Nominatim(user_agent="monsoon-readiness-hackathon")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)

    # remove brackets / quotes
    s = re.sub(r"[\[\]']", "", s)

    junk_words = [
        "Near", "near",
        "towards",
        "both carriageway",
        "carriageway",
        "Redlight",
        "roundabout",
        "loop",
        "--"
    ]
    for w in junk_words:
        s = s.replace(w, "")

    # collapse double spaces
    s = re.sub(r"\s+", " ", s)

    return s.strip()

latitudes = []
longitudes = []

print("\n🚀 Starting geocoding (Delhi-biased, no bounding box)...\n")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding"):
    loc = clean_text(row.get("Specific location"))
    road = clean_text(row.get("Name of the Road"))

    candidates = [
        f"{loc}, {road}, New Delhi, Delhi, India",
        f"{road}, New Delhi, Delhi, India",
        f"{loc}, New Delhi, Delhi, India",
        f"{road}, Delhi, India",
        f"{loc}, Delhi, India"
    ]

    found = False

    for q in candidates:
        if len(q.replace(",", "").strip()) == 0:
            continue

        print(f"🔍 Trying: {q}")

        try:
            location = geocode(q)

            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
                print(f"✅ Found: ({location.latitude}, {location.longitude})\n")
                found = True
                break

        except Exception as e:
            print(f"❌ Error: {e}")

    if not found:
        latitudes.append(None)
        longitudes.append(None)
        print("⚠️ No match after all fallbacks\n")

df["latitude"] = latitudes
df["longitude"] = longitudes

df.to_csv("waterlogging_with_coordinates_WORKING.csv", index=False)

print("\n🎉 Done! Saved as waterlogging_with_coordinates_WORKING.csv\n")


# %%
df = pd.read_csv("D:\me\CODING\Github\DeepRed\waterlogging_with_coordinates_WORKING.csv")   

# %%
df[df.isnull().any(axis=1)]

# %%
import pandas as pd
import ast   # to safely parse list-like strings

def count_dates(x):
    if pd.isna(x):
        return None
    
    try:
        # convert "['a','b']" -> ['a','b']
        parsed = ast.literal_eval(x)
        
        # if it's a list, return its length
        if isinstance(parsed, list):
            return len(parsed)
        
        # otherwise default to 1
        return 1
    
    except Exception:
        return 1

# apply only where Frequency is null
mask = df["Frequency"].isna()

df.loc[mask, "Frequency"] = df.loc[mask, "Date"].apply(count_dates)

# optionally convert Frequency to int
df["Frequency"] = df["Frequency"].astype(int)

# save back
df.to_csv("waterlogging_updated.csv", index=False)

print("Done — saved as waterlogging_updated.csv")


# %%
import pandas as pd
df = pd.read_csv("D:\me\CODING\Github\DeepRed\waterlogging_updated.csv")

# %%
df[df.isnull().any(axis=1)]

# %%
location = geocode("Kasturba Nagar, Delhi, India")

# %%
location.latitude, location.longitude

# %%
df.loc[df["Sl. No."] == 207, ["latitude", "longitude"]] = [location.latitude, location.longitude]

# %%
df.to_csv("waterlogging_final.csv", index=False)

# %%
df = pd.read_csv("D:\me\CODING\Github\DeepRed\waterlogging_final.csv")

# %%
df = pd.read_csv("D:\me\CODING\Github\DeepRed\waterlogging_final.csv")

# %%
df[df.isnull().any(axis=1)]

# %%
df.describe().to_markdown()

# %%
# %pip install pandas geopandas shapely h3 folium matplotlib

# %%
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import h3
import folium
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load your CSV
# -----------------------------
df = pd.read_csv("waterlogging_final.csv")

# -----------------------------
# 2) Combine duplicate coordinates
# -----------------------------
df = df.groupby(["latitude", "longitude"], as_index=False)["Frequency"].sum()

# -----------------------------
# 3) Convert to H3 hexagons
# -----------------------------
resolution = 7  # smaller number = bigger hexagons

df["h3_index"] = df.apply(
    lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], resolution),
    axis=1
)

# -----------------------------
# 4) Aggregate frequency per hex
# -----------------------------
hex_agg = df.groupby("h3_index")["Frequency"].sum().reset_index()

# -----------------------------
# 5) Convert H3 hex to polygons
# -----------------------------
def h3_to_polygon(h):
    boundary = h3.cell_to_boundary(h)  # returns list of (lat, lon)

    # Convert to (lon, lat) for Shapely
    boundary_lonlat = [(lng, lat) for lat, lng in boundary]

    return Polygon(boundary_lonlat)

hex_agg["geometry"] = hex_agg["h3_index"].apply(h3_to_polygon)

gdf = gpd.GeoDataFrame(hex_agg, geometry="geometry", crs="EPSG:4326")

# -----------------------------
# 6) Assign colors
# -----------------------------
def freq_to_color(f):
    if f == 0:
        return "#351998"  # white
    elif f == 1:
        return "#2ECC71"  # green
    elif 2 <= f <= 3:
        return "#F1C40F"  # yellow
    else:
        return "#E74C3C"  # red

gdf["color"] = gdf["Frequency"].apply(freq_to_color)

# -----------------------------
# 7) SAVE GEOJSON (for JS dashboards)
# -----------------------------
gdf.to_file("hexbin_data.geojson", driver="GeoJSON")

print("Saved GeoJSON → hexbin_data.geojson")

# -----------------------------
# 8) STATIC PNG MAP
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, color=gdf["color"], edgecolor="black", linewidth=0.2)

ax.set_title("Water-Logging Risk Hexbin Map — Delhi", fontsize=12)
ax.set_axis_off()

plt.savefig("hexbin_map.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved PNG → hexbin_map.png")

# -----------------------------
# 9) INTERACTIVE HTML MAP
# -----------------------------
center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

for _, row in gdf.iterrows():
    geojson = gpd.GeoSeries([row["geometry"]]).__geo_interface__
    folium.GeoJson(
        geojson,
        style_function=lambda x, color=row["color"]: {
            "color": "black",
            "weight": 1,
            "fillColor": color,
            "fillOpacity": 0.6,
        },
        tooltip=f"Frequency: {row['Frequency']}"
    ).add_to(m)

m.save("hexbin_map.html")

print("Saved interactive map → hexbin_map.html")


# %%
