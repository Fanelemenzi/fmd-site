# streamlit_fmd_map_app.py
# Streamlit app to map Foot-and-Mouth Disease (FMD) outbreaks
# Features:
# - Load CSV from a default path or via uploader
# - Auto-detect latitude/longitude and key columns (infections, vaccinations, country, date)
# - Interactive Folium map with clustered markers and clickable popups showing record details
# - Sidebar statistics: total infections per country, vaccination totals, number of outbreaks
# - Filters: country, species (if present), date range (if present)
# - Download filtered data as CSV

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import altair as alt
from io import StringIO
from datetime import datetime
from cordon_polyline import add_cordon_to_map, get_cordon_bounds

st.set_page_config(page_title="FMD Outbreak Map", layout="wide")

# Initialize session state for map stability
if 'map_center' not in st.session_state:
    st.session_state.map_center = [-26.0, 28.0]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6

# ---------- Helper functions ----------
@st.cache_data
def read_csv_auto(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    return df


def detect_lat_lon(df):
    # Common column names
    lat_candidates = [c for c in df.columns if c.lower() in ('lat','latitude','y','gps_latitude','latitude_deg')]
    lon_candidates = [c for c in df.columns if c.lower() in ('lon','long','longitude','x','gps_longitude','longitude_deg')]

    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]

    # Try fuzzy matching
    for c in df.columns:
        low = c.lower()
        if 'lat' in low and not lat_candidates:
            lat_candidates.append(c)
        if ('lon' in low or 'long' in low) and not lon_candidates:
            lon_candidates.append(c)

    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]

    # Try a single 'coordinates' column like "lat, lon"
    coords_cols = [c for c in df.columns if 'coord' in c.lower() or 'location'==c.lower()]
    for c in coords_cols:
        sample = df[c].dropna().astype(str)
        if sample.str.contains(',').any():
            # parse
            lat_col = df[c].astype(str).str.split(',', expand=True)[0].str.strip().astype(float)
            lon_col = df[c].astype(str).str.split(',', expand=True)[1].str.strip().astype(float)
            df['_derived_lat'] = lat_col
            df['_derived_lon'] = lon_col
            return '_derived_lat', '_derived_lon'

    return None, None


def detect_column(df, keywords):
    keywords = [k.lower() for k in keywords]
    for c in df.columns:
        low = c.lower()
        for k in keywords:
            if k in low:
                return c
    return None


def build_popup_html(row, is_geocoded=False):
    # Build a compact HTML popup showing key information
    html = '<div style="max-width:350px; font-family: Arial, sans-serif;">'
    
    # Header with outbreak ID
    outbreak_id = row.get('outbreak_id', 'Unknown')
    html += f'<h4 style="margin:0 0 10px 0; color:#333; border-bottom:2px solid #e74c3c; padding-bottom:5px;">{outbreak_id}</h4>'
    
    if is_geocoded:
        html += '<div style="background:#fff3cd; border:1px solid #ffeaa7; padding:5px; margin:5px 0; border-radius:3px; font-size:11px;"><em>📍 Estimated location</em></div>'
    
    # Key information in a clean format
    key_fields = {
        'Country': row.get('country', 'N/A'),
        'Province/State': row.get('admin1_province_state', 'N/A'),
        'District': row.get('admin2_district_muni', 'N/A'),
        'Status': row.get('status', 'N/A'),
        'Serotype': row.get('serotype', 'N/A'),
        'Host Species': row.get('host_species', 'N/A'),
        'Premises Affected': row.get('num_premises_affected', 'N/A'),
        'Animals Affected': row.get('num_animals_affected', 'N/A'),
        'Date Reported': row.get('date_reported', 'N/A'),
        'Control Measures': row.get('control_measures', 'N/A')
    }
    
    html += '<table style="width:100%; font-size:12px; border-collapse:collapse; margin-top:10px;">'
    for label, value in key_fields.items():
        if pd.notna(value) and str(value).strip() != '' and str(value) != 'N/A':
            display_val = str(value)
            # Truncate long values
            if len(display_val) > 40:
                display_val = display_val[:37] + '...'
            html += f'<tr><td style="padding:3px 8px 3px 0; font-weight:bold; color:#555; vertical-align:top;">{label}:</td><td style="padding:3px 0; color:#333;">{display_val}</td></tr>'
    
    html += '</table></div>'
    return html

# ---------- UI ----------
st.title("Foot-and-Mouth Disease — Outbreak Map (Eswatini | South Africa | Mozambique)")

st.write("Upload a CSV with outbreak records, or the app will try to load the default CSV bundled on the server.")

with st.sidebar:
    st.header("Map Options")
    show_cordon = st.checkbox("Show FMD Cordon Line", value=True, help="Display the Foot-and-Mouth Disease cordon boundary")
    
    st.header("Summary statistics")
    # Statistics will be populated after data load
    
    st.header("Filters")
    # Filters will be populated after data load

# Load data
default_path = 'fmd_example.csv'
try:
    df = read_csv_auto(default_path)
except Exception as e:
    st.error(f"Could not read CSV file at {default_path}: {e}")
    st.stop()

st.write(f"Data loaded — {len(df)} rows and {len(df.columns)} columns.")



# Show first rows collapsed
with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df.head(10))

# Detect important columns
lat_col, lon_col = detect_lat_lon(df)
country_col = detect_column(df, ['country','nation','state','iso'])
infect_col = detect_column(df, ['infect','case','confirmed','count'])
vacc_col = detect_column(df, ['vaccin','vax','dose','vaccinated'])
date_col = detect_column(df, ['date','day','reported'])
species_col = detect_column(df, ['species','animal','host'])

# Convert types where possible
if infect_col:
    df['_infect_count'] = pd.to_numeric(df[infect_col], errors='coerce')
else:
    df['_infect_count'] = np.nan

if vacc_col:
    df['_vacc_count'] = pd.to_numeric(df[vacc_col], errors='coerce')
else:
    df['_vacc_count'] = np.nan

if date_col:
    try:
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        df['_date'] = pd.NaT
else:
    df['_date'] = pd.NaT

# Alert if lat/lon missing
if lat_col is None or lon_col is None:
    st.warning('Could not automatically detect latitude/longitude columns.\nPlease ensure your CSV includes latitude/longitude columns (e.g., "latitude", "longitude") or a "coordinates" column formatted as "lat, lon".')
    st.stop()

# Safe numeric cast
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

# Geocoding fallback for missing coordinates
def get_province_coordinates():
    """Return approximate coordinates for South African provinces and other regions"""
    return {
        'Free State': (-28.5, 26.5),
        'Gauteng': (-26.2, 28.0),
        'KwaZulu-Natal': (-29.0, 30.5),
        'Mpumalanga': (-25.5, 30.0),
        'North West': (-26.0, 25.5),
        'Eastern Cape': (-32.0, 26.5),
        'Western Cape': (-33.5, 19.0),
        'Northern Cape': (-29.0, 21.0),
        'Limpopo': (-24.0, 29.5),
        'Eswatini': (-26.5, 31.5),
        'Mozambique': (-18.0, 35.0)
    }

# Fill missing coordinates using province/location data
province_coords = get_province_coordinates()
admin1_col = detect_column(df, ['admin1', 'province', 'state'])

missing_coords_mask = df[lat_col].isna() | df[lon_col].isna()
original_missing_count = missing_coords_mask.sum()

if original_missing_count > 0:
    st.info(f"Found {missing_coords_mask.sum()} records with missing coordinates. Attempting to geocode using location information...")
    
    for idx, row in df[missing_coords_mask].iterrows():
        # Try to match province/admin1
        if admin1_col and pd.notna(row[admin1_col]):
            province = row[admin1_col].strip()
            if province in province_coords:
                df.at[idx, lat_col] = province_coords[province][0]
                df.at[idx, lon_col] = province_coords[province][1]
                continue
        
        # Try to match country as fallback
        if country_col and pd.notna(row[country_col]):
            country = row[country_col].strip()
            if country in province_coords:
                df.at[idx, lat_col] = province_coords[country][0]
                df.at[idx, lon_col] = province_coords[country][1]

# Check again for valid coordinates after geocoding
coords_mask = df[lat_col].notna() & df[lon_col].notna()
if coords_mask.sum() == 0:
    st.error('No rows have valid latitude/longitude values even after geocoding attempt. Cannot build map.')
    st.stop()

geocoded_count = coords_mask.sum() - (len(df) - original_missing_count)
if geocoded_count > 0:
    st.success(f"Successfully geocoded {geocoded_count} records using location information.")

# Show data quality metrics
original_coords_count = len(df) - original_missing_count
total_mappable = coords_mask.sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Records with original coordinates", original_coords_count)
with col2:
    st.metric("Records geocoded", geocoded_count)
with col3:
    st.metric("Total mappable records", total_mappable)

# Filters
countries = df[country_col].dropna().unique().tolist() if country_col else []
species_opts = df[species_col].dropna().unique().tolist() if species_col else []

with st.sidebar:
    selected_countries = st.multiselect('Country', options=sorted(countries), default=sorted(countries))
    selected_species = st.multiselect('Species', options=sorted(species_opts), default=sorted(species_opts))
    
    # Date filter
    if not df['_date'].isna().all():
        min_date = df['_date'].min()
        max_date = df['_date'].max()
        start_date, end_date = st.date_input('Date range', value=(min_date.date(), max_date.date()))
        # filter
        mask_date = (df['_date'] >= pd.to_datetime(start_date)) & (df['_date'] <= pd.to_datetime(end_date))
    else:
        mask_date = pd.Series([True]*len(df))

mask_country = pd.Series([True]*len(df))
if country_col and selected_countries:
    mask_country = df[country_col].isin(selected_countries)

mask_species = pd.Series([True]*len(df))
if species_col and selected_species:
    mask_species = df[species_col].isin(selected_species)

mask_coords = df[lat_col].notna() & df[lon_col].notna()

filtered = df[mask_country & mask_species & mask_date & mask_coords].copy()

# Update sidebar stats at the top
with st.sidebar:
    # Clear the header and add stats
    total_infections = int(filtered['_infect_count'].sum(skipna=True)) if not filtered['_infect_count'].isna().all() else 'Unknown'
    total_vacc = int(filtered['_vacc_count'].sum(skipna=True)) if not filtered['_vacc_count'].isna().all() else 'Unknown'
    total_outbreaks = len(filtered)
    
    st.metric('Total outbreaks', total_outbreaks)
    st.metric('Total premises affected', total_infections)
    st.metric('Total animals slaughtered', total_vacc)

# Infections per country
if country_col:
    country_agg = filtered.groupby(country_col)['_infect_count'].sum(min_count=1).reset_index().sort_values('_infect_count', ascending=False)
else:
    country_agg = pd.DataFrame()

st.header('Map of Outbreaks')

# Show legend for marker types
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔴 **Red markers**: Precise GPS coordinates")
with col2:
    st.markdown("🟠 **Orange markers**: Estimated locations (province/region level)")
with col3:
    st.markdown("❌ **Blue X markers**: Cordon checkpoints (line connects Maphiveni→Lukhula→Siteki→Mambane)")

# Track which coordinates were geocoded for different marker styles
if 'original_missing_count' in locals() and original_missing_count > 0:
    # Mark rows that had missing coordinates originally
    df['_is_geocoded'] = missing_coords_mask
else:
    df['_is_geocoded'] = False

# Build stable folium map
mean_lat = filtered[lat_col].mean()
mean_lon = filtered[lon_col].mean()

# Use session state for stable map center, fallback to calculated center
if not np.isnan(mean_lat) and not np.isnan(mean_lon):
    calculated_center = [mean_lat, mean_lon]
else:
    calculated_center = [-26.0, 28.0]

# Update session state only if significantly different
current_center = st.session_state.map_center
if abs(current_center[0] - calculated_center[0]) > 2 or abs(current_center[1] - calculated_center[1]) > 2:
    st.session_state.map_center = calculated_center

# Create map with stable configuration
m = folium.Map(
    location=st.session_state.map_center, 
    zoom_start=st.session_state.map_zoom,
    tiles='OpenStreetMap',
    prefer_canvas=True,
    control_scale=True,
    zoom_control=True,
    scrollWheelZoom=True,
    doubleClickZoom=False,  # Disable double-click zoom for stability
    dragging=True,
    keyboard=False,  # Disable keyboard controls for stability
    worldCopyJump=False
)

# Add marker cluster with stable configuration
marker_cluster = MarkerCluster(
    name="FMD Outbreaks",
    overlay=True,
    control=True,
    show_coverage_on_hover=False,  # Reduce visual noise
    zoom_to_bounds_on_click=False,  # Prevent auto-zoom on cluster click
    max_cluster_radius=50
).add_to(m)

for idx, row in filtered.iterrows():
    is_geocoded = row.get('_is_geocoded', False)
    popup_html = build_popup_html(row, is_geocoded)
    
    # Use different icons for geocoded vs original coordinates
    if is_geocoded:
        # Orange marker for estimated locations
        icon = folium.Icon(color='orange', icon='info-sign', prefix='glyphicon')
    else:
        # Red marker for precise locations
        icon = folium.Icon(color='red', icon='warning-sign', prefix='glyphicon')
    
    folium.Marker(
        location=[row[lat_col], row[lon_col]], 
        popup=folium.Popup(
            popup_html, 
            max_width=400,
            sticky=True  # Keep popup open until explicitly closed
        ),
        icon=icon,
        tooltip=f"Outbreak: {row.get('outbreak_id', 'Unknown ID')}"  # Quick preview on hover
    ).add_to(marker_cluster)

# Add cordon polyline if enabled
if show_cordon:
    m = add_cordon_to_map(m)

# Add layer control for better user experience
folium.LayerControl().add_to(m)

# Render map in Streamlit with stable configuration
st_data = st_folium(
    m, 
    width=800, 
    height=600,
    returned_objects=["last_object_clicked"],  # Only return essential data
    key="fmd_map"  # Stable key to prevent unnecessary re-renders
)

# Right column: charts and table
right_col, left_col = st.columns((1,1))
with right_col:
    st.subheader('Infections by country')
    if not country_agg.empty:
        chart = alt.Chart(country_agg).mark_bar().encode(x=alt.X('_infect_count:Q', title='Infections'), y=alt.Y(f'{country_col}:N', sort='-x', title='Country'))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info('Country information not found in the dataset.')

    st.subheader('Filtered data')
    st.dataframe(filtered.head(50))

with left_col:
    st.subheader('Aggregates')
    if country_col:
        st.dataframe(country_agg.rename(columns={country_col:'Country', '_infect_count':'Infections'}).head(20))
    else:
        st.info('No country column detected.')

# Download filtered CSV
csv_buffer = StringIO()
filtered.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

st.download_button('Download filtered CSV', data=csv_buffer.getvalue(), file_name='fmd_filtered.csv', mime='text/csv')

st.write('---')
st.markdown('**Notes & tips:**\n- If the app cannot find latitude/longitude, add columns named `latitude` and `longitude` or a `coordinates` column in the format `lat, lon`.\n- Column detection attempts to find sensible columns for infections/vaccinations/dates using common keywords. If detection fails, consider renaming columns to include keywords like `infect`, `cases`, `vaccin`, `date`, `country`.')

# End of app
