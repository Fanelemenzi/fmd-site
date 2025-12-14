import pandas as pd
import folium

def load_cordon_nodes(csv_path='cordon_polyline_nodes.csv'):
    """Load cordon polyline nodes from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Sort by sequence_order to ensure proper polyline connection
        df = df.sort_values('sequence_order')
        return df
    except Exception as e:
        print(f"Error loading cordon nodes: {e}")
        return None

def create_cordon_polyline(df):
    """Create coordinate pairs for the cordon polyline - only connecting specific cordon areas"""
    if df is None or df.empty:
        return []
    
    # Filter for only the specific cordon areas: Maphiveni, Lukhula, Siteki, Mambane
    cordon_areas = ['Maphiveni', 'Lukhula', 'Siteki', 'Mambane']
    filtered_df = df[df['location_name'].isin(cordon_areas)].copy()
    
    # Sort by sequence order to ensure proper connection
    filtered_df = filtered_df.sort_values('sequence_order')
    
    # Extract coordinates in sequence order
    coordinates = []
    for _, row in filtered_df.iterrows():
        coordinates.append([row['latitude'], row['longitude']])
    
    return coordinates

def add_cordon_to_map(folium_map, csv_path='cordon_polyline_nodes.csv'):
    """Add cordon polyline and markers to existing folium map"""
    # Load the cordon nodes
    df = load_cordon_nodes(csv_path)
    if df is None:
        return folium_map
    
    # Create polyline coordinates
    coordinates = create_cordon_polyline(df)
    
    if coordinates:
        # Add the polyline to the map
        folium.PolyLine(
            locations=coordinates,
            color='blue',
            weight=3,
            opacity=0.8,
            popup='FMD Cordon Line',
            tooltip='Foot-and-Mouth Disease Cordon Boundary'
        ).add_to(folium_map)
        
        # Add markers for ALL cordon nodes (connected and unconnected)
        cordon_areas = ['Maphiveni', 'Lukhula', 'Siteki', 'Mambane']
        
        for _, row in df.iterrows():
            # Create popup with node information
            popup_html = f"""
            <div style="max-width:300px; font-family: Arial, sans-serif;">
                <h4 style="margin:0 0 10px 0; color:#1f77b4; border-bottom:2px solid #1f77b4; padding-bottom:5px;">
                    {row['node_id']}
                </h4>
                <table style="width:100%; font-size:12px; border-collapse:collapse;">
                    <tr><td style="padding:3px 8px 3px 0; font-weight:bold; color:#555;">Location:</td>
                        <td style="padding:3px 0; color:#333;">{row['location_name']}</td></tr>
                    <tr><td style="padding:3px 8px 3px 0; font-weight:bold; color:#555;">Country:</td>
                        <td style="padding:3px 0; color:#333;">{row['country']}</td></tr>
                    <tr><td style="padding:3px 8px 3px 0; font-weight:bold; color:#555;">Region:</td>
                        <td style="padding:3px 0; color:#333;">{row['admin1_region']}</td></tr>
                    <tr><td style="padding:3px 8px 3px 0; font-weight:bold; color:#555;">Coordinates:</td>
                        <td style="padding:3px 0; color:#333;">{row['latitude']:.4f}, {row['longitude']:.4f}</td></tr>
                    <tr><td style="padding:3px 8px 3px 0; font-weight:bold; color:#555;">Notes:</td>
                        <td style="padding:3px 0; color:#333;">{row['notes']}</td></tr>
                </table>
            </div>
            """
            
            # Determine if this node is part of the connected cordon line
            is_connected = row['location_name'] in cordon_areas
            status_text = "Connected Cordon" if is_connected else "Cordon Checkpoint"
            
            # Add marker with 'x' icon for all cordon nodes
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=350),
                icon=folium.Icon(color='blue', icon='remove', prefix='glyphicon'),
                tooltip=f"{status_text}: {row['location_name']}"
            ).add_to(folium_map)
    
    return folium_map

def get_cordon_bounds(csv_path='cordon_polyline_nodes.csv'):
    """Get the bounding box of the cordon polyline for map centering"""
    df = load_cordon_nodes(csv_path)
    if df is None or df.empty:
        return None
    
    return {
        'min_lat': df['latitude'].min(),
        'max_lat': df['latitude'].max(),
        'min_lon': df['longitude'].min(),
        'max_lon': df['longitude'].max(),
        'center_lat': df['latitude'].mean(),
        'center_lon': df['longitude'].mean()
    }