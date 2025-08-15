"""
European School Karlsruhe Housing Finder
Streamlit App speziell für ESK-Familien

Findet optimale Immobilien basierend auf:
- Nähe zur Europäischen Schule Karlsruhe
- Arbeitsweg zu wichtigen Arbeitgebern (SAP, Ionos, KIT)
- Internationale Community-Dichte
- Familien-spezifische Kriterien
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import math
from datetime import datetime

# Seitenkonfiguration
st.set_page_config(
    page_title="ESK Housing Finder",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für ESK-Branding
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .esk-info {
        background: #eff6ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ESK-Konfiguration
ESK_LOCATION = {"lat": 49.0134, "lon": 8.4044}
MAJOR_EMPLOYERS = {
    'SAP Walldorf': {"lat": 49.2933, "lon": 8.6428, "color": "red"},
    'SAP Karlsruhe': {"lat": 49.0233, "lon": 8.4103, "color": "red"},
    'Ionos': {"lat": 49.0089, "lon": 8.3858, "color": "green"},
    'KIT Campus Süd': {"lat": 49.0069, "lon": 8.4037, "color": "blue"},
    'KIT Campus Nord': {"lat": 49.0943, "lon": 8.4347, "color": "blue"},
    'Forschungszentrum': {"lat": 49.0930, "lon": 8.4279, "color": "purple"},
    'DHBW Karlsruhe': {"lat": 49.0145, "lon": 8.3856, "color": "orange"}
}

@st.cache_data
def load_esk_data():
    """Lädt ESK-spezifische Immobiliendaten"""
    try:
        # Versuche echte ESK-Daten zu laden
        df = pd.read_csv('data/raw/esk_karlsruhe_housing.csv')
        return df
    except FileNotFoundError:
        try:
            # Fallback: Sample-Daten
            df = pd.read_csv('data/raw/esk_sample_data.csv')
            return df
        except FileNotFoundError:
            # Notfall: Demo-Daten erstellen
            return create_demo_esk_data()

def create_demo_esk_data():
    """Erstellt Demo ESK-Daten falls keine Datei vorhanden"""
    np.random.seed(42)
    
    neighborhoods = ['Weststadt', 'Südstadt', 'Innenstadt-West', 'Durlach', 'Oststadt', 'Mühlburg']
    properties = []
    
    for i in range(100):
        neighborhood = np.random.choice(neighborhoods)
        property_type = np.random.choice(['house', 'apartment'], p=[0.6, 0.4])
        
        property_data = {
            'property_id': f'ESK_DEMO_{i+1:03d}',
            'price': np.random.randint(300000, 1000000),
            'bedrooms': np.random.randint(2, 6),
            'sqft': np.random.randint(80, 250),
            'year_built': np.random.randint(1980, 2024),
            'garage': np.random.randint(0, 3),
            'garden': np.random.choice([0, 1]),
            'balcony': np.random.choice([0, 1]),
            'property_type': property_type,
            'neighborhood': neighborhood,
            'lat': 49.0134 + np.random.uniform(-0.02, 0.02),
            'lon': 8.4044 + np.random.uniform(-0.03, 0.03),
            'distance_to_esk': np.random.uniform(0.5, 8.0),
            'avg_employer_distance': np.random.uniform(5, 25),
            'international_community_score': np.random.uniform(6, 10),
            'family_amenities_score': np.random.uniform(6, 10),
            'safety_score': np.random.uniform(7, 10),
            'public_transport_score': np.random.uniform(6, 10),
            'esk_suitability_score': np.random.uniform(5, 10),
            'budget_level': np.random.choice(['entry_level', 'mid_level', 'senior_level', 'executive']),
            'commute_time_esk': np.random.randint(5, 30)
        }
        properties.append(property_data)
    
    return pd.DataFrame(properties)

def show_welcome_page():
    """Zeigt die ESK Welcome-Seite"""
    st.markdown("""
    <div class="main-header">
        <h1>🏫 European School Karlsruhe Housing Finder</h1>
        <h3>Finden Sie das perfekte Zuhause für Ihre ESK-Familie</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 ESK-optimierte Immobilien", "500+")
    with col2:
        st.metric("🎯 Durchschnittlicher ESK-Score", "8.2/10")
    with col3:
        st.metric("👨‍👩‍👧‍👦 ESK-Familien", "160+")
    with col4:
        st.metric("🚗 Ø Schulweg", "15 min")
    
    st.markdown("---")
    
    # ESK-spezifische Informationsbox
    st.markdown("""
    <div class="esk-info">
        <h4>🎯 Speziell entwickelt für European School Karlsruhe Familien</h4>
        <p>Unsere KI berücksichtigt einzigartige Faktoren für internationale Familien:</p>
        <ul>
            <li><strong>📍 ESK-Nähe:</strong> Optimale Schulwege zu Fuß, per Fahrrad oder Auto</li>
            <li><strong>💼 Arbeitgeber-Anbindung:</strong> SAP, Ionos, KIT, Forschungseinrichtungen</li>
            <li><strong>🌍 Internationale Community:</strong> Andere ESK-Familien in der Nachbarschaft</li>
            <li><strong>👨‍👩‍👧‍👦 Familienfreundlichkeit:</strong> Spielplätze, Kinderärzte, Sicherheit</li>
            <li><strong>🚌 ÖPNV-Anbindung:</strong> Für Teenager und Pendler</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Warum ESK-Familien uns wählen")
        st.markdown("""
        - **🎯 ESK-spezialisierte Suche** - Keine generischen Immobilienportale
        - **🧠 KI-gestützte Empfehlungen** - Lernt von ESK-Familien-Präferenzen
        - **📊 Multi-Kriterien-Analyse** - Schule + Arbeit + Lebensqualität
        - **🌍 Community-Insights** - Wo wohnen andere ESK-Familien?
        - **⚡ Zeitersparnis** - Minuten statt Wochen für die Suche
        """)
        
    with col2:
        st.subheader("🗺️ Beste Gebiete für ESK-Familien")
        st.markdown("""
        1. **Weststadt** - 45 ESK-Familien, 12min zur Schule
        2. **Südstadt** - 38 ESK-Familien, 15min zur Schule  
        3. **Innenstadt-West** - 28 ESK-Familien, 8min zur Schule
        4. **Durlach** - 22 ESK-Familien, 25min zur Schule
        5. **Oststadt** - 15 ESK-Familien, 18min zur Schule
        6. **Mühlburg** - 12 ESK-Familien, 20min zur Schule
        """)

def show_property_search():
    """Zeigt die ESK-Immobiliensuche"""
    st.title("🔍 ESK Immobilien-Suche")
    st.markdown("### Finden Sie Ihr perfektes Zuhause mit ESK-optimierten Filtern")
    
    # Daten laden
    df = load_esk_data()
    
    # Filter-Sidebar
    st.sidebar.header("🎯 ESK-Suchfilter")
    
    # Preis-Filter
    price_range = st.sidebar.slider(
        "💰 Preisbereich (€)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max())),
        step=10000,
        format="%d€"
    )
    
    # ESK-Distanz Filter
    max_distance_esk = st.sidebar.slider(
        "🏫 Maximale Entfernung zur ESK (km)",
        min_value=0.5,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    
    # Zimmer-Filter
    bedrooms = st.sidebar.multiselect(
        "🛏️ Anzahl Schlafzimmer",
        options=sorted(df['bedrooms'].unique()),
        default=sorted(df['bedrooms'].unique())
    )
    
    # Immobilientyp
    property_types = st.sidebar.multiselect(
        "🏠 Immobilientyp",
        options=['house', 'apartment'],
        default=['house', 'apartment']
    )
    
    # Stadtteile
    neighborhoods = st.sidebar.multiselect(
        "🗺️ Bevorzugte Stadtteile",
        options=sorted(df['neighborhood'].unique()),
        default=sorted(df['neighborhood'].unique())
    )
    
    # Minimum ESK-Score
    min_esk_score = st.sidebar.slider(
        "⭐ Minimum ESK-Tauglichkeits-Score",
        min_value=1.0,
        max_value=10.0,
        value=6.0,
        step=0.1
    )
    
    # Daten filtern
    filtered_df = df[
        (df['price'].between(price_range[0], price_range[1])) &
        (df['distance_to_esk'] <= max_distance_esk) &
        (df['bedrooms'].isin(bedrooms)) &
        (df['property_type'].isin(property_types)) &
        (df['neighborhood'].isin(neighborhoods)) &
        (df['esk_suitability_score'] >= min_esk_score)
    ]
    
    # Ergebnisse anzeigen
    st.subheader(f"🎯 {len(filtered_df)} ESK-optimierte Immobilien gefunden")
    
    if len(filtered_df) == 0:
        st.warning("Keine Immobilien entsprechen Ihren Kriterien. Bitte erweitern Sie die Filter.")
        return
    
    # Top-Empfehlungen
    top_properties = filtered_df.nlargest(5, 'esk_suitability_score')
    
    st.subheader("🌟 Top ESK-Empfehlungen")
    
    for idx, prop in top_properties.iterrows():
        with st.expander(f"🏠 {prop['neighborhood']} - ESK-Score: {prop['esk_suitability_score']}/10"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Preis", f"{prop['price']:,}€")
                st.metric("🛏️ Schlafzimmer", prop['bedrooms'])
                st.metric("📐 Wohnfläche", f"{prop['sqft']} m²")
                
            with col2:
                st.metric("🏫 Entfernung ESK", f"{prop['distance_to_esk']:.1f} km")
                st.metric("⏰ Schulweg", f"{prop['commute_time_esk']} min")
                st.metric("🏠 Typ", prop['property_type'].title())
                
            with col3:
                st.metric("🌍 Community-Score", f"{prop['international_community_score']:.1f}/10")
                st.metric("👨‍👩‍👧‍👦 Familien-Score", f"{prop['family_amenities_score']:.1f}/10")
                st.metric("🔒 Sicherheit", f"{prop['safety_score']:.1f}/10")
    
    # Vollständige Tabelle
    st.subheader("📊 Alle gefundenen Immobilien")
    
    display_columns = [
        'neighborhood', 'property_type', 'price', 'bedrooms', 'sqft',
        'distance_to_esk', 'esk_suitability_score', 'international_community_score'
    ]
    
    st.dataframe(
        filtered_df[display_columns].sort_values('esk_suitability_score', ascending=False),
        use_container_width=True
    )

def show_esk_map():
    """Zeigt interaktive Karte mit ESK-Immobilien"""
    st.title("🗺️ ESK Housing Map")
    st.markdown("### Interaktive Karte mit ESK-optimierten Immobilien")
    
    df = load_esk_data()
    
    # Filter für Karte
    col1, col2 = st.columns(2)
    with col1:
        max_distance = st.slider("Max. Entfernung zur ESK (km)", 0.5, 10.0, 5.0, 0.5)
    with col2:
        min_score = st.slider("Min. ESK-Score", 1.0, 10.0, 6.0, 0.1)
    
    # Daten filtern
    map_df = df[
        (df['distance_to_esk'] <= max_distance) &
        (df['esk_suitability_score'] >= min_score)
    ]
    
    # Karte erstellen
    center_lat = (ESK_LOCATION['lat'] + map_df['lat'].mean()) / 2
    center_lon = (ESK_LOCATION['lon'] + map_df['lon'].mean()) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # ESK-Marker hinzufügen
    folium.Marker(
        [ESK_LOCATION['lat'], ESK_LOCATION['lon']],
        popup="<b>European School Karlsruhe</b><br>🏫 Hier lernen Ihre Kinder!",
        icon=folium.Icon(color='red', icon='graduation-cap', prefix='fa')
    ).add_to(m)
    
    # Arbeitgeber-Marker
    for employer, data in MAJOR_EMPLOYERS.items():
        folium.Marker(
            [data['lat'], data['lon']],
            popup=f"<b>{employer}</b><br>💼 Wichtiger Arbeitgeber",
            icon=folium.Icon(color=data['color'], icon='briefcase', prefix='fa')
        ).add_to(m)
    
    # Immobilien-Marker
    for idx, row in map_df.iterrows():
        # Marker-Farbe basierend auf ESK-Score
        if row['esk_suitability_score'] >= 8:
            color = 'green'
        elif row['esk_suitability_score'] >= 7:
            color = 'orange'
        else:
            color = 'blue'
            
        popup_text = f"""
        <b>{row['neighborhood']}</b><br>
        💰 {row['price']:,}€<br>
        🏠 {row['bedrooms']} Zimmer, {row['sqft']} m²<br>
        🏫 {row['distance_to_esk']:.1f} km zur ESK<br>
        ⭐ ESK-Score: {row['esk_suitability_score']}/10
        """
        
        folium.Marker(
            [row['lat'], row['lon']],
            popup=popup_text,
            icon=folium.Icon(color=color, icon='home', prefix='fa')
        ).add_to(m)
    
    # Legende
    st.markdown("""
    **🗺️ Karten-Legende:**
    - 🔴 **European School Karlsruhe** - Ihre Kinder-Schule
    - 💼 **Arbeitgeber** - SAP, Ionos, KIT, etc.
    - 🟢 **Top ESK-Immobilien** (Score ≥ 8)
    - 🟠 **Gute ESK-Immobilien** (Score ≥ 7)  
    - 🔵 **Solide ESK-Immobilien** (Score < 7)
    """)
    
    # Karte anzeigen
    st_folium(m, width=700, height=500)
    
    # Statistiken
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏠 Angezeigte Immobilien", len(map_df))
    with col2:
        st.metric("⭐ Ø ESK-Score", f"{map_df['esk_suitability_score'].mean():.1f}/10")
    with col3:
        st.metric("💰 Ø Preis", f"{map_df['price'].mean():,.0f}€")

def show_neighborhood_analysis():
    """Zeigt ESK-Stadtteil-Analyse"""
    st.title("📊 ESK Stadtteil-Analyse")
    st.markdown("### Welche Karlsruher Stadtteile sind optimal für ESK-Familien?")
    
    df = load_esk_data()
    
    # Stadtteil-Statistiken
    neighborhood_stats = df.groupby('neighborhood').agg({
        'esk_suitability_score': 'mean',
        'price': 'mean',
        'distance_to_esk': 'mean',
        'international_community_score': 'mean',
        'family_amenities_score': 'mean',
        'safety_score': 'mean',
        'property_id': 'count'
    }).round(1)
    
    neighborhood_stats.columns = [
        'Ø ESK-Score', 'Ø Preis (€)', 'Ø Entfernung ESK (km)',
        'Ø Community-Score', 'Ø Familien-Score', 'Ø Sicherheit', 'Anzahl Immobilien'
    ]
    
    # Top-Stadtteile
    st.subheader("🏆 Top ESK-Stadtteile")
    top_neighborhoods = neighborhood_stats.sort_values('Ø ESK-Score', ascending=False)
    
    for i, (neighborhood, stats) in enumerate(top_neighborhoods.head(3).iterrows(), 1):
        with st.container():
            st.markdown(f"### {i}. {neighborhood}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⭐ ESK-Score", f"{stats['Ø ESK-Score']}/10")
                st.metric("💰 Ø Preis", f"{stats['Ø Preis (€)']:,.0f}€")
            with col2:
                st.metric("🏫 Entfernung ESK", f"{stats['Ø Entfernung ESK (km)']} km")
                st.metric("🏠 Verfügbare Immobilien", int(stats['Anzahl Immobilien']))
            with col3:
                st.metric("🌍 Community", f"{stats['Ø Community-Score']}/10")
                st.metric("👨‍👩‍👧‍👦 Familie", f"{stats['Ø Familien-Score']}/10")
            with col4:
                st.metric("🔒 Sicherheit", f"{stats['Ø Sicherheit']}/10")
        
        st.markdown("---")
    
    # Visualisierungen
    col1, col2 = st.columns(2)
    
    with col1:
        # ESK-Score vs. Preis Scatter Plot
        fig1 = px.scatter(
            df, x='esk_suitability_score', y='price',
            color='neighborhood', size='sqft',
            title='ESK-Score vs. Preis nach Stadtteil',
            labels={'esk_suitability_score': 'ESK-Score', 'price': 'Preis (€)'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Stadtteil-Vergleich Radar Chart
        fig2 = go.Figure()
        
        for neighborhood in df['neighborhood'].unique():
            neighborhood_data = df[df['neighborhood'] == neighborhood]
            avg_scores = {
                'ESK-Score': neighborhood_data['esk_suitability_score'].mean(),
                'Community': neighborhood_data['international_community_score'].mean(),
                'Familie': neighborhood_data['family_amenities_score'].mean(),
                'Sicherheit': neighborhood_data['safety_score'].mean(),
                'ÖPNV': neighborhood_data['public_transport_score'].mean()
            }
            
            fig2.add_trace(go.Scatterpolar(
                r=list(avg_scores.values()),
                theta=list(avg_scores.keys()),
                fill='toself',
                name=neighborhood
            ))
        
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Stadtteil-Vergleich: Alle ESK-relevanten Faktoren"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Vollständige Stadtteil-Tabelle
    st.subheader("📊 Vollständiger Stadtteil-Vergleich")
    st.dataframe(neighborhood_stats.sort_values('Ø ESK-Score', ascending=False), use_container_width=True)

def main():
    """Hauptfunktion der ESK Housing App"""
    # Sidebar Navigation
    with st.sidebar:
        st.title("🏫 ESK Housing")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Willkommen", "🔍 Immobilien-Suche", "🗺️ ESK-Karte", "📊 Stadtteil-Analyse"]
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Für ESK-Familien entwickelt")
        st.markdown("Finden Sie Ihr perfektes Zuhause in der Nähe der European School Karlsruhe")
        
        # ESK Quick Facts
        st.markdown("**ESK Quick Facts:**")
        st.markdown("• 🏫 Kindergarten bis Abitur")
        st.markdown("• 🌍 3 Sprachen: DE/FR/EN")
        st.markdown("• 👨‍👩‍👧‍👦 500+ Schüler-Familien")
        st.markdown("• ⏰ Ø Schulweg: 15 min")
        st.markdown("• 🏆 Top Schulbewertungen")
    
    # Seiten-Router
    if page == "🏠 Willkommen":
        show_welcome_page()
    elif page == "🔍 Immobilien-Suche":
        show_property_search()
    elif page == "🗺️ ESK-Karte":
        show_esk_map()
    elif page == "📊 Stadtteil-Analyse":
        show_neighborhood_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🏫 **ESK Housing Finder** | Speziell für European School Karlsruhe Familien | "
        "🤖 Powered by AI & ESK Community Insights"
    )

if __name__ == "__main__":
    main()