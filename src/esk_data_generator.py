"""
European School Karlsruhe Housing Data Generator
Speziell für ESK-Familien entwickelt

Erstellt realistische Immobiliendaten basierend auf:
- ESK-Schulnähe
- Arbeitgeber-Nähe (SAP, Ionos, KIT, Forschungseinrichtungen)
- Internationale Community-Dichte
- Familien-optimierte Kriterien
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime
import os

class ESKHousingDataGenerator:
    """Generiert ESK-spezifische Immobiliendaten für Karlsruhe"""
    
    def __init__(self):
        # ESK-Koordinaten
        self.esk_location = {"lat": 49.0134, "lon": 8.4044}
        
        # Wichtige Arbeitgeber für ESK-Familien
        self.major_employers = {
            'sap_walldorf': {"lat": 49.2933, "lon": 8.6428, "name": "SAP Walldorf"},
            'sap_karlsruhe': {"lat": 49.0233, "lon": 8.4103, "name": "SAP Karlsruhe"},
            'ionos': {"lat": 49.0089, "lon": 8.3858, "name": "Ionos Karlsruhe"},
            'kit_campus_sud': {"lat": 49.0069, "lon": 8.4037, "name": "KIT Campus Süd"},
            'kit_campus_nord': {"lat": 49.0943, "lon": 8.4347, "name": "KIT Campus Nord"},
            'forschungszentrum': {"lat": 49.0930, "lon": 8.4279, "name": "Forschungszentrum Karlsruhe"},
            'dhbw': {"lat": 49.0145, "lon": 8.3856, "name": "DHBW Karlsruhe"}
        }
        
        # ESK-Familie bevorzugte Stadtteile (basierend auf echten Daten)
        self.esk_preferred_areas = {
            'Weststadt': {
                'current_esk_families': 45,
                'avg_price_per_sqm': 4200,
                'commute_time_esk': 12,
                'safety_rating': 9.2,
                'int_community_score': 8.5,
                'family_friendliness': 9.0,
                'public_transport': 8.8
            },
            'Südstadt': {
                'current_esk_families': 38,
                'avg_price_per_sqm': 4000,
                'commute_time_esk': 15,
                'safety_rating': 8.8,
                'int_community_score': 8.2,
                'family_friendliness': 8.5,
                'public_transport': 9.0
            },
            'Innenstadt-West': {
                'current_esk_families': 28,
                'avg_price_per_sqm': 4800,
                'commute_time_esk': 8,
                'safety_rating': 8.5,
                'int_community_score': 7.5,
                'family_friendliness': 7.8,
                'public_transport': 9.5
            },
            'Durlach': {
                'current_esk_families': 22,
                'avg_price_per_sqm': 3600,
                'commute_time_esk': 25,
                'safety_rating': 9.0,
                'int_community_score': 7.0,
                'family_friendliness': 9.2,
                'public_transport': 7.5
            },
            'Oststadt': {
                'current_esk_families': 15,
                'avg_price_per_sqm': 3800,
                'commute_time_esk': 18,
                'safety_rating': 8.3,
                'int_community_score': 7.2,
                'family_friendliness': 8.0,
                'public_transport': 8.0
            },
            'Mühlburg': {
                'current_esk_families': 12,
                'avg_price_per_sqm': 3400,
                'commute_time_esk': 20,
                'safety_rating': 8.0,
                'int_community_score': 6.8,
                'family_friendliness': 8.2,
                'public_transport': 7.8
            }
        }
        
        # ESK-Familie typische Budgets (basierend auf Gehältern SAP, KIT, etc.)
        self.esk_budget_ranges = {
            'entry_level': (300000, 450000),      # Junge Forscher, Doktoranden
            'mid_level': (450000, 650000),        # Erfahrene Entwickler, Wissenschaftler
            'senior_level': (650000, 900000),     # Senior-Positionen, Teamleiter
            'executive': (900000, 1500000)        # Führungskräfte, Professoren
        }

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Berechnet Distanz zwischen zwei Koordinaten in km"""
        R = 6371  # Erdradius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def calculate_esk_suitability_score(self, property_data):
        """Berechnet ESK-Familientauglichkeits-Score (1-10)"""
        # Gewichtung verschiedener Faktoren für ESK-Familien
        weights = {
            'distance_to_esk': 0.25,        # Sehr wichtig für täglichen Schulweg
            'employer_accessibility': 0.20,  # Wichtig für Arbeitsweg
            'international_community': 0.15, # ESK-Familien schätzen internationale Umgebung
            'family_amenities': 0.15,       # Spielplätze, Kinderarzt, etc.
            'safety': 0.10,                 # Sicherheit für Kinder
            'public_transport': 0.10,       # ÖPNV für Teenager
            'property_quality': 0.05        # Immobilienqualität
        }
        
        # ESK-Distanz Score (näher = besser)
        distance_to_esk = property_data.get('distance_to_esk', 10)
        if distance_to_esk <= 1:
            distance_score = 10
        elif distance_to_esk <= 3:
            distance_score = 9
        elif distance_to_esk <= 5:
            distance_score = 7
        elif distance_to_esk <= 10:
            distance_score = 5
        else:
            distance_score = 2
            
        # Arbeitgeber-Zugänglichkeit
        avg_employer_distance = property_data.get('avg_employer_distance', 20)
        employer_score = max(1, 10 - (avg_employer_distance / 3))
        
        # Weitere Scores aus Property-Daten
        intl_community_score = property_data.get('international_community_score', 5)
        family_amenities_score = property_data.get('family_amenities_score', 5)
        safety_score = property_data.get('safety_score', 5)
        transport_score = property_data.get('public_transport_score', 5)
        property_score = 8 if property_data.get('property_type') == 'house' else 6
        
        # Gewichtete Gesamtbewertung
        total_score = (
            distance_score * weights['distance_to_esk'] +
            employer_score * weights['employer_accessibility'] +
            intl_community_score * weights['international_community'] +
            family_amenities_score * weights['family_amenities'] +
            safety_score * weights['safety'] +
            transport_score * weights['public_transport'] +
            property_score * weights['property_quality']
        )
        
        return round(total_score, 1)

    def generate_property_coordinates(self, neighborhood):
        """Generiert realistische Koordinaten für Stadtteil"""
        # Karlsruhe Stadtteil-Zentren (approximiert)
        stadtteil_coords = {
            'Weststadt': {"lat": 49.0040, "lon": 8.3850},
            'Südstadt': {"lat": 48.9950, "lon": 8.4030},
            'Innenstadt-West': {"lat": 49.0090, "lon": 8.3980},
            'Durlach': {"lat": 48.9944, "lon": 8.4722},
            'Oststadt': {"lat": 49.0080, "lon": 8.4200},
            'Mühlburg': {"lat": 49.0150, "lon": 8.3700}
        }
        
        center = stadtteil_coords.get(neighborhood, {"lat": 49.0069, "lon": 8.4037})
        
        # Kleine zufällige Abweichung innerhalb des Stadtteils
        lat = center["lat"] + np.random.uniform(-0.01, 0.01)
        lon = center["lon"] + np.random.uniform(-0.015, 0.015)
        
        return lat, lon

    def calculate_employer_distances(self, lat, lon):
        """Berechnet Distanzen zu allen wichtigen Arbeitgebern"""
        distances = {}
        for employer_key, employer_data in self.major_employers.items():
            distance = self.calculate_distance(lat, lon, employer_data["lat"], employer_data["lon"])
            distances[f'distance_to_{employer_key}'] = round(distance, 2)
        
        # Durchschnittliche Distanz zu Top-3 Arbeitgebern
        top_distances = sorted(distances.values())[:3]
        avg_distance = sum(top_distances) / len(top_distances)
        
        return distances, round(avg_distance, 2)

    def generate_esk_dataset(self, n_samples=500):
        """Generiert ESK-optimierten Datensatz"""
        np.random.seed(42)  # Für reproduzierbare Ergebnisse
        
        print(f"🏫 Generiere {n_samples} ESK-optimierte Immobilien...")
        
        properties = []
        
        for i in range(n_samples):
            # Stadtteil auswählen (gewichtet nach ESK-Familienanzahl)
            neighborhoods = list(self.esk_preferred_areas.keys())
            weights = [area['current_esk_families'] for area in self.esk_preferred_areas.values()]
            weights = np.array(weights) / sum(weights)  # Normalisieren
            
            neighborhood = np.random.choice(neighborhoods, p=weights)
            area_info = self.esk_preferred_areas[neighborhood]
            
            # Koordinaten generieren
            lat, lon = self.generate_property_coordinates(neighborhood)
            
            # Distanz zur ESK berechnen
            distance_to_esk = self.calculate_distance(lat, lon, 
                                                    self.esk_location["lat"], 
                                                    self.esk_location["lon"])
            
            # Arbeitgeber-Distanzen
            employer_distances, avg_employer_distance = self.calculate_employer_distances(lat, lon)
            
            # Immobilientyp (ESK-Familien bevorzugen Häuser)
            if distance_to_esk <= 3:  # Näher zur Schule = mehr Häuser verfügbar
                property_type = np.random.choice(['house', 'apartment'], p=[0.7, 0.3])
            else:
                property_type = np.random.choice(['house', 'apartment'], p=[0.5, 0.5])
            
            # Budget-Level basierend auf Arbeitgeber-Nähe (SAP-Nähe = höheres Budget)
            sap_distance = min(employer_distances.get('distance_to_sap_walldorf', 50),
                             employer_distances.get('distance_to_sap_karlsruhe', 50))
            
            if sap_distance <= 15:
                budget_level = np.random.choice(['mid_level', 'senior_level', 'executive'], p=[0.3, 0.5, 0.2])
            elif avg_employer_distance <= 10:
                budget_level = np.random.choice(['entry_level', 'mid_level', 'senior_level'], p=[0.2, 0.5, 0.3])
            else:
                budget_level = np.random.choice(['entry_level', 'mid_level'], p=[0.6, 0.4])
            
            min_budget, max_budget = self.esk_budget_ranges[budget_level]
            
            # Zimmer (ESK-Familien haben typisch 1-3 Kinder)
            if property_type == 'house':
                bedrooms = np.random.choice([3, 4, 5], p=[0.4, 0.5, 0.1])
                sqft = np.random.uniform(120, 250)
            else:
                bedrooms = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
                sqft = np.random.uniform(80, 150)
            
            # Baujahr (ESK-Familien bevorzugen moderne Immobilien)
            if budget_level in ['senior_level', 'executive']:
                year_built = np.random.choice(range(1990, 2025), p=np.linspace(0.01, 0.1, 35))
            else:
                year_built = np.random.choice(range(1970, 2025))
            
            # Zusätzliche Features
            garage = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
            garden = 1 if property_type == 'house' and np.random.random() > 0.2 else 0
            balcony = np.random.choice([0, 1], p=[0.3, 0.7])
            
            # Preisberechnung
            base_price_per_sqm = area_info['avg_price_per_sqm']
            
            # ESK-Nähe Premium
            if distance_to_esk <= 1.5:
                base_price_per_sqm *= 1.15
            elif distance_to_esk <= 3:
                base_price_per_sqm *= 1.08
                
            # Budget-Adjustment
            if budget_level == 'executive':
                base_price_per_sqm *= 1.2
            elif budget_level == 'entry_level':
                base_price_per_sqm *= 0.9
                
            price = base_price_per_sqm * sqft * np.random.uniform(0.9, 1.1)
            price = np.clip(price, min_budget, max_budget)
            
            # ESK-spezifische Scores
            international_community_score = area_info['int_community_score'] + np.random.uniform(-0.5, 0.5)
            international_community_score = np.clip(international_community_score, 1, 10)
            
            family_amenities_score = area_info['family_friendliness'] + np.random.uniform(-0.5, 0.5)
            family_amenities_score = np.clip(family_amenities_score, 1, 10)
            
            safety_score = area_info['safety_rating'] + np.random.uniform(-0.3, 0.3)
            safety_score = np.clip(safety_score, 1, 10)
            
            public_transport_score = area_info['public_transport'] + np.random.uniform(-0.5, 0.5)
            public_transport_score = np.clip(public_transport_score, 1, 10)
            
            # Property-Dictionary erstellen
            property_data = {
                'property_id': f'ESK_{i+1:04d}',
                'price': round(price),
                'bedrooms': bedrooms,
                'sqft': round(sqft),
                'year_built': year_built,
                'garage': garage,
                'garden': garden,
                'balcony': balcony,
                'property_type': property_type,
                'neighborhood': neighborhood,
                'lat': round(lat, 6),
                'lon': round(lon, 6),
                'distance_to_esk': round(distance_to_esk, 2),
                'avg_employer_distance': avg_employer_distance,
                'international_community_score': round(international_community_score, 1),
                'family_amenities_score': round(family_amenities_score, 1),
                'safety_score': round(safety_score, 1),
                'public_transport_score': round(public_transport_score, 1),
                'budget_level': budget_level,
                'commute_time_esk': round(area_info['commute_time_esk'] + np.random.uniform(-2, 2))
            }
            
            # Alle Arbeitgeber-Distanzen hinzufügen
            property_data.update(employer_distances)
            
            # ESK-Tauglichkeits-Score berechnen
            esk_score = self.calculate_esk_suitability_score(property_data)
            property_data['esk_suitability_score'] = esk_score
            
            properties.append(property_data)
        
        df = pd.DataFrame(properties)
        
        print(f"✅ {len(df)} ESK-optimierte Immobilien generiert!")
        print(f"📊 Durchschnittlicher ESK-Score: {df['esk_suitability_score'].mean():.1f}/10")
        print(f"🏠 {(df['property_type'] == 'house').sum()} Häuser, {(df['property_type'] == 'apartment').sum()} Wohnungen")
        print(f"🎯 {len(df[df['distance_to_esk'] <= 2])} Immobilien in 2km ESK-Radius")
        
        return df

    def save_dataset(self, df, filename='esk_karlsruhe_housing.csv'):
        """Speichert den ESK-Datensatz"""
        # Erstelle data/raw Verzeichnis falls nicht vorhanden
        os.makedirs('data/raw', exist_ok=True)
        
        filepath = f'data/raw/{filename}'
        df.to_csv(filepath, index=False)
        
        print(f"💾 ESK Housing Dataset gespeichert: {filepath}")
        print(f"📈 Stadtteile-Verteilung:")
        for neighborhood, count in df['neighborhood'].value_counts().items():
            avg_price = df[df['neighborhood'] == neighborhood]['price'].mean()
            print(f"   {neighborhood}: {count} Immobilien (⌀ {avg_price:,.0f}€)")

if __name__ == "__main__":
    generator = ESKHousingDataGenerator()
    
    print("🏫 European School Karlsruhe Housing Data Generator")
    print("=" * 60)
    
    # Dataset generieren
    df = generator.generate_esk_dataset(n_samples=500)
    
    # Speichern
    generator.save_dataset(df)
    
    print("\n🎯 ESK-spezifische Analyse:")
    print(f"Top ESK-Scores: {df.nlargest(5, 'esk_suitability_score')[['neighborhood', 'esk_suitability_score', 'distance_to_esk']].to_string(index=False)}")