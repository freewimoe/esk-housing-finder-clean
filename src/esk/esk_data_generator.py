import pandas as pd
import numpy as np
import math

class ESKHousingAnalyzer:
    def __init__(self):
        self.esk_location = {'lat': 49.0134, 'lon': 8.4044}
        
        self.family_profiles = {
            'sap_family': {'budget': 600000, 'areas': ['Durlach', 'Bruchsal', 'Südstadt'], 'type': 'house'},
            'kit_family': {'budget': 450000, 'areas': ['Weststadt', 'Südstadt', 'Innenstadt-West'], 'type': 'apartment'},
            'ionos_family': {'budget': 550000, 'areas': ['Innenstadt-West', 'Weststadt', 'Mühlburg'], 'type': 'apartment'},
            'research_family': {'budget': 400000, 'areas': ['Nordweststadt', 'Neureut', 'Weststadt'], 'type': 'house'},
            'eu_commuter': {'budget': 800000, 'areas': ['Innenstadt-West', 'Südstadt', 'Durlach'], 'type': 'house'}
        }
        
        self.neighborhoods = {
            'Weststadt': {'families': 52, 'safety': 9.2, 'community': 8.5, 'price_sqm': 3800},
            'Südstadt': {'families': 45, 'safety': 8.8, 'community': 9.1, 'price_sqm': 3600},
            'Innenstadt-West': {'families': 35, 'safety': 9.0, 'community': 8.8, 'price_sqm': 4200},
            'Durlach': {'families': 28, 'safety': 9.3, 'community': 7.2, 'price_sqm': 3200},
            'Oststadt': {'families': 18, 'safety': 8.5, 'community': 7.8, 'price_sqm': 3400},
            'Nordweststadt': {'families': 15, 'safety': 8.8, 'community': 6.5, 'price_sqm': 2900}
        }
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
    
    def generate_esk_dataset(self, n_samples=800):
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            profiles = ['sap_family', 'kit_family', 'ionos_family', 'research_family', 'eu_commuter']
            weights = [0.35, 0.25, 0.15, 0.15, 0.10]
            family_profile = np.random.choice(profiles, p=weights)
            profile_info = self.family_profiles[family_profile]
            
            # Neighborhood
            preferred_areas = profile_info['areas']
            all_areas = list(self.neighborhoods.keys())
            
            area_weights = []
            for area in all_areas:
                base_weight = self.neighborhoods[area]['families']
                if area in preferred_areas:
                    area_weights.append(base_weight * 2)
                else:
                    area_weights.append(base_weight)
            
            area_weights = np.array(area_weights) / sum(area_weights)
            neighborhood = np.random.choice(all_areas, p=area_weights)
            area_info = self.neighborhoods[neighborhood]
            
            # Property type
            preferred_type = profile_info['type']
            if preferred_type == 'house':
                property_type = np.random.choice(['house', 'apartment'], p=[0.8, 0.2])
            else:
                property_type = np.random.choice(['house', 'apartment'], p=[0.3, 0.7])
            
            # Property details
            if property_type == 'house':
                bedrooms = np.random.choice([3, 4, 5], p=[0.4, 0.5, 0.1])
                bathrooms = np.random.choice([2, 3, 4], p=[0.4, 0.5, 0.1])
                sqft = np.random.normal(140, 30)
                garage = np.random.choice([1, 2], p=[0.4, 0.6])
                garden = 1
            else:
                bedrooms = np.random.choice([2, 3, 4], p=[0.3, 0.6, 0.1])
                bathrooms = np.random.choice([1, 2, 3], p=[0.2, 0.7, 0.1])
                sqft = np.random.normal(95, 20)
                garage = np.random.choice([0, 1], p=[0.6, 0.4])
                garden = np.random.choice([0, 1], p=[0.7, 0.3])
            
            sqft = max(60, min(sqft, 280))
            year_built = np.random.randint(1980, 2024)
            
            # Coordinates
            base_coords = {
                'Weststadt': (49.0069, 8.3737), 'Südstadt': (49.0000, 8.3937),
                'Innenstadt-West': (49.0069, 8.3937), 'Durlach': (48.9969, 8.4737),
                'Oststadt': (49.0100, 8.4200), 'Nordweststadt': (49.0300, 8.3500)
            }
            
            base_lat, base_lon = base_coords.get(neighborhood, (49.0069, 8.4037))
            lat = base_lat + np.random.uniform(-0.012, 0.012)
            lon = base_lon + np.random.uniform(-0.012, 0.012)
            
            distance_to_esk = self.calculate_distance(lat, lon, self.esk_location['lat'], self.esk_location['lon'])
            
            # Price calculation
            base_price_per_sqm = area_info['price_sqm']
            
            if family_profile == 'sap_family':
                base_price_per_sqm *= 1.1
            elif family_profile == 'eu_commuter':
                base_price_per_sqm *= 1.15
            
            if distance_to_esk < 1.5:
                base_price_per_sqm *= 1.2
            elif distance_to_esk < 3:
                base_price_per_sqm *= 1.1
            
            price = base_price_per_sqm * sqft * np.random.uniform(0.9, 1.1)
            budget = profile_info['budget']
            price = np.clip(price, budget * 0.6, budget * 1.3)
            
            # ESK score
            distance_score = max(0, 10 - distance_to_esk * 1.5)
            safety_score = area_info['safety']
            community_score = area_info['community']
            budget_score = 9.0 if price <= budget * 0.8 else 7.0 if price <= budget else 5.0
            type_score = 9 if property_type == preferred_type else 6
            area_match = 1.2 if neighborhood in preferred_areas else 1.0
            
            esk_score = (distance_score * 0.30 + safety_score * 0.20 + community_score * 0.15 + budget_score * 0.15 + type_score * 0.20) * area_match
            esk_score = round(min(10.0, esk_score), 1)
            
            data.append({
                'property_id': f'ESK_{i+1:04d}',
                'price': round(price),
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft': round(sqft),
                'year_built': year_built,
                'property_type': property_type,
                'neighborhood': neighborhood,
                'latitude': lat,
                'longitude': lon,
                'garage': garage,
                'garden': garden,
                'family_profile': family_profile,
                'distance_to_esk': round(distance_to_esk, 2),
                'safety_score': round(safety_score + np.random.uniform(-0.3, 0.3), 1),
                'international_community': round(community_score + np.random.uniform(-0.5, 0.5), 1),
                'esk_suitability_score': esk_score
            })
        
        df = pd.DataFrame(data)
        df['price'] = df['price'].clip(lower=180000, upper=2000000)
        return df

if __name__ == "__main__":
    analyzer = ESKHousingAnalyzer()
    df = analyzer.generate_esk_dataset()
    df.to_csv('data/esk/karlsruhe_esk_properties.csv', index=False)
    print(f' Generated {len(df)} ESK-optimized properties')
    print(f' Employer distribution:')
    print(df['family_profile'].value_counts())
