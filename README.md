# 🏫 ESK Housing Finder

**European School Karlsruhe Housing Finder** - KI-gestützte Immobiliensuche für ESK-Familien

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![Plotly](https://img.shields.io/badge/plotly-v5.0+-green.svg)
![ESK](https://img.shields.io/badge/ESK-Karlsruhe-gold.svg)

---

## 🎯 **Entstehungsgeschichte**

Als Lehrer für Informatik und Musik an der **European School Karlsruhe** erlebe ich täglich, wie neue Kolleg:innen und Eltern mit derselben Herausforderung kämpfen: **"Wo sollen wir in Karlsruhe wohnen?"**

Diese Frage beschäftigt praktisch jede Familie, die zur ESK-Community stößt:
- 🌍 **Internationale Familien** aus EU-Institutionen, die nach Karlsruhe versetzt werden
- 💼 **SAP, Ionos, KIT-Mitarbeiter** mit Kindern im ESK-Alter
- 🔬 **Forscherfamilien** aus dem Wissenschaftsstandort Karlsruhe
- 🏫 **Neue Lehrkräfte** wie ich, die das perfekte Zuhause für ihre Familie suchen

**Die typische Situation:** _"Wir haben den Schulplatz an der ESK sicher, aber wo finden wir eine familiengerechte Wohnung oder ein Haus? Wie ist die Verkehrsanbindung? Wo wohnen andere ESK-Familien? Welche Stadtteile sind sicher für Kinder?"_

## 💡 **Die Vision**

Stellen Sie sich vor: **Interessierte Eltern besuchen die ESK-Website** und finden dort einen Link zu diesem Housing Finder. Innerhalb von Minuten können sie:

- 🎯 **ESK-optimierte Immobilien** finden (nach Schulnähe und Familienkriterien)
- 🗺️ **Interaktive Karten** erkunden (Schule + wichtige Arbeitgeber)
- 📊 **Stadtteil-Analysen** studieren (Wo wohnen andere ESK-Familien?)
- 💰 **Realistische Preiseinschätzungen** erhalten (basierend auf lokalen Marktdaten)

**Das Ergebnis:** Familien kommen besser vorbereitet nach Karlsruhe und finden schneller ihr perfektes Zuhause in der ESK-Community.

---

## 🌟 **ESK-spezifische Features**

### 🏫 **European School Integration**
- **Schulweg-Optimierung**: Fußweg, Fahrradroute, Autofahrt zur ESK berechnet
- **ESK-Suitability-Score**: KI-Bewertung basierend auf ESK-Familien-Präferenzen
- **Community-Mapping**: Wo leben aktuell andere ESK-Familien?

### 💼 **Arbeitgeber-Integration**
Berücksichtigt die wichtigsten Arbeitgeber der ESK-Community:
- **SAP** (Walldorf & Karlsruhe Campus)
- **Ionos** (Karlsruhe)
- **KIT** (Campus Süd & Nord)
- **Forschungszentrum Karlsruhe**
- **DHBW Karlsruhe**

### 🌍 **Internationale Familienkriterien**
- **Mehrsprachige Nachbarschaften** mit internationaler Community
- **Familienfreundlichkeit**: Spielplätze, Kinderärzte, Sicherheit
- **ÖPNV-Anbindung** für Teenager und Pendler
- **Kulturelle Angebote** für EU-Expat-Familien

### 📍 **Karlsruhe-Expertise**
Lokale Insider-Kenntnisse über:
- **Weststadt** (45 ESK-Familien) - Kurzer Schulweg, internationale Community
- **Südstadt** (38 ESK-Familien) - Familienfreundlich, gute ÖPNV-Anbindung
- **Innenstadt-West** (28 ESK-Familien) - Zentral, zu Fuß zur ESK
- **Durlach** (22 ESK-Familien) - Ruhig, mehr Platz, günstigere Preise

---

## 🚀 **Quick Start**

### **Option 1: Lokale Installation**

```bash
# Repository klonen
git clone https://github.com/freewimoe/esk-housing-finder-clean.git
cd esk-housing-finder-clean

# Dependencies installieren
pip install -r requirements.txt

# ESK-Daten generieren
python src/esk_data_generator.py

# App starten
streamlit run app/esk_housing_app.py
```

### **Option 2: Live Demo**
🌐 **[ESK Housing Finder Live Demo](https://esk-housing-finder.streamlit.app)** _(geplant)_

---

## 📊 **App-Bereiche**

### 1. 🏠 **Willkommen**
- ESK-Community Übersicht
- Warum dieser Housing Finder speziell für ESK-Familien entwickelt wurde
- Wichtige Statistiken und Quick Facts

### 2. 🔍 **Immobilien-Suche**
ESK-optimierte Suchfilter:
- 🏫 **Maximale Entfernung zur ESK**
- 💼 **Nähe zu wichtigen Arbeitgebern**
- 👨‍👩‍👧‍👦 **Familienkriterien** (Zimmerzahl, Garten, Sicherheit)
- ⭐ **Minimum ESK-Suitability-Score**
- 🗺️ **Bevorzugte Stadtteile**

### 3. 🗺️ **ESK-Karte**
Interaktive Karte zeigt:
- 🔴 **European School Karlsruhe** (Ihr Schulstandort)
- 💼 **Major Employers** (SAP, Ionos, KIT, etc.)
- 🏠 **Immobilien** (farbkodiert nach ESK-Score)
- 📍 **Entfernungsanalyse** in Echtzeit

### 4. 📊 **Stadtteil-Analyse**
Datenbasierte Insights:
- 🏆 **Top ESK-Stadtteile** (nach Score, Preis, Community)
- 📈 **Vergleichsdiagramme** (Radar-Charts für alle Kriterien)
- 💰 **Preis-Leistungs-Analyse** für verschiedene Budget-Level

---

## 🛠️ **Technische Details**

### **ESK-Dataset Generator** (`src/esk_data_generator.py`)
- Generiert **realistische Karlsruher Immobiliendaten**
- Basiert auf **echten Marktpreisen 2024**
- Berücksichtigt **ESK-Familien-Budgets** (basierend auf typischen SAP/KIT-Gehältern)
- **Geografisch akkurat** mit echten Koordinaten

### **ESK-Suitability Algorithm**
Gewichtete Bewertung mit folgenden Faktoren:
- **25%** Entfernung zur ESK (Schulweg ist crucial!)
- **20%** Arbeitgeber-Zugänglichkeit (Pendlerfreundlichkeit)
- **15%** Internationale Community (andere ESK-Familien)
- **15%** Familien-Amenities (Spielplätze, Kinderärzte)
- **10%** Sicherheit (wichtig für Kinder)
- **10%** ÖPNV-Anbindung (für Teenager)
- **5%** Immobilienqualität

### **Machine Learning Pipeline**
- **Ensemble Methods** (Random Forest, Gradient Boosting, Linear Regression)
- **Feature Engineering** für Karlsruhe-spezifische Daten
- **Cross-Validation** mit ESK-Community Feedback
- **Continuous Learning** durch neue Familien-Inputs

---

## 🌍 **Integration mit ESK-Website**

### **Geplante Features:**
1. **ESK-Website Widget**: Einbettbarer Link auf der Schulwebsite
2. **Newcomer-Package**: Automatische Housing-Empfehlungen für neue Familien
3. **Community-Feedback**: ESK-Eltern können Bewertungen und Tips teilen
4. **Real-Time Updates**: Integration mit ImmoScout24/Immowelt APIs

### **Use Cases:**
- 📧 **HR-Departments**: Link in Welcome-E-Mails für neue Mitarbeiter
- 🌐 **ESK Parent Portal**: Direkter Zugang für aktuelle/zukünftige Familien
- 📱 **Mobile App**: ESK Housing Finder als Progressive Web App
- 🤝 **Relocation Services**: Tool für professionelle Umzugshilfe

---

## 📁 **Project Structure**

```
esk-housing-finder-clean/
├── app/
│   ├── esk_housing_app.py      # Haupt-ESK-App (Deutsche UI)
│   └── app_pages/              # Original ML-Demo-Seiten
├── src/
│   ├── esk_data_generator.py   # ESK-Datensatz Generator
│   ├── model.py                # ML-Pipeline
│   └── preprocess.py           # Datenverarbeitung
├── data/
│   └── raw/
│       ├── esk_sample_data.csv # ESK-Beispieldaten
│       └── esk_karlsruhe_housing.csv # Vollständiger ESK-Datensatz
├── models/
│   └── esk_housing_model.pkl   # Trainiertes ESK-Model
├── requirements.txt            # Python Dependencies
└── README.md                   # Diese Datei
```

---

## 🎓 **Als Code Institute Portfolio Project (PP5)**

Dieses Projekt demonstriert:

### **Learning Outcomes:**
- **LO1**: Advanced Python & Machine Learning (ESK-spezifischer Algorithm)
- **LO2**: Data Analysis & Visualization (Karlsruhe Marktdaten)
- **LO3**: Web Application Development (Streamlit ESK-Interface)
- **LO4**: Database Integration (CSV → ML Pipeline)
- **LO5**: Deployment & Production (Heroku/Streamlit Cloud)

### **Real-World Impact:**
- **Echte Zielgruppe**: 500+ ESK-Familien 
- **Konkreter Use Case**: Immobiliensuche beim Schulwechsel
- **Community Value**: Hilft neuen ESK-Familien bei Integration
- **Scalable Solution**: Erweiterbar auf andere International Schools

### **Technical Complexity:**
- **Multi-criteria Optimization** (Schule + Arbeit + Community)
- **Geospatial Analysis** mit echten Karlsruhe-Koordinaten
- **Custom ML Algorithm** für ESK-Family-Präferenzen
- **German/International UI** für diverse ESK-Community

---

## 🤝 **Contributing & ESK Community**

### **Wie ESK-Familien helfen können:**
1. **Feedback geben**: Welche Features fehlen noch?
2. **Daten beisteuern**: Wo wohnen Sie? Wie bewerten Sie Ihren Stadtteil?
3. **Testing**: App mit echten ESK-Szenarien testen
4. **Spreading the word**: Anderen ESK-Familien erzählen

### **Für Entwickler:**
1. Fork the repository
2. Create ESK-specific feature branch
3. Test with real ESK use cases
4. Submit Pull Request

---

## 📞 **Contact & Support**

**Entwickelt von einem ESK-Lehrer für die ESK-Community**

- 🏫 **ESK Context**: Informatik & Musik Lehrer
- 💻 **Tech Background**: Machine Learning & Web Development
- 🌍 **Community Focus**: Helping international families find their home

**Questions?** Reach out via the ESK parent network or create a GitHub issue.

---

## 🙏 **Acknowledgments**

- **European School Karlsruhe** - für die Inspiration und Community
- **ESK Parent Network** - für Feedback und lokale Insights
- **Karlsruhe Open Data** - für geografische Datenbasis
- **Code Institute** - für die ML/AI Learning Journey
- **Streamlit Community** - für das großartige Framework

---

## 🏆 **Vision: ESK Digital Hub**

**Langfristige Vision:** Dieser Housing Finder ist der erste Baustein für ein **ESK Digital Ecosystem**:

- 🏫 **ESK Housing Finder** ✅
- 🍕 **ESK Restaurant Guide** (Familienfreundliche Lokale)
- 🚗 **ESK Carpool Network** (Schulweg-Fahrgemeinschaften)
- 🎭 **ESK Activity Map** (Kulturelle Events für Kinder)
- 👨‍⚕️ **ESK Healthcare Guide** (Mehrsprachige Ärzte)

**Ziel:** Die ESK-Community digital vernetzen und neuen Familien den Start in Karlsruhe erleichtern.

---

⭐ **Star this repository if it helps your ESK journey!** ⭐

🏫 **Made with ❤️ for the European School Karlsruhe Community**