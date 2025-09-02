# WHAT Category Structure (TREE.md)

## 🔒 AUTHORITY NOTICE
**This TREE.md file may ONLY be modified by the AUDITOR.**
- Other agents may ONLY update the contents of existing .txt files
- Other agents may NOT create new directories or files
- Other agents may NOT rename directories or files
- Other agents may NOT modify this TREE.md file

---

## 📁 WHAT - Data and Subject Matter
**Definition**: The data, information, or subject matter that AI processes (the NOUNS)
**Key Question**: "What kind of DATA does the AI work with?"
**Examples**: medische gegevens, krediet geschiedenis, biometrische data, film voorkeuren
**NOT**: AI capabilities (→FUNCTION), usage locations (→CONTEXT), affected people (→TARGET)

## 📋 GENERAL RULES FOR ALL FILES

1. **Language**: ALL vocabulary entries must be in Dutch (unless English term is commonly used in Dutch)
2. **Format**: Each term on its own line, no translations
3. **Sorting**: Sort by commonality (most common terms first)
4. **Size**: Each file must contain 20-200 lines total
5. **Quality**: Only include terms with HIGH confidence they belong here
6. **No Mixing**: NEVER include terms from other dimensions

## EXAMPLE FILE HEADER:
```
# Medical Data - HIGH RISK WHAT
# Medische gegevens
# 
# PATH: what/hr/persoonlijke_data/medische_gegevens.txt
# DIMENSION: WHAT (Data processed by AI)
# RISK: HIGH (Sensitive health information)
#
# HEADER RULES - DO NOT MODIFY - ONLY AUDITOR MAY CHANGE THIS SECTION
#
# WHAT dimension contains: Data types, information, and content that AI processes
# THIS FILE contains: Medical records, health data, and clinical information
#
# VALIDATION RULES:
# - Must be data or information (not an action)
# - Must be medical/health related content
# - Must be what AI processes (not who or where)
# - Must be sensitive enough for high-risk classification
#
# FORMAT: 
# - Dutch terms only (unless English term is commonly used in Dutch)
# - Each term on its own line
# - Sort by commonality (most common first)
#
# END HEADER RULES
```

---

## 📂 Directory Structure

### 📂 hr/ (hoog_risico / high_risk)
*Sensitive data that can impact fundamental rights or safety*

#### 📂 persoonlijke_data/ (personal_data)
- `fysiologische_data.txt` - Physical characteristic measurements
- `gezichts_biometrische_data.txt` - Facial feature data and measurements
- `medische_gegevens.txt` - Medical records and health data
- `genetische_informatie.txt` - Genetic markers and DNA data
- `gedrag_patronen.txt` - Personal behavior patterns
- `emotionele_data.txt` - Emotional state data
- `stem_patronen.txt` - Voice characteristics and patterns

#### 📂 beslissings_data/ (decision_data)
- `beslissings_gegevens.txt` - General decision-related data
- `financiele_beslissingen.txt` - Financial decision data
- `personeels_beslissingen.txt` - HR decision data
- `hypotheek_krediet.txt` - Mortgage and credit data
- `aanwervingsproces.txt` - Hiring process data
- `prestatie_evaluatie.txt` - Performance evaluation data
- `juridische_gegevens.txt` - Legal status and document data
- `toelating_criteria.txt` - Admission criteria and thresholds
- `examen_resultaten.txt` - Exam performance data
- `student_beoordelingen.txt` - Student assessment data

#### 📂 surveillance_data/ (surveillance_data)
- `dna_profilering.txt` - DNA profile databases
- `gedrag_analyse.txt` - Behavior pattern data
- `iris_scannen.txt` - Iris scan data
- `loop_analyse.txt` - Gait analysis data
- `stem_herkenning.txt` - Voice pattern data
- `vingerafdruk_analyse.txt` - Fingerprint analysis data
- `bewijs_analyse.txt` - Evidence analysis data
- `misdaad_voorspelling.txt` - Crime prediction data
- `surveillance_systemen.txt` - Surveillance system data
- `verkiezings_gegevens.txt` - Voting records and electoral data
- `locatie_tracking.txt` - Location and movement data
- `communicatie_metadata.txt` - Communication metadata
- `netwerk_verkeer.txt` - Network traffic data

#### 📂 infrastructuur_data/ (infrastructure_data)
- `energie_systemen.txt` - Energy system operational data
- `stroomnet_controle.txt` - Power grid control data
- `transport_systemen.txt` - Transportation network data
- `water_systemen.txt` - Water system sensor data
- `telecom_verkeer.txt` - Telecommunications data
- `sensor_metingen.txt` - IoT sensor measurements
- `milieu_data.txt` - Environmental monitoring data

### 📂 lr/ (laag_risico / low_risk)
*General-purpose data for commercial or personal use*

#### 📂 commerciële_data/ (commercial_data)
- `financiële_rapportage.txt` - Financial reporting data
- `klant_beheer.txt` - Customer management data
- `klanten_service.txt` - Customer service data
- `kwaliteit_beheer.txt` - Quality management data
- `leverancier_beheer.txt` - Vendor management data
- `logistiek_data.txt` - Logistics data
- `marketing_analyse.txt` - Marketing analytics data
- `operationele_data.txt` - Operational data
- `retail_gegevens.txt` - Retail transaction data
- `supply_chain.txt` - Supply chain data
- `verkoop_data.txt` - Sales data
- `voorraad_beheer.txt` - Inventory management data
- `business_intelligence.txt` - Business intelligence data
- `compliance_data.txt` - Compliance information
- `data_beheer.txt` - Data management information
- `hr_analyse.txt` - HR analytics data
- `project_management.txt` - Project management data
- `technische_formaten.txt` - Technical format data
- `werkplek_data.txt` - Workplace data

#### 📂 content_data/ (content_data)
- `aanbevelingen.txt` - Recommendation data
- `digitale_content.txt` - Digital content assets
- `kennis_beheer.txt` - Knowledge management data
- `vertalingen.txt` - Translation data
- `artistieke_data.txt` - Artistic content data
- `leermateriaal.txt` - Learning materials and content
- `gaming_content.txt` - Gaming content data
- `media_content.txt` - Media content data
- `afspeellijst_sociale_data.txt` - Playlist and social data
- `ar_entertainment.txt` - AR entertainment data
- `entertainment_filters.txt` - Entertainment filter data
- `gaming_data.txt` - Gaming data
- `media_bestanden.txt` - Media file data
- `tekst_documenten.txt` - Text documents
- `beeld_bibliotheek.txt` - Image libraries
- `audio_collecties.txt` - Audio collections

#### 📂 analyse_data/ (analytics_data)
- `kern_analyse.txt` - Core analytics data
- `technologie_metrieken.txt` - Technology metrics
- `voorspellende_analyse.txt` - Predictive analytics data
- `website_analyse.txt` - Website analytics data
- `gebruiker_gedrag.txt` - User behavior data
- `prestatie_metrieken.txt` - Performance metrics
- `trend_analyse.txt` - Trend analysis data
- `markt_onderzoek.txt` - Market research data

#### 📂 persoonlijke_data/ (personal_data)
- `berichten_data.txt` - Message and chat data
- `sociale_media_data.txt` - Social media content data
- `persoonlijke_productiviteit.txt` - Personal productivity data
- `hobby_data.txt` - Hobby and lifestyle data
- `agenda_gegevens.txt` - Calendar and scheduling data
- `contact_lijsten.txt` - Contact information
- `voorkeur_instellingen.txt` - Preference settings
- `persoonlijke_documenten.txt` - Personal documents

---

## ⚠️ COMMON ERRORS TO AVOID

1. **Adding functions instead of data**:
   ❌ "gezichtsherkenning systeem" - this is FUNCTION
   ✅ "gezichts kenmerken" - this is the data

2. **Adding locations/contexts**:
   ❌ "ziekenhuis data" - hospital is CONTEXT
   ✅ "medische gegevens" - this is the data type

3. **Adding people/targets**:
   ❌ "werknemers informatie" - werknemers is TARGET
   ✅ "personeels dossiers" - this is the data

4. **Being too vague**:
   ❌ "data", "informatie", "gegevens"
   ✅ "krediet scores", "medische diagnoses", "stem patronen"

## 🚫 NEUTRAL WORDS TO EXCLUDE

**CRITICAL**: The following generic/neutral words MUST NOT appear in vocabulary files:

### Banned Generic Terms:
- ❌ **"data"** (standalone) - Always specify what KIND of data
- ❌ **"informatie"** / **"information"** - Too vague
- ❌ **"gegevens"** (standalone) - Must have specific qualifier
- ❌ **"content"** (standalone) - Specify type of content
- ❌ **"documenten"** / **"documentation"** - Too generic
- ❌ **"bestanden"** / **"files"** - Specify file type/content
- ❌ **"metrieken"** / **"metrics"** - Must specify what is measured
- ❌ **"statistieken"** / **"statistics"** - Too broad without context
- ❌ **"rapporten"** / **"reports"** - Specify report type
- ❌ **"materialen"** / **"materials"** - Too vague
- ❌ **"resources"** - Marketing term, not data type

### Examples of Fixing Neutral Terms:
- ❌ "voice data" → ✅ "stem biometrie patronen"
- ❌ "session data" → ✅ "gebruiker interactie logs"
- ❌ "metadata" → ✅ "foto EXIF locatie data"
- ❌ "process efficiency" → ✅ "productie doorvoer tijden"
- ❌ "data quality" → ✅ "patiënt record volledigheid scores"

### Special Rules for WHAT:
1. **"data" suffix is banned** - Replace with specific descriptors:
   - ❌ "audio data" → ✅ "gesprek opnames", "stem samples"
   - ❌ "video data" → ✅ "bewakingsbeelden", "gezichts opnames"

2. **Abstract concepts need context**:
   - ❌ "workflow analysis" → ✅ "order verwerkings tijdlijnen"
   - ❌ "resource planning" → ✅ "personeels bezetting schema's"

3. **Compound generic terms are still generic**:
   - ❌ "data governance" → Remove entirely (meta-concept)
   - ❌ "data architecture" → Remove entirely (IT infrastructure)

## 📏 VALIDATION CHECKLIST

Before adding any term to a WHAT file:
- [ ] Is it data, information, or content?
- [ ] Does it describe what is being PROCESSED?
- [ ] Is it in Dutch?
- [ ] Is it specific enough?
- [ ] Are you 100% sure it's not an action (FUNCTION)?
- [ ] Are you 100% sure it's not a location (CONTEXT)?
- [ ] Are you 100% sure it's not a person/group (TARGET)?

---

**Last Updated**: [Date]
**Version**: 3.0 - RESTRUCTURED
**Authority**: AUDITOR ONLY