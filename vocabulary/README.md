# Vocabulary Organization for EU AI Act Risk Classification

## Overview
This vocabulary system organizes AI-related terms into a 3-dimensional structure for EU AI Act risk classification. The system uses a genie-based approach where specialized agents review and maintain each dimension.

- **Structure**: 3 orthogonal dimensions (FUNCTION, WHAT, TARGET)
- **Instructions**: See CLAUDE.md for detailed classification rules and genie instructions
- **Category Organization**: See individual TREE.md files in each dimension folder

This vocabulary is organized into three orthogonal dimensions that capture the essential elements for determining AI system risk levels according to the EU AI Act.

## The Three Dimensions

### 1. FUNCTION - AI Capabilities and Processes
**Definition**: The AI capability, technology, or process being performed (the AI "verb" and "action").

**Examples of AI capabilities**:
- `gezichtsherkenning` / `facial recognition` - Face identification technology
- `geautomatiseerde besluitvorming` / `automated decision making` - AI making decisions
- `emotie detectie` / `emotion detection` - Detecting emotional states
- `gedragsanalyse` / `behavior analysis` - Analyzing behavior patterns
- `risicobeoordeling algoritme` / `risk assessment algorithm` - Algorithmic risk scoring
- `aanbeveling systeem` / `recommendation system` - Suggesting content/products

**Examples of AI-enabled processes**:
- `beoordelen sollicitanten` - Evaluating job applicants
- `screenen kredietaanvragen` - Screening credit applications  
- `diagnosticeren patiënten` - Diagnosing patients
- `identificeren verdachten` - Identifying suspects
- `aanbevelen films` - Recommending movies

**What belongs here**: AI functions, algorithms, capabilities, and the processes they perform

**What does NOT belong here**: 
- Subject matter (credit, medical) → goes in WHAT
- Affected parties (students, employees) → goes in TARGET

### 2. WHAT - Subject Matter/Domain Content
**Definition**: The subject matter, data, or content the AI is processing (the "noun" being processed).

**Examples**:
- `hypotheek` / `mortgage` - Mortgage-related data
- `medische gegevens` / `medical data` - Health information
- `krediet` / `credit` - Credit/financial data
- `stemgedrag` / `voting behavior` - Electoral data
- `sollicitatie` / `job application` - Employment applications
- `biometrische data` / `biometric data` - Physical characteristics
- `film voorkeuren` / `movie preferences` - Entertainment data

**What belongs here**: The data, content, or subject matter being processed

**What does NOT belong here**:
- AI capabilities (recognition, detection) → goes in FUNCTION
- Who it affects (patients, voters) → goes in TARGET

### 3. TARGET - Who/What is Affected
**Definition**: The individuals, groups, or entities that are subject to or affected by the AI system.

**Examples of human targets**:
- `werknemers` / `employees` - Workers in an organization
- `studenten` / `students` - Learners in educational settings
- `patiënten` / `patients` - People receiving medical care
- `burgers` / `citizens` - Members of the public
- `klanten` / `customers` - People purchasing goods/services
- `kinderen` / `children` - Minors (vulnerable group)
- `ouderen` / `elderly` - Senior citizens (vulnerable group)
- `sollicitanten` / `job_applicants` - People seeking employment
- `reizigers` / `travelers` - People in transit

**Examples of entity targets**:
- `bedrijven` / `companies` - Business organizations
- `voertuigen` / `vehicles` - Cars, trucks, etc.
- `transacties` / `transactions` - Financial operations
- `systemen` / `systems` - IT infrastructure
- `producten` / `products` - Goods being analyzed

**What belongs here**: Anyone or anything that is analyzed, monitored, assessed, or decided upon by the AI

**What does NOT belong here**:
- The AI technology itself (monitoring, scoring) → goes in FUNCTION
- The data being processed (medical data, CV data) → goes in WHAT

## Risk Classification Logic

The combination of these three dimensions determines the risk level:

### High Risk Combinations (AI Act regulated applications):
- `gezichtsherkenning` + `biometrische data` + `verdachten`
- `geautomatiseerde besluitvorming` + `kredietgegevens` + `klanten`  
- `voorspelling algoritme` + `medische symptomen` + `patiënten`
- `beoordelen sollicitanten` + `sollicitatie gegevens` + `sollicitanten`
- `screenen kredietaanvragen` + `financiële gegevens` + `aanvragers`

### Low Risk Combinations (non-regulated applications):
- `gezichtsherkenning` + `foto's` + `vrienden`
- `aanbeveling systeem` + `filmgegevens` + `gebruikers`
- `zoek algoritme` + `productgegevens` + `shoppers`
- `chatbot` + `klantvragen` + `bezoekers`
- `vertaal systeem` + `tekstgegevens` + `lezers`

### Key Insight: 
The **same AI capability** can be high-risk or low-risk depending on the application context:
- `gezichtsherkenning` + `biometrische data` + `verdachten` = **HIGH-RISK** (AI Act regulated)  
- `gezichtsherkenning` + `foto's` + `vrienden` = **LOW-RISK** (not regulated)

## Directory Structure

```
vocabulary/
├── function/
│   ├── hr/ (high_risk)
│   │   ├── algoritmen/
│   │   │   ├── geautomatiseerde_besluitvorming.txt
│   │   │   ├── krediet_scoring.txt
│   │   │   └── medische_diagnose.txt
│   │   ├── biometrisch/
│   │   │   ├── gezichtsherkenning.txt
│   │   │   ├── vingerafdruk_herkenning.txt
│   │   │   └── emotie_detectie.txt
│   │   └── surveillance/
│   │       ├── surveillance_monitoring.txt
│   │       └── gedragsanalyse.txt
│   └── lr/ (low_risk)
│       ├── aanbevelingen/
│       │   ├── aanbevelingen.txt
│       │   └── content_aanbevelingen.txt
│       ├── assistentie/
│       │   ├── ai_assistenten.txt
│       │   └── chatbots.txt
│       └── generatie/
│           ├── tekst_generatie.txt
│           └── beeld_generatie.txt
├── what/
│   ├── hr/ (high_risk)
│   │   ├── beslissings_data/
│   │   │   ├── beslissings_gegevens.txt
│   │   │   ├── krediet_scoring.txt
│   │   │   └── medische_diagnose.txt
│   │   ├── persoonlijke_data/
│   │   │   ├── biometrische_data.txt
│   │   │   ├── medische_gegevens.txt
│   │   │   └── emotionele_data.txt
│   │   └── surveillance_data/
│   │       ├── surveillance_systemen.txt
│   │       └── gedrag_analyse.txt
│   └── lr/ (low_risk)
│       ├── content_data/
│       │   ├── aanbevelingen.txt
│       │   ├── media_content.txt
│       │   └── gaming_data.txt
│       ├── commerciële_data/
│       │   ├── klant_service_data.txt
│       │   └── marketing_analyse.txt
│       └── persoonlijke_data/
│           ├── voorkeur_instellingen.txt
│           └── sociale_media_data.txt
└── target/
    ├── hr/ (high_risk)
    │   ├── kwetsbare_bevolking/
    │   │   ├── kinderen.txt
    │   │   ├── ouderen.txt
    │   │   └── gehandicapten.txt
    │   ├── beslissings_subjecten/
    │   │   ├── werknemers.txt
    │   │   ├── sollicitanten.txt
    │   │   └── studenten.txt
    │   └── gecontroleerde_personen/
    │       ├── surveillance_subjecten.txt
    │       └── publieke_ruimte_gebruikers.txt
    └── lr/ (low_risk)
        ├── vrijwillige_gebruikers/
        │   ├── ai_gebruikers.txt
        │   ├── spelers.txt
        │   └── abonnees.txt
        └── professionele_gebruikers/
            ├── klanten.txt
            ├── zakelijke_gebruikers.txt
            └── gasten.txt
```

## How to Use This Vocabulary System

1. **Understanding the Structure**: Each AI application is described by combining terms from all three dimensions
2. **Risk Assessment**: The combination of dimensions determines if an AI system is high-risk or low-risk
3. **Classification Rules**: See CLAUDE.md for detailed classification rules and quality standards
4. **Maintenance**: The vocabulary is maintained by specialized genies focusing on specific dimensions

## Key Principles

- **Dutch-first**: All terms are listed with Dutch first, then English translation where applicable
- **High Confidence**: Only include terms that clearly belong to their dimension and risk level
- **Specificity**: Avoid generic terms that could apply to multiple contexts
- **Quality over Quantity**: Better to have fewer, accurate terms than many questionable ones
- **Orthogonality**: Each dimension captures distinct aspects - no overlap between FUNCTION, WHAT, and TARGET

For detailed classification rules, review processes, and genie instructions, see **CLAUDE.md**.