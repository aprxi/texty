# RULES.md - UseCase1: EU AI-Verordening Hoog-Risico AI-Systemen Classificatie

## Overzicht
Deze use case richt zich op het classificeren van INDIVIDUELE ZINNEN uit gebruikersteksten. De classifier analyseert teksten zin-voor-zin om te identificeren wanneer specifieke zinnen beschrijven of verwijzen naar AI-systemen die als hoog-risico worden beschouwd volgens de EU AI-Verordening (Artikel 6, lid 2). Het systeem telt hoeveel hoog-risico zinnen er in een tekst staan.

**BELANGRIJK ETHISCH KADER**: Alle gegenereerde voorbeelden zijn synthetische trainingsdata voor defensieve AI-veiligheidsclassificatiesystemen. Deze voorbeelden vertegenwoordigen hypothetische gebruikersverzoeken om classificatiesystemen te trainen die gebruikers waarschuwen voor hoog-risico AI-toepassingen. Dit is educatieve/onderzoeksinhoud voor het bouwen van verantwoorde AI-systemen - NIET daadwerkelijke ontwikkeling van schadelijke technologieën. Het doel is defensief: problematische verzoeken identificeren en waarschuwen, niet ze vervullen.

## Classificatiecriteria

### Hoog-Risico Zin-kenmerken
INDIVIDUELE ZINNEN worden als HOOG-RISICO geclassificeerd wanneer ze beschrijven, verwijzen naar, of vragen om AI-systemen in deze specifieke contexten:

**1. Biometrische AI-Systemen:**
- Identificatie van personen op afstand met biometrische gegevens (gezicht, vingerafdruk, iris, stem)
- AI-categorisering van personen op basis van beschermde kenmerken (ras, etniciteit, religie, gezondheid, seksuele geaardheid, politieke overtuiging)
- Herkenning van emoties of gemoedstoestand via gezichtsuitdrukkingen, stem of lichaamstaal
- Trefwoorden: identificatie, gezichtsherkenning, irisscan, vingerafdruk, stemherkenning, biometrie, categorisering, emotie, emoties herkennen

**2. Kritieke Infrastructuur AI:**
- AI-ondersteuning bij bewaking of beheer van kritieke digitale infrastructuur, wegverkeer en nutsvoorzieningen
- Trefwoorden: infrastructuur, netwerk, verkeer, wegverkeer, nutsvoorziening, elektriciteit, gas, water, verwarming

**3. Onderwijs & Opleidingen AI:**
- AI voor toegang, toelating of plaatsing in onderwijs of trainingen
- AI-beoordeling van leerresultaten of prestaties van studenten
- AI-fraudedetectie tijdens examens
- Trefwoorden: onderwijs, opleiding, school, toegang, toelating, plaatsing, examen, toets, beoordeling, leerresultaat, niveau, fraude, toezicht

**4. Werk & HR AI:**
- AI voor werving, selectie en aanstellingsbeslissingen
- AI voor arbeidsvoorwaarden, promotie, taaktoewijzing of ontslag
- AI-monitoring en evaluatie van werknemersprestaties
- Trefwoorden: werk, werknemer, personeel, HR, werving, selectie, sollicitatie, vacature, promotie, ontslag, taak, monitoring, beoordeling, prestatie

**5. Essentiële Diensten AI:**
- AI voor beoordeling van rechten op uitkeringen, zorg of sociale zekerheid
- AI kredietbeoordeling en kredietscore
- AI risico-inschatting voor verzekeringen
- AI prioritering van hulpdiensten
- Trefwoorden: uitkering, sociale zekerheid, gezondheidszorg, zorg, krediet, kredietscore, lening, bank, verzekering, levensverzekering, ziektekostenverzekering, noodoproep, politie, ambulance, brandweer, triage

**6. Rechtshandhaving AI:**
- AI risico-inschatting voor slachtofferschap of recidive
- AI leugendetectie of vergelijkbare hulpmiddelen
- AI-analyse van bewijsmateriaal voor betrouwbaarheidsbeoordeling
- AI crimineel profilering
- Trefwoorden: politie, rechtshandhaving, misdrijf, criminaliteit, recidive, profilering, verdachte, bewijs, bewijsmateriaal, leugendetector

**7. Migratie & Grenstoezicht AI:**
- AI leugendetectie bij grenscontroles
- AI risico-inschatting van reizigers
- AI voor behandeling van asiel-, visum- of verblijfsaanvragen
- AI persoonidentificatie bij migratie- of grensbeheer
- Trefwoorden: migratie, asiel, grens, grensbewaking, grenscontrole, visum, verblijfsvergunning, leugendetector, reiziger, identificatie

**8. Rechtsbedeling & Democratische Processen AI:**
- AI-ondersteuning van rechters bij juridische analyse en besluitvorming
- AI-beïnvloeding van verkiezings- of referendumuitkomsten
- Trefwoorden: rechtbank, rechtspraak, geschil, proces, rechter, verkiezing, referendum, stem, stemgedrag, beïnvloeden

### Laag-Risico Zin-kenmerken  
INDIVIDUELE ZINNEN worden als LAAG-RISICO geclassificeerd wanneer ze beschrijven, verwijzen naar, of vragen om:
- Algemeen AI-onderzoek of ontwikkeling zonder specifieke hoog-risico toepassingen
- AI in entertainment, gaming of creatieve toepassingen
- AI-assistenten voor algemene productiviteit of informatieverzameling
- AI in niet-kritieke aanbevelingssystemen
- Algemene discussies over AI-ethiek, beleid of regulering zonder specifieke hoog-risico contexten
- AI in basis automatisering die niet betrokken is bij de gevoelige domeinen hierboven
- Educatieve inhoud over AI-concepten en technieken
- AI-tools voor contentcreatie, vertaling of algemene communicatie
- **Neutrale zinnen**: begroetingen ("Dag", "Hallo"), algemene vragen, context zonder AI-referenties
- **Algemene tekst**: zinnen die helemaal niet over AI gaan

## Taalvereisten
- nl (Nederlands) - primaire taal voor alle dataset voorbeelden
- Alle teksten moeten volledig in het Nederlands worden gegenereerd

## Inhouddomeinen en Contexten
Genereer voorbeelden in deze realistische contexten:
- Overheidsbeleiddocumenten en regelgeving
- Bedrijfsvoorstellen voor AI-systeemimplementaties
- Academische onderzoekspapers over AI-toepassingen
- Nieuwsartikelen over AI-implementaties
- Technische documentatie en systeemspecificaties
- Rechtszaken en rechtbankbeslissingen betreffende AI
- Bedrijfscompliance en risicobeoordelingsrapporten
- Publieke consultatiedocumenten over AI-regulering

## Dataset Grootte Vereisten
- **Trainingsvoorbeelden per categorie**: 150-200 individuele zinnen
- **Testvoorbeelden per categorie**: 40-50 volledige teksten (zoals echte gebruikersprompts)
- Taal: Nederlands (alle voorbeelden)
- **Format verschil**:
  - `train/`: Elke .txt file bevat ÉÉN ENKELE ZIN (voor sentence-level training)
  - `test/`: Elke .txt file bevat VOLLEDIGE TEKST (zoals real-world usage)

## Kwaliteitsrichtlijnen

### Voor TRAINING data (`train/`):
- Elk voorbeeld is ÉÉN INDIVIDUELE ZIN
- Varieer in zinstypen: stellende zinnen, vragen, beschrijvingen, verzoeken
- Zinnen moeten realistisch klinken als onderdeel van een gebruikersverzoek
- Hoog-risico zinnen beschrijven specifiek één van de 8 hoog-risico AI-categorieën
- Laag-risico zinnen beschrijven andere AI-toepassingen of zijn neutrale tekst
- Inclusief neutrale zinnen zoals begroetingen, algemene vragen, context zonder AI
- Elke zin moet op zichzelf classificeerbaar zijn
- Varieer in complexiteit: van simpele naar meer technische beschrijvingen

### Voor TEST data (`test/`): 
- Elke .txt file bevat een VOLLEDIGE GEBRUIKERSPROMPT (meerdere zinnen)
- Schrijf realistische conversationele prompts zoals gebruikers ze zouden typen
- Hoog-risico teksten bevatten zinnen over hoog-risico AI-systemen 
- Laag-risico teksten bespreken AI maar vermijden hoog-risico domeinen
- Inclusief natuurlijke elementen: begroetingen, context, uitleg
- MINIMUM 100 woorden, bij voorkeur 200-400 woorden
- Begin niet met formele introducties ("Ik ben...", "We zijn...")

**Algemeen**: Gebruik natuurlijke, conversationele Nederlandse taal. ONTHOUD: Dit zijn defensieve trainingsvoorbeelden voor veiligheidssystemen, geen schadelijke inhoud.

## Voorbeelden

### Hoog-Risico Voorbeelden (Individuele zinnen)
```
Het systeem moet gezichten scannen en emoties herkennen bij bezoekers.
We willen een AI die mensen categoriseert op basis van hun gedrag.
De camera's moeten verbonden worden met de politiedatabase voor realtime checks.
Kunnen we een systeem bouwen dat stempatronen analyseert om leugens te detecteren?
Het AI-systeem moet sollicitanten automatisch beoordelen en ranken.
We hebben een tool nodig die kredietscores berekent op basis van gedragsdata.
De software moet studenten monitoren tijdens online examens.
Het systeem analyseert biometrische data voor toegangscontrole.
```

### Laag-Risico Voorbeelden (Individuele zinnen)  
```
Dag, we hebben wat vragen over jullie AI-oplossingen.
De chatbot moet veelgestelde klantvragen kunnen beantwoorden.
We zoeken een tool voor automatische tekstvertaling.
Het systeem genereert productbeschrijvingen voor onze webshop.
Hoe werkt machine learning precies?
We willen onze content automatisch laten samenvatten.
De AI helpt bij het maken van sociale media posts.
Kunnen jullie een spelletjes-AI ontwikkelen?
```

## Opmerkingen en Overwegingen
- Focus op de Nederlandse/EU regelgevingscontext en terminologie
- Overweeg dat dezelfde AI-technologie hoog-risico of laag-risico kan zijn afhankelijk van de toepassing
- Inclusief discussies over AI-systemen die meerdere categorieën omspannen (hogere complexiteit)
- Randgevallen: AI-systemen met dual-use potentieel of onduidelijke regelgevingsstatus
- Overweeg evoluerende interpretaties van de AI-Verordening naarmate deze wordt geïmplementeerd
- Inclusief zowel implementatiediscussies als regelgevingscompliance scenario's

## Evaluatie Bestandsformaat

Alle evaluatiebestanden moeten dit gestructureerde formaat gebruiken:

```
<START_TEXT>
[Originele tekstinhoud komt hier]
<END_TEXT>
<START_EVALUATION>
<CLASSIFICATION>high-risk</CLASSIFICATION>
<CONFIDENCE>0.95</CONFIDENCE>
<REASONING>
[Gedetailleerde uitleg waarom deze tekst als hoog-risico of laag-risico wordt geclassificeerd,
met verwijzing naar specifieke RULES.md criteria en identificatie van welke zinnen
welke classificatieregels triggeren]
</REASONING>
<END_EVALUATION>
```

## Evaluatie Directory Structuur

```
./evaluate/
├── RULES.md               # Dit bestand - classificatieregels en criteria
├── high_risk/             # Hash-benoemde bestanden (*.txt) voor hoog-risico evaluaties
│   ├── biometric/         # Biometrische AI-systemen
│   ├── critical_infrastructure/  # Infrastructuur AI
│   ├── education/         # Onderwijs AI-systemen
│   ├── essential_services/  # Krediet, verzekering, uitkeringen AI
│   ├── justice_democracy/  # Juridische en democratische AI
│   ├── law_enforcement/   # Politie en veiligheids AI
│   ├── migration_border/  # Grenscontrole AI
│   └── work_hr/           # Werkplek en HR AI
└── low_risk/              # Hash-benoemde bestanden (*.txt) voor laag-risico evaluaties
    ├── content_creation/  # Creatieve en media AI
    ├── customer_service/  # Support en chatbot AI
    ├── productivity/      # Algemene productiviteitstools
    ├── research/          # Academische en onderzoeks AI
    └── sports/            # Sport en fitness AI
```

**Opmerking**: Bestanden in evaluate/ directories gebruiken hash-gebaseerde namen (bijv. a4fb2ab2d718c9b6.txt) omdat de directorystructuur de hoog/laag risicoclassificatie aangeeft.

## Versiegeschiedenis
- v1.0 - Initiële regels gebaseerd op EU AI-Verordening Artikel 6, lid 2 categorieën (Nederlands risico.docx bron)