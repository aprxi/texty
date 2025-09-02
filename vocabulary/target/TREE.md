# TARGET Category Structure (TREE.md)

## 🔒 AUTHORITY NOTICE
**This TREE.md file may ONLY be modified by the AUDITOR.**
- Other agents may ONLY update the contents of existing .txt files
- Other agents may NOT create new directories or files
- Other agents may NOT rename directories or files
- Other agents may NOT modify this TREE.md file

---

## 📁 TARGET - Affected Individuals and Groups
**Definition**: WHO or WHAT is affected by AI systems - the people, groups, or entities
**Key Question**: "WHO is being analyzed, monitored, or decided upon?"
**Examples**: werknemers, studenten, patiënten, burgers, kinderen
**NOT**: AI capabilities (→FUNCTION), data types (→WHAT), usage locations (→CONTEXT)

## 📋 GENERAL RULES FOR ALL FILES

1. **Language**: ALL vocabulary entries must be in Dutch (unless English term is commonly used in Dutch)
2. **Format**: Each term on its own line, no translations
3. **Sorting**: Sort by commonality (most common terms first)
4. **Size**: Each file must contain 20-200 lines total
5. **Quality**: Only include terms with HIGH confidence they belong here
6. **No Mixing**: NEVER include terms from other dimensions

## EXAMPLE FILE HEADER:
```
# Employees and Workers - HIGH RISK TARGET
# Werknemers
# 
# PATH: target/hr/beslissings_subjecten/werknemers.txt
# DIMENSION: TARGET (Who is affected by AI)
# RISK: HIGH (Subject to employment decisions)
#
# HEADER RULES - DO NOT MODIFY - ONLY AUDITOR MAY CHANGE THIS SECTION
#
# TARGET dimension contains: People, groups, or entities affected by AI systems
# THIS FILE contains: Employees and workers subject to AI-based decisions
#
# VALIDATION RULES:
# - Must be a person or group of people
# - Must be in employment/work context
# - Must NOT include what AI does to them
# - Must NOT include where they work
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
*Vulnerable groups or those subject to significant AI impact*

#### 📂 kwetsbare_bevolking/ (vulnerable_population)
- `gehandicapten.txt` - People with disabilities
- `kinderen.txt` - Children and minors
- `ouderen.txt` - Elderly people
- `minderheidsgroepen.txt` - Ethnic and cultural minorities
- `vluchtelingen.txt` - Refugees and asylum seekers
- `daklozen.txt` - Homeless individuals
- `chronisch_zieken.txt` - Chronically ill patients
- `mentaal_kwetsbaren.txt` - Mentally vulnerable individuals

#### 📂 beslissings_subjecten/ (decision_subjects)
- `werknemers.txt` - Employees and workers
- `sollicitanten.txt` - Job applicants
- `studenten.txt` - Students in educational settings
- `patiënten.txt` - Patients in healthcare
- `burgers.txt` - Citizens in civic participation contexts
- `inwoners.txt` - Residents of jurisdictions
- `kiezers.txt` - Voters and electoral participants
- `ai_professionals.txt` - AI developers and researchers
- `geautomatiseerde_beslissing_subjecten.txt` - People subject to automated decisions
- `bestuurders.txt` - Managers and executives
- `gig_werkers.txt` - Gig economy workers
- `migranten.txt` - Migrants and immigrants
- `krediet_aanvragers.txt` - Credit applicants
- `verzekering_houders.txt` - Insurance holders

#### 📂 gecontroleerde_personen/ (controlled_persons)
- `surveillance_subjecten.txt` - Individuals under surveillance
- `verdachten.txt` - Criminal suspects
- `gevangenen.txt` - Prisoners and detainees
- `gemonitorde_werknemers.txt` - Monitored employees
- `bezoekers.txt` - Monitored visitors to facilities
- `grens_reizigers.txt` - Border travelers
- `publieke_ruimte_gebruikers.txt` - Users of public spaces

### 📂 lr/ (laag_risico / low_risk)
*General users and voluntary participants*

#### 📂 vrijwillige_gebruikers/ (voluntary_users)
- `ai_gebruikers.txt` - General AI system users
- `consumenten.txt` - General consumers
- `klanten.txt` - Customers and clients
- `kijkers.txt` - Viewers and audiences
- `spelers.txt` - Video game players
- `burgers.txt` - General citizens (non-civic context)
- `sociale_media_gebruikers.txt` - Social media platform users
- `winkelaars.txt` - In-store shoppers
- `app_gebruikers.txt` - Mobile app users
- `platform_gebruikers.txt` - Digital platform users
- `abonnees.txt` - Subscribers and members
- `bezoekers.txt` - Website visitors

#### 📂 professionele_gebruikers/ (professional_users)
- `creatieve_professionals.txt` - Artists and creative workers
- `zakelijke_gebruikers.txt` - Business users
- `tech_gebruikers.txt` - Technology users
- `leerlingen.txt` - General learners
- `gasten.txt` - Hotel and restaurant guests
- `content_makers.txt` - Content creators
- `freelancers.txt` - Independent contractors
- `consultants.txt` - Professional consultants
- `ondernemers.txt` - Entrepreneurs
- `specialisten.txt` - Industry specialists

#### 📂 persoonlijke_kringen/ (personal_circles)
- `familie_gebruikers.txt` - Family members
- `gemeenschaps_gebruikers.txt` - Community members
- `hobbyisten.txt` - Hobby enthusiasts
- `persoonlijke_gebruikers.txt` - Individual personal users
- `vrijwilligers.txt` - Volunteers
- `fitness_gebruikers.txt` - Gym and fitness members
- `recreatie_gebruikers.txt` - Recreational participants
- `vrienden.txt` - Friends and social contacts
- `kennissen.txt` - Acquaintances
- `buren.txt` - Neighbors
- `verenigings_leden.txt` - Association members
- `club_leden.txt` - Club members

---

## ⚠️ COMMON ERRORS TO AVOID

1. **Adding contexts to people**:
   ❌ "werknemers in de zorg" - "in de zorg" is CONTEXT
   ✅ "werknemers" - just the people

2. **Adding data about people**:
   ❌ "studenten met cijfers" - "cijfers" is WHAT
   ✅ "studenten" - just the group

3. **Adding what AI does to them**:
   ❌ "gemonitorde werknemers" - "gemonitorde" is FUNCTION
   ✅ "werknemers" - just the target group

4. **Being too specific**:
   ❌ "mannelijke werknemers tussen 25-35 jaar"
   ✅ "werknemers" (keep it general)

## 🚫 NEUTRAL WORDS TO EXCLUDE

**CRITICAL**: The following generic/neutral words MUST NOT appear in vocabulary files:

### Banned Generic Terms:
- ❌ **"gebruikers"** (standalone) - Always specify type of users
- ❌ **"personen"** / **"people"** - Too generic
- ❌ **"individuen"** / **"individuals"** - Too vague
- ❌ **"groepen"** / **"groups"** - Must specify what kind
- ❌ **"partijen"** / **"parties"** - Legal jargon, too abstract
- ❌ **"actoren"** / **"actors"** - Academic term, too generic
- ❌ **"stakeholders"** - Business jargon
- ❌ **"deelnemers"** / **"participants"** - Specify participants in what
- ❌ **"leden"** / **"members"** - Members of what?
- ❌ **"subjecten"** / **"subjects"** - Too clinical/abstract

### Examples of Fixing Neutral Terms:
- ❌ "users" → ✅ "sociale media gebruikers", "app gebruikers"
- ❌ "active users" → ✅ "dagelijkse app bezoekers"
- ❌ "new users" → ✅ "nieuwe klanten", "eerste keer kopers"
- ❌ "participants" → ✅ "wedstrijd deelnemers", "cursisten"
- ❌ "members" → ✅ "fitness club leden", "vakbonds leden"

### Special Rules for TARGET:
1. **Generic user types need context**:
   - ❌ "registered users" → ✅ "geregistreerde klanten"
   - ❌ "premium users" → ✅ "betaalde abonnees"
   - ❌ "guest users" → ✅ "website bezoekers"

2. **Avoid meta-categories**:
   - ❌ "all users" → Remove entirely
   - ❌ "various groups" → Remove entirely
   - ❌ "different personas" → Remove entirely

3. **Be specific about relationships**:
   - ❌ "related parties" → ✅ "familie leden", "zakenpartners"
   - ❌ "connected individuals" → ✅ "sociale netwerk contacten"

## 📏 VALIDATION CHECKLIST

Before adding any term to a TARGET file:
- [ ] Is it a person, group, or entity?
- [ ] Is it WHO/WHAT is affected by AI?
- [ ] Is it in Dutch?
- [ ] Is it general enough (not over-specified)?
- [ ] Are you 100% sure it's not an AI capability (FUNCTION)?
- [ ] Are you 100% sure it's not data (WHAT)?
- [ ] Are you 100% sure it's not a location (CONTEXT)?

---

**Last Updated**: [Date]
**Version**: 3.0 - RESTRUCTURED
**Authority**: AUDITOR ONLY