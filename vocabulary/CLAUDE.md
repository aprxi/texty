# Vocabulary Organization Instructions for EU AI Act Risk Classification

## Overview
This document provides detailed instructions for organizing vocabulary into a 4-dimensional structure for EU AI Act compliance. Each dimension captures different aspects of AI systems to determine risk levels.

**Note**: For the overall system structure and directory organization, see README.md. This document focuses on classification rules and quality standards.

## The Four Dimensions

### 1. FUNCTION - AI Capabilities/Technologies
**Definition**: The AI capability or technology being used (the AI "verb").

**Examples**:
- `gezichtsherkenning` / `facial recognition` - Face identification technology
- `geautomatiseerde besluitvorming` / `automated decision making` - AI making decisions
- `emotie detectie` / `emotion detection` - Detecting emotional states
- `gedragsanalyse` / `behavior analysis` - Analyzing behavior patterns
- `risicobeoordeling algoritme` / `risk assessment algorithm` - Algorithmic risk scoring

**What belongs here**: AI functions, algorithms, and capabilities (VERBS/ACTIONS)

**What does NOT belong here**: 
- Subject matter (credit, medical) → goes in WHAT
- Application domains (education, workplace) → goes in CONTEXT
- Affected parties (students, employees) → goes in TARGET

### 2. WHAT - Subject Matter/Domain Content
**Definition**: The subject matter or content the AI is processing (the "noun" being processed).

**Examples**:
- `hypotheek` / `mortgage` - Mortgage-related data
- `medische gegevens` / `medical data` - Health information
- `krediet` / `credit` - Credit/financial data
- `stemgedrag` / `voting behavior` - Electoral data
- `biometrische data` / `biometric data` - Physical characteristics

**What belongs here**: The data, content, or subject matter being processed (NOUNS/DATA TYPES)

**What does NOT belong here**:
- AI capabilities (recognition, detection) → goes in FUNCTION
- Where it's used (workplace, hospital) → goes in CONTEXT
- Who it affects (patients, voters) → goes in TARGET

### 3. CONTEXT - Domain/Setting/Actor
**Definition**: The domain, setting, or sector where the AI is being applied.

**Examples**:
- `workplace` / `werkplek` - Employment settings
- `education` / `onderwijs` - Educational settings
- `healthcare` / `gezondheidszorg` - Medical settings
- `law_enforcement` / `rechtshandhaving` - Police/justice
- `entertainment` / `entertainment` - Games/media
- `sports` / `sport` - Athletic/fitness

**What belongs here**: Sectors, industries, physical or virtual settings, organizational contexts

**What does NOT belong here**:
- AI technologies → goes in FUNCTION
- People being affected → goes in TARGET

### 4. TARGET - Who/What is Affected
**Definition**: The individuals, groups, or entities that are subject to or affected by the AI system.

**Examples**:
- `employees` / `werknemers` - Workers in an organization
- `students` / `studenten` - Learners in educational settings
- `patients` / `patiënten` - People receiving medical care
- `citizens` / `burgers` - Members of the public
- `children` / `kinderen` - Minors (vulnerable group)

**What belongs here**: Anyone or anything that is analyzed, monitored, assessed, or decided upon by the AI

**What does NOT belong here**:
- The organizations using AI → goes in CONTEXT
- The AI technology itself → goes in FUNCTION

## ⚠️ CRITICAL CLASSIFICATION RULES ⚠️

### Every single term MUST meet ALL criteria:

1. **CORRECT DIMENSION** - Each term must belong to its dimension with HIGH CONFIDENCE:
   - **FUNCTION**: What the AI system DOES (verbs/actions/capabilities)
   - **WHAT**: What data/information is PROCESSED (nouns/data types)
   - **CONTEXT**: WHERE the AI is used (domains/sectors/settings)
   - **TARGET**: WHO/WHAT is affected (people/groups/entities)

2. **CORRECT RISK LEVEL** - Each term must be classified based on ACTUAL RISK:
   - **HIGH RISK**: Only if the term represents an application that:
     - Makes decisions affecting fundamental rights
     - Operates in critical infrastructure
     - Affects safety, health, or legal status
     - Has significant impact on life opportunities
     - Examples: biometric identification, credit scoring, asylum decisions, medical diagnosis
   - **LOW RISK**: General purpose, entertainment, or voluntary-use applications
     - Examples: photo filters, game recommendations, search functions, chatbots

3. **HIGH CONFIDENCE RULE**: If there's ANY doubt about classification:
   - Remove the term
   - Move to a more appropriate file
   - Or add context to make it specific

4. **NO GENERIC TERMS**: Avoid terms that could apply to many contexts:
   - ❌ "processing", "analysis", "system", "data", "information"
   - ✅ "medical diagnosis", "credit scoring", "biometric identification"

5. **CONTEXT MATTERS**: Some terms can be both high/low risk:
   - "image recognition" → LOW risk in photo apps, HIGH risk in surveillance
   - "account verification" → LOW risk for social media, HIGH risk for government services
   - Place in the appropriate file based on the SPECIFIC USE CASE

6. **LANGUAGE RULES**:
   - Always list Dutch terms first, then English translations
   - Sort by commonality (most common first) within each section
   - Include variations (singular/plural, related terms)

## Review Checklist for Each Term
- [ ] Does it clearly belong to this dimension?
- [ ] Is the risk level appropriate for this specific use?
- [ ] Is it specific enough to be meaningful?
- [ ] Would an auditor agree with this classification?
- [ ] Is it in the right language order (Dutch first)?

## Common Mistakes to Avoid

1. **Mixing Dimensions**:
   - ❌ Putting "facial recognition" (FUNCTION) in WHAT
   - ❌ Putting "employees" (TARGET) in CONTEXT
   - ❌ Putting "workplace data" (WHAT) in CONTEXT

2. **Wrong Risk Level**:
   - ❌ "account verification" in high-risk asylum context (too generic)
   - ❌ "movie recommendations" in high-risk (it's entertainment)
   - ❌ Generic "data processing" in high-risk (too vague)

3. **Too Generic**:
   - ❌ "system", "platform", "solution", "tool"
   - ❌ "data", "information", "content"
   - ❌ "processing", "handling", "management"

## File Organization Rules

1. **File Structure**:
   ```
   # [Category Description]
   # Sorted by commonality (most common first)
   
   [most common Dutch term]
   [English translation]
   [second most common Dutch term]
   [English translation]
   ...
   
   # [Subcategory if needed]
   [terms continue...]
   ```

2. **Maximum ~100 terms per file** - Split large categories into subcategories

3. **File Naming**:
   - Use descriptive names (e.g., `krediet_scoring.txt` not `financial.txt`)
   - Keep names short but specific
   - Use underscores for multi-word names

## Examples of Correct Classification

### HIGH RISK Examples:
1. **FUNCTION**: `gezichtsherkenning` (facial recognition) - Biometric identification capability
2. **WHAT**: `medische diagnose` (medical diagnosis) - Health-critical data
3. **CONTEXT**: `rechtshandhaving` (law enforcement) - Justice/police setting
4. **TARGET**: `asielzoekers` (asylum seekers) - Vulnerable group

### LOW RISK Examples:
1. **FUNCTION**: `aanbeveling systeem` (recommendation system) - Suggestion capability
2. **WHAT**: `film voorkeuren` (movie preferences) - Entertainment data
3. **CONTEXT**: `entertainment` (entertainment) - Leisure setting
4. **TARGET**: `app gebruikers` (app users) - Voluntary users

## Instructions for Genies

When working on a specific category:

1. **Read every line critically** - Ask yourself:
   - Does this term truly belong in this dimension?
   - Is the risk level correct for this specific use?
   - Is it specific enough to be meaningful?

2. **Remove misclassified terms** - If a term doesn't belong:
   - Delete it from the current file
   - Note where it should go (if anywhere)
   - Don't try to fix other files - focus on your assigned category

3. **Check for duplicates** - Within your category:
   - Remove exact duplicates
   - Consolidate similar terms
   - Keep the most common/clear version

4. **Maintain quality** - For each file:
   - Ensure proper header comments
   - Sort by commonality (most common first)
   - Keep Dutch/English pairs together
   - Group related concepts

5. **When in doubt** - Apply the HIGH CONFIDENCE rule:
   - If unsure about classification → remove it
   - If too generic → remove it
   - If could be multiple risk levels → pick the most specific use case

Remember: It's better to have fewer, high-quality terms than many questionable ones. Quality over quantity!