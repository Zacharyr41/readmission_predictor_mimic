# Ontology Reference

This document describes the RDF ontology used by the Temporal Knowledge Graph for hospital readmission prediction.

The ontology design follows the OWL-Time-compliant approach presented by Vannieuwenhuyze, Mimouni, and Du Mouza in ["A Conceptual Model for Discovering Implicit Temporal Knowledge in Clinical Data"](https://doi.org/10.1007/978-3-032-08620-4_6) (ER 2025). See the original implementation at [github.com/avannieuwenhuyze/clinical-tkg-cmls2025](https://github.com/avannieuwenhuyze/clinical-tkg-cmls2025).

## Namespaces

The knowledge graph uses the following namespaces:

| Prefix | URI | Description |
|--------|-----|-------------|
| `mimic` | `http://www.cnam.fr/MIMIC4-ICU-BSI/V1#` | MIMIC-IV clinical entities |
| `time` | `http://www.w3.org/2006/time#` | W3C OWL-Time ontology |
| `rdf` | `http://www.w3.org/1999/02/22-rdf-syntax-ns#` | RDF syntax |
| `rdfs` | `http://www.w3.org/2000/01/rdf-schema#` | RDF Schema |
| `owl` | `http://www.w3.org/2002/07/owl#` | OWL Web Ontology Language |
| `xsd` | `http://www.w3.org/2001/XMLSchema#` | XML Schema datatypes |

In Python code:
```python
from src.graph_construction.ontology import MIMIC_NS, TIME_NS

# Example usage
patient_uri = MIMIC_NS[f"Patient_{subject_id}"]
```

## Class Hierarchy

```
owl:Thing
├── mimic:Patient
├── time:TemporalEntity
│   ├── time:Instant
│   ├── time:Interval
│   │   ├── mimic:HospitalAdmission
│   │   ├── mimic:ICUStay
│   │   └── mimic:ICUDay
│   └── mimic:Event
│       ├── mimic:BioMarkerEvent
│       ├── mimic:ClinicalSignEvent
│       ├── mimic:MicrobiologyEvent
│       ├── mimic:AntibioticAdmissionEvent
│       └── mimic:DiagnosisEvent
├── mimic:Comorbidity
├── mimic:Organism
├── mimic:SpecimenType
└── mimic:Unit
```

### Class Descriptions

| Class | Description | Superclass |
|-------|-------------|------------|
| `mimic:Patient` | A patient in the MIMIC-IV database | `owl:Thing` |
| `mimic:HospitalAdmission` | A hospital admission episode from admit to discharge | `time:Interval` |
| `mimic:ICUStay` | An ICU stay within a hospital admission | `time:Interval` |
| `mimic:ICUDay` | A single calendar day within an ICU stay | `time:Interval` |
| `mimic:Event` | Base class for clinical events | `time:TemporalEntity` |
| `mimic:BioMarkerEvent` | Lab test result (biomarker measurement) | `mimic:Event` |
| `mimic:ClinicalSignEvent` | Vital sign measurement | `mimic:Event` |
| `mimic:MicrobiologyEvent` | Microbiology culture result | `mimic:Event` |
| `mimic:AntibioticAdmissionEvent` | Medication administration event | `mimic:Event` |
| `mimic:DiagnosisEvent` | ICD diagnosis code | `mimic:Event` |
| `mimic:Comorbidity` | Named comorbidity condition | `owl:Thing` |

## Object Properties

Properties linking entities together:

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasAdmission` | Patient | HospitalAdmission | Links patient to their admissions |
| `mimic:admissionOf` | HospitalAdmission | Patient | Inverse of hasAdmission |
| `mimic:containsICUStay` | HospitalAdmission | ICUStay | Links admission to ICU stays |
| `mimic:hasICUStay` | Patient | ICUStay | Links patient to ICU stays |
| `mimic:hasICUDay` | ICUStay | ICUDay | Links ICU stay to its days |
| `mimic:partOf` | ICUDay | ICUStay | Inverse: day belongs to stay |
| `mimic:hasICUDayEvent` | ICUDay | Event | Links day to events |
| `mimic:associatedWithICUDay` | Event | ICUDay | Inverse: event belongs to day |
| `mimic:hasICUStayEvent` | ICUStay | Event | Links stay to events |
| `mimic:associatedWithICUStay` | Event | ICUStay | Inverse: event belongs to stay |
| `mimic:hasEvent` | Patient | Event | Links patient to events |
| `mimic:concernsPatient` | (any) | Patient | General patient reference |
| `mimic:followedBy` | HospitalAdmission | HospitalAdmission | Links to subsequent admission |
| `mimic:hasDiagnosis` | HospitalAdmission | DiagnosisEvent | Links admission to diagnoses |
| `mimic:diagnosisOf` | DiagnosisEvent | HospitalAdmission | Inverse of hasDiagnosis |
| `mimic:hasComorbidity` | Patient | Comorbidity | Links patient to comorbidities |
| `mimic:cultureResult` | MicrobiologyEvent | CultureResult | Microbiology culture result |
| `mimic:oragnismFound` | MicrobiologyEvent | Organism | Organism identified |

### OWL-Time Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `time:hasBeginning` | Interval | Instant | Start time of interval |
| `time:hasEnd` | Interval | Instant | End time of interval |
| `time:hasDuration` | Interval | Duration | Duration of interval |
| `time:inXSDDateTimeStamp` | Instant | xsd:dateTimeStamp | Timestamp value |
| `time:numericDuration` | Duration | xsd:decimal | Numeric duration value |

## Datatype Properties

Properties with literal values:

### Patient Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasSubjectId` | Patient | xsd:integer | MIMIC subject_id |
| `mimic:hasGender` | Patient | xsd:string | Gender ("M" or "F") |
| `mimic:hasAge` | Patient | xsd:decimal | Age at admission |

### Hospital Admission Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasAdmissionId` | HospitalAdmission | xsd:integer | MIMIC hadm_id |
| `mimic:hasAdmissionType` | HospitalAdmission | xsd:string | Admission type (EMERGENCY, ELECTIVE, etc.) |
| `mimic:hasDischargeLocation` | HospitalAdmission | xsd:string | Discharge location (HOME, SNF, etc.) |
| `mimic:readmittedWithin30Days` | HospitalAdmission | xsd:boolean | 30-day readmission flag |
| `mimic:readmittedWithin60Days` | HospitalAdmission | xsd:boolean | 60-day readmission flag |
| `mimic:daysToReadmission` | HospitalAdmission | xsd:integer | Days until next admission |
| `mimic:hospitalExpireFlag` | HospitalAdmission | xsd:boolean | In-hospital mortality |

### ICU Stay Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasStayId` | ICUStay | xsd:integer | MIMIC stay_id |

### ICU Day Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasDayNumber` | ICUDay | xsd:integer | Day number (1-indexed) |

### Event Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasItemId` | Event | xsd:integer | MIMIC item_id |
| `mimic:hasValue` | Event | xsd:decimal/string | Measurement value |
| `mimic:hasUnit` | Event | xsd:string | Unit of measurement |
| `mimic:hasBiomarkerType` | BioMarkerEvent | xsd:string | Biomarker name |
| `mimic:hasFluid` | BioMarkerEvent | xsd:string | Body fluid tested |
| `mimic:hasRefRangeLower` | BioMarkerEvent | xsd:decimal | Lower reference range |
| `mimic:hasRefRangeUpper` | BioMarkerEvent | xsd:decimal | Upper reference range |
| `mimic:hasClinicalSignName` | ClinicalSignEvent | xsd:string | Vital sign name |
| `mimic:hasCategory` | Event | xsd:string | Event category |

### Medication Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasDrugName` | AntibioticAdmissionEvent | xsd:string | Drug name |
| `mimic:hasDoseValue` | AntibioticAdmissionEvent | xsd:decimal | Dose value |
| `mimic:hasDoseUnit` | AntibioticAdmissionEvent | xsd:string | Dose unit |
| `mimic:hasRoute` | AntibioticAdmissionEvent | xsd:string | Administration route |

### Diagnosis Properties

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `mimic:hasIcdCode` | DiagnosisEvent | xsd:string | ICD code |
| `mimic:hasIcdVersion` | DiagnosisEvent | xsd:integer | ICD version (9 or 10) |
| `mimic:hasLongTitle` | DiagnosisEvent | xsd:string | Diagnosis description |
| `mimic:hasSequenceNumber` | DiagnosisEvent | xsd:integer | Diagnosis priority (1 = primary) |

## Allen Temporal Relations

The system uses Allen's interval algebra to model temporal relationships between clinical events. These relations are computed for events within each ICU stay.

### Relation Definitions

```
Allen Relation    Visual Representation    OWL-Time Predicate
─────────────────────────────────────────────────────────────
before            A---|     |---B          time:before
meets             A---|B--------           time:intervalMeets
overlaps          A------|                 time:intervalOverlaps
                     |---B----
starts            |A---|                   time:intervalStarts
                  |----B----|
during               |--A--|               time:inside
                  |----B----|
finishes               |---A|              time:intervalFinishes
                  |----B----|
```

### Classification Logic

The `_classify_allen_relation()` function in `allen_relations.py` implements the classification:

```python
def _classify_allen_relation(a_start, a_end, b_start, b_end) -> str | None:
    # A ends before B starts -> before
    if a_end < b_start:
        return "before"

    # A ends exactly when B starts (and A is a proper interval) -> meets
    if a_end == b_start and a_start < b_start:
        return "meets"

    # A starts before B, A ends during B -> overlaps
    if a_start < b_start < a_end < b_end:
        return "overlaps"

    # Same start, A ends first -> starts
    if a_start == b_start and a_end < b_end:
        return "starts"

    # A entirely within B -> during
    if b_start < a_start and a_end < b_end:
        return "during"

    # A entirely within B (A is instant) -> during
    if a_start == a_end and b_start < a_start < b_end:
        return "during"

    # Same end, A starts later -> finishes
    if a_end == b_end and a_start > b_start:
        return "finishes"

    return None
```

### OWL-Time Predicate Mapping

```python
ALLEN_PREDICATES = {
    "before": TIME_NS.before,
    "during": TIME_NS.inside,
    "overlaps": TIME_NS.intervalOverlaps,
    "meets": TIME_NS.intervalMeets,
    "starts": TIME_NS.intervalStarts,
    "finishes": TIME_NS.intervalFinishes,
}
```

## Example SPARQL Queries

### 1. Get All Patients with Readmission Status

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?subjectId ?hadmId ?readmitted30
WHERE {
    ?patient rdf:type mimic:Patient ;
             mimic:hasSubjectId ?subjectId ;
             mimic:hasAdmission ?admission .
    ?admission mimic:hasAdmissionId ?hadmId ;
               mimic:readmittedWithin30Days ?readmitted30 .
}
ORDER BY ?subjectId ?hadmId
```

### 2. Get Patient Demographics

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?hadmId ?age ?gender
WHERE {
    ?patient rdf:type mimic:Patient ;
             mimic:hasAge ?age ;
             mimic:hasGender ?gender ;
             mimic:hasAdmission ?admission .
    ?admission mimic:hasAdmissionId ?hadmId .
}
```

### 3. Count Events per ICU Stay

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?stayId (COUNT(?event) AS ?eventCount)
WHERE {
    ?stay rdf:type mimic:ICUStay ;
          mimic:hasStayId ?stayId .
    ?event mimic:associatedWithICUStay ?stay .
}
GROUP BY ?stayId
ORDER BY DESC(?eventCount)
```

### 4. Get Lab Values with Reference Ranges

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?hadmId ?biomarkerType ?value ?refLower ?refUpper ?charttime
WHERE {
    ?admission rdf:type mimic:HospitalAdmission ;
               mimic:hasAdmissionId ?hadmId ;
               mimic:containsICUStay ?icuStay .
    ?event rdf:type mimic:BioMarkerEvent ;
           mimic:associatedWithICUStay ?icuStay ;
           mimic:hasBiomarkerType ?biomarkerType ;
           mimic:hasValue ?value ;
           time:inXSDDateTimeStamp ?charttime .
    OPTIONAL { ?event mimic:hasRefRangeLower ?refLower . }
    OPTIONAL { ?event mimic:hasRefRangeUpper ?refUpper . }
}
ORDER BY ?hadmId ?biomarkerType ?charttime
```

### 5. Count Allen Temporal Relations

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?stayId (COUNT(*) AS ?beforeCount)
WHERE {
    ?stay rdf:type mimic:ICUStay ;
          mimic:hasStayId ?stayId .
    ?eventA mimic:associatedWithICUStay ?stay ;
            time:before ?eventB .
    ?eventB mimic:associatedWithICUStay ?stay .
}
GROUP BY ?stayId
```

### 6. Get Diagnoses with ICD Codes

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?hadmId ?icdCode ?seqNum ?longTitle
WHERE {
    ?admission rdf:type mimic:HospitalAdmission ;
               mimic:hasAdmissionId ?hadmId ;
               mimic:hasDiagnosis ?diagnosis .
    ?diagnosis mimic:hasIcdCode ?icdCode ;
               mimic:hasSequenceNumber ?seqNum .
    OPTIONAL { ?diagnosis mimic:hasLongTitle ?longTitle . }
}
ORDER BY ?hadmId ?seqNum
```

### 7. Get ICU Stay Duration

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?hadmId ?stayId ?losDays
WHERE {
    ?admission rdf:type mimic:HospitalAdmission ;
               mimic:hasAdmissionId ?hadmId ;
               mimic:containsICUStay ?icuStay .
    ?icuStay mimic:hasStayId ?stayId ;
             time:hasDuration ?duration .
    ?duration time:numericDuration ?losDays .
}
```

### 8. Count Nodes by Type

```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?type (COUNT(?node) AS ?count)
WHERE {
    ?node rdf:type ?type .
}
GROUP BY ?type
ORDER BY DESC(?count)
```

### 9. Get Medication Events

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?hadmId ?drugName ?startTime ?endTime ?route
WHERE {
    ?admission rdf:type mimic:HospitalAdmission ;
               mimic:hasAdmissionId ?hadmId ;
               mimic:containsICUStay ?icuStay .
    ?event rdf:type mimic:AntibioticAdmissionEvent ;
           mimic:associatedWithICUStay ?icuStay ;
           mimic:hasDrugName ?drugName ;
           time:hasBeginning ?begin ;
           time:hasEnd ?end .
    ?begin time:inXSDDateTimeStamp ?startTime .
    ?end time:inXSDDateTimeStamp ?endTime .
    OPTIONAL { ?event mimic:hasRoute ?route . }
}
```

### 10. Find Sequential Admissions

```sparql
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?subjectId ?firstAdmId ?secondAdmId
WHERE {
    ?patient rdf:type mimic:Patient ;
             mimic:hasSubjectId ?subjectId ;
             mimic:hasAdmission ?firstAdm .
    ?firstAdm mimic:hasAdmissionId ?firstAdmId ;
              mimic:followedBy ?secondAdm .
    ?secondAdm mimic:hasAdmissionId ?secondAdmId .
}
ORDER BY ?subjectId ?firstAdmId
```

## Running SPARQL Queries in Python

```python
from rdflib import Graph

# Load the graph
graph = Graph()
graph.parse("data/processed/knowledge_graph.rdf")

# Define the query
query = """
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?subjectId ?hadmId
WHERE {
    ?patient rdf:type mimic:Patient ;
             mimic:hasSubjectId ?subjectId ;
             mimic:hasAdmission ?admission .
    ?admission mimic:hasAdmissionId ?hadmId .
}
LIMIT 10
"""

# Execute and iterate results
for row in graph.query(query):
    print(f"Patient {row.subjectId}, Admission {row.hadmId}")
```

## Ontology Files

The ontology is defined in two RDF/XML files:

### `ontology/definition/base_ontology.rdf`

Contains the core MIMIC-IV ontology:
- Patient, ICUStay, ICUDay classes
- Event classes (BioMarkerEvent, ClinicalSignEvent, etc.)
- Basic object and datatype properties

### `ontology/definition/extended_ontology.rdf`

Contains extensions for readmission prediction:
- HospitalAdmission class (subclass of time:Interval)
- hasAdmission, admissionOf, followedBy properties
- readmittedWithin30Days, readmittedWithin60Days properties
- Additional clinical event properties
