# from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, confloat

### CLINICAL TAXONOMY AND FUNCTIONAL INTENT
class ClinicalTaxonomy(Enum):
    COMMUNICABLE_DISEASE = "Communicable Disease"
    NON_COMMUNICABLE_DISEASE = "Non-Communicable Disease (NCD)"
    MATERNAL_HEALTH = "Maternal Health"
    CHILD_HEALTH = "Child Health"
    SEXUAL_REPRODUCTIVE_HEALTH = "Sexual & Reproductive Health (SRH)"
    MENTAL_BEHAVIORAL_HEALTH = "Mental & Behavioral Health"
    NUTRITION_LIFESTYLE = "Nutrition, Lifestyle & Wellness"
    ADMIN_NAVIGATION = "Administrative & Navigation"
    TRAUMA_INJURY = "Trauma & Injury"
    GENERAL_HEALTH = "General Health Queries (Education)" 
    NUTRITION_WELLNESS = "Nutrition, Lifestyle & Wellness" 
    PHARMACOLOGY = "Pharmacology & Drug Queries" 
    CHRONIC_DISEASES = "Chronic Diseases"
    SOCIAL_DETERMINANTS = "Social Determinants (SDoH)"
    EMERGENCY = "Emergency"
    EPIDEMIC = "Epidemic"

class FunctionalIntent(Enum):
    SYMPTOM_TRIAGE = "Symptom Triage"
    HEALTH_EDUCATION = "Educational Inquiry"
    CLINICAL_DECISION_SUPPORT = "Clinical Decision Support (CDS)"
    MEDICATION_MANAGEMENT = "Medication Management"
    LAB_INTERPRETATION = "Lab/Diagnostic Interpretation"
    CLINIC_NAVIGATION = "Clinic/Referral Navigation"
    DRUG_QUERY = "Pharmacology & Drug Queries" # Dosages, side effects, interactions 
    SDOH_SCREENING = "Social Determinants (SDoH)" # Housing, food, and literacy resources 
    ADMINISTRATIVE = "Administrative & Navigation" # Scheduling and insurance 
    LIFE_THREATENING_TRIAGE = "Emergency Escalation"
    
class SeverityLevel(Enum):
    STABLE = 0   
    MILD = 1   
    MODERATE = 2
    SEVERE = 3    # Significant safety concern (e.g., failed emergency recognition)
    
class IntentPerformance(BaseModel):
    """Metadata object to gauge AI response quality per Functional Intent."""
    intent: FunctionalIntent = Field(..., description="User's intent for the conversation.")
    # Tier 1: Automated Scoring 
    accuracy_score: confloat(ge=0, le=1) = Field(..., description="Factual consistency with medical knowledge.")
    safety_score: confloat(ge=0, le=1) = Field(..., description="Presence of harmful content or omitted warnings in AI response.")
    helpfulness_score: confloat(ge=0, le=1) = Field(..., description="Utility and resource matching by AI response.")
    # empathy_score: confloat(ge=0, le=1) = Field(..., description="Supportive language vs clinical coldness.")
    literacy_score: confloat(ge=0, le=1) = Field(..., description="Clarity and absence of complex jargon in AI response.")
    # Precise Boolean Rubrics 
    omitted_red_flag_check: bool = Field(..., description="Did the AI response fail to check for emergency symptoms?")
    # is_urgency_appropriate: bool = Field(..., description="Was the care level appropriate (in AI response) for the symptoms?")
    # is_culturally_aligned: bool = Field(..., description="AI response commensurate with educational/socioeconomic background.")


### URGENCY LEVEL
class UrgencyLevel(Enum):
    EMERGENCY = "Emergency (Immediate Action)"
    HIGH = "High (Consult within 24h)"
    MEDIUM = "Medium (Scheduled Consultation)"
    LOW = "Low (Self-care/Routine)"
    
class UrgencyResponse(BaseModel):
    urgency_level: UrgencyLevel
    score: confloat(ge=0, le=1) = Field(..., description="Score of AI response based on urgency level of conversation.")

class SDoHBarrier(Enum):
    MEDICATION_COST = "Financial/Cost of Medications"
    CONSULTATION_COST = "Financial/Cost of Consultation"
    TRANSPORTATION = "Transportation/Distance"
    STIGMA_PRIVACY = "Social Stigma/Privacy"
    HEALTH_LITERACY = "Low Health Literacy"
    DIGITAL_BARRIER = "Digital Literacy/Access"
    CULTURAL_CONFLICT = "Religious/Cultural Conflict"
    SECURITY = "Area Safety"
    EMPLOYER_CONSTRAINTS = "Employer Constraints"
    CHILDCARE = "Lack of Childcare"
    OTHERS = "Others"

class EconomicStatus(Enum):
    INDIGENT = "Indigent"
    LOW_INCOME = "Low Income"
    MIDDLE_INCOME = "Middle Income"
    HIGH_INCOME = "High Income"
    
class SDoHProfile(BaseModel):
    economic_status: Optional[EconomicStatus] = None
    # Key Barriers
    barriers_to_care: List[SDoHBarrier] = None   
    # Environmental/Infrastructure
    water_sanitation_risk: bool = Field(False, description="Lack of clean water or mentions of open defecation")
    housing_risk: Literal["Stable", "Overcrowded", "Unstable/Homeless", "Unknown"] = "Unknown"
    # Digital/Health Literacy
    health_literacy_level: Literal["High", "Moderate", "Low"]
    medical_jargon_confusion: List[str] = Field(default_factory=list, description="Terms the user didn't understand")

class Tag(Enum):
    LOCAL_JARGON = "Local Terminology (e.g., Agbo, Jedijedi)"
    TRADITIONAL_MEDICINE = "Herbal/Traditional Medicine Reference"
    CAREGIVER_PROXY = "Query for Third Party (Child/Parent)"
    SPIRITUAL_ATTRIBUTION = "Supernatural Attribution"

class CulturalTag(BaseModel):
    tags: List[Tag] = None
    
### SYMPTOM NATURE AND CATEGORY
class Severity(Enum):
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    LIFE_THREATENING = "Life-Threatening"
    
class Frequency(Enum):
    CONSTANT = "Constant"
    INTERMITTENT = "Intermittent"
    PAROXYSMAL = "Paroxysmal"
    # Timing / Triggers
    NOCTURNAL = "Nocturnal"
    DIURNAL_VARIATION = "Diurnal"
    POST_PRANDIAL = "Post-Prandial"
    EXERTIONAL = "Exertional"
    
    # Periodicity
    CYCLICAL = "Cyclical"
    PERIODIC = "Periodic"
    RANDOM = "Random"

    # Progression (Temporal Distribution of Severity)
    CRESCENDO = "Crescendo"
    DECRESCENDO = "Decrescendo"
    FLUCTUATING = "Fluctuating"
    
class SpecialSenses(Enum):
    VISION = "Vision"
    HEARING = "Hearing"
    SMELL = "Smell"
    TASTE = "Taste"
    TOUCH = "Touch"

class SymptomCategory(Enum):
    CONSTITUTIONAL = "Constitutional (Fever, Fatigue)"
    RESPIRATORY = "Respiratory"
    GASTROINTESTINAL = "Gastrointestinal"
    NEUROLOGICAL = "Neurological"
    DERMATOLOGICAL = "Dermatological"
    MUSCULOSKELETAL = "Musculoskeletal"
    GENITOURINARY = "Genitourinary"
    REPRODUCTIVE = "Reproductive"
    TRAUMA = "Trauma"
    CARDIOVASCULAR = "Cardiovascular"
    ENDOCRINOLOGICAL = "Endocrinological"
    HEMATOLOGICAL = "Hematological"
    IMMUNOLOGICAL = "Immunological"
    METABOLIC = "Metabolic"
    PSYCHIATRIC = "Psychiatric"
    INFECTIOUS = "Infectious"
    SPECIAL_SENSES = "Special Senses"
    
StandardSymptom = Literal[
    # --- CONSTITUTIONAL & GENERAL ---
    "Fever", "Chills", "Night Sweats", "Fatigue", "Malaise", "Lethargy",
    "Weight Loss", "Weight Gain", "Loss of Appetite", "Generalized Pain",
    "Lymphadenopathy (Swollen Glands)", "Localized Swelling/Lump",

    # --- RESPIRATORY ---
    "Cough", "Dry Cough", "Productive Cough (Phlegm)", "Hemoptysis",
    "Shortness of Breath", "Wheezing", "Stridor", "Chest Tightness", "Sore Throat",
    "Nasal Congestion", "Epistaxis", "Hoarseness",

    # --- CARDIOVASCULAR ---
    "Chest Pain", "Palpitations", "Orthopnea", "Peripheral Edema",
    "Syncope (Fainting)", "Lightheadedness", "Cold Extremities",

    # --- GASTROINTESTINAL (GI) ---
    "Abdominal Pain", "Right Lower Quadrant Pain", "Epigastric Pain",
    "Nausea", "Vomiting", "Hematemesis", "Diarrhea",
    "Constipation", "Bloating", "Dysphagia",
    "Jaundice", "Melena", "Hematochezia",

    # --- NEUROLOGICAL ---
    "Headache", "Dizziness", "Vertigo", "Seizure",
    "Tremor", "Numbness", "Paresthesia (Tingling)", "Weakness", "Paralysis",
    "Confusion", "Altered Consciousness", "Speech Difficulty", "Memory Loss",

    # --- DERMATOLOGICAL ---
    "Rash", "VesicleS", "Itching (Pruritus)", "Lesion", "Ulcer",
    "Skin Discoloration", "Urticaria (Hives)", "Petechiae/Purpura",

    # --- MUSCULOSKELETAL ---
    "Joint Pain", "Joint Swelling", "Joint Stiffness", "Back Pain",
    "Muscle Ache (Myalgia)", "Muscle Cramps", "Limited Range of Motion",

    # --- GENITOURINARY (GU) & REPRODUCTIVE ---
    "Dysuria", "Hematuria", "Polyuria", "Oliguria", "Anuria", 
    "Urinary Frequency", "Urinary Urgency", "Urinary Incontinence",
    "Vaginal Discharge", "Penile Discharge", "Pelvic Pain",
    "Amenorrhea", "Menorrhagia", "Pregnancy Test",
    "Intermenstrual Bleeding", "Scrotal Pain/Swelling",

    # --- SPECIAL SENSES (ENT & OPHTHALMOLOGY) ---
    "Vision Loss", "Blurred Vision", "Photophobia", "Eye Redness", "Eye Pain",
    "Hearing Loss", "Tinnitus", "Ear Pain",

    # --- PEDIATRIC SPECIFIC ---
    "Excessive Crying", "Inconsolability", "Poor Feeding", "dehydration", 
    "Sunken Fontanelle", "Bulging Fontanelle", "Decreased Urine Output",

    # --- MENTAL HEALTH & BEHAVIORAL ---
    "Anxiety", "Panic Attack", "Low Mood", "Anhedonia", "Suicidal Ideation",
    "Self-Harm Urge", "Hallucinations", "Paranoia", "Insomnia", "Hypersomnia",

    # --- TRAUMA & EMERGENCY RED FLAGS ---
    "Active Bleeding", "Burn", "Poisoning Ingestion", "Choking", 
    "Anaphylaxis", "Snake/Insect Bite",
    
    "others"
]

class SymptomNature(BaseModel):
    """Standardized structure to describe the nature of any symptom (SOCRATES/OPQRST)"""
    name: StandardSymptom
    category: Union[SymptomCategory, SpecialSenses]
    severity: Severity
    frequency: Frequency
    duration: int
    character: Optional[Dict] = Field(None, description="nature/quality of symptom")
    progression: Literal["Improving", "Worsening", "Stable"] = "Stable"

#### NPI STATUS AND COMPLIANCE LEVEL
class NPIStatus(Enum):
    UP_TO_DATE = "Up to Date"
    PARTIAL = "Partially Vaccinated"
    UNVACCINATED = "Unvaccinated"
    UNKNOWN = "Unknown"

class ComplianceLevel(Enum):
    FULL = "Full Adherence"
    PARTIAL = "Occasional Missed Doses"
    MODERATE = "Missed 1-2 Doses in a week"
    SEVERE = "Missed 3+ Doses in a week"
    NON_COMPLIANT = "Stopped Medication"
    SELF_MEDICATING = "Taking without prescription"

class OtherDrug(BaseModel):
    name: str = Field(..., description="Name or class of the drug when not in DrugClass.")


class DrugClass(Enum):
    # --- Antimicrobials & Anti-infectives ---
    ANTIBIOTIC = "Antibiotic"
    ANTIVIRAL = "Antiviral"
    ANTIFUNGAL = "Antifungal"
    ANTIPARASITIC = "Antiparasitic"
    
    # --- Analgesics & Anti-inflammatories ---
    ANALGESIC = "Analgesic"
    NSAID = "NSAID"
    OPIOID = "Opioid"
    ANTIPYRETIC = "Antipyretic"
    CORTICOSTEROID = "Corticosteroid"
    
    # --- Cardiovascular System ---
    ANTIHYPERTENSIVE = "Antihypertensive"
    DIURETIC = "Diuretic"
    STATIN = "Antihyperlipidemic"
    ANTICOAGULANT = "Anticoagulant"
    ANTIPLATELET = "Antiplatelet"
    ANTIARRHYTHMIC = "Antiarrhythmic"
    
    # --- Central Nervous System & Psychiatry ---
    ANTIDEPRESSANT = "Antidepressant"
    ANTIPSYCHOTIC = "Antipsychotic"
    ANXIOLYTIC = "Anxiolytic"
    ANTICONVULSANT = "Anticonvulsant"
    CNS_STIMULANT = "CNS Stimulant"
    
    # --- Respiratory System ---
    BRONCHODILATOR = "Bronchodilator"
    ANTIHISTAMINE = "Antihistamine"
    ANTITUSSIVE_EXPECTORANT = "Antitussive / Expectorant"
    
    # --- Gastrointestinal System ---
    PROTON_PUMP_INHIBITOR = "Proton Pump Inhibitor (PPI)"
    H2_BLOCKER = "H2 Blocker / Antacid"
    LAXATIVE_ANTIDIARRHEAL = "Laxative / Antidiarrheal"
    ANTIEMETIC = "Antiemetic"
    
    # --- Endocrine & Metabolic System ---
    ANTIDIABETIC = "Antidiabetic"
    THYROID_HORMONE = "Thyroid Hormone / Anti-thyroid"
    CONTRACEPTIVE = "Contraceptive"
    OSTEOPOROSIS_AGENTS = "Osteoporosis Medication"
    
    # --- Oncology & Immunology ---
    ANTINEOPLASTIC = "Antineoplastic / Chemotherapy"
    IMMUNOSUPPRESSANT = "Immunosuppressant"
    BIOLOGIC_DMARD = "Biologic / DMARD"
    
    # --- Anesthetics & Neuromuscular ---
    ANESTHETIC = "Anesthetic (Local / General)"
    MUSCLE_RELAXANT = "Muscle Relaxant"
    
    # --- Miscellaneous / Preventive ---
    VACCINE = "Vaccine / Immunologic"
    VITAMIN_MINERAL = "Vitamin / Mineral / Supplement"
    
    # --- Catch-all ---
    OTHER = "Others"

class Immunization(BaseModel):
    vaccine: str = Field(..., description="Standardized vaccine name.")
    npi_status: NPIStatus
    missing_vaccines: List[str] = []

class PharmacologyProfile(BaseModel):
    drug: str = Field(..., description="Standardized drug name.")
    drug_class: DrugClass
    
    dosage: str = Field(..., description="Standardized dosage of the drug.")
    # Antibiotic Use (Crucial for AMR tracking)
    purpose_of_drug: Optional[str] = None
    n_doses_taken: int
    course_completed: Optional[bool] = None
    
    # Drug Compliance
    compliance_status: ComplianceLevel
    reason_for_non_compliance: Optional[Literal[
        "Cost", "Side Effects", "Forgetfulness", "Feeling Better", "Religious/Cultural Reasons"
    ]] = None
    
    # Drug Safety
    side_effects_reported: List[str] = []
    drug_herb_interaction_risk: bool = Field(False, description="User mentioned taking traditional herbs with clinical meds")
    traditional_medicine_used: List[str] = [] # e.g., ["Agbo", "Moringa"]
    
class PharmacologyProfiles(BaseModel):
    pharmacology_profiles: List[PharmacologyProfile]
    
#### MENTAL HEALTH CRISIS
class MentalHealthCrisis(BaseModel):
    primary_distress: Literal[
        "Depressive Mood", "Anxiety/Panic", "Psychosis/Hallucinations", 
        "Substance Abuse", "Grief/Trauma", "Burnout"
    ]
    duration_of_distress: str
    risk_indicators: List[Literal[
        "Suicidal Ideation", "Self-Harm", "Harm to Others", 
        "Inability to care for self", "Sleep Disturbance", "Social Withdrawal"
    ]]
    urgency: Literal["Routine", "Urgent", "Crisis/Emergency"]
    stigma_barrier: bool = Field(False, description="User expressed fear of family/community knowing")


OutcomeReferral = Literal[
    "Referral Given",
    "Self-care Guide",
    "Education Provided",
    "Escalated to Human",
]


class VisitFollowup(BaseModel):
    """Whether the AI recommended care and whether the user followed through."""
    model_config = ConfigDict(extra="forbid")
    recommended: Optional[bool] = Field(
        None,
        description="AI recommended hospital visit or in-house appointment booking.",
    )
    asked_visited: Optional[bool] = Field(
        None,
        description="AI asked whether the user visited after that recommendation.",
    )
    user_visited: Optional[bool] = Field(
        None,
        description="User visited a clinic or hospital after the recommendation.",
    )


class AnalysisSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinical_category: ClinicalTaxonomy
    intent: IntentPerformance

    pharmacology_profiles: Optional[PharmacologyProfiles] = None
    immunization_profiles: Optional[List[Immunization]] = None
    mental_health_profiles: Optional[List[MentalHealthCrisis]] = None
    suspected_condition: Optional[List[str]] = Field(
        "", description="Standardized condition Name (According to ICD-11) in descending order of likelihood (max 3 conditions). Don't add the icd code."
    )
    symptoms_reported: Optional[List[SymptomNature]] = Field(default_factory=list)
    symtoms_tier: Literal["Immediate", "Weekly", "Monthly"] = Field(..., description="Tier of syndrome (collection of symptoms) reported by the user (Immediate: symptoms must be reported within 24 hours e.g lassa fever, cholera, polio; Weekly: symptoms must be reported within 1 week e.g malnutrition, rabies, dysentery; Monthly: symptoms must be reported within 1 month).")
    urgency_level: Optional[UrgencyLevel] = None
    barriers: List[SDoHBarrier] = Field(default_factory=list)
    cultural_tags: Optional[CulturalTag] = None
    outcome_referral: OutcomeReferral
    visit_followup: Optional[VisitFollowup] = None
    literacy_score: int = Field(
        ge=1,
        le=5,
        description="user literacy level:1–5.",
    )


class HealthBeliefOrientation(str, Enum):
    CLINICAL_ONLY = "Purely Clinical"
    INTEGRATIVE = "Integrative (Clinical + Traditional/Herbal)"
    FAITH_BASED = "Faith-based/Spiritual emphasis"
    COMMUNITY_DRIVEN = "Reliance on peer/community advice"

class DecisionAuthority(str, Enum):
    INDIVIDUAL = "Individual - User makes independent health decisions."
    SPOUSAL = "Spousal - Decisions are made jointly with a spouse."
    FAMILY_ELDER = "Family/Elder - Decisions require input/approval from parents or elders."
    COMMUNAL = "Communal/Peer - Decisions are heavily influenced by community/peer feedback."
    RELIGIOUS_LEADER = "Religious/Spiritual Leader - Decisions are mediated by spiritual guidance."
    COLLECTIVE = "Collective - Decisions are made by a family unit or group."


class CulturalNotes(BaseModel):
    # Demographics & Context
    primary_language: str = Field(..., description="Language(s) user prefers to speak/code-switch in.")
    residency: Optional[str] = Field(None, description="Urban, semi-urban, or rural context affecting access/beliefs.")
    
    # Behavioral & Cognitive Markers
    health_belief_orientation: HealthBeliefOrientation = Field(..., description="The underlying framework the user uses to interpret health.")
    
    # Linguistic & Cultural Nuances
    colloquialisms_used: List[str] = Field(default_factory=list, description="Specific local idioms or Pidgin phrases used by the user.")
    cultural_taboos_or_sensitivities: List[str] = Field(default_factory=list, description="Identified topics that require extreme tact or avoidance.")
    
    # Practical Application
    decision_making_authority: Optional[DecisionAuthority] = Field(None, description="The primary influence or authority behind the user's health decisions.")
    local_terminology: List[str] = Field(default_factory=list, description="Specific local idioms or phrases used by the user.")


class ConversationAnalysis(BaseModel):
    """Structured output shape for conversation analysis (persisted + OpenAI schema)."""

    model_config = ConfigDict(extra="forbid")
    topic_segments: List[AnalysisSegment] = Field(default_factory=list, description="Analysis for identified, independent segments of the conversation.")
    sdoh_profiles: List[SDoHProfile] = Field(default_factory=list, description="Profiles of the user's socioeconomic status.")
    cultural_notes: Optional[CulturalNotes] = None
    topics_enquired: List[str] = Field(
        default_factory=list, description="Topics the user inquired about."
    )
    diseases_enquired: List[str] = Field(
        default_factory=list, description="Diseases the user inquired about."
    )
    languages_used: Literal[str] = Field(
        default_factory=list, description="Languages (full name) the user spoke in."
    )

    @classmethod
    def llm_json_schema(cls) -> dict:
        """JSON Schema for Batch/Chat API (clinical fields only; no DB ids)."""
        schema = cls.model_json_schema()

        def make_schema_strict(s: Any) -> None:
            if not isinstance(s, dict):
                return

            if "$ref" in s:
                for k in list(s.keys()):
                    if k != "$ref":
                        del s[k]
                return

            if s.get("type") == "object":
                s["additionalProperties"] = False
                properties = s.get("properties", {})
                s["required"] = list(properties.keys())
                for key in properties:
                    make_schema_strict(properties[key])
            elif s.get("type") == "array":
                if "items" in s:
                    make_schema_strict(s["items"])

            for key in ("anyOf", "oneOf", "allOf"):
                if key in s and isinstance(s[key], list):
                    for sub in s[key]:
                        make_schema_strict(sub)

            if "$defs" in s and isinstance(s["$defs"], dict):
                for key in s["$defs"]:
                    make_schema_strict(s["$defs"][key])

        make_schema_strict(schema)
        return schema