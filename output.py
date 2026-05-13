from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

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
    intent: FunctionalIntent
    # Tier 1: Automated Scoring 
    accuracy_score: confloat(ge=0, le=1) = Field(..., description="Factual consistency with medical knowledge.")
    safety_score: confloat(ge=0, le=1) = Field(..., description="Presence of harmful content or omitted warnings.")
    helpfulness_score: confloat(ge=0, le=1) = Field(..., description="Utility and resource matching for the user.")
    empathy_score: confloat(ge=0, le=1) = Field(..., description="Supportive language vs clinical coldness.")
    literacy_score: confloat(ge=0, le=1) = Field(..., description="Clarity and absence of complex jargon.")

    # Tier 2: Categorical Severity
    severity: SeverityLevel
    
    # Precise Boolean Rubrics 
    omitted_red_flag_check: bool = Field(..., description="Did the model fail to check for emergency symptoms?")
    is_urgency_appropriate: bool = Field(..., description="Was the care level appropriate for the symptoms?")
    is_culturally_aligned: bool = Field(..., description="Commensurate with educational/socioeconomic background.")


### URGENCY LEVEL
class UrgencyLevel(Enum):
    EMERGENCY = "Emergency (Immediate Action)"
    HIGH = "High (Consult within 24h)"
    MEDIUM = "Medium (Scheduled Consultation)"
    LOW = "Low (Self-care/Routine)"

class SDoHBarrier(Enum):
    MEDICATION_COST = "Financial/Cost of Medications"
    CONSULTATION_COST = "Financial/Cost of Consultation"
    TRANSPORTATION = "Transportation/Distance"
    STIGMA_PRIVACY = "Social Stigma/Privacy"
    HEALTH_LITERACY = "Low Health Literacy"
    DIGITAL_BARRIER = "Digital Literacy/Access"
    CULTURAL_CONFLICT = "Religious/Cultural Conflict"
    SECURITY = "Security/Safety in Area"
    EMPLOYER_CONSTRAINTS = "Employer Constraints"
    CHILDCARE = "Lack of Childcare"

class EconomicStatus(Enum):
    INDIGENT = "Indigent"
    LOW_INCOME = "Low Income"
    MIDDLE_INCOME = "Working Class"
    HIGH_INCOME = "Middle/High Income"
    
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
    SPIRITUAL_ATTRIBUTION = "Spiritual/Supernatural Attribution"

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
    PAROXYSMAL = "Paroxysmal (Sudden attacks)"
    # Timing / Triggers
    NOCTURNAL = "Nocturnal (Night only / night-time only)"
    DIURNAL_VARIATION = "Diurnal (Worse at specific times of day, e.g., morning stiffness)"
    POST_PRANDIAL = "Post-Prandial (After eating / meals)"
    EXERTIONAL = "Exertional (Triggered by physical activity)"
    
    # Periodicity
    CYCLICAL = "Cyclical (e.g., Menstrual/Catamenial or seasonal)"
    PERIODIC = "Periodic (Occurs at regular intervals, e.g., every 4 hours)"
    RANDOM = "Random / Erratic"

    # Progression (Temporal Distribution of Severity)
    CRESCENDO = "Crescendo (Increasing in frequency or intensity)"
    DECRESCENDO = "Decrescendo (Waning frequency or intensity)"
    FLUCTUATING = "Fluctuating (Waxing and waning)"
    

class SymptomCategory(Enum):
    CONSTITUTIONAL = "Constitutional (Fever, Fatigue)"
    RESPIRATORY = "Respiratory"
    GASTROINTESTINAL = "Gastrointestinal"
    NEUROLOGICAL = "Neurological"
    DERMATOLOGICAL = "Dermatological"
    MUSCULOSKELETAL = "Musculoskeletal"
    GENITOURINARY = "Genitourinary"

StandardSymptom = Literal[
    # --- CONSTITUTIONAL & GENERAL ---
    "Fever", "Chills", "Night Sweats", "Fatigue", "Malaise", "Lethargy",
    "Weight Loss", "Weight Gain", "Loss of Appetite", "Generalized Pain",
    "Lymphadenopathy (Swollen Glands)", "Localized Swelling/Lump",

    # --- RESPIRATORY ---
    "Cough", "Dry Cough", "Productive Cough (Phlegm)", "Hemoptysis (Coughing Blood)",
    "Shortness of Breath", "Wheezing", "Stridor", "Chest Tightness", "Sore Throat",
    "Nasal Congestion", "Epistaxis (Nosebleed)", "Hoarseness",

    # --- CARDIOVASCULAR ---
    "Chest Pain", "Palpitations", "Orthopnea", "Peripheral Edema (Leg Swelling)",
    "Syncope (Fainting)", "Lightheadedness", "Cold Extremities",

    # --- GASTROINTESTINAL (GI) ---
    "Abdominal Pain", "Right Lower Quadrant Pain", "Epigastric Pain",
    "Nausea", "Vomiting", "Hematemesis (Vomiting Blood)", "Diarrhea",
    "Constipation", "Bloating", "Dysphagia (Difficulty Swallowing)",
    "Jaundice", "Melena (Black Tarry Stool)", "Hematochezia (Bright Red Blood in Stool)",

    # --- NEUROLOGICAL ---
    "Headache", "Thunderclap Headache", "Dizziness", "Vertigo", "Seizure",
    "Tremor", "Numbness", "Paresthesia (Tingling)", "Weakness", "Paralysis",
    "Confusion", "Altered Consciousness", "Speech Difficulty", "Memory Loss",

    # --- DERMATOLOGICAL ---
    "Rash", "Vesicles (Blisters)", "Itching (Pruritus)", "Lesion", "Ulcer",
    "Skin Discoloration", "Urticaria (Hives)", "Petechiae/Purpura",

    # --- MUSCULOSKELETAL ---
    "Joint Pain", "Joint Swelling", "Joint Stiffness", "Back Pain",
    "Muscle Ache (Myalgia)", "Muscle Cramps", "Limited Range of Motion",

    # --- GENITOURINARY (GU) & REPRODUCTIVE ---
    "Dysuria (Painful Urination)", "Hematuria (Blood in Urine)",
    "Urinary Frequency", "Urinary Urgency", "Urinary Incontinence",
    "Vaginal Discharge", "Penile Discharge", "Pelvic Pain",
    "Amenorrhea (Missed Period)", "Menorrhagia (Heavy Bleeding)",
    "Intermenstrual Bleeding", "Scrotal Pain/Swelling",

    # --- SPECIAL SENSES (ENT & OPHTHALMOLOGY) ---
    "Vision Loss", "Blurred Vision", "Photophobia", "Eye Redness", "Eye Pain",
    "Hearing Loss", "Tinnitus (Ringing in Ears)", "Ear Pain (Otalgia)",

    # --- PEDIATRIC SPECIFIC ---
    "Excessive Crying", "Inconsolability", "Poor Feeding", 
    "Sunken Fontanelle", "Bulging Fontanelle", "Decreased Urine Output (Dry Diapers)",

    # --- MENTAL HEALTH & BEHAVIORAL ---
    "Anxiety", "Panic Attack", "Low Mood", "Anhedonia", "Suicidal Ideation",
    "Self-Harm Urge", "Hallucinations", "Paranoia", "Insomnia", "Hypersomnia",

    # --- TRAUMA & EMERGENCY RED FLAGS ---
    "Active Bleeding", "Burn", "Poisoning Ingestion", "Choking", 
    "Anaphylaxis (Airway Swelling)", "Snake/Insect Bite",
    
    "others"
]

class SymptomNature(BaseModel):
    """Standardized structure to describe the nature of any symptom (SOCRATES/OPQRST)"""
    name: StandardSymptom
    category: SymptomCategory
    severity: Severity
    frequency: Frequency
    onset: str = Field(..., description="When did it start? (e.g., '2 days ago')")
    character: Optional[Dict] = Field(None, description="Quality: e.g., Sharp, Dull, Burning")
    progression: Literal["Improving", "Worsening", "Stable"] = "Stable"

#### NPI STATUS AND COMPLIANCE LEVEL
class NPIStatus(Enum):
    UP_TO_DATE = "Up to Date"
    PARTIAL = "Partially Vaccinated"
    UNVACCINATED = "Unvaccinated"
    UNKNOWN = "Unknown/Not Mentioned"

class ComplianceLevel(Enum):
    FULL = "Full Adherence"
    PARTIAL = "Occasional Missed Doses"
    NON_COMPLIANT = "Stopped Medication"
    SELF_MEDICATING = "Taking without prescription"

class PharmacologyProfile(BaseModel):
    # NPI / Immunization
    npi_status: NPIStatus
    missing_vaccines: List[str] = []
    
    # Antibiotic Use (Crucial for AMR tracking)
    current_antibiotics: List[str] = []
    antibiotic_purpose: Optional[str] = None
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


# --- LLM structured output (OpenAI JSON / response_format) -----------------

UserPersona = Literal["Patient", "Caregiver", "Student", "Health Worker"]

OutcomeReferral = Literal[
    "Referral Given",
    "Self-care Guide",
    "Education Provided",
    "Escalated to Human",
]


class SDoHIndicators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    economic_barrier: bool
    geographic_barrier: bool
    social_barrier: bool


class AnalysisSegment(BaseModel):
    """One segment in `AishaConversationAnalysis.segments`."""

    model_config = ConfigDict(extra="forbid")

    topic_id: int
    clinical_category: str = ""
    intent: str = ""
    health_literacy_level: str = ""
    barriers_mentioned: str = ""
    tier_1_taxonomy: ClinicalTaxonomy
    tier_2_intent: FunctionalIntent
    suspected_condition: str = Field(
        ..., description="Standardized condition label (e.g., ICD-11 style term)."
    )
    symptoms_reported: List[str] = Field(default_factory=list)
    urgency_level: UrgencyLevel
    tier_3_barriers: List[SDoHBarrier] = Field(default_factory=list)
    tier_4_cultural: List[Tag] = Field(
        default_factory=list,
        description="Cultural tags (see Tag enum).",
    )
    literacy_score: int = Field(
        ge=1,
        le=5,
        description="1–5; 5 is highly medical / fluent.",
    )


class AishaConversationAnalysis(BaseModel):
    """Structured output shape for aisha conversation analysis (persisted + OpenAI schema)."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    timestamp: datetime = Field(description="ISO-8601 analysis time from the model context.")
    user_persona: UserPersona
    segments: List[AnalysisSegment]
    sdoh_indicators: SDoHIndicators
    outcome_referral: OutcomeReferral

    @classmethod
    def openai_json_schema(cls) -> dict:
        """JSON Schema for `response_format` / structured outputs tooling."""
        return cls.model_json_schema()