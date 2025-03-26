from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum


############################################################
# 1) ENUMERATIONS
############################################################

class DesignForDataCollection(str, Enum):
    COHORT = "Cohort"
    CASE_CONTROL = "Case-control"
    EHR_RETROSPECTIVE = "EHR retrospective"
    EHR_PROSPECTIVE = "EHR prospective"
    CROSS_SECTIONAL = "Cross-sectional"
    RCT = "Randomized Controlled Trial"
    LONGITUDINAL = "Longitudinal"
    CASE_SERIES = "Case-series"
    MULTI_CENTER = "Multi-center"
    NATIONAL_REGISTRY = "National Registry"
    DATA_SHARING = "Data sharing"
    OTHER = "Other"

class ModelInputInstanceType(str, Enum):
    IMAGE = "Image"
    VIDEO = "Video"
    TABULAR = "Tabular"
    GENOMIC = "Genomic"
    TEXT = "Text"
    SENSOR = "Sensor"
    AUDIO = "Audio"
    TIME_SERIES = "Time-series"
    OTHER = "Other"

class TimeUnit(str, Enum):
    MINUTES = "minutes"
    DAYS = "days"
    MONTHS = "months"
    YEARS = "years"
    OTHER = "Other"

class ClinicalTask(str, Enum):
    DIAGNOSIS = "Diagnosis"
    TREATMENT_PLANNING = "Treatment_planning"
    SCREENING = "Screening"
    PROGNOSIS = "Prognosis"
    RISK_STRATIFICATION = "Risk stratification"
    MONITORING = "Monitoring"
    PREDICTIVE_ANALYSIS = "Predictive Analysis"
    POPULATION_HEALTH = "Population Health"
    OUTCOMES_RESEARCH = "Outcomes Research"
    MEDICATION_ADHERENCE = "Medication Adherence"
    OTHER = "Other"

class AIDataType(str, Enum):
    IMAGE = "Image"
    VIDEO = "Video"
    TABULAR = "Tabular"
    TEXT = "Text"
    GENOMIC = "Genomic"
    SENSOR = "Wearable sensor"
    AUDIO = "Audio"
    TIME_SERIES = "Time-series"
    MULTIMODAL = "Multimodal"
    SEQUENCE = "Sequence"
    OTHER = "Other"

############################################################
# 2) DATASET & ENROLLMENT
############################################################

class Enrollment(BaseModel):
    """
    Defines how participants and data sources are included in the study.
    """
    target_population: Optional[List[str]] = Field(
        None,
        description="Which population/patients the AI model targets, e.g., 'patients with diabetes'."
    )
    target_disease: Optional[List[str]] = Field(
        None,
        description="Disease(s) the AI model focuses on, e.g. 'COVID-19', 'breast cancer'."
    )
    control_population: Optional[List[str]] = Field(
        None,
        description="Population used as a control or negative group."
    )
    cohort_definition: Optional[str] = Field(
        None,
        description="Inclusion/exclusion criteria for the entire cohort (patients or not)."
    )
    case_definition: Optional[str] = Field(
        None,
        description="How positive cases are defined/validated, i.e. 'PCR confirmed', 'histopathology', etc."
    )
    number_of_centers: Optional[int] = Field(
        None,
        description="Number of centers/sites from which data was collected."
    )
    data_source_countries: Optional[List[str]] = Field(
        None,
        description="Country or list of countries where the data is sourced."
    )
    design_for_data_collection: Optional[DesignForDataCollection] = Field(
        None,
        description="Design for data collection, e.g. 'Cohort', 'Case-control', etc."
    )
    design_for_data_collection_other: Optional[str] = Field(
        None,
        description="If design_for_data_collection == 'Other', specify details here."
    )

############################################################
# 3) SAMPLE SIZE
############################################################

class SampleSize(BaseModel):
    """
    Information about sample sizes for patients, controls, etc.
    """
    cohort_count: Optional[int] = Field(
        None,
        description="Total number of people in the dataset (patients + non-patients)."
    )
    patient_count: Optional[int] = Field(
        None,
        description="Number of positive (patient) cases in the dataset."
    )
    model_input_instance_type: Optional[ModelInputInstanceType] = Field(
        None,
        description="The type of input instances used by the model (image, video, tabular...)."
    )
    model_input_instance_type_custom: Optional[str] = Field(
        None,
        description="If model_input_instance_type is 'Other', specify the custom type here."
    )
    number_of_all_instances: Optional[int] = Field(
        None,
        description="Number of total data instances (e.g., total images, total records)."
    )
    number_of_positive_instances: Optional[int] = Field(
        None,
        description="Number of positive instances among all data instances."
    )

############################################################
# 4) DATA TYPE
############################################################

class DataTypeInfo(BaseModel):
    """
    Data type from biomedical perspective and AI perspective.
    """
    medical_data_type: Optional[str] = Field(
        None,
        description="e.g. 'CT scan', 'MRI', 'EHR clinical notes', 'genomics'."
    )
    ai_data_type: Optional[AIDataType] = Field(
        None,
        description="Data type from an AI perspective, e.g. 'Image', 'Video', 'Text', etc."
    )
    ai_data_type_custom: Optional[str] = Field(
        None,
        description="If ai_data_type is 'Other', specify the custom type here."
    )

############################################################
# 5) TASK
############################################################

class OutcomeTimeHorizon(BaseModel):
    """
    The interval between input data and the outcome event.
    """
    interval: Optional[int] = Field(
        None,
        description="The numeric length of the interval (e.g., '365' for one year)."
    )
    unit: Optional[TimeUnit] = Field(
        None,
        description="Unit of time for the interval (days, months, years, etc.)."
    )
    other_unit_description: Optional[str] = Field(
        None,
        description="If 'OTHER', specify the unit here."
    )

class TaskInfo(BaseModel):
    """
    AI model's clinical/technical tasks and predicted outcome.
    """
    prediction_task: Optional[str] = Field(
        None,
        description="Aim of the model in general terms, e.g. 'predict pneumonia from chest X-ray'."
    )
    clinical_task: Optional[ClinicalTask] = Field(
        None,
        description="e.g. 'Diagnosis', 'Treatment_planning', 'Screening', etc."
    )
    clinical_task_other: Optional[str] = Field(
        None,
        description="If clinical_task is 'Other', specify here."
    )
    outcome_to_predict: Optional[str] = Field(
        None,
        description="Briefly describe the outcome, e.g. 'Mortality', 'Cancer occurrence'."
    )
    outcome_time_horizon: Optional[OutcomeTimeHorizon] = None

############################################################
# 6) AI MODEL
############################################################

class AIModelInfo(BaseModel):
    """
    Captures details about the AI approach used.
    """
    ai_field: Optional[str] = Field(
        None,
        description="Main field of AI for the project (e.g. 'deep learning', 'reinforcement learning'). Avoid abbreviations."
    )
    algorithms: Optional[List[str]] = Field(
        None,
        description="List of algorithms used in this study, spelled out (e.g. 'Convolutional Neural Network', 'Random Forest')."
    )
    best_performing_model: Optional[str] = Field(
        None,
        description="Which model had the best performance among the algorithms tested."
    )

############################################################
# 7) EVALUATION
############################################################

class PerformanceMetric(BaseModel):
    """
    Single performance metric (e.g. AUC, accuracy).
    """
    metric_name: Optional[str] = Field(
        None,
        description="Metric name, e.g. 'AUC', 'Accuracy', 'F1-score'."
    )
    value_percent: Optional[float] = Field(
        None,
        description="Value from 0 to 100, e.g. 92.5. Convert if needed (0.925 => 92.5)."
    )

class ModelPerformance(BaseModel):
    """
    Performance results for a single model.
    """
    model_name: Optional[str] = Field(
        None,
        description="Full name of the model (no abbreviations)."
    )
    performance: Optional[List[PerformanceMetric]] = Field(
        None,
        description="List of metrics reported for this model."
    )

class Evaluation(BaseModel):
    """
    Overall evaluation approach, metrics, and external validation.
    """
    validation_method: Optional[str] = Field(
        None,
        description="e.g. '10-fold cross validation', 'hold-out test set', etc."
    )
    reported_metrics: Optional[List[str]] = Field(
        None,
        description="High-level list of metrics (e.g. 'AUC, F1, Precision, Recall')."
    )
    external_validation: Optional[bool] = Field(
        None,
        description="True if they performed external validation on a separate dataset."
    )
    all_models_performance: Optional[List[ModelPerformance]] = Field(
        None,
        description="Performance details for each model tested."
    )


############################################################
# 8) ENUMERATIONS
############################################################

class TRIPODField(str, Enum):
    TITLE_MODEL_DEVELOPMENT = "Title - mention of developing or evaluating prediction model"
    TITLE_TARGET_POPULATION = "Title - mention target population"
    TITLE_OUTCOME_TO_PREDICT = "Title - mention outcome to predict"
    BACKGROUND_HEALTHCARE_CONTEXT = "Background - a brief explanation of the healthcare context and rationale for developing or evaluating"
    OBJECTIVE_STUDY_TYPE = "Objective - describe object and whether the study is development, evaluation, or both"
    METHOD_SOURCE_OF_DATA = "Method - Source of data"
    METHOD_ELIGIBILITY_CRITERIA = "Method - eligibility criteria"
    METHOD_OUTCOME_PREDICTED = "Method - outcome to be predicted"
    METHOD_OUTCOME_TIME_HORIZON = "Method - outcome time horizon if relevant (prognostic)"
    METHOD_MODEL_TYPE = "Method - Model type"
    METHOD_MODEL_BUILDING_STEPS = "Method - model building steps"
    METHOD_VALIDATION_METHOD = "Method - method for validation"
    METHOD_MEASURES_FOR_MODEL_PERFORMANCE = "Method - measures used to assess model performance"
    RESULT_PARTICIPANTS_OUTCOME_EVENTS = "Result - number of participants and outcome events"
    RESULT_PREDICTORS_IN_FINAL_MODEL = "Result - summary of predictors in the final model"
    RESULT_MODEL_PERFORMANCE_ESTIMATE = "Result - model performance estimate"
    RESULT_MODEL_PERFORMANCE_CI = "Result - model performance confidence interval"
    REGISTRATION_NUMBER_REPOSITORY = "Registration - give the registration number or repository"

class TimeUnit(str, Enum):
    MINUTES = "minutes"
    DAYS = "days"
    MONTHS = "months"
    YEARS = "years"
    OTHER = "Other"


############################################################
# 9) AI RELEVANT FIELDS
############################################################

class AIRelevanceCheck(BaseModel):
    """
    This model handles checking whether the article is related to AI in medicine.
    """
    is_related_to_ai_in_medicine: bool = Field(
        ...,
        description="Is this article related to Artificial Intelligence in Medicine?"
    )
    exclusion_reason: Optional[str] = Field(
        None,
        description="Reason for exclusion if the paper is not related to AI in Medicine."
    )


############################################################
# 10) TRIPOD-AI CHECKLIST
############################################################

class TRIPODChecklist(BaseModel):
    """
    This model tracks the TRIPOD-AI checklist items for the paper.
    Each item corresponds to a boolean indicating whether it's mentioned in the abstract.
    """
    title_model_development: bool = Field(..., description="Title - mention of developing or evaluating prediction model")
    title_target_population: bool = Field(..., description="Title - mention target population")
    title_outcome_to_predict: bool = Field(..., description="Title - mention outcome to predict")
    background_healthcare_context: bool = Field(..., description="Background - a brief explanation of the healthcare context and rationale for developing or evaluating")
    objective_study_type: bool = Field(..., description="Objective - describe object and whether the study is development, evaluation or both")
    method_source_of_data: bool = Field(..., description="Method - Source of data")
    method_eligibility_criteria: bool = Field(..., description="Method - eligibility criteria")
    method_outcome_predicted: bool = Field(..., description="Method - outcome to be predicted")
    method_outcome_time_horizon: bool = Field(..., description="Method - outcome time horizon if relevant (prognostic)")
    method_model_type: bool = Field(..., description="Method - Model type")
    method_model_building_steps: bool = Field(..., description="Method - model building steps")
    method_validation_method: bool = Field(..., description="Method - method for validation")
    method_measures_for_model_performance: bool = Field(..., description="Method - measures used to assess model performance")
    result_participants_outcome_events: bool = Field(..., description="Result - number of participants and outcome events")
    result_predictors_in_final_model: bool = Field(..., description="Result - summary of predictors in the final model")
    result_model_performance_estimate: bool = Field(..., description="Result - model performance estimate")
    result_model_performance_ci: bool = Field(..., description="Result - model performance confidence interval")
    registration_number_repository: bool = Field(..., description="Registration - give the registration number or repository")


############################################################
# 4) FULL PUBLISHING MODEL FOR THE AI STUDY
############################################################

class AIStudy(BaseModel):
    """
    This is the top-level model that captures all necessary information 
    for analyzing an AI in medicine paper, including its relevance to AI,
    and the TRIPOD-AI checklist details.
    """
    ai_relevance_check: AIRelevanceCheck
    tripod_checklist: Optional[TRIPODChecklist] = None  # Optional, filled only if the paper is related to AI
    enrollment: Optional[Enrollment] = None
    sample_size: Optional[SampleSize] = None
    data_type_info: Optional[DataTypeInfo] = None
    task_info: Optional[TaskInfo] = None
    ai_model_info: Optional[AIModelInfo] = None
    evaluation: Optional[Evaluation] = None

    class Config:
        # Example output to illustrate typical usage
        schema_extra = {
            "example": {
                "ai_relevance_check": {
                    "is_related_to_ai_in_medicine": True,
                    "exclusion_reason": None
                },
                "tripod_checklist": {
                    "title_model_development": True,
                    "title_target_population": True,
                    "title_outcome_to_predict": True,
                    "background_healthcare_context": True,
                    "objective_study_type": True,
                    "method_source_of_data": True,
                    "method_eligibility_criteria": True,
                    "method_outcome_predicted": True,
                    "method_outcome_time_horizon": True,
                    "method_model_type": True,
                    "method_model_building_steps": True,
                    "method_validation_method": True,
                    "method_measures_for_model_performance": True,
                    "result_participants_outcome_events": True,
                    "result_predictors_in_final_model": True,
                    "result_model_performance_estimate": True,
                    "result_model_performance_ci": True,
                    "registration_number_repository": True
                },
                "enrollment": {
                    "target_population": ["patients with diabetes"],
                    "target_disease": ["Type 2 diabetes"],
                    "control_population": ["healthy adults"],
                    "cohort_definition": "Adults aged 40-60 with diagnosed Type 2 diabetes.",
                    "case_definition": "Diagnosed based on blood sugar levels and A1C test results.",
                    "number_of_centers": 2,
                    "data_source_countries": ["USA", "Canada"],
                    "design_for_data_collection": "Cohort"
                },
                "sample_size": {
                    "cohort_count": 1000,
                    "patient_count": 500,
                    "model_input_instance_type": "IMAGE",
                    "number_of_all_instances": 5000,
                    "number_of_positive_instances": 2500
                },
                "data_type_info": {
                    "medical_data_type": "X-ray images",
                    "ai_data_type": "Image"
                },
                "task_info": {
                    "prediction_task": "Classify diabetic retinopathy",
                    "clinical_task": "Diagnosis",
                    "outcome_to_predict": "Presence of diabetic retinopathy",
                    "outcome_time_horizon": {
                        "interval": 365,
                        "unit": "days"
                    }
                },
                "ai_model_info": {
                    "ai_field": "Deep learning",
                    "algorithms": ["Convolutional Neural Network", "Logistic Regression"],
                    "best_performing_model": "Convolutional Neural Network"
                },
                "evaluation": {
                    "validation_method": "Cross-validation",
                    "reported_metrics": ["AUC", "Accuracy", "F1-score"],
                    "external_validation": True,
                    "all_models_performance": [
                        {
                            "model_name": "Convolutional Neural Network",
                            "performance": [
                                {"metric_name": "AUC", "value_percent": 94.3},
                                {"metric_name": "Accuracy", "value_percent": 90.0}
                            ]
                        }
                    ]
                }
            },
            "example_for_not_relevant_paper": {
                "ai_relevance_check": {
                    "is_related_to_ai_in_medicine": False,
                    "exclusion_reason": "Not related to Artificial Intelligence in Medicine but related to analysis and epimiology of diabetes."
                },
                "tripod_checklist": None,
                "enrollment": None,
                "sample_size": None,
                "data_type_info": None,
                "task_info": None,
                "ai_model_info": None,
                "evaluation": None
            }
        }
