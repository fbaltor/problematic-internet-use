from enums import FeatureGroup


DATASET_PATH = "data/problematic-internet-use/"
ID_COL = "id"
LABEL_COL = "sii"
GROUP_NAME_TO_FEATURE_GROUP = {
    "Demographics": FeatureGroup.DEMOGRAPHIC,
    "Children's Global Assessment Scale": FeatureGroup.ASSESSMENT_SCALE,
    "Physical Measures": FeatureGroup.PHYSICAL_MEASURES,
    "FitnessGram Vitals and Treadmill": FeatureGroup.FITNESSGRAM_VITALS,
    "FitnessGram Child": FeatureGroup.FITNESSGRAM_CHILD,
    "Bio-electric Impedance Analysis": FeatureGroup.IMPEDANCE,
    "Physical Activity Questionnaire (Adolescents)": FeatureGroup.PHYSICAL_ACTIVITY_QA_ADOLECENT,
    "Physical Activity Questionnaire (Children)": FeatureGroup.PHYSICAL_ACTIVITY_QA_CHILD,
    "Parent-Child Internet Addiction Test": FeatureGroup.INTERNET_ADDICTION_TEST,
    "Sleep Disturbance Scale": FeatureGroup.SLEEP_DISTURBANCE,
    "Internet Use": FeatureGroup.INTERNET_USAGE,
}
