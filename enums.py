from enum import Enum

class SII(Enum):
    SII_NONE = 0
    SII_MILD = 1
    SII_MODERATE = 2
    SII_SEVERE = 3
    SII_UNKNOWN = 4


class Gender(Enum):
    MALE = 0
    FEMALE = 1


class FeatureGroup(Enum):
    DEMOGRAPHIC = "DEMOGRAPHIC"  # Information about age and sex of participants.
    INTERNET_USAGE = (
        "INTERNET_USAGE"  # Number of hours of using computer/internet per day.
    )
    ASSESSMENT_SCALE = "ASSESSMENT_SCALE"  # Numeric scale used by mental health clinicians to rate the general functioning of youths under the age of 18.
    PHYSICAL_MEASURES = "PHYSICAL_MEASURES"  # Collection of blood pressure, heart rate, height, weight and waist, and hip measurements.
    FITNESSGRAM_VITALS = "FITNESSGRAM_VITALS"  # Measurements of cardiovascular fitness assessed using the NHANES treadmill protocol.
    FITNESSGRAM_CHILD = "FITNESSGRAM_CHILD"  #  Health related physical fitness assessment measuring five different parameters including aerobic capacity, muscular strength, muscular endurance, flexibility, and body composition.
    IMPEDANCE = "IMPEDANCE"  # Measure of key body composition elements, including BMI, fat, muscle, and water content.
    PHYSICAL_ACTIVITY_QA_CHILD = "PHYSICAL_ACTIVITY_QA_CHILD"  # Information about children's participation in vigorous activities over the last 7 days.
    PHYSICAL_ACTIVITY_QA_ADOLECENT = "PHYSICAL_ACTIVITY_QA_ADOLECENT"  # Information about children's participation in vigorous activities over the last 7 days.
    SLEEP_DISTURBANCE = (
        "SLEEP_DISTURBANCE"  # Scale to categorize sleep disorders in children.
    )
    INTERNET_ADDICTION_TEST = "INTERNET_ADDICTION_TEST"  # 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency.
