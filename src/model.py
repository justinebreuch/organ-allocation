from enum import Enum, auto
"""
Renamed columns from datasources.
"""
class Column(Enum):
    DESCRIPTION = auto()
    DONOR_ID = auto()
    DONOR_TYPE = auto()
    END_DATE = auto()
    INIT_MELD_PELD_LAB_SCORE = auto()
    ORGAN_RECOVERY_DATE = auto()
    ORGAN_TRANSPLANT_ID = auto()
    RECIPIENT_STATUS = auto()
    TRANSPLANT_DATE = auto()
    VARIABLE_NAME = auto()
    WAITLIST_ID = auto()
    WAITLIST_REGISTRATION_DATE = auto()
