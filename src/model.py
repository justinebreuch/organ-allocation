from dataclasses import dataclass
from enum import Enum, auto


class Action(Enum):
    allocate = auto()
    skip = auto()


class Column(Enum):
    """
    Renamed columns from datasources.
    """
    DAYS_ON_WAITLIST = auto()
    DESCRIPTION = auto()
    DIAGNOSIS = auto()
    DIAGNOSIS_CODE = auto()
    DONOR_ADMISSION_DATE = auto()
    DONOR_AGE = auto()
    DONOR_BLOOD_TYPE = auto()
    DONOR_GENDER = auto()
    DONOR_ID = auto()
    DONOR_LIVER_QUALITY = auto()
    DONOR_RECIPIENT_BLOOD_TYPE_MATCH = auto()
    DONOR_TYPE = auto()
    END_DATE = auto()
    FOLLOWUP_DATE = auto()
    FOLLOWUP_ID = auto()
    FUNCTIONAL_STATUS = auto()
    FUNCTIONAL_CODE = auto()
    FUNCTIONAL_STATUS_AT_FOLLOW_UP = auto()
    FUNCTIONAL_STATUS_AT_REGISTRATION = auto()
    FUNCTIONAL_STATUS_AT_TRANSPLANT = auto()
    GRAFT_FUNCTIONING = auto()
    GRAFT_LIFESPAN = auto()
    INIT_MELD_OR_PELD = auto()
    INIT_MELD_PELD_LAB_SCORE = auto()
    FINAL_MELD_PELD_LAB_SCORE = auto()
    NUM_ACUTE_REJECTIONS = auto()
    ORGAN_RECOVERY_DATE = auto()
    ORGAN_TRANSPLANT_ID = auto()
    PATIENT_SURVIVAL_TIME = auto()
    REASON_REMOVED_WAITLIST = auto()
    RECIPIENT_AGE = auto()
    RECIPIENT_BLOOD_TYPE = auto()
    RECIPIENT_GENDER = auto()
    RECIPIENT_ID = auto()
    RECIPIENT_STATUS = auto()
    REJECTED_WITHIN_YEAR = auto()
    STATUS_AT_REGISTRATION = auto()
    TRANSPLANT_DATE = auto()
    VARIABLE_NAME = auto()
    WAITLIST_ID = auto()
    WAITLIST_REGISTRATION_DATE = auto()
    RECIPIENT_BLOOD_TYPE_AS_CODE = auto()
    DONOR_BLOOD_TYPE_AS_CODE = auto()

    @classmethod
    def from_code(cls, code: str):
        code_map = {
            'PX_STAT': cls.RECIPIENT_STATUS,
            'TRR_ID_CODE': cls.ORGAN_TRANSPLANT_ID,
            'TRR_FOL_ID_CODE': cls.FOLLOWUP_ID,
            'PX_STAT_DATE': cls.FOLLOWUP_DATE,
            'FUNC_STAT': cls.FUNCTIONAL_STATUS,
            'ACUTE_REJ_EPI': cls.NUM_ACUTE_REJECTIONS,
            'PT_CODE': cls.RECIPIENT_ID,
            'WL_ID_CODE': cls.WAITLIST_ID,
            'FUNC_STAT_TCR': cls.FUNCTIONAL_STATUS_AT_REGISTRATION,
            'FUNC_STAT_TRF': cls.FUNCTIONAL_STATUS_AT_FOLLOW_UP,
            'FUNC_STAT_TRR': cls.FUNCTIONAL_STATUS_AT_TRANSPLANT,
            'INIT_STAT': cls.STATUS_AT_REGISTRATION,
            'DIAG': cls.DIAGNOSIS,
            'PTIME': cls.PATIENT_SURVIVAL_TIME,
            'DAYSWAIT_CHRON': cls.DAYS_ON_WAITLIST,
            'AGE': cls.RECIPIENT_AGE,
            'AGE_DON': cls.DONOR_AGE,
            'TRTREJ1Y': cls.REJECTED_WITHIN_YEAR,
            'TX_DATE': cls.TRANSPLANT_DATE,
            'LI_BIOPSY': cls.DONOR_LIVER_QUALITY,
            'ABO': cls.RECIPIENT_BLOOD_TYPE,
            'ABO_DON': cls.DONOR_BLOOD_TYPE,
            'ABO_MAT': cls.DONOR_RECIPIENT_BLOOD_TYPE_MATCH,
            'GRF_STAT': cls.GRAFT_FUNCTIONING,
            'GTIME': cls.GRAFT_LIFESPAN,
            'GENDER': cls.RECIPIENT_GENDER,
            'ADMIT_DATE_DON': cls.DONOR_ADMISSION_DATE,
            'DON_TY': cls.DONOR_TYPE,
            'GENDER_DON': cls.DONOR_GENDER,
            'RECOVERY_DATE_DON': cls.ORGAN_RECOVERY_DATE,
            'INIT_DATE': cls.WAITLIST_REGISTRATION_DATE,
            'REM_CD': cls.REASON_REMOVED_WAITLIST,
            'INIT_MELD_OR_PELD': cls.INIT_MELD_OR_PELD,
            'FINAL_MELD_PELD_LAB_SCORE': cls.FINAL_MELD_PELD_LAB_SCORE,
        }
        return code_map.get(code, None)


class RecipientStatus(Enum):
    ALIVE = auto()
    LOST = auto()
    DIED = auto()
    RETRANSPLANTED = auto()
    NOT_SEEN = auto()
    OTHER = auto()

    @classmethod
    def from_code(cls, code: str):
        code_map = {
            'A': cls.ALIVE,
            'L': cls.LOST,
            'D': cls.DIED,
            'R': cls.RETRANSPLANTED,
            'N': cls.NOT_SEEN,
        }
        return code_map.get(code, None)


class DonorType(Enum):
    DECEASED = auto()
    LIVING = auto()
    OTHER = auto()

    @classmethod
    def from_code(cls, code: str):
        code_map = {
            'C': cls.DECEASED,
            'L': cls.LIVING,
        }
        return code_map.get(code, cls.OTHER)


class Urgency(Enum):
    MELD = auto()
    PELD = auto()

    @classmethod
    def from_code(cls, code: str):
        code_map = {
            'MELD': cls.MELD,
            'PELD': cls.PELD,
        }
        return code_map.get(code, None)


class WaitlistRemovalReason(Enum):
    TRANSPLANT = auto()
    MEDICALLY_UNSUITABLE = auto()
    REFUSED_TRANSPLANT = auto()
    TRANSFERRED = auto()
    DIED = auto()
    OTHER = auto()
    LISTED_IN_ERROR = auto()
    UNACCEPTABLE_ANTIGENS = auto()
    CONDITION_IMPROVED = auto()
    CONDITION_DETERIORATED = auto()
    REMOVED_IN_ERROR = auto()
    CHANGED_ORGAN = auto()
    PROGRAM_INACTIVE = auto()
    DIED_DURING_TRANSPLANT = auto()
    UNABLE_TO_CONTACT = auto()
    WAITING_FOR_KP = auto()
    WAITING_FOR_OTHER_ORGAN = auto()

    @classmethod
    def from_code(cls, code: int):
        code_map = {
            2: cls.TRANSPLANT,
            3: cls.TRANSPLANT,
            4: cls.TRANSPLANT,
            5: cls.MEDICALLY_UNSUITABLE,
            6: cls.REFUSED_TRANSPLANT,
            7: cls.TRANSFERRED,
            8: cls.DIED,
            9: cls.OTHER,
            10: cls.LISTED_IN_ERROR,
            11: cls.UNACCEPTABLE_ANTIGENS,
            12: cls.CONDITION_IMPROVED,
            13: cls.CONDITION_DETERIORATED,
            14: cls.TRANSPLANT,
            15: cls.TRANSPLANT,
            16: cls.REMOVED_IN_ERROR,
            17: cls.CHANGED_ORGAN,
            18: cls.TRANSPLANT,
            19: cls.TRANSPLANT,
            20: cls.PROGRAM_INACTIVE,
            21: cls.DIED_DURING_TRANSPLANT,
            22: cls.TRANSPLANT,
            23: cls.DIED_DURING_TRANSPLANT,
            24: cls.UNABLE_TO_CONTACT,
            40: cls.WAITING_FOR_KP,
            41: cls.WAITING_FOR_OTHER_ORGAN,
            42: cls.WAITING_FOR_OTHER_ORGAN,
            43: cls.WAITING_FOR_OTHER_ORGAN,
            44: cls.WAITING_FOR_OTHER_ORGAN,
            45: cls.WAITING_FOR_OTHER_ORGAN,
        }
        return code_map.get(code, None)


class FunctionalStatus(Enum):
    # Performs activities of daily living with NO assistance (1)
    NO_ASSISTANCE = auto()

    # Performs activities of daily living with SOME assistance (2)
    SOME_ASSISTANCE = auto()

    # Performs activities of daily living with TOTAL assistance (3)
    TOTAL_ASSISTANCE = auto()

    # Not Applicable (patient < 1 year old) (996)
    NOT_APPLICABLE = auto()

    # Unknown (998)
    UNKNOWN = auto()

    # 10% - Moribund, fatal processes progressing rapidly (2010)
    MORIBUND = auto()

    # 20% - Very sick, hospitalization necessary: active treatment necessary (2020)
    VERY_SICK = auto()

    # 30% - Severely disabled: hospitalization is indicated, death not imminent (2030)
    SEVERELY_DISABLED = auto()

    # 40% - Disabled: requires special care and assistance (2040)
    DISABLED = auto()

    # 50% - Requires considerable assistance and frequent medical care (2050)
    CONSIDERABLE_ASSISTANCE = auto()

    # 60% - Requires occasional assistance but is able to care for needs (2060)
    OCCASIONAL_ASSISTANCE = auto()

    # 70% - Cares for self: unable to carry on normal activity or active work (2070)
    SELF_CARE = auto()

    # 80% - Normal activity with effort: some symptoms of disease (2080)
    NORMAL_WITH_EFFORT = auto()

    # 90% - Able to carry on normal activity: minor symptoms of disease (2090)
    NORMAL_WITH_SYMPTOMS = auto()

    # 100% - Normal, no complaints, no evidence of disease (2100)
    NORMAL = auto()

    # 10% - No play; does not get out of bed (4010)
    CHILD_BEDRIDDEN = auto()

    # 20% - Often sleeping; play entirely limited to very passive activities (4020)
    CHILD_MOSTLY_SLEEPING = auto()

    # 30% - In bed; needs assistance even for quiet play (4030)
    CHILD_IN_BED = auto()

    # 40% - Mostly in bed; participates in quiet activities (4040)
    CHILD_MOSTLY_IN_BED = auto()

    # 50% - Can dress but lies around much of day; no active play; can take part in quiet play/activities (4050)
    CHILD_INACTIVE = auto()

    # 60% - Up and around, but minimal active play; keeps busy with quieter activities (4060)
    CHILD_MINIMAL_PLAY = auto()

    # 70% - Both greater restriction of and less time spent in play activity (4070)
    CHILD_RESTRICTED_PLAY = auto()

    # 80% - Active, but tires more quickly (4080)
    CHILD_TIRES_QUICKLY = auto()

    # 90% - Minor restrictions in physically strenuous activity (4090)
    CHILD_MINOR_RESTRICTIONS = auto()

    # 100% - Fully active, normal (4100)
    CHILD_FULLY_ACTIVE = auto()

    @classmethod
    def from_code(cls, code: int):
        code_map = {
            1: cls.NO_ASSISTANCE,
            2: cls.SOME_ASSISTANCE,
            3: cls.TOTAL_ASSISTANCE,
            996: cls.NOT_APPLICABLE,
            998: cls.UNKNOWN,
            2010: cls.MORIBUND,
            2020: cls.VERY_SICK,
            2030: cls.SEVERELY_DISABLED,
            2040: cls.DISABLED,
            2050: cls.CONSIDERABLE_ASSISTANCE,
            2060: cls.OCCASIONAL_ASSISTANCE,
            2070: cls.SELF_CARE,
            2080: cls.NORMAL_WITH_EFFORT,
            2090: cls.NORMAL_WITH_SYMPTOMS,
            2100: cls.NORMAL,
            4010: cls.CHILD_BEDRIDDEN,
            4020: cls.CHILD_MOSTLY_SLEEPING,
            4030: cls.CHILD_IN_BED,
            4040: cls.CHILD_MOSTLY_IN_BED,
            4050: cls.CHILD_INACTIVE,
            4060: cls.CHILD_MINIMAL_PLAY,
            4070: cls.CHILD_RESTRICTED_PLAY,
            4080: cls.CHILD_TIRES_QUICKLY,
            4090: cls.CHILD_MINOR_RESTRICTIONS,
            4100: cls.CHILD_FULLY_ACTIVE
        }
        return code_map.get(code, cls.UNKNOWN)
