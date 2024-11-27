import os
from typing import Callable, Dict, List, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from .model import Column
import numpy as np


DATES = [Column.ORGAN_RECOVERY_DATE.name, Column.WAITLIST_REGISTRATION_DATE.name,
         Column.TRANSPLANT_DATE.name, Column.END_DATE.name]

"""
VALUES
"""
DECEASED_VALUE = 'Deceased'
""""
DATE CLEANING
"""
VARIABLE_NAME_UPDATES = {
    'PX_STAT': Column.RECIPIENT_STATUS.name,
    'TRR_ID_CODE': Column.ORGAN_TRANSPLANT_ID.name,
    'TRR_FOL_ID_CODE': 'FOLLOWUP_ID',
    'PX_STAT_DATE': 'FOLLOWUP_DATE',
    'FUNC_STAT': 'FUNCTIONAL_STATUS',
    'ACUTE_REJ_EPI': 'NUM_ACUTE_REJECTIONS',
    'PT_CODE': 'RECIPIENT_ID',
    'WL_ID_CODE': Column.WAITLIST_ID.name,
    'FUNC_STAT_TCR': 'FUNCTIONAL_STATUS_AT_REGISTRATION',
    'FUNC_STAT_TRF': 'FUNCTIONAL_STATUS_AT_FOLLOW_UP',
    'FUNC_STAT_TRR': 'FUNCTIONAL_STATUS_AT_TRANSPLANT',
    'INIT_STAT': 'STATUS_AT_REGISTRATION',
    'DIAG': 'DIAGNOSIS',
    'PTIME': 'PATIENT_SURVIVAL_TIME',
    'DAYSWAIT_CHRON': 'DAYS_ON_WAITLIST',
    'AGE': 'RECIPIENT_AGE',
    'AGE_DON': 'DONOR_AGE',
    'TRTREJ1Y': 'REJECTED_WITHIN_YEAR',
    'TX_DATE': Column.TRANSPLANT_DATE.name,
    'LI_BIOPSY': 'DONOR_LIVER_QUALITY',
    'ABO': 'RECIPIENT_BLOOD_TYPE',
    'ABO_DON': 'DONOR_BLOOD_TYPE',
    'ABO_MAT': 'DONOR-RECIPIENT_BLOOD_LEVEL',
    'GRF_STAT': 'GRAFT_FUNCTIONING',
    'GTIME': 'GRAFT_LIFESPAN',
    'GENDER': 'RECIPIENT_GENDER',
    'ADMIT_DATE_DON': 'DONOR_ADMISSION_DATE',
    'DON_TY': Column.DONOR_TYPE.name,
    'GENDER_DON': 'DONOR_GENDER',
    'RECOVERY_DATE_DON': Column.ORGAN_RECOVERY_DATE.name,
    'INIT_DATE': Column.WAITLIST_REGISTRATION_DATE.name
}
VALUE_UPDATES = {
    Column.RECIPIENT_STATUS.name: {
        'A': 'Alive',
        'L': 'Lost',
        'D': 'Died',
        'R': 'Retransplanted',
        'N': 'Not Seen',
    },
    Column.VARIABLE_NAME.name: VARIABLE_NAME_UPDATES,
    Column.DONOR_TYPE.name: {'C': DECEASED_VALUE, 'L': 'Living'},
}

"""
COLUMNS OF RELEVANCE.
Reflect original names in spreadsheet.
"""
METADATA_NAMES = ['VARIABLE NAME', 'DESCRIPTION', 'SAS ANALYSIS FORMAT']
LIVER_ORGAN_RELEVANT_NAMES = [
    'TRR_ID_CODE',
    'WL_ID_CODE',
    'PT_CODE',
    Column.DONOR_ID.name,
    'GENDER',
    'AGE',
    'AGE_DON',
    'INIT_MELD_OR_PELD',
    Column.INIT_MELD_PELD_LAB_SCORE.name,
    'FINAL_MELD_OR_PELD',
    'FINAL_MELD_PELD_LAB_SCORE',
    'FUNC_STAT_TCR',
    'FUNC_STAT_TRF',
    'FUNC_STAT_TRR',
    'INIT_STAT',
    'DIAG',
    'PTIME',
    'DAYSWAIT_CHRON',
    'TRTREJ1Y',
    'TX_DATE',
    'LI_BIOPSY',
    'ABO',
    'ABO_DON',
    'ABO_MAT',
    'GRF_STAT',
    'GTIME',
    'ADMIT_DATE_DON',
    'DON_TY',
    'RECOVERY_DATE_DON',
    'INIT_DATE',
    'INIT_STAT',
    Column.END_DATE.name
]
LIVER_FOLLOW_UP_RELEVANT_NAMES = [
    'TRR_ID_CODE',
    'PX_STAT',
    'PX_STAT_DATE',
    'TRR_FOL_ID_CODE',
    'TRT_REJ_NUM',
    'FUNC_STAT',
    'ACUTE_REJ_EPI']
DONOR_RELEVANT_NAMES = [Column.DONOR_ID.name, 'RECOVERY_DATE_DON']


@staticmethod
def renames(original_df: pd.DataFrame) -> pd.DataFrame:
    original_df.rename(columns=VARIABLE_NAME_UPDATES, inplace=True)
    return original_df


@staticmethod
def rename_variable_name_rows(original_df: pd.DataFrame) -> pd.DataFrame:
    original_df[Column.VARIABLE_NAME.name] = original_df[Column.VARIABLE_NAME.name].map(
        VARIABLE_NAME_UPDATES)
    return original_df


@staticmethod
def rename_row_values(original_df: pd.DataFrame) -> pd.DataFrame:
    for column, value_map in VALUE_UPDATES.items():
        if column in original_df.columns:
            original_df[column] = original_df[column].map(
                # Preserve NaN and unmapped values
                lambda row: value_map.get(row, row))
    return original_df


@staticmethod
def read_csv_to_dataframe(csv_file_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_file_path, delimiter='\t', encoding='ISO-8859-1')


@staticmethod
def read_metadata() -> Dict[str, pd.DataFrame]:
    """
    Returns dataframe of information for STAR File Data Dictionary.
    """
    cleaned_metadata = {}
    sheets = pd.read_excel(
        'STAR File User Guide/STAR File Data Dictionary.xlsx', sheet_name=None)
    sheet_names = list(sheets.keys())
    if sheet_names:
        first_sheet_name = sheet_names[0]
        del sheets[first_sheet_name]
    for sheet, value in sheets.items():
        dataframe_for_sheet = pd.DataFrame(value)
        # Set column names based on first row
        dataframe_for_sheet.columns = dataframe_for_sheet.iloc[0]
        dataframe_for_sheet = dataframe_for_sheet[1:]
        dataframe_for_sheet.reset_index(drop=True, inplace=True)
        cleaned_metadata[sheet] = dataframe_for_sheet
    return cleaned_metadata


@staticmethod
def get_metadata_for_sheet(sheet_name: str) -> pd.DataFrame:
    metadata_df = read_metadata()
    metadata_df = metadata_df[sheet_name]
    metadata_df = metadata_df[METADATA_NAMES]
    metadata_df.columns = metadata_df.columns.str.replace(' ', '_')
    metadata_df[Column.VARIABLE_NAME.name] = metadata_df[Column.VARIABLE_NAME.name].map(
        lambda row: VARIABLE_NAME_UPDATES.get(row, row))
    return metadata_df


@staticmethod
def get_dictionary_for_sheet(sheet_name: str) -> Dict[str, str]:
    """
    Returns a map of variables to their descriptions based on a sheet in the STAR File Data Dictionary.
    """
    metadata_df = get_metadata_for_sheet(sheet_name)
    return metadata_df.set_index(Column.VARIABLE_NAME.name)[Column.DESCRIPTION.name].to_dict()


@staticmethod
def process_dataframe(original_df: pd.DataFrame, columns_to_extract: List[str] = None) -> pd.DataFrame:
    original_df = original_df[columns_to_extract] if columns_to_extract else original_df
    original_df = renames(original_df)
    original_df = rename_row_values(original_df)
    original_df.replace('.', np.nan, inplace=True)
    return original_df


@staticmethod
def filter_dataframe(original_df: pd.DataFrame, predicate: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    return original_df[predicate(original_df)]


@staticmethod
def get_names_from_html(html_file_path: str) -> List[str]:
    html_content = ''
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    rows = soup.find_all('tr')
    first_td_values = []
    for row in rows:
        tds = row.find_all('td')
        if len(tds) > 0:
            first_td_values.append(tds[0].get_text(strip=True))
    return first_td_values


@staticmethod
def get_data_dictionary_from_dataframe(sheet_in_metadata: str, data_df: pd.DataFrame) -> Dict[str, str]:
    variable_description_map = get_dictionary_for_sheet(sheet_in_metadata)
    return {column: variable_description_map.get(column, "Description not found") for column in data_df.columns}


@staticmethod
def read_organ_data(sheet_in_metadata: str = 'LIVER_DATA', data_file_path: str = 'Delimited Text File 202409/Liver/LIVER_DATA.DAT') -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Contains information on all waiting list registrations and transplants of that organ type that have been listed or performed. 
    There is one record per waiting list  registration/transplant event, and each record includes the most recent
    follow-up information.

    Waiting list registrations can be selected by choosing records where WAITLIST_ID is not null,
    and transplants performed can be selected by choosing records where ORGAN_TRANSPLANT_ID is not null. 
    """
    data_df = read_csv_to_dataframe(data_file_path)
    data_df.columns = get_names_from_html(
        f'{os.path.splitext(data_file_path)[0]}.htm')
    data_df = process_dataframe(data_df, LIVER_ORGAN_RELEVANT_NAMES)
    # Filter on deceased donors only
    data_df = filter_dataframe(data_df, lambda df: (
        df[Column.DONOR_TYPE.name].isna()) | (df[Column.DONOR_TYPE.name] == DECEASED_VALUE))
    data_df[DATES] = data_df[DATES].apply(pd.to_datetime)
    return data_df, get_data_dictionary_from_dataframe(sheet_in_metadata, data_df)


@staticmethod
def read_follow_up_data(sheet_in_metadata: str = 'LIVER_FOLLOWUP_DATA', data_file_path: str = 'Delimited Text File 202409/Liver/Individual Follow-up Records/LIVER_FOLLOWUP_DATA.DAT') -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Contains one record per follow-up per transplant
    event. 
    Therefore, in most cases, you will find multiple records per transplant.
    """
    data_df = read_csv_to_dataframe(data_file_path)
    data_df.columns = get_names_from_html(
        f'{os.path.splitext(data_file_path)[0]}.htm')
    data_df = process_dataframe(data_df, LIVER_FOLLOW_UP_RELEVANT_NAMES)
    return data_df, get_data_dictionary_from_dataframe(sheet_in_metadata, data_df)


@staticmethod
def read_donor_data(sheet_in_metadata: str = 'DECEASED_DONOR_DATA', data_file_path: str = 'Delimited Text File 202409/Deceased Donor/DECEASED_DONOR_DATA.DAT') -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Contains information on all deceased donors that have donated organs for 
    There is one record per donor.
    """
    data_df = read_csv_to_dataframe(data_file_path)
    data_df.columns = get_names_from_html(
        f'{os.path.splitext(data_file_path)[0]}.htm')
    data_df = process_dataframe(data_df, DONOR_RELEVANT_NAMES)
    data_df[Column.DONOR_ID.name] = data_df[Column.DONOR_ID.name].astype(
        'int64')
    return data_df, get_data_dictionary_from_dataframe(sheet_in_metadata, data_df)


@staticmethod
def get_available_organs_on_date(date: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Provides the amount of livers available for donation by a given date
    based on deceased donors.

    This should be a feature in our state. 
    """
    donor_df, _ = read_donor_data()
    organ_df, _ = read_organ_data()
    transplants_with_donors = organ_df.dropna(subset=[Column.DONOR_ID.name])
    transplants_with_donors[Column.DONOR_ID.name] = transplants_with_donors[Column.DONOR_ID.name].astype(
        'int64')
    transplants_with_donors = transplants_with_donors.dropna(
        subset=[Column.ORGAN_TRANSPLANT_ID.name])
    transplants_with_donors = transplants_with_donors[[
        Column.ORGAN_TRANSPLANT_ID.name, Column.TRANSPLANT_DATE.name, Column.DONOR_ID.name]]
    transplants_with_donors = pd.merge(
        donor_df, transplants_with_donors, on=Column.DONOR_ID.name, how='left')
    transplants_with_donors = transplants_with_donors.sort_values(
        Column.TRANSPLANT_DATE.name).groupby(Column.ORGAN_TRANSPLANT_ID.name, as_index=False).first()

    # TODO: Figure out why I need to do this again after the groupby.
    transplants_with_donors[Column.ORGAN_RECOVERY_DATE.name] = pd.to_datetime(
        transplants_with_donors[Column.ORGAN_RECOVERY_DATE.name])
    transplants_with_donors[Column.TRANSPLANT_DATE.name] = pd.to_datetime(
        transplants_with_donors[Column.TRANSPLANT_DATE.name])

    transplants_with_donors = transplants_with_donors[
        (transplants_with_donors[Column.ORGAN_RECOVERY_DATE.name] <= date) &
        ((transplants_with_donors[Column.TRANSPLANT_DATE.name] > date)
         | transplants_with_donors[Column.TRANSPLANT_DATE.name].isna())
    ]
    return transplants_with_donors


@staticmethod
def get_waitlist_members_on_date(date: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Provides the amount of livers available for donation by a given date
    based on deceased donors.

    This should be a feature in our state. 
    """
    organ_df, _ = read_organ_data()
    waitlist_members_without_transplant = organ_df.groupby(Column.WAITLIST_ID.name).filter(
        lambda group: group[Column.ORGAN_TRANSPLANT_ID.name].isna().all())
    waitlist_members_without_transplant = waitlist_members_without_transplant.dropna(
        subset=[Column.INIT_MELD_PELD_LAB_SCORE.name])
    waitlist_members_without_transplant = waitlist_members_without_transplant[
        (waitlist_members_without_transplant[Column.WAITLIST_REGISTRATION_DATE.name] <= date) &
        # Make sure they haven't died yet or been removed.
        (waitlist_members_without_transplant[Column.END_DATE.name] > date)]
    return waitlist_members_without_transplant
