import os
from typing import Callable, Dict, List, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from .model import Column, DonorType, FunctionalStatus, RecipientStatus, WaitlistRemovalReason
import numpy as np


DATES = [Column.ORGAN_RECOVERY_DATE.name, Column.WAITLIST_REGISTRATION_DATE.name,
         Column.TRANSPLANT_DATE.name, Column.END_DATE.name]

""""
DATE CLEANING
"""
VALUES_TO_UPDATE = {
    Column.RECIPIENT_STATUS.name: RecipientStatus,
    Column.VARIABLE_NAME.name: Column,
    Column.DONOR_TYPE.name: DonorType,
    Column.REASON_REMOVED_WAITLIST.name: WaitlistRemovalReason,
    Column.FUNCTIONAL_STATUS.name: FunctionalStatus,
    Column.FUNCTIONAL_STATUS_AT_REGISTRATION.name: FunctionalStatus,
    Column.FUNCTIONAL_STATUS_AT_FOLLOW_UP.name: FunctionalStatus,
    Column.FUNCTIONAL_STATUS_AT_TRANSPLANT.name: FunctionalStatus,
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
    'DONOR_ID',
    'GENDER',
    'AGE',
    'AGE_DON',
    'INIT_MELD_OR_PELD',
    'INIT_MELD_PELD_LAB_SCORE',
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
    'END_DATE',
    'REM_CD',
    'PX_STAT'
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
RELEVANT_WAITLIST_RESONS = [
    WaitlistRemovalReason.TRANSPLANT.name,
    WaitlistRemovalReason.DIED.name,
    WaitlistRemovalReason.DIED_DURING_TRANSPLANT.name]


@staticmethod
def rename_columns(original_df: pd.DataFrame) -> pd.DataFrame:
    original_df.rename(columns={col: Column.from_code(col).name if Column.from_code(
        col)
        else col for col in original_df.columns}, inplace=True)
    return original_df


@staticmethod
def rename_variable_name_rows(original_df: pd.DataFrame) -> pd.DataFrame:
    original_df[Column.VARIABLE_NAME.name] = original_df[Column.VARIABLE_NAME.name].map(
        lambda old_value: Column.from_code(old_value).name)
    return original_df


@staticmethod
def rename_row_values(original_df: pd.DataFrame) -> pd.DataFrame:
    for column, enum in VALUES_TO_UPDATE.items():
        if column in original_df.columns:
            original_df[column] = original_df[column].map(
                # Preserve NaN and unmapped values
                lambda row: enum.from_code(pd.to_numeric(row, errors='ignore')).name if enum.from_code(pd.to_numeric(row, errors='ignore')) else row)
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
        lambda old_value: Column.from_code(old_value).name if Column.from_code(old_value) else old_value)
    return metadata_df


@staticmethod
def get_dictionary_for_sheet(sheet_name: str) -> Dict[str, str]:
    """
    Returns a map of variables to their descriptions based on a sheet in the STAR File Data Dictionary.
    """
    metadata_df = get_metadata_for_sheet(sheet_name)
    return metadata_df.set_index(Column.VARIABLE_NAME.name)[Column.DESCRIPTION.name].to_dict()


@staticmethod
def process_dataframe(original_df: pd.DataFrame, columns_to_extract: List[str] = None, sample_frac: float = 0.25) -> pd.DataFrame:
    original_df = original_df[columns_to_extract] if columns_to_extract else original_df
    original_df = rename_columns(original_df)
    original_df = rename_row_values(original_df)
    original_df.replace('.', np.nan, inplace=True)
    original_df = original_df.sample(frac=sample_frac).reset_index(drop=True)
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
    There is one record per waiting list registration/transplant event, and each record includes the most recent
    follow-up information.

    Waiting list registrations can be selected by choosing records where RECIPIENT_ID is not null,
    and transplants performed can be selected by choosing records where ORGAN_TRANSPLANT_ID is not null.
    """
    data_df = read_csv_to_dataframe(data_file_path)
    data_df.columns = get_names_from_html(
        f'{os.path.splitext(data_file_path)[0]}.htm')
    data_df = process_dataframe(
        data_df, LIVER_ORGAN_RELEVANT_NAMES)
    data_df = data_df.dropna(subset=[Column.RECIPIENT_ID.name])

    # Filter on deceased donors and relevant reasons for the end of the transplant.
    valid_recipient_ids = data_df.groupby(Column.RECIPIENT_ID.name).filter(
        lambda group:
        (
            (group[Column.REASON_REMOVED_WAITLIST.name].isin(
                RELEVANT_WAITLIST_RESONS)).all()
        )
        & (
            group[Column.DONOR_TYPE.name].isin(
                [DonorType.OTHER.name, DonorType.DECEASED.name]).all()
        )
        # Remove 'OTHER' and 'RETRANSPLANTED'
        & (
            group[Column.RECIPIENT_STATUS.name].isin(
                [RecipientStatus.DIED.name, RecipientStatus.ALIVE.name]).all()
        )
    )[Column.RECIPIENT_ID.name].unique()

    data_df = data_df[data_df[Column.RECIPIENT_ID.name].isin(
        valid_recipient_ids)]

    data_df[Column.DONOR_ID.name] = data_df[Column.DONOR_ID.name].astype(
        'Int64')
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
    data_df = process_dataframe(
        data_df, DONOR_RELEVANT_NAMES, sample_frac=0.25)
    data_df[Column.DONOR_ID.name] = data_df[Column.DONOR_ID.name].astype(
        'Int64')
    for column in DATES:
        if column in data_df.columns:
            data_df[column] = data_df[column].apply(pd.to_datetime)
    return data_df, get_data_dictionary_from_dataframe(sheet_in_metadata, data_df)


@staticmethod
def get_available_organs(by_date: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Provides the amount of livers available for donation by a given date
    based on deceased donors.

    This should be a feature in our state.
    """
    organ_df, _ = read_organ_data()
    transplants_with_donors = organ_df.dropna(subset=[Column.DONOR_ID.name])

    # Identify RECIPIENT_IDs that have at least one non-null ORGAN_TRANSPLANT_ID.
    # They should eventually get a transplant or pass to be in our dataset.
    valid_recipient_ids = organ_df.groupby(Column.RECIPIENT_ID.name).filter(
        lambda group:
        (
            (group[Column.REASON_REMOVED_WAITLIST.name] == WaitlistRemovalReason.DIED).any() |
            group[Column.ORGAN_TRANSPLANT_ID.name].notna().any()
        ) &
        (
            (group[Column.ORGAN_RECOVERY_DATE.name] <= by_date).any() &
            (group[Column.END_DATE.name] > by_date).any() &
            (group[Column.TRANSPLANT_DATE.name] >= by_date).any()
        ) if by_date else True
    )[Column.RECIPIENT_ID.name].unique()

    transplants_with_donors = transplants_with_donors[transplants_with_donors[Column.RECIPIENT_ID.name].isin(
        valid_recipient_ids)]

    # Add donor data to transplant data.
    donor_df, _ = read_donor_data()
    transplants_with_donors = pd.merge(
        transplants_with_donors, donor_df, on=[
            Column.DONOR_ID.name, Column.ORGAN_RECOVERY_DATE.name], how='left'
    )
    return transplants_with_donors


@staticmethod
def get_waitlist_members(by_date: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Provides the amount of livers available for donation by a given date
    based on deceased donors.

    This should be a feature in our state.
    """
    organ_df, _ = read_organ_data()
    organ_df = organ_df.dropna(subset=[Column.RECIPIENT_ID.name])

    # They should also have been registered prior to date but
    # not yet have died or been removed.
    if by_date:
        organ_df = organ_df[
            (organ_df[Column.WAITLIST_REGISTRATION_DATE.name] <= by_date) &
            # Make sure they haven't died yet or been removed.
            (organ_df[Column.END_DATE.name] > by_date)]

    # Group by RECIPIENT_ID and filter groups that have at least one
    # INIT_MELD_PELD_LAB_SCORE non-null
    # and either has an ORGAN_TRANSPLANT_ID or have died.
    valid_recipient_ids = organ_df.groupby(Column.RECIPIENT_ID.name).filter(
        lambda group:
            ((group[Column.TRANSPLANT_DATE.name] >= by_date).any() &
             (group[Column.ORGAN_TRANSPLANT_ID.name].notna()).any() |
             (group[Column.REASON_REMOVED_WAITLIST.name] == WaitlistRemovalReason.DIED).any()) &
        group[Column.INIT_MELD_PELD_LAB_SCORE.name].notna().any()
    )[Column.RECIPIENT_ID.name].unique()

    # Filter to keep only rows with these RECIPIENT_IDs.
    waitlist_members = organ_df[organ_df[Column.RECIPIENT_ID.name].isin(
        valid_recipient_ids)]
    return waitlist_members
