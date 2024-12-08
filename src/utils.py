import os
from typing import Any, Callable, Dict, List, Set, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from .model import Column, DonorType, FunctionalStatus, RecipientStatus, Urgency, WaitlistRemovalReason
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
RELEVANT_WAITLIST_REASONS = [
    WaitlistRemovalReason.TRANSPLANT.name,
    WaitlistRemovalReason.DIED.name,
    WaitlistRemovalReason.DIED_DURING_TRANSPLANT.name]

RELEVANT_URGENCY = [
    Urgency.MELD.name,
]


def rename_columns(original_df: pd.DataFrame) -> pd.DataFrame:
    original_df.rename(columns={col: Column.from_code(col).name if Column.from_code(
        col)
        else col for col in original_df.columns}, inplace=True)
    return original_df


def rename_variable_name_rows(original_df: pd.DataFrame) -> pd.DataFrame:
    original_df[Column.VARIABLE_NAME.name] = original_df[Column.VARIABLE_NAME.name].map(
        lambda old_value: Column.from_code(old_value).name)
    return original_df


def rename_row_values(original_df: pd.DataFrame) -> pd.DataFrame:
    for column, enum in VALUES_TO_UPDATE.items():
        if column in original_df.columns:
            original_df[column] = original_df[column].map(
                # Preserve NaN and unmapped values
                lambda row: enum.from_code(pd.to_numeric(row, errors='ignore')).name if enum.from_code(pd.to_numeric(row, errors='ignore')) else row)
    return original_df


def read_csv_to_dataframe(csv_file_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_file_path, delimiter='\t', encoding='ISO-8859-1')


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


def get_metadata_for_sheet(sheet_name: str) -> pd.DataFrame:
    metadata_df = read_metadata()
    metadata_df = metadata_df[sheet_name]
    metadata_df = metadata_df[METADATA_NAMES]
    metadata_df.columns = metadata_df.columns.str.replace(' ', '_')
    metadata_df[Column.VARIABLE_NAME.name] = metadata_df[Column.VARIABLE_NAME.name].map(
        lambda old_value: Column.from_code(old_value).name if Column.from_code(old_value) else old_value)
    return metadata_df


def get_dictionary_for_sheet(sheet_name: str) -> Dict[str, str]:
    """
    Returns a map of variables to their descriptions based on a sheet in the STAR File Data Dictionary.
    """
    metadata_df = get_metadata_for_sheet(sheet_name)
    return metadata_df.set_index(Column.VARIABLE_NAME.name)[Column.DESCRIPTION.name].to_dict()


def process_dataframe(original_df: pd.DataFrame, columns_to_extract: List[str] = None, sample_frac: float = 1) -> pd.DataFrame:
    original_df = original_df[columns_to_extract] if columns_to_extract else original_df
    original_df = rename_columns(original_df)
    original_df = rename_row_values(original_df)
    original_df.replace('.', np.nan, inplace=True)
    original_df = original_df.sample(
        frac=sample_frac, random_state=42).reset_index(drop=True)
    return original_df


def filter_dataframe(original_df: pd.DataFrame, predicate: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    return original_df[predicate(original_df)]


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


def get_data_dictionary_from_dataframe(sheet_in_metadata: str, data_df: pd.DataFrame) -> Dict[str, str]:
    variable_description_map = get_dictionary_for_sheet(sheet_in_metadata)
    return {column: variable_description_map.get(column, "Description not found") for column in data_df.columns}


def read_organ_data(_: str = 'LIVER_DATA', data_file_path: str = 'Delimited Text File 202409/all_transplants.csv', sample_frac=1.0) -> Tuple[pd.DataFrame, Dict[str, str]]:
    data_df = pd.read_csv(data_file_path)
    data_df = data_df.sample(
        frac=sample_frac, random_state=42).reset_index(drop=True)
    data_df = data_df.astype({
        Column.ORGAN_TRANSPLANT_ID.name: 'str',
        Column.INIT_MELD_PELD_LAB_SCORE.name: 'Int64',
        Column.FINAL_MELD_PELD_LAB_SCORE.name: 'Int64',
        Column.PATIENT_SURVIVAL_TIME.name: 'Int64',
        Column.GRAFT_LIFESPAN.name: 'Int64',
        Column.DAYS_ON_WAITLIST.name: 'Int64',
        Column.RECIPIENT_BLOOD_TYPE.name: 'str',
        Column.RECIPIENT_STATUS.name: 'str',
        Column.TRANSPLANT_DATE.name: 'datetime64[ns]',
        Column.END_DATE.name: 'datetime64[ns]',
        Column.DIAGNOSIS.name: 'Int64'
    })
    data_df[DATES] = data_df[DATES].apply(pd.to_datetime)
    data_df = add_blood_type_code(
        data_df, to_column=Column.RECIPIENT_BLOOD_TYPE_AS_CODE, from_column=Column.RECIPIENT_BLOOD_TYPE)
    data_df = add_blood_type_code(
        data_df, to_column=Column.DONOR_BLOOD_TYPE_AS_CODE, from_column=Column.DONOR_BLOOD_TYPE)
    data_df[Column.DIAGNOSIS.name] = data_df[Column.DIAGNOSIS.name].fillna(
        -1).astype('Int64')
    return data_df, None


def read_organ_data_computed(sheet_in_metadata: str = 'LIVER_DATA', data_file_path: str = 'Delimited Text File 202409/Liver/LIVER_DATA.DAT') -> Tuple[pd.DataFrame, Dict[str, str]]:
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
                RELEVANT_WAITLIST_REASONS)).all()
        )
        & (
            (group[Column.INIT_MELD_OR_PELD.name].isin(
                RELEVANT_URGENCY)).all()
        )
        & (
            group[Column.DONOR_TYPE.name].isin(
                [DonorType.OTHER.name, DonorType.DECEASED.name]).all()
        )
        # Remove 'OTHER' and 'RETRANSPLANTED'
        & (
            group[Column.RECIPIENT_STATUS.name].isna().all() |
            group[Column.RECIPIENT_STATUS.name].isin(
                [RecipientStatus.DIED.name, RecipientStatus.ALIVE.name]).all()
        )
    )[Column.RECIPIENT_ID.name].unique()

    data_df = data_df[data_df[Column.RECIPIENT_ID.name].isin(
        valid_recipient_ids)]

    data_df[DATES] = data_df[DATES].apply(pd.to_datetime)
    return data_df, get_data_dictionary_from_dataframe(sheet_in_metadata, data_df)


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


def read_donor_data(_: str = 'DECEASED_DONOR_DATA', data_file_path: str = 'Delimited Text File 202409/all_donors.csv', sample_frac=0.1) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Contains information on all deceased donors that have donated organs for
    There is one record per donor.
    """
    data_df = pd.read_csv(data_file_path)
    data_df = data_df.sample(
        frac=sample_frac, random_state=42).reset_index(drop=True)
    data_df[Column.DONOR_ID.name] = data_df[Column.DONOR_ID.name].astype(
        'Int64')
    for column in DATES:
        if column in data_df.columns:
            data_df[column] = data_df[column].apply(pd.to_datetime)
    return data_df, None


def read_donor_data_computed(sheet_in_metadata: str = 'DECEASED_DONOR_DATA', data_file_path: str = 'Delimited Text File 202409/Deceased Donor/DECEASED_DONOR_DATA.DAT') -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Contains information on all deceased donors that have donated organs for
    There is one record per donor.
    """
    data_df = read_csv_to_dataframe(data_file_path)
    data_df.columns = get_names_from_html(
        f'{os.path.splitext(data_file_path)[0]}.htm')
    data_df = process_dataframe(
        data_df, DONOR_RELEVANT_NAMES, sample_frac=1.0)
    data_df[Column.DONOR_ID.name] = data_df[Column.DONOR_ID.name].astype(
        'Int64')
    for column in DATES:
        if column in data_df.columns:
            data_df[column] = data_df[column].apply(pd.to_datetime)
    return data_df, get_data_dictionary_from_dataframe(sheet_in_metadata, data_df)


def add_blood_type_code(df: pd.DataFrame, from_column: Column, to_column: Column):
    blood_types = get_all_blood_types()
    blood_type_to_index = {bt: idx for idx, bt in enumerate(blood_types)}
    df[to_column.name] = df[from_column.name].map(
        blood_type_to_index).fillna(-1).astype(int)
    return df


def get_available_organs(by_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Provides the amount of livers available for donation by a given date
    based on deceased donors.

    This should be a feature in our state.
    """
    transplant_data, _ = read_organ_data()
    transplants_with_donors = transplant_data.dropna(
        subset=[Column.DONOR_ID.name])

    # Use boolean indexing instead of groupby and filter for performance
    if by_date:
        valid_transplants = (
            transplants_with_donors[Column.ORGAN_TRANSPLANT_ID.name].notna() &
            (transplants_with_donors[Column.ORGAN_RECOVERY_DATE.name] <= by_date) &
            (transplants_with_donors[Column.END_DATE.name] >= by_date) &
            (transplants_with_donors[Column.TRANSPLANT_DATE.name] >= by_date)
        )
    else:
        valid_transplants = transplants_with_donors[Column.ORGAN_TRANSPLANT_ID.name].notna(
        )

    valid_recipient_ids = transplants_with_donors.loc[valid_transplants, Column.RECIPIENT_ID.name].unique(
    )
    transplants_with_donors = transplants_with_donors[transplants_with_donors[Column.RECIPIENT_ID.name].isin(
        valid_recipient_ids)]

    donor_df, _ = read_donor_data()
    transplants_with_donors = pd.merge(
        transplants_with_donors, donor_df, on=[
            Column.DONOR_ID.name, Column.ORGAN_RECOVERY_DATE.name], how='left'
    )
    transplants_with_donors[Column.DONOR_ID.name] = transplants_with_donors[Column.DONOR_ID.name].astype(
        'Int64')
    return transplants_with_donors


def get_waitlist_members(by_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Provides the amount of livers available for donation by a given date
    based on deceased donors.

    This should be a feature in our state.
    """
    organ_df, _ = read_organ_data()
    organ_df = organ_df.dropna(subset=[Column.RECIPIENT_ID.name])

    if by_date:
        organ_df = organ_df[
            (organ_df[Column.WAITLIST_REGISTRATION_DATE.name] <= by_date) &
            (organ_df[Column.END_DATE.name] >= by_date) &
            (organ_df[Column.TRANSPLANT_DATE.name] >= by_date)
        ]

    valid_recipient_ids = organ_df.loc[
        organ_df[Column.TRANSPLANT_DATE.name] >= by_date, Column.RECIPIENT_ID.name
    ].unique()

    waitlist_members = organ_df[organ_df[Column.RECIPIENT_ID.name].isin(
        valid_recipient_ids)]
    waitlist_members.loc[:, Column.DONOR_ID.name] = waitlist_members[Column.DONOR_ID.name].astype(
        'Int64')
    return waitlist_members


def get_compatible_blood_types(blood_type: str) -> Dict[str, str]:
    """
    Given a blood type and the recipient matches and compatibility.

    Args:
        blood_type (str): The blood type to check compatibility for.

    Returns:
        Dict: compatible blood types as strings to their match value.
    """
    # all_transplants, _ = read_organ_data(sample_frac=1.0)
    # compatibility_df = all_transplants[[Column.DONOR_BLOOD_TYPE.name, Column.RECIPIENT_BLOOD_TYPE.name,
    #                                     Column.DONOR_RECIPIENT_BLOOD_TYPE_MATCH.name]].drop_duplicates().fillna({Column.DONOR_RECIPIENT_BLOOD_TYPE_MATCH.name: -1})
    # matches = compatibility_df[compatibility_df[Column.DONOR_BLOOD_TYPE.name] == blood_type][[
    #     Column.RECIPIENT_BLOOD_TYPE.name, Column.DONOR_RECIPIENT_BLOOD_TYPE_MATCH.name]]
    # compatibility_map = matches[matches[Column.DONOR_RECIPIENT_BLOOD_TYPE_MATCH.name] >= 1].set_index(
    #     Column.RECIPIENT_BLOOD_TYPE.name)[Column.DONOR_RECIPIENT_BLOOD_TYPE_MATCH.name].to_dict()
    # return compatibility_map
    matches = {'A': {'A': 1.0,
                     'AB': 2.0,
                     'A1': 1.0,
                     'B': 3.0,
                     'O': 3.0,
                     'A2B': 2.0,
                     'A1B': 2.0,
                     'A2': 1.0},
               'A1': {'A': 1.0,
                      'AB': 2.0,
                      'A1': 1.0,
                      'A2': 1.0,
                      'O': 3.0,
                      'A1B': 2.0,
                      'A2B': 2.0,
                      'B': 3.0},
               'A1B': {'AB': 1.0, 'O': 3.0, 'B': 3.0, 'A2B': 1.0, 'A': 3.0, 'A1B': 1.0},
               'A2': {'A': 1.0, 'AB': 2.0, 'O': 3.0, 'A1': 1.0, 'A2': 1.0, 'B': 3.0},
               'A2B': {'AB': 1.0, 'A': 3.0, 'O': 3.0, 'B': 3.0, 'A2B': 1.0},
               'AB': {'AB': 1.0, 'A1B': 1.0, 'B': 3.0, 'O': 3.0, 'A': 3.0},
               'B': {'B': 1.0, 'AB': 2.0, 'O': 3.0, 'A2B': 2.0, 'A1B': 2.0, 'A': 3.0},
               'O': {'O': 1.0, 'A': 2.0, 'B': 2.0, 'AB': 2.0, 'A1': 2.0}}
    return matches.get(blood_type)


def get_match_value(donor_blood_type: str, recipient_blood_type) -> int:
    matches = get_compatible_blood_types(donor_blood_type)
    return matches.get(recipient_blood_type, -100)


def get_all_blood_types() -> List[str]:
    # all_transplants, _ = read_organ_data()
    # blood_types = pd.concat([
    #     all_transplants[Column.DONOR_BLOOD_TYPE.name],
    #     all_transplants[Column.RECIPIENT_BLOOD_TYPE.name]
    # ]).unique()
    # blood_types = [bt for bt in blood_types.tolist() if pd.notna(bt)]
    # blood_types.sort()
    # return blood_types
    return ['A', 'A1', 'A1B', 'A2', 'A2B', 'AB', 'B', 'O']


def get_mininal_columns_available_organs(by_date: pd.Timestamp) -> pd.DataFrame:
    AVAILABLE_ORGAN_FEATURES = [
        Column.DONOR_ID.name,
        Column.DONOR_BLOOD_TYPE.name,
        Column.DONOR_BLOOD_TYPE_AS_CODE.name,
        Column.ORGAN_RECOVERY_DATE.name,
    ]
    available_organs = get_available_organs(by_date=by_date)
    return available_organs[AVAILABLE_ORGAN_FEATURES]


def get_mininal_columns_waitlist(by_date: pd.Timestamp) -> pd.DataFrame:
    WAITLIST_PREVIEW_FEATURES = [
        Column.ORGAN_TRANSPLANT_ID.name,
        Column.RECIPIENT_ID.name,
        Column.DONOR_ID.name,
        Column.INIT_MELD_PELD_LAB_SCORE.name,
        Column.FINAL_MELD_PELD_LAB_SCORE.name,
        Column.PATIENT_SURVIVAL_TIME.name,
        Column.GRAFT_LIFESPAN.name,
        Column.DAYS_ON_WAITLIST.name,
        Column.RECIPIENT_BLOOD_TYPE.name,
        Column.RECIPIENT_BLOOD_TYPE_AS_CODE.name,
        Column.RECIPIENT_AGE.name,
        Column.TRANSPLANT_DATE.name,
        Column.END_DATE.name,
        Column.DIAGNOSIS.name
    ]
    waitlist_members = get_waitlist_members(by_date=by_date)
    return waitlist_members[WAITLIST_PREVIEW_FEATURES].sort_values(by=[Column.INIT_MELD_PELD_LAB_SCORE.name], ascending=False)


def get_next_day(current_date: pd.Timestamp, allocated_ids: Set[int], max_waitlist: int = 10) -> Tuple[pd.Timestamp, List[pd.DataFrame]]:
    date = current_date
    daily_organs, daily_waitlist_members = pd.DataFrame(), pd.DataFrame()
    while daily_organs.empty and daily_waitlist_members.empty:
        # available_organs = get_mininal_columns_available_organs(by_date=date)
        # available_organs = available_organs[~available_organs[Column.DONOR_ID.name].isin(
        #     allocated_ids)]

        # waitlist_members = get_mininal_columns_waitlist(
        #     by_date=date)
        available_organs, waitlist_members = read_from_file(date)
        waitlist_members = waitlist_members[~waitlist_members[Column.DONOR_ID.name].isin(
            allocated_ids)]
        waitlist_members = waitlist_members[waitlist_members[Column.RECIPIENT_BLOOD_TYPE.name].apply(
            lambda recipient_blood_type: any(
                get_match_value(donor_blood_type=donor_blood_type,
                                recipient_blood_type=recipient_blood_type) >= 1.0
                for donor_blood_type in available_organs[Column.DONOR_BLOOD_TYPE.name]
            )
        )]
        if (len(available_organs) > 0 and len(waitlist_members) >= max_waitlist):
            daily_organs = available_organs
            daily_waitlist_members = waitlist_members.head(max_waitlist)
        else:
            print(f'''WARNING: Could not find compatibles for {
                  date} because only found {len(waitlist_members)}''')
            date += pd.Timedelta(days=1)
    return date, [daily_organs, daily_waitlist_members]


def read_from_file(date: pd.Timestamp):
    organs_filename = f"data/{date.strftime('%Y-%m-%d')}_organs.csv"
    waitlist_filename = f"data/{date.strftime('%Y-%m-%d')}_waitlist.csv"

    try:
        organs_df = pd.read_csv(organs_filename)
        waitlist_df = pd.read_csv(waitlist_filename)
        return [organs_df, waitlist_df]
    except FileNotFoundError:
        print(f"Data files for {date.strftime('%Y-%m-%d')} not found.")
    return pd.DataFrame.empty(), pd.DataFrame.empty()
