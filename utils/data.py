import os
import pathlib
from collections import Counter
from pathlib import Path
from typing import Union

import pandas as pd

from utils.variable_codes import VariableCodes


def process_transcripts(variable_code: str, data_folder: Union[pathlib.Path, str]) -> pd.DataFrame:
    """

    Args:
        variable_code: Variable code used for training, see /utils/variable_codes.py
        data_folder: Path to training data.

    Returns:
        pre-processed training dataframe
    """
    data_folder = Path(data_folder) if isinstance(data_folder, str) else data_folder
    transcript_folder = data_folder / "broadcast"
    all_transcripts = os.listdir(transcript_folder)
    abrv_months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]

    contents = []
    id_ = []
    temp = ""
    new_row = 0

    for month in all_transcripts:
        with open(transcript_folder / month, encoding="utf-8") as transcripts:
            for line in transcripts:
                check = False
                if "B_" in line:
                    id_.append(line.replace("\n", ""))
                    new_row = 1
                    continue
                if "(CBS)" in line or "(Radio)" in line or "(NBC)" in line or "(ABC)" in line or \
                        "KONG-SEA" in line or "(FOX)" in line or "FM News" in line or "News Radio" in line or \
                        "AM 860" in line or "KOBI" in line or "(PBS)" in line or "KBNZ" in line or \
                        "KNKX" in line or "Portland's CW" in line or "Fox 12 Plus" in line or \
                        "KINK Radio" in line or "OPB" in line or "Fox Sports Radio" in line:
                    continue
                for i in abrv_months:
                    if "AM" in line and i in line and "•" in line \
                            or "PM" in line and i in line and "•" in line:
                        check = True
                        break
                if check is True:
                    continue
                if len(line.strip()) < 4:
                    continue
                if new_row == 1:
                    temp = temp.replace("\n", " ")
                    contents.append(temp)
                    temp = ""
                    new_row = 34
                if new_row == 34:
                    temp += line

    ser = pd.Series(contents[1:])
    ser1 = pd.Series(id_)
    d = {"AC01_01": ser1.str.upper(), "text": ser}
    df = pd.DataFrame(data=d)
    df = df[:-1]

    # add here the latest Excel with the manually coded variable codes
    f = data_folder / "V!brant_data_all_161220.xlsx"
    df_classes = pd.read_excel(f)

    df_classes = df_classes.iloc[1:]

    # check selected variable code mapping
    #VariableCodes.ALTERNATIVES_TO_SUICIDE.value

    # upper case ID for Excel and extracted ID's to match
    df_classes.AC01_01 = df_classes.AC01_01.str.upper()
    # combine extracted text and variable codes from excel, by joining on ID
    fin = pd.merge(df, df_classes, on="AC01_01")
    print(f"Number of samples from joined texts and Excel  by ID: {fin.shape}")
    # select relevant columns
    df_var = fin.filter(items=["AC01_01", "text", variable_code, "CO01"])
    # filter for coder -9 for consistently labeled training data
    df_var = df_var[df_var["CO01"] == -9]
    # remove texts that dont have a assigned number (-9)
    df_var = df_var[df_var[variable_code] != -9]
    print(f"Number of samples with only coder -9: {df_var.shape}")
    print(f"Frequencies of labels: {Counter(df_var[variable_code]).most_common()}")

    if variable_code == VariableCodes.PROBLEM_SOLUTION.value:
        df_var[variable_code] = df_var[variable_code].replace(1, "Problem").\
            replace(2, "Solution").replace(3, "Both").replace(4, "Neither")
    elif variable_code == VariableCodes.MAIN_FOCUS.value:
        df_var[variable_code] = df_var[variable_code].replace(1, "completed").replace(2,
                                                                                      "attempted"). \
            replace(3, "ideation").replace(4, "mur_sui_indi").replace(5, "mass_mur_sui"). \
            replace(6, "assisted").replace(7, "cluster").replace(9, "policy_pvt").replace(10,
                                                                                          "research"). \
            replace(11, "legal_issues").replace(12, "healing_story").replace(13, "other").replace(
            14, "advocacy"). \
            replace(15, "prevention")
    else:
        df_var[variable_code] = df_var[variable_code].replace(1, "no").replace(2, "yes")
    # lowercasing text
    df_var.text = df_var.text.apply(str.lower)

    return df_var
