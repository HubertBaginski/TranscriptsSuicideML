from enum import Enum


class VariableCodes(Enum):
    """ Supported varaible codes for model training and inference. """
    COMPLETED_SUICIDE = "MF02_01"
    CELEBRITY_SUICIDE = "ID05_01"
    SUICIDAL_IDEATION = "MF02_03"
    ALTERNATIVES_TO_SUICIDE = "AU01_01"
    MONOCAUSALITY = "CS02_01"
    HEALING_STORY = "MF02_12"
    POSITIVE_OUTCOME_CRISIS = "PO01_01"
    ENHANCING_MYTHS = "PR01_01"
    PROBLEM_SOLUTION = "PS01"
    MAIN_FOCUS = "MF01"