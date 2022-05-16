# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from enum import IntEnum


class MathMLAnnotationType(IntEnum):
    """
    Types of MathML markup notation.
    """
    PRESENTATION = 0
    CONTENT = 1
