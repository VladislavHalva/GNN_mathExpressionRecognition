# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from enum import IntEnum


class SltEdgeTypes(IntEnum):
    """
    Possible SLT edge types.
    """
    PARENT_CHILD = 0
    GRANDPARENT_GRANDCHILD = 1
    LEFTBROTHER_RIGHTBROTHER = 2
    CURRENT_CURRENT = 3
