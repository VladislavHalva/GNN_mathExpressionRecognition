from enum import IntEnum


class SltEdgeTypes(IntEnum):
    PARENT_CHILD = 1
    GRANDPARENT_GRANDCHILD = 2
    LEFTBROTHER_RIGHTBROTHER = 3
    CURRENT_CURRENT = 4
