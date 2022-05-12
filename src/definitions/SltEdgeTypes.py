from enum import IntEnum


class SltEdgeTypes(IntEnum):
    PARENT_CHILD = 0
    GRANDPARENT_GRANDCHILD = 1
    LEFTBROTHER_RIGHTBROTHER = 2
    CURRENT_CURRENT = 3
