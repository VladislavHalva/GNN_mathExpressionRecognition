from enum import IntEnum


class SrtEdgeTypes(IntEnum):
    UNDEFINED = 0
    RIGHT = 1
    ABOVE = 2
    BELOW = 3
    INSIDE = 4
    SUPERSCRIPT = 5
    SUBSCRIPT = 6
    TO_ENDNODE = 7

    @classmethod
    def from_string(cls, string):
        string = string.strip().lower()
        switcher = {
            'right': cls.RIGHT,
            'above': cls.ABOVE,
            'below': cls.BELOW,
            'inside': cls.INSIDE,
            'superscript': cls.SUPERSCRIPT,
            'subscript': cls.SUBSCRIPT
        }
        return switcher.get(string, cls.UNDEFINED)
