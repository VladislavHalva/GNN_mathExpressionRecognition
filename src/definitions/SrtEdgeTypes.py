# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from enum import IntEnum


class SrtEdgeTypes(IntEnum):
    """
    Possible extSLT edge relations. Only for parent-child nodes.
    """
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

    @classmethod
    def to_string(cls, idx):
        switcher = {
            cls.UNDEFINED: 'undefined',
            cls.RIGHT: 'right',
            cls.ABOVE: 'above',
            cls.BELOW: 'below',
            cls.INSIDE: 'inside',
            cls.SUPERSCRIPT: 'superscript',
            cls.SUBSCRIPT: 'subscript',
            cls.TO_ENDNODE: 'to-endnode'
        }
        return switcher.get(idx, 'unk')
