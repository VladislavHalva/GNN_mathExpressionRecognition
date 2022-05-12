from enum import IntEnum

from src.definitions.SrtEdgeTypes import SrtEdgeTypes


class SrtEdgeTypesExt(IntEnum):
    UNDEFINED = 0
    RIGHT = 1
    ABOVE = 2
    BELOW = 3
    INSIDE = 4
    SUPERSCRIPT = 5
    SUBSCRIPT = 6
    LEFT = 7
    AROUND = 8
    ROOT_SUPERSCRIPT = 9
    ROOT_SUBSCRIPT = 10
    SELF = 11

    @classmethod
    def from_string(cls, string):
        string = string.strip().lower()
        switcher = {
            'right': cls.RIGHT,
            'above': cls.ABOVE,
            'below': cls.BELOW,
            'inside': cls.INSIDE,
            'superscript': cls.SUPERSCRIPT,
            'subscript': cls.SUBSCRIPT,
            'left': cls.LEFT,
            'around': cls.AROUND,
            'root_superscript': cls.ROOT_SUPERSCRIPT,
            'root_subscript': cls.ROOT_SUBSCRIPT,
            'self': cls.SELF
        }
        return switcher.get(string, cls.UNDEFINED)

    @classmethod
    def get_reverse(cls, etype):
        switcher = {
            cls.RIGHT: cls.LEFT,
            cls.ABOVE: cls.BELOW,
            cls.BELOW: cls.ABOVE,
            cls.INSIDE: cls.AROUND,
            cls.SUPERSCRIPT: cls.ROOT_SUPERSCRIPT,
            cls.SUBSCRIPT: cls.ROOT_SUBSCRIPT,
            cls.LEFT: cls.RIGHT,
            cls.AROUND: cls.INSIDE,
            cls.ROOT_SUPERSCRIPT: cls.SUPERSCRIPT,
            cls.ROOT_SUBSCRIPT: cls.SUBSCRIPT,
            cls.SELF: cls.SELF,
            cls.UNDEFINED: cls.UNDEFINED
        }
        return switcher.get(etype, cls.UNDEFINED)

    @classmethod
    def from_srt_edge_type(cls, srt_etype):
        switcher = {
            SrtEdgeTypes.RIGHT: cls.RIGHT,
            SrtEdgeTypes.ABOVE: cls.ABOVE,
            SrtEdgeTypes.BELOW: cls.BELOW,
            SrtEdgeTypes.INSIDE: cls.INSIDE,
            SrtEdgeTypes.SUPERSCRIPT: cls.SUPERSCRIPT,
            SrtEdgeTypes.SUBSCRIPT: cls.SUBSCRIPT,
            SrtEdgeTypes.TO_ENDNODE: cls.UNDEFINED
        }
        return switcher.get(srt_etype, cls.UNDEFINED)

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
            cls.LEFT: 'left',
            cls.AROUND: 'around',
            cls.ROOT_SUPERSCRIPT: 'root_superscript',
            cls.ROOT_SUBSCRIPT: 'root_subscript',
            cls.SELF: 'self'
        }
        return switcher.get(idx, 'unk')
