import logging
import os
from pathlib import Path

from tqdm import tqdm

from src.data.CrohmeDataset import CrohmeDataset
from src.data.LatexVocab import LatexVocab
import xml.etree.ElementTree as ET

from src.definitions.SrtEdgeTypes import SrtEdgeTypes


def mathml_dfs(xml_ns, mathml_ns, root):
    s, r = [], []
    if root.tag == mathml_ns+'math':
        for child in root:
            s_a, r_a, _ = mathml_dfs(xml_ns, mathml_ns, child)
            s.extend(s_a)
            r.extend(r_a)
        return s, r, None
    elif root.tag == mathml_ns+'mrow':
        # just connect all children to row (right relation)
        linearly_connect = []
        for child in root:
            s_a, r_a, subtree_root_id = mathml_dfs(xml_ns, mathml_ns, child)
            s.extend(s_a)
            r.extend(r_a)
            linearly_connect.append(subtree_root_id)
        for src, tgt in zip(linearly_connect, linearly_connect[1:]):
            r.append({
                'src_id': src,
                'tgt_id': tgt,
                'type': SrtEdgeTypes.RIGHT
            })
        return s, r, linearly_connect[0]
    elif root.tag == mathml_ns+'msqrt':
        # append \sqrt symbol and connect its first child as inside
        # all children connect to row - linear (just like mrow)
        sqrt_symbol_id = root.attrib.get(xml_ns+'id')
        s.append({
            'id': sqrt_symbol_id,
            'symbol': r"\sqrt"
        })
        linearly_connect = []
        for i, child in enumerate(root):
            s_a, r_a, subtree_root_id = mathml_dfs(xml_ns, mathml_ns, child)
            s.extend(s_a)
            r.extend(r_a)
            linearly_connect.append(subtree_root_id)
            if i == 0:
                r.append({
                    'src_id': sqrt_symbol_id,
                    'tgt_id': subtree_root_id,
                    'type': SrtEdgeTypes.INSIDE
                })
        for src, tgt in zip(linearly_connect, linearly_connect[1:]):
            r.append({
                'src_id': src,
                'tgt_id': tgt,
                'type': SrtEdgeTypes.RIGHT
            })
        return s, r, sqrt_symbol_id
    elif root.tag in [mathml_ns+'msub', mathml_ns+'msup']:
        # process subtrees and add subscript connection between their roots
        basis_id = None
        script_id = None
        for i, child in enumerate(root):
            s_a, r_a, subtree_root_id = mathml_dfs(xml_ns, mathml_ns, child)
            s.extend(s_a)
            r.extend(r_a)
            if i == 0:
                basis_id = subtree_root_id
            elif i == 1:
                script_id = subtree_root_id
        if not basis_id or not script_id:
            logging.warning('MathML sub/superscript syntax error')
            return s, r, None
        if root.tag == mathml_ns+'msub':
            relation_type = SrtEdgeTypes.SUBSCRIPT
        else:
            relation_type = SrtEdgeTypes.SUPERSCRIPT
        r.append({
            'src_id': basis_id,
            'tgt_id': script_id,
            'type': relation_type
        })
        return s, r, basis_id
    elif root.tag in [mathml_ns + 'msubsup', mathml_ns + 'munderover']:
        # process subtrees and add sub+superscript/under+over connection between their roots
        basis_id = None
        subscript_id = None
        superscript_id = None
        for i, child in enumerate(root):
            s_a, r_a, subtree_root_id = mathml_dfs(xml_ns, mathml_ns, child)
            s.extend(s_a)
            r.extend(r_a)
            if i == 0:
                basis_id = subtree_root_id
            elif i == 1:
                subscript_id = subtree_root_id
            elif i == 2:
                superscript_id = subtree_root_id
        if not basis_id or not subscript_id or not superscript_id:
            logging.warning('MathML sub+superscript syntax error')
            return s, r, None

        if root.tag == mathml_ns + 'msubsup':
            relation1 = SrtEdgeTypes.SUBSCRIPT
            relation2 = SrtEdgeTypes.SUPERSCRIPT
        else:
            relation1 = SrtEdgeTypes.BELOW
            relation2 = SrtEdgeTypes.ABOVE

        r.append({
            'src_id': basis_id,
            'tgt_id': subscript_id,
            'type': relation1
        })
        r.append({
            'src_id': basis_id,
            'tgt_id': superscript_id,
            'type': relation2
        })
        return s, r, basis_id
    elif root.tag == mathml_ns+'mfrac':
        # process subtrees, add \frac symbol and add above/below
        # relation to numerator/denominator
        frac_symbol_id = root.attrib.get(xml_ns+'id')
        s.append({
            'id': frac_symbol_id,
            'symbol': r"\frac"
        })
        numerator_root_id = None
        denominator_root_id = None
        for i, child in enumerate(root):
            s_a, r_a, subtree_root_id = mathml_dfs(xml_ns, mathml_ns, child)
            s.extend(s_a)
            r.extend(r_a)
            if i == 0:
                numerator_root_id = subtree_root_id
            elif i == 1:
                denominator_root_id = subtree_root_id
        if not numerator_root_id or not denominator_root_id:
            logging.warning('MathML fraction syntax error')
            return s, r, None
        r.append({
            'src_id': frac_symbol_id,
            'tgt_id': numerator_root_id,
            'type': SrtEdgeTypes.ABOVE
        })
        r.append({
            'src_id': frac_symbol_id,
            'tgt_id': denominator_root_id,
            'type': SrtEdgeTypes.BELOW
        })
        return s, r, frac_symbol_id
    elif root.tag in [mathml_ns+'mi', mathml_ns+'mn', mathml_ns+'mo', mathml_ns+'mtext', mathml_ns+'mspace', mathml_ns+'ms']:
        id = root.attrib.get(xml_ns+'id')
        s.append({
            'id': id,
            'symbol': root.text
        })
        return s, r, id
    else:
        print('unknown MathML element: ' + root.tag)
        exit()

def test():
    inkml_path = 'assets/crohme/train/inkml/test_2012/formulaire040-equation013.inkml'

    if not os.path.isfile(inkml_path) and Path(inkml_path).suffix != '.inkml':
        logging.warning("Inkml file does not exists: " + inkml_path)
        return ""

    xml_namespace = '{http://www.w3.org/XML/1998/namespace}'
    doc_namespace = '{http://www.w3.org/2003/InkML}'
    mathml_namespace = '{http://www.w3.org/1998/Math/MathML}'
    tree = ET.parse(inkml_path)
    root = tree.getroot()

    annotation_mathml = root.find(doc_namespace + 'annotationXML[@type="truth"][@encoding="Content-MathML"]')
    if not annotation_mathml:
        logging.warning("Inkml file does not contain MathML annotation: " + inkml_path)
        return ""

    math_root = annotation_mathml.find(mathml_namespace + 'math')
    if not math_root:
        logging.warning("Inkml file does not contain math description root: " + inkml_path)
        return ""

    # s, r, _ = mathml_dfs(xml_namespace, mathml_namespace, math_root)

    components_shape = (32, 32)
    images_root = 'assets/crohme/train/img/'
    inkmls_root = 'assets/crohme/train/inkml/'
    lgs_root = 'assets/crohme/train/lg/'
    tokenizer = LatexVocab.load_tokenizer('assets/tokenizer.json')
    #
    dataset = CrohmeDataset(images_root, inkmls_root, lgs_root, tokenizer, components_shape)
    #
    for dataitem in tqdm(dataset):
        pass
