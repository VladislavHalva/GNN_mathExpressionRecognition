# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import logging
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re

from tokenizers.pre_tokenizers import Split
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

from src.definitions.MathMLAnnotationType import MathMLAnnotationType
from src.utils.utils import mathml_unicode_to_latex_label


class LatexVocab:
    """
    Represents LaTeX vocabulary and manages tokenizer.
    """
    @staticmethod
    def generate_formulas_file_from_inkmls(inkmls_root, tgt_file, substitute_terms=False, latex_gt=True, mathml_gt=False):
        """
        Parses given InkML files and files of all symbols in formulas.
        :param inkmls_root: InkML filepath
        :param tgt_file: filepath, where the symbols shall be stored
        :param substitute_terms: whether to substitute identifiers, numbers and text element with special tokens (only for MathML parsing)
        :param latex_gt: whether to parse LaTeX groundtruth
        :param mathml_gt: whether to parse MathML groundtruth
        """
        if not os.path.exists(inkmls_root):
            raise FileNotFoundError('Inkmls directory not found')

        xml_namespace = '{http://www.w3.org/XML/1998/namespace}'
        doc_namespace = '{http://www.w3.org/2003/InkML}'
        mathml_namespace = '{http://www.w3.org/1998/Math/MathML}'

        # parse formulas
        formulas = []
        for subdir, _, files in tqdm(os.walk(inkmls_root)):
            for file in files:
                file_ext = file.split('.')[-1]
                if file_ext == 'inkml':
                    filepath = os.path.join(subdir, file)
                    tree = ET.parse(filepath)
                    root = tree.getroot()

                    if latex_gt:
                        # parse formulas from latex groundtruth
                        try:
                            latex_gt = root.find(doc_namespace + 'annotation[@type="truth"]').text
                            latex_gt = latex_gt.replace('$', '')
                            latex_gt = LatexVocab.split_to_tokens(latex_gt)
                            formulas.append(latex_gt)
                        except AttributeError:
                            # element not found
                            pass
                    if mathml_gt:
                        # parse formulas from mathml groundtruth
                        # get mathml annotation section and determine type
                        annotation_mathml_content = root.find(
                            doc_namespace + 'annotationXML[@type="truth"][@encoding="Content-MathML"]')
                        annotation_mathml_presentation = root.find(
                            doc_namespace + 'annotationXML[@type="truth"][@encoding="Presentation-MathML"]')
                        if annotation_mathml_content:
                            annotation_type = MathMLAnnotationType.CONTENT
                            annotation_mathml = annotation_mathml_content
                        elif annotation_mathml_presentation:
                            annotation_type = MathMLAnnotationType.PRESENTATION
                            annotation_mathml = annotation_mathml_presentation
                        else:
                            continue
                        # find mathml definition root
                        if annotation_type == MathMLAnnotationType.CONTENT:
                            math_root = annotation_mathml.find(mathml_namespace + 'math')
                        else:
                            math_root = annotation_mathml.find(doc_namespace + 'math')
                        if not math_root:
                            continue
                        try:
                            # append curly brackets - not in mathml, but definitely should be in tokens
                            formulas.append("{ }")
                            # different namespaces in various types of annotation
                            if annotation_type == MathMLAnnotationType.CONTENT:
                                file_symbols = LatexVocab.mathml_symbols_dfs(xml_namespace, mathml_namespace, math_root, substitute_terms)
                            else:
                                file_symbols = LatexVocab.mathml_symbols_dfs(xml_namespace, doc_namespace, math_root, substitute_terms)
                            for i, symbol in enumerate(file_symbols):
                                file_symbols[i] = mathml_unicode_to_latex_label(symbol, True)
                            file_symbols = " ".join(file_symbols)
                            formulas.append(file_symbols)
                        except AttributeError as e:
                            continue

        # get unique formulas
        formulas = set(formulas)
        logging.info(str(len(formulas)) + ' different formulas found')
        # split formulas to symbols and get unique symbols
        symbols = []
        for formula in formulas:
            symbols.extend(formula.split(' '))
        # write result to file
        with open(tgt_file, 'w') as fd:
            fd.write(' '.join(symbols))
        logging.info('Formulas written to ' + tgt_file)

    @staticmethod
    def mathml_symbols_dfs(xml_ns, mathml_ns, root, substitute_terms=False):
        """
        DFS traversal of MathML notation to retrieve symbols.
        :param xml_ns: xml namespace
        :param mathml_ns: mathml namespace
        :param root: current subtree root element
        :param substitute_terms: whether to substitute identifiers, numbers and text elements
        :return: list of symbols
        """
        if root.tag in [mathml_ns + 'mi', mathml_ns + 'mn', mathml_ns + 'mo', mathml_ns + 'mtext',
                        mathml_ns + 'mspace', mathml_ns + 'ms']:
            if substitute_terms:
                if root.tag == mathml_ns + 'mn':
                    return ['<NUM>']
                elif root.tag == mathml_ns + 'mi':
                    return ['<ID>']
                elif root.tag == mathml_ns + 'mtext':
                    return ['<TEXT>']
                else:
                    return [root.text]
            else:
                return [root.text]
        else:
            subtree_symbols = []
            if root.tag == mathml_ns + 'msqrt':
                subtree_symbols.append(r'\sqrt')
            elif root.tag == mathml_ns + 'mroot':
                subtree_symbols.append(r'\sqrt')
            elif root.tag == mathml_ns + 'msub':
                subtree_symbols.append(r'_')
            elif root.tag == mathml_ns + 'msup':
                subtree_symbols.append(r'^')
            elif root.tag == mathml_ns + 'mover':
                subtree_symbols.append(r'\overset')
            elif root.tag == mathml_ns + 'munder':
                subtree_symbols.append(r'\underset')
            elif root.tag == mathml_ns + 'msubsup':
                subtree_symbols.append(r'^')
                subtree_symbols.append(r'_')
            elif root.tag == mathml_ns + 'munderover':
                subtree_symbols.append(r'\overset')
                subtree_symbols.append(r'\underset')
            elif root.tag == mathml_ns + 'mfrac':
                subtree_symbols.append(r'\frac')

            for child in root:
                subtree_symbols.extend(LatexVocab.mathml_symbols_dfs(xml_ns, mathml_ns, child, substitute_terms))
            return subtree_symbols

    @staticmethod
    def split_to_tokens(latex_formula):
        """
        Splits LaTeX formula to tokens (on spaces) using regular expressions.
        :param: latex_formula: input LaTeX formula (string)
        :return: LaTeX formula with separated tokens (string)
        """
        # remove whitespace at the beginning and the end
        latex_formula = latex_formula.strip()
        # classic commands
        latex_formula = re.sub(r"(\\[a-zA-Z]+)(?![a-zA-Z])", r" \1 ", latex_formula)
        # special commands
        latex_formula = re.sub(r"(\\[\{\}\|\#\,])", r" \1 ", latex_formula)
        # variables
        latex_formula = re.sub(r"(?<![\\\a-zA-Z])([a-zA-Z]+)(?![a-zA-Z])", r" \1 ", latex_formula)
        # numbers
        latex_formula = re.sub(r"([0-9]+)", r" \1 ", latex_formula)
        # special characters
        latex_formula = re.sub(r"(?<!\\)([^a-zA-Z0-9\s\\])", r" \1 ", latex_formula)
        # remove multispaces
        latex_formula = re.sub(r"(\s\s+)", " ", latex_formula)
        # remove whitespaces at the beginning and the end
        latex_formula = latex_formula.strip()
        return latex_formula

    @staticmethod
    def create_tokenizer(formulas_file, min_freq=2):
        """
        Trains tokenizer on file of extracted symbols. Append special tokens
        :param formulas_file: file containing symbols separated with spaces
        :param min_freq: minimal frequence of symbol occurrence to be added to vocabulary
        :return: tokenizer object
        """
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Split(pattern=" ", behavior='isolated')
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [PAD] $B:1 [EOS]:1",
            special_tokens=[("[PAD]", 1), ("[BOS]", 2), ("[EOS]", 3)],
        )
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            min_frequency=min_freq,
            show_progress=True
        )
        tokenizer.train(trainer=trainer, files=[formulas_file])
        return tokenizer

    @staticmethod
    def save_tokenizer(tokenizer, tgt_file):
        """
        Saves tokenizer to a file.
        :param tokenizer: tokenizer object
        :param tgt_file: target filepath
        """
        tokenizer.save(tgt_file)

    @staticmethod
    def load_tokenizer(tokenizer_file):
        """
        Loads trained tokenizer from a file.
        :param tokenizer_file: tokenizer filepath
        :return: loaded tokenizer
        """
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError('Tokenizer file not found')
        return Tokenizer.from_file(tokenizer_file)
