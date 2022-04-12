import logging
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re

from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence, Split
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

from pylatexenc.latexwalker import LatexWalker, LatexMathNode

from src.data.LTokenizer import LTokenizer
from src.definitions.MathMLAnnotationType import MathMLAnnotationType
from src.definitions.exceptions.ItemLoadError import ItemLoadError


class LatexVocab:
    @staticmethod
    def generate_formulas_file_from_inkmls(inkmls_root, tgt_file):
        if not os.path.exists(inkmls_root):
            raise FileNotFoundError('Inkmls directory not found')

        doc_namespace = '{http://www.w3.org/2003/InkML}'

        formulas = []
        for subdir, _, files in tqdm(os.walk(inkmls_root)):
            for file in files:
                file_ext = file.split('.')[-1]
                if file_ext == 'inkml':
                    filepath = os.path.join(subdir, file)
                    tree = ET.parse(filepath)
                    root = tree.getroot()

                    try:
                        latex_gt = root.find(doc_namespace + 'annotation[@type="truth"]').text
                        latex_gt = latex_gt.replace('$', '')
                        latex_gt = LatexVocab.split_to_tokens(latex_gt)
                        formulas.append(latex_gt)
                    except AttributeError:
                        # element not found
                        pass

        formulas = set(formulas)
        logging.info(str(len(formulas)) + ' different formulas found')

        with open(tgt_file, 'w') as fd:
            fd.write('\n'.join(formulas))

        logging.info('Formulas written to ' + tgt_file)

    @staticmethod
    def mathml_symbols_dfs(xml_ns, mathml_ns, root):
        if root.tag in [mathml_ns + 'mi', mathml_ns + 'mn', mathml_ns + 'mo', mathml_ns + 'mtext',
                        mathml_ns + 'mspace', mathml_ns + 'ms']:
            return [root.text]
        else:
            children_symbols = []
            for child in root:
                children_symbols.extend(LatexVocab.mathml_symbols_dfs(xml_ns, mathml_ns, child))
            return children_symbols

    @staticmethod
    def generate_formulas_file_from_inkmls_mathml(inkmls_root, tgt_file):
        if not os.path.exists(inkmls_root):
            raise FileNotFoundError('Inkmls directory not found')

        xml_namespace = '{http://www.w3.org/XML/1998/namespace}'
        doc_namespace = '{http://www.w3.org/2003/InkML}'
        mathml_namespace = '{http://www.w3.org/1998/Math/MathML}'

        symbols = []
        for subdir, _, files in tqdm(os.walk(inkmls_root)):
            for file in files:
                file_ext = file.split('.')[-1]
                if file_ext == 'inkml':
                    filepath = os.path.join(subdir, file)
                    tree = ET.parse(filepath)
                    root = tree.getroot()

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

                    # find math definition root
                    if annotation_type == MathMLAnnotationType.CONTENT:
                        math_root = annotation_mathml.find(mathml_namespace + 'math')
                    else:
                        math_root = annotation_mathml.find(doc_namespace + 'math')
                    if not math_root:
                        continue

                    try:
                        # different namespaces in various types of annotation
                        if annotation_type == MathMLAnnotationType.CONTENT:
                            file_symbols = LatexVocab.mathml_symbols_dfs(xml_namespace, mathml_namespace, math_root)
                        else:
                            file_symbols = LatexVocab.mathml_symbols_dfs(xml_namespace, doc_namespace, math_root)
                        symbols.extend(file_symbols)
                    except AttributeError as e:
                        continue

        symbols = set(symbols)
        symbols = ' '.join(symbols)

        with open(tgt_file, 'a') as fd:
            fd.write("\n")
            fd.write(symbols)

        logging.info('Symbols written to ' + tgt_file)

    @staticmethod
    def split_to_tokens(latex_formula):
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

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Split(pattern=" ", behavior='isolated')
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [PAD] $B:1 [EOS]:1",
            special_tokens=[("[PAD]", 1), ("[BOS]", 2), ("[EOS]", 3)],
        )

        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            min_frequency=min_freq,
            show_progress=True
        )

        tokenizer.train(trainer=trainer, files=[formulas_file])
        return tokenizer

    @staticmethod
    def save_tokenizer(tokenizer, tgt_file):
        tokenizer.save(tgt_file)

    @staticmethod
    def load_tokenizer(tokenizer_file):
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError('Tokenizer file not found')

        return Tokenizer.from_file(tokenizer_file)

    # @staticmethod
    # def create_tokenizer(formulas_file, min_freq=1):
    #     words = []
    #     with open(formulas_file, 'r') as file:
    #         for line in file:
    #             line_words = line.split(' ')
    #             line_words = [word.strip() for word in line_words]
    #             words.extend(line_words)
    #     words = list(set(words))
    #     words.extend(['[PAD]', '[EOS]', '[UNK]'])
    #     return LTokenizer(words)
    #
    # @staticmethod
    # def load_tokenizer(tokenizer_file):
    #     if not os.path.exists(tokenizer_file):
    #         raise FileNotFoundError('Tokenizer file not found')
    #
    #     return LTokenizer.from_file(tokenizer_file)
