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
