import logging

import torch

from src.Trainer import Trainer

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    train = False
    evaluate = True

    trainer = Trainer(
        model_name='MER_tSimple_eSimple',
        tokenizer_path='assets/tokenizer.json',
        vocab_path='assets/vocab.txt',
        load_vocab=True,
        inkml_folder_vocab='assets/crohme/train/inkml',
        load_model='checkpoints/MER_tSimple_eSimple_22-05-08_18-57-46_final.pth',
        writer='runs/',
        temp_path='temp/'
    )

    if train:
        trainer.set_eval_during_training(
            images_root='assets/crohme/simple/img/',
            inkmls_root='assets/crohme/simple/inkml/',
            batch_size=4,
            print_stats=True,
            print_item_level_stats=False,
            each_nth_epoch=20
        )
        trainer.train(
            images_root='assets/crohme/simple/img/',
            inkmls_root='assets/crohme/simple/inkml/',
            epochs=100,
            batch_size=4,
            save_model_dir='checkpoints/',
            save_checkpoint_each_nth_epoch=50
        )

    if evaluate:
        trainer.evaluate(
            images_root='assets/crohme/simple/img/',
            inkmls_root='assets/crohme/simple/inkml/',
            batch_size=1,
            print_stats=True,
            print_item_level_stats=True,
            store_results_dir=None,
            results_author='Vladislav Halva'
        )

