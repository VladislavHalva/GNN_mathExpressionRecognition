import logging

from src.Trainer import Trainer

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    trainer = Trainer(
        model_name='MER',
        tokenizer_path='assets/tokenizer.json',
        vocab_path='assets/vocab.txt',
        load_vocab=True,
        inkml_folder_vocab='assets/crohme/train/inkml',
        load_model='MER_22-05-03_15-32-51_final.pth',
        writer='runs/',
        temp_path=None
    )

    trainer.set_eval_during_training(
        images_root='assets/crohme/simple/img/',
        inkmls_root='assets/crohme/simple/inkml/',
        batch_size=4,
        print_stats=True,
        print_item_level_stats=True,
        each_nth_epoch=20
    )
    trainer.train(
        images_root='assets/crohme/simple/img/',
        inkmls_root='assets/crohme/simple/inkml/',
        epochs=150,
        batch_size=4,
        save_model_dir='checkpoints/'
    )

    trainer.evaluate(
        images_root='assets/crohme/simple/img/',
        inkmls_root='assets/crohme/simple/inkml/',
        batch_size=4,
        print_stats=True,
        print_item_level_stats=True
    )

