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
        load_model=None,
        writer='runs/'
    )

    trainer.set_eval_during_training(
        images_root='assets/crohme/simple/img/',
        inkmls_root='assets/crohme/simple/inkml/',
        batch_size=4,
        print_stats=True,
        print_item_level_stats=False,
        each_nth_epoch=1
    )
    trainer.train(
        images_root='assets/crohme/simple/img/',
        inkmls_root='assets/crohme/simple/inkml/',
        epochs=100,
        batch_size=4,
        temp_path=None,
        save_model_dir='checkpoints/'
    )

    trainer.evaluate(
        images_root='assets/crohme/simple/img/',
        inkmls_root='assets/crohme/simple/inkml/',
        batch_size=4,
        print_stats=True,
        print_item_level_stats=False
    )

