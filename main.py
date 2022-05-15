import logging

from src.Trainer import Trainer

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    train = True
    evaluate = True

    trainer = Trainer(
        model_name='MER_tSimple_eSimple',
        tokenizer_path='assets/tokenizer.json',
        vocab_path='assets/vocab.txt',
        load_vocab=True,
        inkml_folder_vocab='assets/crohme/train/inkml',
        load_model=None,
        writer='runs/',
        temp_path=None
    )

    if train:
        loss_config = {
            'loss_convnet': 0.0,
            'loss_encoder_nodes': 0.0,
            'loss_attention': 0.0,
            'loss_decoder_nodes': 1.0,
            'loss_decoder_edges': 1.0,
            'loss_decoder_end_nodes': 0.0
        }

        trainer.set_eval_during_training(
            images_root='assets/crohme/simple/img/',
            inkmls_root='assets/crohme/simple/inkml/',
            batch_size=1,
            print_stats=True,
            print_item_level_stats=False,
            each_nth_epoch=5,
            beam_search=True,
            beam_width=3
        )
        # trainer.set_second_eval_during_training(
        #     images_root='assets/crohme/simple/img/',
        #     inkmls_root='assets/crohme/simple/inkml/',
        #     batch_size=1,
        #     print_stats=True,
        #     print_item_level_stats=False,
        #     each_nth_epoch=3,
        #     beam_search=False,
        #     beam_width=3
        # )
        trainer.train(
            images_root='assets/crohme/simple/img/',
            inkmls_root='assets/crohme/simple/inkml/',
            epochs=100,
            batch_size=6,
            loss_config=loss_config,
            save_model_dir='checkpoints/',
            save_checkpoint_each_nth_epoch=0
        )

    if evaluate:
        trainer.evaluate(
            images_root='assets/crohme/simple/img/',
            inkmls_root='assets/crohme/simple/inkml/',
            batch_size=1,
            print_stats=True,
            print_item_level_stats=True,
            store_results_dir=None,
            results_author='Vladislav Halva',
            beam_search=True,
            beam_width=3
        )

