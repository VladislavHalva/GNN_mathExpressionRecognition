import argparse
import json
import logging

from src.Trainer import Trainer
from src.utils.Config import Config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    # parse arguments
    parser = argparse.ArgumentParser(description='Mathematical expressions recognition tool.')
    parser.add_argument('-c', '--config', help='Path to configuration file', required=True)
    parser.add_argument('-p', '--print_config', help='Description for bar argument', required=False, action='store_true')
    args = vars(parser.parse_args())
    # build configuration
    configurator = Config(args['config'])
    config = configurator.get()
    if config is None:
        exit(1)
    if args['print_config']:
        configurator.print()
    # set-up run
    train = 'train' in config
    evaluate = 'evaluate' in config
    # init trainer
    trainer = Trainer(config)
    # set training and evaluation during training
    if train:
        if 'eval1' in config['train']:
            trainer.set_eval_during_training(
                images_root=config['train']['eval1']['images_folder'],
                inkmls_root=config['train']['eval1']['inkmls_folder'],
                batch_size=config['train']['eval1']['batch_size'],
                print_stats=config['train']['eval1']['print_stats_stdout'],
                print_item_level_stats=config['train']['eval1']['print_item_level_stats_stdout'],
                each_nth_epoch=config['train']['eval1']['each_nth_epoch'],
                beam_search=config['train']['eval1']['beam_search'],
                beam_width=config['train']['eval1']['beam_width']
            )
        if 'eval2' in config['train']:
            trainer.set_second_eval_during_training(
                images_root=config['train']['eval2']['images_folder'],
                inkmls_root=config['train']['eval2']['inkmls_folder'],
                batch_size=config['train']['eval2']['batch_size'],
                print_stats=config['train']['eval2']['print_stats_stdout'],
                print_item_level_stats=config['train']['eval2']['print_item_level_stats_stdout'],
                each_nth_epoch=config['train']['eval2']['each_nth_epoch'],
                beam_search=config['train']['eval2']['beam_search'],
                beam_width=config['train']['eval2']['beam_width']
            )
            trainer.train(
                images_root=config['train']['images_folder'],
                inkmls_root=config['train']['inkmls_folder'],
                epochs=config['train']['epochs'],
                batch_size=config['train']['batch_size'],
                loss_config=config['train']['loss'],
                save_model_dir=config['train']['save_model_folder'],
                save_checkpoint_each_nth_epoch=config['train']['save_checkpoint_each_nth_epoch']
            )

    # set evaluation
    if evaluate:
        trainer.evaluate(
            images_root=config['evaluate']['images_folder'],
            inkmls_root=config['evaluate']['inkmls_folder'],
            batch_size=config['evaluate']['batch_size'],
            print_stats=config['evaluate']['print_stats_stdout'],
            print_item_level_stats=config['evaluate']['print_item_level_stats_stdout'],
            store_results_dir=config['evaluate']['store_results_folder'],
            results_author=config['evaluate']['results_author'],
            beam_search=config['evaluate']['beam_search'],
            beam_width=config['evaluate']['beam_width'],
        )

