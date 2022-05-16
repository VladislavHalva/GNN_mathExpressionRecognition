# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import json
import logging
import os
from json import JSONDecodeError


class Config:
    """
    App configuration handler.
    """
    def __init__(self, filepath):
        """
        Load configuration from JSON file and initialize parsing.
        :param filepath:
        """
        self.config = {}
        self.error = False
        self.error_messages = []

        if os.path.exists(filepath):
            with open(filepath) as json_file:
                try:
                    data = json.load(json_file)
                    self.data = data
                except JSONDecodeError as e:
                    logging.error("Config file parse error:")
                    logging.error(" " + str(e))
                    return

            self.parse_config()

    def getset_error(self, error_message=None):
        """
        Set error and print all error, that occurred.
        :param error_message: new error message
        """
        logging.error("Configuration error: ")
        if error_message is not None:
            self.error_messages.append(error_message)
            self.error = True
        for error_message in self.error_messages:
            logging.error(error_message)

    def get_error(self):
        """
        Print errors.
        """
        logging.error("Configuration error: ")
        for error_message in self.error_messages:
            logging.error(error_message)

    def parse_config(self):
        """
        Parsing of configuration file.
        """
        d = self.data
        # DEVICE
        if 'device' in d and d['device'] in ['gpu', 'cpu']:
            self.config['device'] = d['device']
        else:
            self.config['device'] = 'gpu'
        # SUMMARY WRITER
        if 'writer_path' in d and d['writer_path'] is not None and os.path.exists(d['writer_path']):
            self.config['writer_path'] = d['writer_path']
        else:
            self.config['writer_path'] = None
        # TMP DATA STORAGE
        if 'tmp_data_storage_folder' in d and d['tmp_data_storage_folder'] is not None and os.path.exists(d['tmp_data_storage_folder']):
            self.config['tmp_data_storage_folder'] = d['tmp_data_storage_folder']
        else:
            self.config['tmp_data_storage_folder'] = None
        # VOCABULARY CONFIG
        if 'vocabulary' not in d:
            self.getset_error('Vocabulary configuration missing')
            return
        self.config['vocabulary'] = {}
        if 'load_tokenizer' not in d['vocabulary']:
            self.getset_error('Vocabulary configuration missing load tokenizer')
            return
        if d['vocabulary']['load_tokenizer']:
            self.config['vocabulary']['load_tokenizer'] = True
        else:
            self.config['vocabulary']['load_tokenizer'] = False
        if 'inkml_folder_for_vocab' in d['vocabulary'] and d['vocabulary']['inkml_folder_for_vocab'] is not None and os.path.exists(d['vocabulary']['inkml_folder_for_vocab']):
            self.config['vocabulary']['inkml_folder_for_vocab'] = d['vocabulary']['inkml_folder_for_vocab']
        elif self.config['vocabulary']['load_tokenizer']:
            self.getset_error('Folder with InkML files not specified or does not exist for tokenizer.')
            return
        if 'vocab_filepath' in d['vocabulary'] and d['vocabulary']['vocab_filepath'] is not None and os.path.exists(d['vocabulary']['vocab_filepath']):
            self.config['vocabulary']['vocab_filepath'] = d['vocabulary']['vocab_filepath']
        elif self.config['vocabulary']['load_tokenizer']:
            self.getset_error('Path for vocabulary not specified or does not exist for tokenizer.')
            return
        if 'tokenizer_filepath' not in d['vocabulary']:
            self.getset_error('Tokenizer filepath not specified.')
            return
        if self.config['vocabulary']['load_tokenizer'] and d['vocabulary']['tokenizer_filepath'] is not None and not os.path.exists(d['vocabulary']['tokenizer_filepath']):
            self.getset_error("Specified tokenizer file not found")
            return
        self.config['vocabulary']['tokenizer_filepath'] = d['vocabulary']['tokenizer_filepath']
        # MODEL CONFIGURATION
        if 'model' not in d:
            self.getset_error("Model specification missing")
            return
        self.config['model'] = {}
        model_required_fields = [
            'model_name',
            'encoder_edge_fsize',
            'encoder_in_node_fsize',
            'encoder_hidden_node_fsize',
            'encoder_out_node_fsize',
            'decoder_in_fsize',
            'decoder_hidden_fsize',
            'decoder_embed_fsize',
            'decoder_attn_size',
            'dropout_encoder_vgg',
            'dropout_encoder_gat',
            'dropout_decoder_init_embed',
            'dropout_decoder_attention'
        ]
        for model_param in model_required_fields:
            if model_param not in d['model']:
                self.getset_error(model_param + 'specification missing')
                return
            else:
                self.config['model'][model_param] = d['model'][model_param]
        if 'load_state_dict' in d['model'] and d['model']['load_state_dict'] is not None and os.path.exists(d['model']['load_state_dict']):
            self.config['model']['load'] = True
            self.config['model']['load_state_dict'] = d['model']['load_state_dict']
        else:
            self.config['model']['load'] = False
            self.config['model']['load_state_dict'] = None
        # TRAINING CONFIGURATION
        if 'train' in d:
            self.config['train'] = {}
            if 'images_folder' in d['train'] and os.path.exists(d['train']['images_folder']):
                self.config['train']['images_folder'] = d['train']['images_folder']
            else:
                self.getset_error("Training images folder not set or not found")
                return
            if 'inkmls_folder' in d['train'] and os.path.exists(d['train']['inkmls_folder']):
                self.config['train']['inkmls_folder'] = d['train']['inkmls_folder']
            else:
                self.getset_error("Training InkML files folder not set or not found")
                return
            if 'epochs' not in d['train']:
                self.getset_error("Epochs count not set")
                return
            self.config['train']['epochs'] = d['train']['epochs']
            if 'batch_size' in d['train']:
                self.config['train']['batch_size'] = d['train']['batch_size']
            else:
                self.config['train']['batch_size'] = 1
            if 'save_model_folder' in d['train'] and d['train']['save_model_folder'] is not None and os.path.exists(
                    d['train']['save_model_folder']):
                self.config['train']['save_model_folder'] = d['train']['save_model_folder']
            if 'save_checkpoint_each_nth_epoch' in d['train']:
                self.config['train']['save_checkpoint_each_nth_epoch'] = d['train']['save_checkpoint_each_nth_epoch']
            else:
                self.config['train']['save_checkpoint_each_nth_epoch'] = 0
            if 'loss' in d['train']:
                self.config['train']['loss'] = {}
                loss_options = [
                    'loss_convnet',
                    'loss_encoder_nodes',
                    'loss_attention',
                    'loss_decoder_nodes',
                    'loss_decoder_edges',
                    'loss_decoder_end_nodes'
                ]
                for loss_option in loss_options:
                    if loss_option not in d['train']['loss']:
                        self.getset_error(loss_option + ' not set')
                        return
                    else:
                        self.config['train']['loss'][loss_option] = d['train']['loss'][loss_option]
            else:
                self.config['loss'] = {
                    'loss_convnet': 0.0,
                    'loss_encoder_nodes': 0.0,
                    'loss_attention': 0.0,
                    'loss_decoder_nodes': 1.0,
                    'loss_decoder_edges': 1.0,
                    'loss_decoder_end_nodes': 0.0
                }
            # EVAL DURING TRAINING 1
            if 'evaluation_during_train_1' in d['train']:
                self.config['train']['eval1'] = {}
                if 'each_nth_epoch' not in d['train']['evaluation_during_train_1']:
                    self.getset_error('Each nth epoch not set for train evaluation 1')
                    return
                else:
                    self.config['train']['eval1']['each_nth_epoch'] = d['train']['evaluation_during_train_1'][
                        'each_nth_epoch']
                if 'images_folder' not in d['train']['evaluation_during_train_1'] or not os.path.exists(
                        d['train']['evaluation_during_train_1']['images_folder']):
                    self.getset_error('Evaluation 1 images not set or found')
                    return
                else:
                    self.config['train']['eval1']['images_folder'] = d['train']['evaluation_during_train_1'][
                        'images_folder']
                if 'inkmls_folder' not in d['train']['evaluation_during_train_1'] or not os.path.exists(
                        d['train']['evaluation_during_train_1']['inkmls_folder']):
                    self.getset_error('Evaluation 1 InkML files not set or found')
                    return
                else:
                    self.config['train']['eval1']['inkmls_folder'] = d['train']['evaluation_during_train_1'][
                        'inkmls_folder']
                if 'batch_size' in d['train']['evaluation_during_train_1']:
                    self.config['train']['eval1']['batch_size'] = d['train']['evaluation_during_train_1']['batch_size']
                else:
                    self.config['train']['eval1']['batch_size'] = 1
                if 'print_stats_stdout' in d['train']['evaluation_during_train_1']:
                    self.config['train']['eval1']['print_stats_stdout'] = d['train']['evaluation_during_train_1'][
                        'print_stats_stdout']
                else:
                    self.config['train']['eval1']['print_stats_stdout'] = True
                if 'print_item_level_stats_stdout' in d['train']['evaluation_during_train_1']:
                    self.config['train']['eval1']['print_item_level_stats_stdout'] = \
                    d['train']['evaluation_during_train_1']['print_item_level_stats_stdout']
                else:
                    self.config['train']['eval1']['print_item_level_stats_stdout'] = False
                if 'beam_search' in d['train']['evaluation_during_train_1']:
                    self.config['train']['eval1']['beam_search'] = d['train']['evaluation_during_train_1'][
                        'beam_search']
                else:
                    self.config['train']['eval1']['beam_search'] = True
                if 'beam_width' in d['train']['evaluation_during_train_1']:
                    self.config['train']['eval1']['beam_width'] = d['train']['evaluation_during_train_1']['beam_width']
                else:
                    self.config['train']['eval1']['beam_width'] = 3
                # batch_size to 1 if beam search
                if self.config['train']['eval1']['beam_search']:
                    self.config['train']['eval1']['batch_size'] = 1
            # EVAL DURING TRAINING 2
            if 'evaluation_during_train_2' in d['train']:
                self.config['train']['eval2'] = {}
                if 'each_nth_epoch' not in d['train']['evaluation_during_train_2']:
                    self.getset_error('Each nth epoch not set for train evaluation 1')
                    return
                else:
                    self.config['train']['eval2']['each_nth_epoch'] = d['train']['evaluation_during_train_2'][
                        'each_nth_epoch']
                if 'images_folder' not in d['train']['evaluation_during_train_2'] or not os.path.exists(
                        d['train']['evaluation_during_train_2']['images_folder']):
                    self.getset_error('Evaluation 1 images not set or found')
                    return
                else:
                    self.config['train']['eval2']['images_folder'] = d['train']['evaluation_during_train_2'][
                        'images_folder']
                if 'inkmls_folder' not in d['train']['evaluation_during_train_2'] or not os.path.exists(
                        d['train']['evaluation_during_train_2']['inkmls_folder']):
                    self.getset_error('Evaluation 1 InkML files not set or found')
                    return
                else:
                    self.config['train']['eval2']['inkmls_folder'] = d['train']['evaluation_during_train_2'][
                        'inkmls_folder']
                if 'batch_size' in d['train']['evaluation_during_train_2']:
                    self.config['train']['eval2']['batch_size'] = d['train']['evaluation_during_train_2'][
                        'batch_size']
                else:
                    self.config['train']['eval2']['batch_size'] = 1
                if 'print_stats_stdout' in d['train']['evaluation_during_train_2']:
                    self.config['train']['eval2']['print_stats_stdout'] = \
                        d['train']['evaluation_during_train_2']['print_stats_stdout']
                else:
                    self.config['train']['eval2']['print_stats_stdout'] = True
                if 'print_item_level_stats_stdout' in d['train']['evaluation_during_train_2']:
                    self.config['train']['eval2']['print_item_level_stats_stdout'] = \
                        d['train']['evaluation_during_train_2']['print_item_level_stats_stdout']
                else:
                    self.config['train']['eval2']['print_item_level_stats_stdout'] = False
                if 'beam_search' in d['train']['evaluation_during_train_2']:
                    self.config['train']['eval2']['beam_search'] = d['train']['evaluation_during_train_2'][
                        'beam_search']
                else:
                    self.config['train']['eval2']['beam_search'] = True
                if 'beam_width' in d['train']['evaluation_during_train_2']:
                    self.config['train']['eval2']['beam_width'] = d['train']['evaluation_during_train_2'][
                        'beam_width']
                else:
                    self.config['train']['eval2']['beam_width'] = 3
                # batch_size to 1 if beam search
                if self.config['train']['eval2']['beam_search']:
                    self.config['train']['eval2']['batch_size'] = 1
        # EVALUATION CONFIGURATION
        if 'evaluate' in d:
            self.config['evaluate'] = {}
            if 'images_folder' not in d['evaluate'] or not os.path.exists(
                    d['evaluate']['images_folder']):
                self.getset_error('Evaluation images not set or found')
                return
            else:
                self.config['evaluate']['images_folder'] = d['evaluate']['images_folder']
            if 'inkmls_folder' not in d['evaluate'] or not os.path.exists(d['evaluate']['inkmls_folder']):
                self.getset_error('Evaluation InkML files not set or found')
                return
            else:
                self.config['evaluate']['inkmls_folder'] = d['evaluate']['inkmls_folder']
            if 'batch_size' in d['evaluate']:
                self.config['evaluate']['batch_size'] = d['evaluate']['batch_size']
            else:
                self.config['evaluate']['batch_size'] = 1
            if 'print_stats_stdout' in d['evaluate']:
                self.config['evaluate']['print_stats_stdout'] = \
                    d['evaluate']['print_stats_stdout']
            else:
                self.config['evaluate']['print_stats_stdout'] = True
            if 'print_item_level_stats_stdout' in d['evaluate']:
                self.config['evaluate']['print_item_level_stats_stdout'] = \
                    d['evaluate']['print_item_level_stats_stdout']
            else:
                self.config['evaluate']['print_item_level_stats_stdout'] = False
            if 'beam_search' in d['evaluate']:
                self.config['evaluate']['beam_search'] = d['evaluate']['beam_search']
            else:
                self.config['evaluate']['beam_search'] = True
            if 'beam_width' in d['evaluate']:
                self.config['evaluate']['beam_width'] = d['evaluate']['beam_width']
            else:
                self.config['evaluate']['beam_width'] = 3
            if 'store_results_folder' in d['evaluate'] and d['evaluate']['store_results_folder'] is not None and os.path.exists(d['evaluate']['store_results_folder']):
                self.config['evaluate']['store_results_folder'] = d['evaluate']['store_results_folder']
            else:
                self.config['evaluate']['store_results_folder'] = None
            if 'results_author' in d['evaluate']:
                self.config['evaluate']['results_author'] = d['evaluate']['results_author']
            else:
                self.config['evaluate']['results_author'] = 'mer_g2g_model'
            # batch_size to 1 if beam search
            if self.config['evaluate']['beam_search']:
                self.config['evaluate']['batch_size'] = 1

    def get(self):
        """
        Getter for configuration dictionary.
        :return: config or None if error
        """
        if not self.error:
            return self.config
        else:
            self.get_error()
            return None

    def print(self):
        """
        Print loaded configuration to STDOUT.
        """
        if self.error:
            self.get_error()
        else:
            print(json.dumps(self.config, indent=2, sort_keys=False))

