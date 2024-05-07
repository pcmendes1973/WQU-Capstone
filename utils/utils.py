import configparser
import os


def load_config():
    config = configparser.ConfigParser()

    # Get the directory path of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the parent directory
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

    # Construct the path to config.ini in the parent directory
    config_path = os.path.join(parent_directory, 'config.ini')
    config.read(config_path)

    return config


def set_seeds(seed_state):
    np.random.seed(seed_state)
    random.seed(seed_state)
    tf.random.set_seed(seed_state)