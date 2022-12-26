import pytest
import os

def test_if_emnist_data_present():
    # check that_data_dir_exists
    DATA_DIR = "./emnist_data/"
    assert os.path.exists(DATA_DIR) 
    filenames = os.listdir(DATA_DIR)
    assert len(filenames) > 10
    assert any("digits" in name for name in filenames)
