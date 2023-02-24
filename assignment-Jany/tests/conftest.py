# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:58:01 2023

@author: pircajan
"""

import pytest

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_titanic_data():
    return load_dataset(file_name=config.app_config.test_data_file)
