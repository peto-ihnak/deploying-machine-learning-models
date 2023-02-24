# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:54:18 2023

@author: pircajan
"""
import math
import numpy as np

from classification_model.predict import make_predictions

def test_make_predictions(sample_titanic_data):
    expected_first_prediction_value = 0
    expected_no_predictions = 262
    
    # When
    result = make_predictions(input_data=sample_titanic_data)
    
    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    #assert isinstance(predictions[0], int)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=1)
    assert predictions[0] == 0
    assert predictions[1] == 1
