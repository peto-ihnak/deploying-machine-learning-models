# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:43:57 2023

@author: pircajan
"""
import typing as t

from config.core import config
from regression_model import __version__ as _version
from processing.data_manager import load_pipeline
from processing.validation import validate_inputs
import pandas as pd

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe = load_pipeline(pipeline_file_name)


def make_predictions(*, input_data: t.Union[pd.DataFrame, dict], ) -> dict:
    
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    if not errors:
        predictions = titanic_pipe.predict(
            X=validated_data[config.model_config.features]  # Duplicated .. but never mind
        )
        results = {
            "predictions": [pred for pred in predictions],  # type: ignore
            "version": _version,
            "errors": errors,
        }

    return results