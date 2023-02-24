# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:48:41 2023

@author: pircajan
"""
from typing import List, Optional, Tuple
import pandas as pd 
import numpy as np
from config.core import config
from pydantic import BaseModel, ValidationError # Validation of variables...
'''
Package for data validation or preprocessing before inference
'''


def validate_inputs(*, input_data: pd.Dataframe) -> pd.DataFrame:

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None
    
    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records") )
    except ValidationError as error:
        errors = error.json()
    
    return validated_data, errors
    

def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    
    input_data.dropna(inplace=True)

    return input_data




class TitanicDataInputSchema(BaseModel):
    pclass: Optional[str]
    survived: Optional[str]
    sex: Optional[str]
    age: Optional[str]
    sibsp: Optional[str]
    parch: Optional[str]
    fare: Optional[str]
    cabin: Optional[str]
    embarked: Optional[str]
    title: Optional[str]



class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
