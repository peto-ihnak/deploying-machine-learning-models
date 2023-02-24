# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:13:38 2023

@author: pircajan
"""
from config.core import config
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split
from pipeline import titanic_pipe
from sklearn.metrics import accuracy_score, roc_auc_score



def run_training() -> None:
    """Train the model."""
    
    data = load_dataset(file_name=config.app_config.data_file)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
       data[config.model_config.features],  # predictors
       data[config.model_config.target],
       test_size=config.model_config.test_size,
       random_state=config.model_config.random_state,
       )
    
    titanic_pipe.fit(X_train, y_train)
    
    
    
    class_ = titanic_pipe.predict(X_train)
    pred = titanic_pipe.predict_proba(X_train)[:,1]
    
    print('Model of Titanic has been trained sucessfully !')
    # determine mse and rmse
    print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
    print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
    print()
    
    # make predictions for test set
    class_ = titanic_pipe.predict(X_test)
    pred = titanic_pipe.predict_proba(X_test)[:,1]
    
    # determine mse and rmse
    print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
    print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
    print()
    
    
    save_pipeline(pipeline_to_persist=titanic_pipe)
    


if __name__ == "__main__":
    run_training()