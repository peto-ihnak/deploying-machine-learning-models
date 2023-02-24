from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer

def test_ExtractLetterTransformer(sample_titanic_data):
    trans = ExtractLetterTransformer(variables=config.model_config.cabin)
    
    assert sample_titanic_data['cabin'].iat[1] == 'C78'
    
    subject = trans.fit_transform(sample_titanic_data)
    
    assert subject["cabin"].iat[1] == 'C'
    # assert subject["cabin"].iat[0] == ''