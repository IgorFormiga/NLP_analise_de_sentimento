from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

def extract_features_from_corpus(texto_lista, vectorizer, df=False):
    '''
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto em que será normalizada
    vectorizer: object
        Engenharia utilizada para extração das features de texto
    '''

    # Extracting features
    corpus_features = vectorizer.fit_transform(texto_lista).toarray()
    features_names = vectorizer.get_feature_names_out()
    df_corpus_features = None
    if df:
        # Transforming into a dataframe to give interpetability to the process
        df_corpus_features = DataFrame(corpus_features, columns=features_names)
    
    return corpus_features, df_corpus_features

# Criação da classe para extrair features 
class ExtracaoFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, vectorizer):
        # vectorizer é a engenharia utilizada para extração das features de texto
        self.vectorizer = vectorizer
        
    def fit(self, X, y=None):
        return self
    
    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        return self.vectorizer.fit_transform(X).toarray()