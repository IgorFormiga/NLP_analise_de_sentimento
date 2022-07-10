from unidecode import unidecode
from sklearn.base import BaseEstimator, TransformerMixin

# Defining a function to remove the stopwords and to lower the comments
def normalization_process(texto_lista):
    '''
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto em que será normalizada
    '''
    return [unidecode(c) for c in texto_lista.split()]

# Criação de classe para normalização dos dados
class ProcessoNormalizacao(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        return [' '.join(normalization_process(review)) for review in X]
        