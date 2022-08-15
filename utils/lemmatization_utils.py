from sklearn.base import BaseEstimator, TransformerMixin
import spacy


def lemmatization_process(texto_lista):
    '''
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto em que será normalizada
    stemmer: class, default: RSLPStemmer()
        Tipo de Stemmer que será aplicado ao texto
    '''
    # Realizando download para a linguagem Pt-br
    nlp =  spacy.load('pt_core_news_sm')

    return [token.lemma_ for token in nlp(texto_lista)]


# Criação da classe para aplicar o processo de stemming
class ProcessoLemmatization(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        return [' '.join(lemmatization_process(comment)) for comment in X]