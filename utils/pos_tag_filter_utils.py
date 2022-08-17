from sklearn.base import BaseEstimator, TransformerMixin
import spacy


def filter_pos_tag(texto_lista, list_filter_pos_tag, select='not in'):
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

    remove_verbs = list()
    if select == 'in':
        for token in nlp(texto_lista):
            if token.pos_ in (list_filter_pos_tag):
                remove_verbs.append(token.lemma_)

    else:
        for token in nlp(texto_lista):
            if token.pos_ not in (list_filter_pos_tag):
                remove_verbs.append(token.lemma_)

    return remove_verbs


# Criação da classe para aplicar o processo de stemming
class ProcessoFilterPosTag(BaseEstimator, TransformerMixin):

    def __init__(self, list_filter_pos_tag, select='not in'):
        self.filter_pos_tag = list_filter_pos_tag
        self.select = select
    
    def fit(self, X, y=None):
        return self

    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        return [' '.join(filter_pos_tag(comment, self.filter_pos_tag, self.select)) for comment in X]