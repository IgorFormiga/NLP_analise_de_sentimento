from re import sub
from sklearn.base import BaseEstimator, TransformerMixin

def re_dates(texto_lista):
    '''
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto para ser tratado
    '''
    # Criando a RegEx para representação de datas dd/mm/YY
    # \  -> indica que o caractere seguinde não é um metacaractere
    # () -> indica que os caracteres fazem parte de um grupo
    # | -> indica OU
    # [ - ] -> indica que a informação encontra dentro de um intervalo indo de um valor a outro
    # { - } -> indica o minimo e o máximo de repetições de um caractere
    # \d -> indica qualquer algarismo de 0 a 9
    padrao = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [sub(padrao, ' data ', r) for r in texto_lista]


def re_money(text_list):
    """
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto para ser tratado
    """
    
    # Criando a RegEx para representação de valores em reais (R$__,_ _)
    # + -> indica que o caractere antecessor pode ser repetido de uma vez até n vezes 
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [sub(pattern, ' dinheiro ', r) for r in text_list]

def re_negation(text_list):
    """
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto para ser tratado
    """
    
    # Criando a RegEx para termos que representam 'não' [Não, não, NÃO...]
    return [sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', r) for r in text_list]

def re_special_chars(text_list):
    """
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto para ser tratado
    """
    
    # Criando a RegEx
    # \W -> indica que o caratere não é uma palavra, ou seja, caracteres que não são de a a z, A a Z, 0 a 9. 
    return [sub('\W', ' ', r) for r in text_list]

def re_whitespaces(text_list):
    """
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto para ser tratado
    """
    
    # Criando a RegEx
    # \s -> [ \t\n\r\f\v]
    # ' ' (espaço)
    # \t TAB
    # \r retono de carro (volta o cursos para o inicio da linha)
    # \f avanço de pagina
    # \v vertical TAB
    white_spaces = [sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end

def re_numbers(text_list):
    """
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto para ser tratado
    """
    
    # Criando a RegEx
    # \d -> indica qualquer algarismo de 0 a 9
    return [sub('\d+', ' numero ', r) for r in text_list]

# Criação de classe para a aplicação da remoção das expressões regulares
class RemoverRegex(BaseEstimator, TransformerMixin):
    
    def __init__(self, trasformacoes_regex):
        # trasformacoes_regex é um dicionário no qual vai conter todas as traformações de Regex
        self.trasformacoes_regex = trasformacoes_regex
        
    def fit(self, X, y=None):
        return self
    
    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        # Aplicando todas as funções de regex descritas no dicionario trasformacoes_regex
        for regex_name, regex_function in self.trasformacoes_regex.items():
            X = regex_function(X)
            
        return X
        