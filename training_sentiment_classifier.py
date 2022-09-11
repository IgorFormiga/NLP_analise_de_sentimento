"""
Este script é responsavel pro ler, preparar e trainar o classificador de sentimento binário (negativo, other)
a partir dos reviews (comentários) retirados do web-site de e-comerce brasileiro

* Metadata: https://www.kaggle.com/olistbr/brazilian-ecommerce
* Notebook de referência: *.ipynb

--- SUMÁRIO ---

1. Variáveis de projeto
2. Leitura de dados
3. PREPARAÇÃO INICIAL
4. Pepiline de Trasformação
5. Trainando o modelo
6. Salvando o modelo

---------------------------------------------------------------
Escrito por Igor C. Formiga - Última: 09/09/2022
---------------------------------------------------------------
"""

import os
import pandas as pd
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from utils.regex_utils import *
from utils.stopword_utils import RemoverStopwords
from utils.normalize_utils import ProcessoNormalizacao
from utils.stemming_utils import ProcessoStemming
from utils.lemmatization_utils import ProcessoLemmatization
from joblib import dump

"""
-----------------------------------
------ 1. VARIÁVEIS DO PROJETO -------
-----------------------------------
"""

# Variables for address paths
DATA_PATH = 'datasets'

# Referências 
PIPELINES = '3_analise_sentimento.ipynb' # Take a look at your project structure
MODELS = '3_analise_sentimento.ipynb' # Take a look at your project structure

# Variables for reading the data
FILENAME = 'order_reviews.csv'
COLS_READ = ['review_comment_message', 'review_score']
CORPUS_COL = 'review_comment_message'
TARGET_COL = 'review_score'

# Defining stopwords
PT_STOPWORDS = stopwords.words('portuguese')

# Variables for retrieving model
MODEL_KEY = 'LogisticRegression'

"""
-----------------------------------
------- 2. LEITURA DE DADOS -------
-----------------------------------
"""

# Reading the data with text corpus and score
dataset = pd.read_csv(os.path.join(DATA_PATH, FILENAME), sep=';')


"""
-------------------------------------------
------- 3. PREPARAÇÃO INICIAL--------------
-------------------------------------------
"""

# Removendo todos os dados
dataset.drop(['review_id', 'review_creation_date', 'review_answer_timestamp', 'review_comment_title', 'order_id'], axis=1, inplace=True)
dataset = dataset.dropna().reindex()
print(dataset.shape)
# Dicionario para mapear os valores da coluna target
score_map = {
    1: 'negativo',
    2: 'negativo',
    3: 'other',
    4: 'other',
    5: 'other'}

# Substituindo os valores da coluna 'review_score' (target)
dataset['review_score'] = dataset['review_score'].map(score_map) 

"""
---------------------------------------------
------- 4. Pepiline de Trasformação ---------
---------------------------------------------
"""

# Definindo todas as trafomações regex para serem aplicadas ao pipeline
regex_transformers = {
    'datas': re_dates,
    'valores_dinheiro': re_money,
    'numeros': re_numbers,
    'negacoes': re_negation,
    'caracteres_especiais': re_special_chars,
    'espacos_branco': re_whitespaces
}


# Criando o Pipeline
text_pipeline = Pipeline([
    ('regex', RemoverRegex(regex_transformers)),
    ('stopwords', RemoverStopwords(stopwords.words('portuguese'))),
    ('normalization', ProcessoNormalizacao()),
    ('lemmatization', ProcessoLemmatization()),
    ('vectorizer', TfidfVectorizer(min_df=15))
])


# definindo X e y 
X = dataset[CORPUS_COL].tolist()
y = dataset[TARGET_COL]

# Aplicando o pipelin
X_prep = text_pipeline.fit_transform(X)
print(X_prep.shape)

'''# Saving states before prep pipeline
dataset[CORPUS_COL] = X_prep
dataset[CORPUS_COL].to_csv(os.path.join(DATA_PATH, 'X_data.csv'), index=False)
dataset[TARGET_COL].to_csv(os.path.join(DATA_PATH, 'y_data.csv'), index=False)
'''

"""
--------------------------------------------
--------- 5. Trainando o modelo  -----------
--------------------------------------------
"""

# Criando o modelo de classificação (Hiperparâmetros definidos no 3_analise_sentimento.ipynb)
regressao_logistica = LogisticRegression(C=3.2157736102040038,class_weight=None, max_iter=1000, penalty='l2', solver = "liblinear")
regressao_logistica.fit(X_prep, y)


"""
-------------------------------------------
--------- 6. Salvando o modelo  -----------
-------------------------------------------
"""

# Salvando o modelo
filename = 'regressao_logistica_stemming.sav'
dump(regressao_logistica, filename)


