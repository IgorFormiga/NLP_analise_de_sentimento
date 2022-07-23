from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from skopt import BayesSearchCV


def print_score(model, X_processed, y, tuned_parameters):
	
	# Estratégia para avaliar o desempenho do modelo de validação cruzada no conjunto de testes.
	scorer = make_scorer(f1_score, average = 'weighted')
	'''
	'weighted':
	Calcule as métricas para cada rótulo e encontre sua média ponderada pelo suporte 
	(o número de instâncias verdadeiras para cada rótulo). Isso altera 'macro' para 
	levar em conta o desequilíbrio do rótulo; pode resultar em um F-score que não está 
	entre precisão e recuperação.
	'''


	print(f"# Ajustando hiperparâmetros para {scorer}")
	print()
	
	clf = GridSearchCV(model, 
                        tuned_parameters, 
                        scoring=scorer,
						cv = KFold(n_splits = 3, shuffle=True))
	
	X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=.20, random_state=42)

	clf.fit(X_train, y_train)

	print("Melhor conjunto de parâmetros encontrado: ")
	print()
	print(clf.best_params_)
	print()
	print("Pontuações de grade no conjunto de desenvolvimento:")
	print()
	means = clf.cv_results_["mean_test_score"]
	stds = clf.cv_results_["std_test_score"]
	for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
		print(f'{mean:.3f} (+/-{(std * 2):.3f}) for {params}')
	print()
	print("Relatório de classificação detalhado para o melhor modelo:")
	print()
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))



def print_score_BayesSearchCV(model, X_processed, y, search_spaces, n_iter):
	
	# Estratégia para avaliar o desempenho do modelo de validação cruzada no conjunto de testes.
	scorer = make_scorer(f1_score, average = 'weighted')
	'''
	'weighted':
	Calcule as métricas para cada rótulo e encontre sua média ponderada pelo suporte 
	(o número de instâncias verdadeiras para cada rótulo). Isso altera 'macro' para 
	levar em conta o desequilíbrio do rótulo; pode resultar em um F-score que não está 
	entre precisão e recuperação.
	'''


	print(f"# Ajustando hiperparâmetros: ")
	print()
	
	opt = BayesSearchCV(
		estimator=model,
		scoring=scorer,
		search_spaces=search_spaces,
		cv = KFold(n_splits = 3, shuffle=True),
		n_iter=n_iter,
		n_jobs=-1,
		random_state=42)
	
	X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=.20, random_state=42)

	opt.fit(X_train, y_train)

	print("Melhor conjunto de parâmetros encontrado: ")
	print()
	print(opt.best_params_)
	print(f'f1_score (weighted): {opt.best_score_}')
	print()
	'''	print()
	print("Pontuações de grade no conjunto de desenvolvimento:")
	print()
	means = opt.cv_results_["mean_test_score"]
	stds = opt.cv_results_["std_test_score"]
	for mean, std, params in zip(means, stds, opt.cv_results_["params"]):
		print(f'{mean:.3f} (+/-{(std * 2):.3f}) for {params}')
	print()'''
	print("Relatório de classificação detalhado para o melhor modelo:")
	print()
	y_true, y_pred = y_test, opt.predict(X_test)
	print(classification_report(y_true, y_pred))
