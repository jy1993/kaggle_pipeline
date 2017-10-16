import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
debug = True

def get_top_features(clf_type, X_train, y_train, top=20):
	'''
	return top n features according to a model

	input: 
	clf_type: 'xgboost' (binary)
	X_train, y_train:  train data
	top
	'''
	if clf_type == 'xgboost':
		params = {
			'objective': 'binary:logistic',
			'eval_metric': 'auc',
			'eta': 0.1, 
			'gamma': 1,
			'min_child_weight': 1,
			'subsample': 0.8, 
			'colsamplebytree': 0.8,
			'seed': 42,
			'silent': 1
		}
		dtrain = xgb.DMatrix(X_train, label=y_train)
		clf = xgb.train(params, dtrain, 100, early_stopping_rounds = 20, verbose_eval = 10)
		kv = clf.get_fscore()
		feats = sorted(kv.keys(), key=lambda x: kv[x], reverse=True)[0:top]
		if debug:
			xgb.plot_importance(clf, max_num_features = 30)
			plt.show()
		return feats

def add_2_interactions(df, col1, col2, style='add'):
	'''
	adding 2 way interactions for numerical features

	input:
	df   dataframe
	col1, col2:  colnames

	return:
	df with a new column
	'''
	if style == 'add':
		df[col1 + '_add_' + col2] = df[col1] + df[col2]
	elif style == 'minus':
		df[col1 + '_minus_' + col2] = df[col1] - df[col2]
	elif style == 'multiply':
		df[col1 + '_multiply_' + col2] = df[col1] * df[col2]

def feature_transform(df, col, to_add=1):
	'''
	transform features using boxcox
	data need to be positive
	'''
	if (df[col] > 0).astype(np.int32).sum() != len(df[col]):
		df[col] += to_add
	try:
		df[col + '_' + 'boxcox'], max_log = boxcox(df[col])
	except:
		print 'bad input: negative values'
	return max_log
