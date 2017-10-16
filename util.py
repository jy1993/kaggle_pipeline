import pandas as pd

train_file = ''
test_file = ''

def get_feature_group(df):
	'''
	return category/numerical features list of df
	'''
	cate_list = []
	num_list = []
	for col in df.columns:
		if df[col].dtype == 'object' or len(df[col].unique()) < 100:
			cate_list.append(col)
		else:
			num_list.append(col)
	return cate_list, num_list

def num_2_cate(df, col, num_of_bins=3):
	'''
	transform a numerical features into categorical features
	'''
	df['col' + '_bins_' + num_of_bins] = pd.cut(df[col], num_of_bins)
