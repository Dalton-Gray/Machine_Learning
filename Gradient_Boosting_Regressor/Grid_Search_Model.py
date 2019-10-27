# Import libaries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# Newly added library
from sklearn.model_selection import GridSearchCV

# Read in data from CSV as Pandas dataframe
df = pd.read_csv('~/Downloads\Melbourne_housing_FULL.csv')

# The misspellings of "longitude" and "latitude" are preserved, as the two misspellings were not corrected in the source file.
# Delete unneeded columns
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']

df.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)

# Convert non numeric values to numeric using one-hot encoding
features_df = pd.get_dummies(df, columns =['Suburb', 'CouncilArea', 'Type'])

# Remove dependent variable (Price)
del features_df['Price']

# Independent variables
X = features_df.values

#Dependent variables
y = df['Price'].values

#Split the data into test/train set (70/30 split) and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)

# Select algoritm
model = ensemble.GradientBoostingRegressor()

# Set configurations you wish to test for hyperparameters
if __name__ == '__main__':
	param_grid = {
		'n_estimators': [300, 600],
	    'learning_rate': [0.01, 0.02],
	    'max_depth': [7, 9],
	    'min_samples_split': [3, 4],
	    'min_samples_leaf': [5, 6],
	    'max_features' : [0.8, 0.9],
	    'loss': ['ls', 'lad', 'huber']
	}

	# Define grid search. Run with n_jobs number of CPUs in parallel
	# n_jobs = -1 makes use of all cores
	gs_cv = GridSearchCV(model, param_grid, n_jobs = -1)

	# Run grid search on training data
	gs_cv.fit(X_train, y_train)

	# Print optimal hyperparameters
	print (gs_cv.best_params_)

	# Check model accuracy (up to two decimal places)
	# Evaluate results (training)
	mse = mean_absolute_error(y_train, model.predict(X_train))
	print("Training Set mean Absolute Error: %.2f" % mse)

	# Evaluate results (test)
	mse = mean_absolute_error(y_test, model.predict(X_test))
	print("Test Set mean Absolute Error: %.2f" % mse)