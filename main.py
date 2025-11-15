import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../data/ameshousing.csv')
# reads in the dataset as a dataframe from the csv


df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())
# Imputes the missing values in Lot Frontage with the median

df.loc[2260, 'Garage Yr Blt'] = np.NaN
# sets index 2260, column Garage Yr Blt to NaN, as the year has not passed yet (2207)

df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(df['Garage Yr Blt'].median())
# Imputes garage year built with median year, so as to not affect the correlation too much when filling in missing values. 

df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)
# fills in the missing values in Mas Vnr Area with 0s, as all the rows where it is 0, the house does not have a masonry veneer

df['Pool QC'] = df['Pool QC'].fillna("No Pool")
# fills in the NaN values with "No Pool"

df['Misc Feature'] = df['Misc Feature'].fillna("No Misc Features")
# Fills in misc features NaNs with "No Misc Features"

df['Alley'] = df['Alley'].fillna("No Alley Access")
# Fills in alley NaNs with "No Alley Access"

df['Fence'] = df['Fence'].fillna("No Fence")
# fills in fence NaNs with "No Fence"

df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna("No Masonry Veneer")
# fills in mas vnr type NaNs with "No masonry veneer"

df['Fireplace Qu'] = df['Fireplace Qu'].fillna("No Fireplace")
# fills in fireplace qu NaNs with "No Fireplace"

df[['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']] = df[['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']].fillna("No Garage")
# fills in missing garage features with "No Garage"

df[['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']] = df[['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']].fillna("No Basement")
# fills in missing basement features with "No Basement"

df = df.dropna()
# drops the remaining rows that have NA values

nomial_columns = ['PID', 'MS SubClass', 'MS Zoning', 'Street', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air', 'Garage Type', 'Misc Feature', 'Sale Type', 'Sale Condition']
# list of columns with nominal categories, according to the data dictionary. 
ordinal_columns = ['Lot Shape', 'Utilities', 'Land Slope', 'Overall Qual', 'Overall Cond', 'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating QC', 'Electrical', 'Kitchen Qual', 'Functional', 'Fireplace Qu', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive', 'Pool QC', 'Fence']
# list of columns with ordinal categories, according to the data dictionary.

df[nomial_columns] = df[nomial_columns].astype('category')
# converts each of the nominal columns to category dtype

def to_category(column, categories):
    """
    Creator: Daniel Gallo
    Inputs: column name, cateogies list in order
    Outputs: an ordered categorical column of the column passed in 
    """
    return pd.Categorical(column, categories=categories, ordered=True)

df['Lot Shape'] = pd.Categorical(df['Lot Shape'], categories = ['IR3', 'IR2', 'IR1', 'Reg'], ordered=True)
# converts Lot Shape to an ordered categorical column

df['Utilities'] =  pd.Categorical(df['Utilities'], categories=['ELO', 'NoSeWa', 'NoSewr', 'AllPub'], ordered=True)
# converts Utilities to an ordered categorical column

df['Land Slope'] = to_category(df['Land Slope'], ['Gtl', 'Mod', 'Sev'])
# convert land slope to an ordered categorical column

df['Overall Qual'] = to_category(df['Overall Qual'], [1,2,3,4,5,6,7,8,9,10])
df['Overall Cond'] = to_category(df['Overall Cond'], [1,2,3,4,5,6,7,8,9,10])
# converts overall quality and condition to ordered categorical columns

df['Exter Qual'] = to_category(df['Exter Qual'], ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
df['Exter Cond'] = to_category(df['Exter Cond'], ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
# converts exterior quality and condition to ordered categorical columns 

df['Bsmt Qual'] = to_category(df['Bsmt Qual'], ['No Basement', 'Po' ,'Fa', 'TA', 'Gd', 'Ex'])
df['Bsmt Cond'] = to_category(df['Bsmt Cond'], ['No Basement', 'Po' ,'Fa', 'TA', 'Gd', 'Ex'])
# converts basement quality and condition to ordered categorical columns 

df['Bsmt Exposure'] = to_category(df['Bsmt Exposure'], ['No Basement', 'No', 'Mn' ,'Av', 'Gd'])
# converts basement exposure to ordered categorical column

df['BsmtFin Type 1'] = to_category(df['BsmtFin Type 1'], ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])
df['BsmtFin Type 2'] = to_category(df['BsmtFin Type 2'], ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])
# converts basement finish 1 and 2 columns to ordered categorical columns

df['Heating QC'] = to_category(df['Heating QC'], ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
# converts heating qc columns to an ordered categoircal column

df['Electrical'] = to_category(df['Electrical'], ['FuseP', 'FuseF', 'FuseA', 'SBrkr', 'Mix'])
# converts electrical to an ordered categorical column

df['Kitchen Qual'] = to_category(df['Kitchen Qual'], ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
# converts kitchen quality column to ordered categorical column

df['Functional'] = to_category(df['Functional'], ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod' ,'Min2', 'Min1', 'Typ'])
# converts functional columns to ordered catergorical column

df['Fireplace Qu'] = to_category(df['Fireplace Qu'], ['No Fireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])
# converts fireplace qu to ordered categorical column

df['Garage Finish'] = to_category(df['Garage Finish'], ['No Garage', 'Unf', 'RFn', 'Fin'])
# converts garage finish to ordered categorical column

df['Garage Qual'] = to_category(df['Garage Qual'], ['No Garage', 'Po' ,'Fa', 'TA', 'Gd', 'Ex'])
df['Garage Cond'] = to_category(df['Garage Cond'], ['No Garage', 'Po' ,'Fa', 'TA', 'Gd', 'Ex'])
# converts garage quality and condition to categorical column ordered

df['Paved Drive'] = to_category(df['Paved Drive'], ['N', 'P', 'Y'])
# converts paved drive to categorical column ordered

df['Pool QC'] = to_category(df['Pool QC'], ['No Pool', 'Fa', 'TA', 'Gd', 'Ex'])
# converts pool qc to categorical column ordered

df['Fence'] = to_category(df['Fence'], ['No Fence', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'])
# converts fence to ordered categorical column

df['Total Full Bath'] = df['Full Bath'] + df['Bsmt Full Bath']
# creates a new column, Total Full Bath, which is the sum of Full Bath (above ground) and Basement Full Bath
df['Total Half Bath'] = df['Half Bath'] + df['Bsmt Half Bath']
# creates a new column, Total Half Bath, which is the sum of Half Bath (above ground) and Basement Half Bath

for col in ordinal_columns:
    df[f"{col}_codes"] = df[col].cat.codes
    # this is creating the category codes for the ordinal columns, so we can start to make comparisons to the sale price


# These are the plots. Commented out to not run every time this is run. 
"""

sns.scatterplot(data=df, x='Year Built', y='SalePrice', color='yellowgreen')
plt.title("Year Built vs Sale Price")
plt.ylabel("Sale Price")
plt.savefig("../images/yr_built_vs_sale.png")
# Creates a scatterplot comparing Year Built to Sale Price

sns.regplot(data=df, x='Lot Area', y='SalePrice', color='orangered')
plt.title("Lot Area (sq ft) vs Sale Price")
plt.ylabel("Sale Price (in millions)")
plt.xlabel("Lot Area (sq ft)")
plt.savefig("../images/lot_area_vs_sale.png")
# Creates a regression plot comparing Lot Area to Sale Price

sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice', color='cornflowerblue')
plt.title("Above Ground Living Area (sq ft) vs Sale Price")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.savefig("../images/abv_gr_liv_area_vs_sale.png")
# creates a scatterplot compare above ground living area to sale price

sns.histplot(data=df.loc[df['Garage Area'] > 0, 'Garage Area'], kde=True)
plt.title("Distribution of Garage Area (sq ft)")
plt.xlabel("Garage Area (sq ft)");
plt.savefig("../images/hist_of_garage_area.png")
# Creates a histogram of the distribution of Garage Area (filtering out rows with no garage)

sns.regplot(data=df, x='Total Full Bath', y='SalePrice', color='violet')
plt.title("Full Bath Count vs Sale Price")
plt.xlabel("Count of Total Full Bath")
plt.ylabel("Sale Price")
plt.savefig("../images/full_bath_vs_sale.png")
# creates a regression plot of the total full bath count and sale price

sns.regplot(data=df, x='Total Half Bath', y='SalePrice', color='gold')
plt.title("Half Bath Count vs Sale Price")
plt.xlabel("Count of Total Half Bath")
plt.ylabel("Sale Price")
plt.savefig("../images/half_bath_vs_sale.png")
# creates a regression plot of the total full bath count and sale price

sns.boxplot(data=df, x='SalePrice', hue='Garage Qual')
plt.title("Distribution of Sale Price by Garage Quality")
plt.legend(labels=['No Garage', 'Poor', 'Fair', 'Typical', 'Good', 'Excellent'])
plt.savefig("../images/garage_qual_sale_dist.png")
# creates boxplots grouped by Garage Quality, and updates the legend to have full label names

sns.regplot(data=df.loc[df['Total Bsmt SF'] > 0, :], x='Total Bsmt SF', y='SalePrice', color='green', line_kws={'color': 'red'})
plt.title("Total Basement Area (sq ft) vs Sale Price")
plt.xlabel("Total Basement Area (sq ft)")
plt.ylabel("Sale Price")
plt.savefig("../images/bsmt_area_vs_sale.png")
# creates a regression plot, comparing the total basement square footage and sale price

sns.boxplot(data=df, x='SalePrice', hue='Bsmt Qual')
plt.title("Distribution of Sale Price by Basement Height")
plt.xlabel("Sale Price")
plt.legend(labels=["No Basement","Poor (< 70 inches)", "Fair (70-79 inches)", "Typical (80-89 inches)", "Good (90-99 inches)", "Excellent (100+ inches)"])
plt.savefig("../images/base_height_sale_dist.png")
# creates boxplots grouped by Garage Quality, and updates the legend to have descriptive label names

ax = sns.countplot(data=df, x='Bsmt Cond')
ax.bar_label(ax.containers[0])
plt.title("Counts of Basement Conditions")
plt.xlabel("Basement Condition")
plt.savefig("../images/count_of_base_cond.png")
# creates a countplot of the different basement conditions

sns.barplot(data=df, x='Bsmt Cond', y='SalePrice', estimator='mean')
plt.title("Average Sale Price by Basement Condition")
plt.xlabel("Basement Condition")
plt.ylabel("Sale Price")
plt.savefig("../images/avg_sale_by_base_cond.png")
# creates a barplot showing the mean sale price by basement conditions

sns.regplot(data=df, x='Fireplaces', y='SalePrice')
plt.title("Relationship between Fireplaces and Sale Price")
plt.xlabel("Number of Fireplaces")
plt.ylabel("Sale Price")
plt.savefig("../images/fireplace_vs_sale.png")
# creates a regression plot between fireplaces and sale price



sns.regplot(data=df, x='Lot Shape_codes', y='SalePrice')
plt.xlabel("Lot Shape Codes")
plt.ylabel("Sale Price")
plt.xticks(
    ticks=range(len(df['Lot Shape'].cat.categories)),
    labels=df['Lot Shape'].cat.categories,
    rotation=45
);
plt.savefig("../images/lot_shape_vs_sale.png")
# creates a regression plot comparing the Lot Shape (codes) and the Sale Price

sns.regplot(data=df, x='Overall Qual_codes', y='SalePrice')
plt.title("Overall Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.xticks(
    ticks=range(len(df['Overall Qual'].cat.categories)),
    labels=df['Overall Qual'].cat.categories,
    rotation=45
);
plt.savefig("../images/overall_qual_vs_sale.png")
# creates a regression plot comparing the Overall House Quality (codes) and the Sale Price

sns.regplot(data=df, x='Exter Qual_codes', y='SalePrice')
plt.xlabel("Exterior Quality")
plt.ylabel("Sale Price")
plt.xticks(
    ticks=range(len(df['Exter Qual'].cat.categories)),
    labels=df['Exter Qual'].cat.categories,
    rotation=45
);
plt.savefig("../images/exter_qual_vs_sale.png")
# creates a regression plot comparing the Exterior Quality (codes) and the Sale Price

sns.regplot(data=df, x='Kitchen Qual_codes', y='SalePrice')
plt.xlabel("Kitchen Quality")
plt.ylabel("Sale Price")
plt.xticks(
    ticks=range(len(df['Kitchen Qual'].cat.categories)),
    labels=df['Kitchen Qual'].cat.categories,
    rotation=45
);
plt.savefig("../images/kitchen_qual_vs_sale.png")
# creates a regression plot comparing the Kitchen Quality (codes) and the Sale Price

sns.regplot(data=df, x='Heating QC_codes', y='SalePrice')
plt.xticks(
    ticks=range(len(df['Heating QC'].cat.categories)),
    labels=df['Heating QC'].cat.categories,
    rotation=45
);
plt.xlabel("Heating Quality Codes")
plt.ylabel("Sale Price")
plt.savefig("../images/heat_qual_vs_sale.png")
# creates a regression plot comparing the Heating Quality (codes) and the Sale Price

saleprice_corrs = df.corr(numeric_only=True)[['SalePrice']].sort_values(by='SalePrice', ascending=False)
# creates the correlations between all the numeric columns and the SalePrice

plt.figure(figsize=(8, 15))
sns.heatmap(saleprice_corrs,
           vmin=-1,
           vmax=1,
           cmap='coolwarm',
           annot=True)
# creates a heatmap of the previously made correlation map



"""

df.to_csv('../data/cleaned_ameshousing.csv', index=False)
# creates a new csv file of the cleaned dataset. 

features = ['Overall Qual_codes', 'Exter Qual_codes', 'Kitchen Qual_codes', 'Bsmt Qual_codes', \
           'Heating QC_codes', 'Garage Qual_codes', 'Gr Liv Area', \
           'Lot Shape_codes', 'Total Full Bath', 'Total Half Bath', \
           'TotRms AbvGrd', 'Garage Area', 'Overall Cond_codes', 'Misc Val', \
           'Total Bsmt SF', 'Bsmt Cond_codes', 'Garage Area', 'Garage Cond_codes', 'Lot Area']
# the list of features to use in the model
nominal_features = ['House Style', 'Bldg Type', 'Sale Condition', \
                    'Neighborhood', 'Misc Feature', 'Sale Type', 'Garage Type'] 
# list of nominal features to use in the model


X = df[features+nominal_features]
# independant variables
y = df['SalePrice']
# target variable

X = pd.get_dummies(X, columns=nominal_features) 
# makes the duummy variables for the nomianl independent variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# splits the dataset into training (75% of the data) and testing (25% of the data) sets.

lr = LinearRegression()
# instantiates the linear regression model

lr.fit(X_train, y_train)
# trains the linear regression model on the training data and labels 

def get_scores(model, X_train, X_test, y_train, y_test):
    """
    Creator: Daniel Gallo
    Inputs:
        model: the trained machine learning model on the X_train and y_train created previously. 
        X_train: the features of the training set from train_test_split
        X_test: the feautres of the testing set from train_test_split
        y_train: the targets of the training set from train_test_split
        y_test: the targets of the testing set from train test split
    Outputs: Prints the r-squared score of the models predictions of the training set and testing set
    Creates a baseline RMSE score, as well as RMSE scores for training and testing set. 
    Outputs all the scores for comparison

    Notes:
    Model passed in MUST BE fit to the data prior. train_test_split must also occur BEFORE this method invocation. 
    """
    print(f"Training R-Squared Score:\t{model.score(X_train, y_train)}")
    print(f"Testing R-Squared Score:\t{model.score(X_test, y_test)}")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    print(f"Training RMSE Score:\t\t{root_mean_squared_error(y_train, train_preds)}")
    print(f"Testing RMSE Score:\t\t{root_mean_squared_error(y_test, test_preds)}")
    test_mean = y_test.mean()
    baseline_preds = np.full_like(y_test, test_mean)
    print(f"Baseline RMSE Score:\t\t{root_mean_squared_error(y_test, baseline_preds)}")

get_scores(lr, X_train, X_test, y_train, y_test)
# gets the R2 and RMSE scores of the linear regression model 


alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
# list of different alphas for ridge regression for regularization
for a in alphas:
    ridge = Ridge(alpha=a)
    # instantiates the ridge model with the alpha
    ridge.fit(X_train, y_train)
    # trains the model 
    print(f"Alpha = {a}")
    print('=' * 20)
    get_scores(ridge, X_train, X_test, y_train, y_test)
    # output the scores. 
    print('=' * 20)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
for a in alphas:
    lasso = Lasso(alpha=a)
    # instantiates lasso regression model
    lasso.fit(X_train, y_train)
    # trains the model
    print(f"Alpha = {a}")
    print('=' * 20)
    get_scores(lasso, X_train, X_test, y_train, y_test)
    print('=' * 20)


rf = RandomForestRegressor()
# instantiates the random forest model
ls = LinearSVR(max_iter=20000)
# instantiates the linear support vector regression model
dtr = DecisionTreeRegressor()
# instantiates the decision tree regression model.

models = {'Random Forest': rf, 'Linear SVR': ls, 'Decision Tree': dtr}
# dictionary of different models to loop through. 

for name, model in models.items():
    print(name)
    print("=" * 20)
    model.fit(X_train, y_train)
    # trains each model on the training set
    print()

for name, model in models.items():
    print(name)
    print("=" * 20)
    get_scores(model, X_train, X_test, y_train, y_test)
    # prints the scores ofr each model
    print()