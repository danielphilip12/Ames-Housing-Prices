import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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