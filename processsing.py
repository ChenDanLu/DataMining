import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats


df=pd.DataFrame(pd.read_csv('./data/Building_Permits.csv', encoding='gbk', header=0, engine='python'))


def isNumeric(feature):
    return True


def dataAbstract(feature):
    if isNumeric(feature):
        numdf=pd.DataFrame(df[feature])
        numdf.dropna().describe()
        # numdf.mean()
        # numdf.min()
        # numdf.quantile(0.25)
        # numdf.quantile(0.5)
        # numdf.quantile(0.75)
        # numdf.max()
        # numdf.isnull().sum()
    else:
        nomdf=pd.DataFrame(df[feature])
        nomdf.groupby(feature).size()


def dataVisual(feature):
    if isNumeric():
        # Histogram
        df[feature].hist(bins=50)
        # Quantile-Quantile Plot
        stats.probplot(df[feature], dist='norm', plot=pylab)
        pylab.show()
        # Box Plot
        plt.boxplot(df[feature].dropna())
        plt.title(feature)
        plt.show()


def dataProcess(feature):
    df[feature].dropna() #drop NAN
    df[feature].fillna(df[feature].mode()[0]) # fill NAN by mode


