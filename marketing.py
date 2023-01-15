# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:14:32 2023

@author: Victor Resende
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from scipy.stats import ttest_ind


class LoadData:
    """
        Carrega os dados, sendo possível demonstrar estatísticas sobre eles.
    """
    
    def __init__(self, path: str):
        """
        Parameters
        ----------
        path : str
            Caminho arquivo .csv.
        """
        
        self.df = pd.read_csv(path, sep=';')

    def view(self, rows: int):
        """
        Description
        -----------
        Método para visualização do dataframe.
        
        Parameters
        ----------
        rows : int
            Quantidade de linhas.
        """
        
        print(f'Colunas: {list(self.df.columns)}\n\n')
        print(self.df.head(rows))

    def stats(self):
        """ 
        Description
        -----------
        Apresenta as estatísticas do conjunto de dados.
        """
        
        print(f'Estatísticas:\n{self.df.describe()}\n\n')
        print(f'Tipos:\n{self.df.dtypes}')

    def missing(self):
        """ 
        Description
        -----------
        Apresenta a porcentagem de dados faltantes por coluna.
        """
        
        self.missingData = ((self.df.isna().sum())/len(self.df))*100
        print('Dados faltantes (%):\n{missingData}\n\n')


class LinearRegressionModel:
    """
        Modelo de regressão linear.
    """
    
    def __init__(self):
        self.reg = LinearRegression()

    def train(self, X_train: int, y_train: int):
        """
        Description
        -----------
        Método para treinar o modelo.
        
        Parameters
        ----------
        X_train : int
            Dados de treino referente a variável independente.
        y_train : int
            Dados de treino referente a variável dependente.
        """
        
        self.reg.fit(X_train, y_train)

    def predict(self, X_test: int):
        """
        Description
        -----------
        Método para predição.
        
        Parameters
        ----------
        X_test : int
            Dados de teste referente a variável independente.

        Returns
        -------
        Predição da regressão linear.
        """
        
        return self.reg.predict(X_test)

    def score(self, X_test: int, y_test: int):
        """
        Description
        -----------
        Método verificar para verificar a variância explicada pelo modelo.
        A métrica R2 varia entre 0-1. Quanto mais próximo de 1 mais explicativo
        o modelo é.
        
        Parameters
        ----------
        X_test : int
            Dados de teste referente a variável independente.
        y_test : int
            Dados de teste referente a variável dependente.

        Returns
        -------
        Métrica R².
        """
        
        return self.reg.score(X_test, y_test)


class ModelAssumptions:
    def __init__(self, data):
        """
        Parameters
        ----------
        data : DataFrame
            Dados dos quais serão utilizados para os presupostos.
        """
        
        self.data = data

    def linear_assumption(self):
        """
        Description
        -----------
        Matriz de correlação de Pearson.
        """
        
        sns.heatmap(self.data.corr(), annot=True)
        plt.title('Correlação entre as variáveis\n')

    def multicollinearity_assumption(self, data):
        """
        Description
        -----------
        Método para verificar o fator de inflação de variância (VIF).
        
        Parameters
        ----------
        data : DataFrame
            Dataframe cru que será utilizado para a modelagem.
        """
        
        y, X = dmatrices('sales ~' + 'youtube', df, return_type='dataframe')
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        
        if vif.loc[1, 'VIF Factor'] > 5:
            print('Existe multicolinearidade entre as variáveis\n')
        else:
            print('Não existe multicolinearidade entre as variáveis\n')

    def normal_errors_assumption(self, alfa: int = 0.05):
        """
        Description
        -----------
        Método para aplicação do teste de Shapiro (Normalidade dos erros).
        
        Parameters
        ----------
        alfa : int, optional
            Nível de significância. O valor padrão é 0.05.
        """
        
        p_valor = shapiro(self.data['Residuo'])[1]
        print(f'Shapiro p-valor: {p_valor}')

        if p_valor < alfa:
            print('Resíduos não são normalmente distribuídos\n')
        else:
            print('Resíduos são normalmente distribuídos\n')

    def residuals_mean_assumption(self, alfa: int = 0.05):
        """
        Description
        -----------
        Método para aplicação do t-teste (Igualdade entre médias).
        
        Parameters
        ----------
        alfa : int, optional
            Nível de significância. O valor padrão é 0.05.
        """

        p_valor = ttest_ind(self.data['Predito'], self.data['Residuo'])[1]
        print(f'T-teste p-valor: {p_valor}')

        if p_valor < alfa:
            print('Os resíduos possuem média 0\n')
        else:
            print('Os resíduos não possuem média 0\n')

    def autocorrelation_assumption(self, alpha: int = 0.05):
        """
        Description
        -----------
        Método para aplicação do teste de Durbin-Watson (Resíduos i.i.d).
        
        Parameters
        ----------
        alfa : int, optional
            Nível de significância. O valor padrão é 0.05.
        """
        
        p_valor = durbin_watson(self.data['Residuo'])
        print(f'Durbin-Watson p-valor: {p_valor}')

        if p_valor < 1.5 or p_valor > 2.5:
            print('Os erros são independentes (i.i.d)\n')
        else:
            print('Os erros não são independentes (não i.i.d)\n')

    def homoscedasticity_assumption(self):
        """
        Description
        -----------
        Gráfico de dispersão dos resíduos.
        """
        
        sns.scatterplot(x=self.data.index, y=self.data
                        ['Residuo'], alpha=0.5)
        plt.plot(np.repeat(0, self.data.index.max()),
                 color='darkorange', linestyle='--')
        plt.title('Residuos')


df = LoadData('marketing.csv').df

X = df.loc[:, ['youtube']].values
y = df.loc[:, 'sales'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=19
                                                    )

model = LinearRegressionModel()
model.train(X_train, y_train)
y_pred = model.predict(X_test)

df_results = pd.DataFrame({'Atual': y_test, 'Predito': y_pred})
df_results['Residuo'] = abs(df_results['Atual']) - abs(df_results['Predito'])


print('Presuposto 1: Linearidade das variáveis independentes com a variável depente')
ModelAssumptions(df).linear_assumption()

print('Presuposto 2: O erro deve ser normalmente distribuído')
ModelAssumptions(df_results).normal_errors_assumption()

print('Presuposto 3: Multicolinearidade entre as variáveis independentes')
ModelAssumptions(df_results).multicollinearity_assumption(df)

print('Presuposto 3: Resíduos devem possuir média 0')
ModelAssumptions(df_results).residuals_mean_assumption()

print('Presuposto 4: Não deve haver correlação entre resíduos')
ModelAssumptions(df_results).autocorrelation_assumption()

print('Presuposto 5: Resíduos devem possuir variância constante')
ModelAssumptions(df_results).homoscedasticity_assumption()
