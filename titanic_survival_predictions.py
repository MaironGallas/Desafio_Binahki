import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#ignora warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def analise_grafica(train_df, test_df):
    # gráfico de barras de sexo por sobrevivência
    plt.figure()
    sns.barplot(x="Sex", y="Survived", data=train_df)

    # gráfico de barras de sobrevivência por Pclass
    plt.figure()
    sns.barplot(x="Pclass", y="Survived", data=train_df)

    # gráfico de barras de SibSp por sobrevivência
    plt.figure()
    sns.barplot(x="SibSp", y="Survived", data=train_df)

    # gráfico de barras de Parch por sobrevivência
    plt.figure()
    sns.barplot(x="Parch", y="Survived", data=train_df)

    # gráfico de barras de idade por sobrevivência
    plt.figure()
    sns.barplot(x="AgeGroup", y="Survived", data=train_df)

    # chances de sobrevivência por porto de embarque
    sns.factorplot(x='Embarked', y='Survived', data=train_df)
    plt.show()


if __name__ == '__main__':

    # importando arquivos CSV train e test
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # classificando as idades em categorias
    train_df["Age"] = train_df["Age"].fillna(-0.5)
    test_df["Age"] = test_df["Age"].fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels=labels)
    test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels=labels)

    #analise_grafica(train_df, test_df)

    # preenchendo os dados vazios de embarked com S
    test_df['Embarked'].fillna('S', inplace=True)
    train_df['Embarked'].fillna('S', inplace=True)

    # preenchendo os dados vazios de idade
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # substitui vários títulos por nomes mais comuns
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # mapear cada um dos grupos de títulos para um valor numérico
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    # tentando prever os valores de idade ausentes a partir da idade mais comum para seu título
    mr_age = train_df[train_df["Title"] == 1]["AgeGroup"].mode()  # Young Adult
    miss_age = train_df[train_df["Title"] == 2]["AgeGroup"].mode()  # Student
    mrs_age = train_df[train_df["Title"] == 3]["AgeGroup"].mode()  # Adult
    master_age = train_df[train_df["Title"] == 4]["AgeGroup"].mode()  # Baby
    royal_age = train_df[train_df["Title"] == 5]["AgeGroup"].mode()  # Adult
    rare_age = train_df[train_df["Title"] == 6]["AgeGroup"].mode()  # Adult

    age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

    for x in range(len(train_df["AgeGroup"])):
        if train_df["AgeGroup"][x] == "Unknown":
            train_df["AgeGroup"][x] = age_title_mapping[train_df["Title"][x]]

    for x in range(len(test_df["AgeGroup"])):
        if test_df["AgeGroup"][x] == "Unknown":
            test_df["AgeGroup"][x] = age_title_mapping[test_df["Title"][x]]

    # mapear cada valor de idade para um valor numérico
    age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
    train_df['AgeGroup'] = train_df['AgeGroup'].map(age_mapping)
    test_df['AgeGroup'] = test_df['AgeGroup'].map(age_mapping)

    # mapear cada valor de sexo para um valor numérico
    sex_mapping = {"male": 0, "female": 1}
    train_df['Sex'] = train_df['Sex'].map(sex_mapping)
    test_df['Sex'] = test_df['Sex'].map(sex_mapping)

    # mapear cada valor embarcado para um valor numérico
    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)
    test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)

    # preencha o valor de fare ausente com base na fare média para aquela pclass
    for x in range(len(test_df["Fare"])):
        if pd.isnull(test_df["Fare"][x]):
            pclass = test_df["Pclass"][x]  # Pclass = 3
            test_df["Fare"][x] = round(train_df[train_df["Pclass"] == pclass]["Fare"].mean(), 4)

    # mapear os valores das Fare em grupos de valores numéricos
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4, labels=[1, 2, 3, 4])
    test_df['FareBand'] = pd.qcut(test_df['Fare'], 4, labels=[1, 2, 3, 4])

    # criando uma variavel tamanho da familia e sozinho
    train_df['FamilySize'] = 0
    train_df['FamilySize'] = train_df['Parch'] + train_df['SibSp']
    train_df['Alone'] = 0
    train_df.loc[train_df.FamilySize == 0, 'Alone'] = 1  # Alone

    test_df['FamilySize'] = 0
    test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp']
    test_df['Alone'] = 0
    test_df.loc[test_df.FamilySize == 0, 'Alone'] = 1  # Alone

    # limpando os dados
    test_df.drop(['Name', 'Cabin', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)
    train_df.drop(['Name', 'Cabin', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)

    # mapa de calor para ver correlação de todas as variaveis
    sns.heatmap(train_df.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size': 20})
    fig = plt.gcf()
    #plt.show()


    # ESCOLHENDO O MELHOR MODELO
    # dividindo os dados de treinamento
    predictors = train_df.drop(['Survived', 'PassengerId'], axis=1)
    target = train_df["Survived"]
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size=0.22, random_state=0)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_val)
    acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
    print(f'Precisao para Logistic Regression: {acc_logreg}%')

    # Decision Tree
    decisiontree = DecisionTreeClassifier()
    decisiontree.fit(x_train, y_train)
    y_pred = decisiontree.predict(x_val)
    acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
    print(f'Precisao para Decision Tree: {acc_decisiontree}%')

    # KNN or k-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_val)
    acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
    print(f'Precisao para KNN: {acc_knn}%')

    # criando arquivo para submissão, foi decidido utilizar o método de k-Nearest Neighbors

    # set ids as PassengerId and predict survival
    ids = test_df['PassengerId']
    predictions = knn.predict(test_df.drop('PassengerId', axis=1))

    # set the output as a dataframe and convert to csv file named submission.csv
    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)