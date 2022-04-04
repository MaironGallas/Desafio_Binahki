import pandas as pd
import numpy as np

from dash import Dash, html, dcc
import plotly.express as px


if __name__ == '__main__':
    # importando arquivo CSV
    index = []
    data = pd.read_csv("data/Input v2.csv", sep=';').dropna()
    validacao = pd.read_csv("data/Validação v2.csv", sep=';')

    # cria indice igual a planilha validacao
    for i in range(0, len(data['DATA'])):
        index.append(i)
    data = data.set_index(pd.Index(index))

    # cria dataframe com a variaveis a serem calculadas
    colunas = ['energia_ativa_f1', 'energia_ativa_f2', 'energia_ativa_f3', 'energia_ativa_total',
               'energia_aparente_f1', 'energia_aparente_f2', 'energia_aparente_f3', 'energia_aparente_total',
               'energia_reativa_f2', 'energia_reativa_f1', 'energia_reativa_f3', 'energia_reativa_total',
               'fator_potencia_f1', 'fator_potencia_f2', 'fator_potencia_f3', 'fator_Potencia_Total',
               'potencia_aparente_f1', 'potencia_aparente_f2', 'potencia_aparente_f3', 'potencia_aparente_total'
               'potencia_ativa_f1', 'potencia_ativa_f2', 'potencia_ativa_f3', 'potencia_ativa_total',
               'potencia_reativa_f1', 'potencia_reativa_f2', 'potencia_reativa_f3', 'potencia_reativa_total',
               'tensao_f1_f2', 'tensao_f2_f3', 'tensao_f3_f1']
    result = pd.DataFrame(columns=colunas)

    # calculos das variaveis elétricas
    # potencia ativa
    result['potencia_ativa_f1'] = validacao['potencia_ativa_f1']
    result['potencia_ativa_f2'] = validacao['potencia_ativa_f2']
    result['potencia_ativa_f3'] = validacao['potencia_ativa_f3']
    result['potencia_ativa_total'] = validacao['potencia_ativa_total']

    # potencia aparente
    result['potencia_aparente_f1'] = round(data['corrente_f1'] * data['tensao_f1'])
    result['potencia_aparente_f2'] = round(data['corrente_f2'] * data['tensao_f2'])
    result['potencia_aparente_f3'] = round(data['corrente_f3'] * data['tensao_f3'])
    result['potencia_aparente_total'] = result['potencia_aparente_f1']+result['potencia_aparente_f2']+result['potencia_aparente_f3'].dropna()

    # fator de potencia
    result['fator_potencia_f1'] = round(result['potencia_ativa_f1']/result['potencia_aparente_f1'], 2)
    result['fator_potencia_f2'] = round(result['potencia_ativa_f2']/result['potencia_aparente_f2'], 2)
    result['fator_potencia_f3'] = round(result['potencia_ativa_f3']/result['potencia_aparente_f3'], 2)
    result['fator_potencia_total'] = result['fator_potencia_f1']+result['fator_potencia_f2']+result['fator_potencia_f3']

    # potencia reativa
    result['potencia_reativa_f1'] = np.sqrt((result['potencia_aparente_f1']**2)-(result['potencia_ativa_f1']**2))
    result['potencia_reativa_f2'] = np.sqrt((result['potencia_aparente_f2'] ** 2) - (result['potencia_ativa_f2'] ** 2))
    result['potencia_reativa_f3'] = np.sqrt((result['potencia_aparente_f3'] ** 2) - (result['potencia_ativa_f3'] ** 2))
    result['potencia_reativa_total'] = result['potencia_reativa_f1']+result['potencia_reativa_f2']+result['potencia_reativa_f3']

    # tensoes de linha
    result['tensao_f1_f2'] = data['tensao_f1']*np.sqrt(3)
    result['tensao_f2_f3'] = data['tensao_f2']*np.sqrt(3)
    result['tensao_f3_f1'] = data['tensao_f3']*np.sqrt(3)

    # criar parametros para servidor dash plotly
    app = Dash(__name__)

    # figuras a serem mostradas
    fig1 = px.line(data, x="DATA", y=["tensao_f1", "tensao_f2", "tensao_f3"], title='Tensões de Fase (V)')
    fig2 = px.line(data, x="DATA", y=["corrente_f1", "corrente_f2", "corrente_f3", "corrente_neutro"], title='Correntes de Fase e Neutro (A)')
    fig3 = px.line(result, x=data["DATA"], y=["potencia_ativa_f1", "potencia_ativa_f2", "potencia_ativa_f3", "potencia_ativa_total"], title='Potencias Ativas (W)')
    fig3.update_xaxes(title_text='DATA')
    fig4 = px.line(result, x=data["DATA"], y=["potencia_reativa_f1", "potencia_reativa_f2", "potencia_reativa_f3", "potencia_reativa_total"], title='Potencias Reativas (VAr)')
    fig4.update_xaxes(title_text='DATA')
    fig5 = px.line(result, x=data["DATA"], y=["potencia_aparente_f1", "potencia_aparente_f2", "potencia_aparente_f3", "potencia_aparente_total"], title='Potencias Aparentes (VA)')
    fig5.update_xaxes(title_text='DATA')
    fig6 = px.line(result, x=data["DATA"], y=["fator_potencia_f1", "fator_potencia_f2", "fator_potencia_f3", "fator_potencia_total"], title='Fator de Potência')
    fig6.update_xaxes(title_text='DATA')

    # criar layot dash
    app.layout = html.Div(children=[
        html.H1(children='Desafio Binahki'),
        html.Div(children='''Um aplicativo Web para mostrar os dados elétricos no desafio Binahki!!'''),
        dcc.Graph(id='example-graph', figure=fig1),
        dcc.Graph(id='example-graph1', figure=fig2),
        dcc.Graph(id='example-graph2', figure=fig3),
        dcc.Graph(id='example-graph4', figure=fig4),
        dcc.Graph(id='example-graph5', figure=fig5),
        dcc.Graph(id='example-graph6', figure=fig6)
    ])

    # roda servidor local
    app.run_server(debug=True)
