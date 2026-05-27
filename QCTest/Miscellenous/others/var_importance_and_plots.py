"""
Script to host variable importance and plots on Giraffe.

For now, only works for Darwin strategy

BQ Table: {strategy}_Production.{opexchange}_importance
Eg: Darwin_Production.ERCOT_importance

:Date: 2023-08-10
:Authors: akhilesh_somani
"""

import os
import sys
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dash import html, Input, Output, State, callback, callback_context, dcc

script_name = os.path.basename(__file__).split(".")[0]

sys.path.append("/var/www/python/Prod/nighthawk/")
from nighthawk.viz.giraffe import common_functions
from nighthawk.util import bigquery_functions, sql_functions
from nighthawk.data.network import node, path

style_data = {
    'whiteSpace': 'normal',
    'maxWidth': '250px',  # just to make sure no column is too wide
    'height': 'auto',
    'overflow': 'hidden'
}

'''Variable Importance Table'''

table_options = {'columns': [],
                 'style_data': style_data,
                 'page_size': 10,
                 'sort_by': [{'column_id': "importance", 'direction': "desc"}],
                 'hidden_columns': ["id"],
                 'active_cell': {"column": 0, "row": 0, "row_id": "0"},
                 }

table_layout_var_importance = common_functions.FCC_card(script_name=script_name, card_name='var_importance',
                                                        fig_or_table='table',
                                                        datepicker_options={}, table_options=table_options,
                                                        card_style={'overflow': 'scroll'},
                                                        title='Variable Importance').get_layout()

fig_layout_var_plot = common_functions.FCC_card(script_name=script_name, card_name='variable', fig_or_table='fig',
                                                datepicker_options={'days_before': 3, 'days_after': 1,
                                                                    'submit_button': False},
                                                title='Variable Plot').get_layout()


def serve_layout():
    """
    Function to serve layout
    """

    title_row = common_functions.create_title_row(script_name=script_name, card_name="",
                                                  datepicker_options={'days_before': -1, 'days_after': 1,
                                                                      'submit_button': False}, )

    return dbc.Container([

        title_row,

        html.Br(),

        dbc.Row(children=[

            html.Br(),

            dbc.Row(children=[

                # common_functions.FCC_datePickerRange(script_name=script_name, id_suffix='datePickerRange',
                #                                      title='Date Range',
                #                                      user_datepicker_options={'days_before': -1, 'days_after': 1}),

                common_functions.FCC_dropdown(
                    user_picker_options={
                        'options': [],
                        'value': ''},
                    script_name=script_name,
                    title="Node",
                    id_suffix="node_num"
                ),

                common_functions.FCC_dropdown(
                    user_picker_options={
                        'options': [{'label': i, 'value': i} for i in ['Darwin', 'Curie']],
                        'value': 'Darwin'},
                    script_name=script_name,
                    title="Strategy",
                    id_suffix="strategy"
                ),

                common_functions.FCC_dropdown(
                    user_picker_options={
                        'options': [],
                        'value': ''},
                    script_name=script_name,
                    title="Y Var List",
                    id_suffix="y_list"
                ),

                common_functions.FCC_dropdown(
                    user_picker_options={
                        'options': [{'label': i, 'value': i} for i in [10, 20, 30, 50, 100,10000]],
                        'value': 10},
                    script_name=script_name,
                    title="Number of Vars",
                    id_suffix="number_of_vars"
                ),

                # common_functions.FCC_button(script_name=script_name, card_name="", id_suffix="submit_button",
                #                             user_button_options={'children': 'Submit', 'n_clicks': 0})

            ],
                justify="start"),

        ]),

        html.Br(),

        dbc.Row(children=[
            dbc.Col([table_layout_var_importance], xs=6),
            dbc.Col([fig_layout_var_plot], xs=6)
        ]),

        dcc.Store("var_importance_and_plots_gcp_csvfile_data")

    ])


# update 'Y Var List' options based on homepage opexchange selection
@callback(
    [
        Output('var_importance_and_plots_y_list', 'options'),
        Output('var_importance_and_plots_y_list', 'value')
    ],
    Input('homepage-opexchange', 'data')
)
def callback_function_set_y_list_options_and_value(opexchange: str):
    if opexchange == 'ERCOT':
        y_list_options = [{'label': i, 'value': i} for i in ['rt_congestion_da_congestion']]
        y_list_value = 'rt_congestion_da_congestion'
    elif opexchange == 'SPP':
        y_list_options = [{'label': i, 'value': i} for i in ['rt_congestion', 'da_congestion', 'rt_total', 'da_total']]
        y_list_value = 'da_total'
    else:
        y_list_options = [{'label': i, 'value': i} for i in ['rt_congestion_da_congestion', 'rt_total_da_total']]
        y_list_value = 'rt_total_da_total'

    return y_list_options, y_list_value


# update node_num options and table column names based on homepage opexchange and Y Var List
@callback(
    [
        Output('var_importance_and_plots_node_num', 'options'),
        Output('var_importance_and_plots_var_importance_table', 'columns')
    ],
    [
        Input('homepage-opexchange', 'data'),
        Input('var_importance_and_plots_strategy', 'value'),
        Input('var_importance_and_plots_y_list', 'value'),
        State('var_importance_and_plots_datePickerRange', 'start_date'),
        Input('var_importance_and_plots_datePickerRange', 'end_date')
    ]
)
def callback_function_set_node_num_options_and_table_cols(opexchange: str, strategy: str, y_list: str, start_date: str,
                                                          end_date: str):
    if strategy == 'Curie' and opexchange == 'MISO':
        # Get distinct nodes from Curie_MISO.featureImportances in date range
        sql_query = f"""
            select distinct node_num
            from Curie_MISO.featureImportances
            where dt >= '{start_date}'
              and dt <= '{end_date}'
            order by node_num asc
        """
        node_num_df = sql_functions.download_df_from_sql_db(sql_query)
        if node_num_df.empty:
            # No data: return empty options and simple columns
            table_columns = [
                {'name': 'id', 'id': 'id', 'type': 'numeric', 'hideable': True},
                {'name': 'Opexchange', 'id': 'opexchange', 'type': 'text'},
                {'name': 'Node', 'id': 'node_num', 'type': 'text'},
                {'name': 'Date', 'id': 'dt', 'type': 'text'},
                {'name': 'Variable', 'id': 'var', 'type': 'text'},
                {'name': 'Importance', 'id': 'importance', 'type': 'numeric'},
            ]
            return [], table_columns

        node_num_df['node_num'] = node_num_df['node_num'].astype(int)

        node_obj = node.Node(market=opexchange, node_nums=list(node_num_df['node_num'].unique()))
        node_details_df = node_obj.get_node_details()

        node_num_df = node_num_df.merge(node_details_df, on=['node_num'], how='left')
        node_num_df['final_node'] = node_num_df['node_num'].astype(str) + "-" + node_num_df['node_name']
        node_num_list = list(node_num_df['final_node'])

        # Simpler table for Curie – no correlation columns
        table_columns = [
            {'name': 'id', 'id': 'id', 'type': 'numeric', 'hideable': True},
            {'name': 'Opexchange', 'id': 'opexchange', 'type': 'text'},
            {'name': 'Node', 'id': 'node_num', 'type': 'text'},
            {'name': 'Date', 'id': 'dt', 'type': 'text'},
            {'name': 'Variable', 'id': 'var', 'type': 'text'},
            {'name': 'Importance', 'id': 'importance', 'type': 'numeric'},
        ]
        return node_num_list, table_columns
    bq_dataset = f'{strategy}_Production'
    bq_table_name = f'{opexchange}_importance'

    bq_query = f"""SELECT DISTINCT node_num FROM {bq_dataset}.{bq_table_name} 
                   WHERE dt >= '{start_date}' AND dt <= '{end_date}' AND y_list = '{y_list}'
                   ORDER BY node_num ASC;"""

    node_num_df = bigquery_functions.download_df_from_bq(bq_query)

    # get node-names
    if opexchange == 'ERCOT':
        node_num_df[['source_num', 'sink_num']] = node_num_df['node_num'].str.split("_", expand=True)
        node_num_df[['source_num', 'sink_num']] = node_num_df[['source_num', 'sink_num']].astype(int)

        path_obj = path.Path(market=opexchange, path_df=node_num_df[['source_num', 'sink_num']].drop_duplicates())
        path_details_df = path_obj.get_path_details()
        # ['source_num', 'sink_num', 'source_name', 'sink_name', 'source_zone', 'sink_zone']

        node_num_df = node_num_df.merge(path_details_df,
                                        on=['source_num', 'sink_num'],
                                        how='left')

        node_num_df['node_name'] = node_num_df['source_name'] + "_" + node_num_df['sink_name']

    else:
        node_num_df['node_num'] = node_num_df['node_num'] .astype(int)

        node_obj = node.Node(market=opexchange, node_nums=list(node_num_df['node_num'].unique()))
        node_details_df = node_obj.get_node_details()
        # ['node_num', 'node_name', 'node_zone', etc.]

        node_num_df = node_num_df.merge(node_details_df,
                                        on=['node_num'],
                                        how='left')

    node_num_df['final_node'] = node_num_df['node_num'].astype(str) + "-" + node_num_df['node_name']

    node_num_list = list(node_num_df['final_node'])

    # table column names
    if 'total' in y_list:
        table_columns = [
            {'name': 'id', 'id': 'id', 'type': 'numeric', 'hideable': True},
            {'name': 'Opexchange', 'id': 'opexchange', 'type': 'text'},
            {'name': 'Node', 'id': 'node_num', 'type': 'text'},
            {'name': 'Variable', 'id': 'var', 'type': 'text'},
            {'name': 'Importance', 'id': 'importance', 'type': 'numeric'},
            {'name': 'DA Corr', 'id': 'da_corr', 'type': 'numeric'},
            {'name': 'RT Corr', 'id': 'rt_corr', 'type': 'numeric'},
        ]
    elif 'congestion' in y_list:
        table_columns = [
            {'name': 'id', 'id': 'id', 'type': 'numeric', 'hideable': True},
            {'name': 'Opexchange', 'id': 'opexchange', 'type': 'text'},
            {'name': 'Node', 'id': 'node_num', 'type': 'text'},
            {'name': 'Variable', 'id': 'var', 'type': 'text'},
            {'name': 'Importance', 'id': 'importance', 'type': 'numeric'},
            {'name': 'DA Congestion Corr', 'id': 'da_congestion_corr', 'type': 'numeric'},
            {'name': 'RT Congestion Corr', 'id': 'rt_congestion_corr', 'type': 'numeric'},
        ]

    return node_num_list, table_columns


# get variable importance data for a particular node_num
@callback(
    [
        Output('var_importance_and_plots_var_importance_table', 'data'),
        Output('var_importance_and_plots_gcp_csvfile_data', 'data')
    ],
    [
        Input('var_importance_and_plots_node_num', 'value'),
        Input('homepage-opexchange', 'data'),
        Input('var_importance_and_plots_strategy', 'value'),
        Input('var_importance_and_plots_y_list', 'value'),
        Input('var_importance_and_plots_number_of_vars', 'value'),
        State('var_importance_and_plots_datePickerRange', 'start_date'),
        Input('var_importance_and_plots_datePickerRange', 'end_date'),
        State('var_importance_and_plots_var_importance_table', 'columns')
    ]
)
def callback_function_get_var_importance(node_info: str, opexchange: str, strategy: str, y_list: str,
                                         number_of_vars: int, start_date: str, end_date: str,
                                         table_columns: list) -> tuple:
    if not node_info:
        cols_table_var_importance = [col_name['id'] for col_name in table_columns]
        return pd.DataFrame(columns=cols_table_var_importance).to_dict("records"), {}

    # Example node_info: "1656-<NODE_NAME>" → we only want the numeric id
    node_num = node_info.split("-")[0]

    if strategy == 'Curie' and opexchange == 'MISO':
        sql_query = f"""
            select
                node_num,
                dt,
                var,
                round(importance, 3) as importance
            from Curie_MISO.featureImportances
            where node_num = {node_num}
              and dt >= '{start_date}'
              and dt <= '{end_date}'
            order by importance desc
            limit {number_of_vars}
        """
        var_imp_df = sql_functions.download_df_from_sql_db(sql_query)

        if var_imp_df.empty:
            cols_table_var_importance = [col_name['id'] for col_name in table_columns]
            return pd.DataFrame(columns=cols_table_var_importance).to_dict("records"), {}

        var_imp_df['opexchange'] = opexchange
        var_imp_df['id'] = range(len(var_imp_df))

        data = None
        for bid_date in pd.date_range(start_date, end_date).strftime("%Y-%m-%d")[::-1]:
            try:
                csv_filename = f"gs://curiemiso/production/trainingcsvs/data_{int(node_num)}_{bid_date}.csv"
                data = pd.read_csv(csv_filename)
                break
            except FileNotFoundError:
                continue

        if data is None:
            return var_imp_df.to_dict("records"), {}

        var_list = [v for v in var_imp_df['var'].values if v in data.columns]
        cols_for_plot = ['dt', 'hr'] + var_list
        var_plot_df = data[cols_for_plot].copy()

        return var_imp_df.to_dict("records"), var_plot_df.to_dict("records")
    
    bq_dataset = f'{strategy}_Production'
    bq_table_name = f'{opexchange}_importance'

    # if node_info:  # eg: 12765_12766-<NODE_A>_<NODE_B> (for ERCOT) OR 635-<NODE_A>
    #     node_num = node_info.split("-")[0]

    if opexchange == 'ERCOT':  # for ERCOT, node_num is a string in BQ tables
        bq_query = f"""SELECT node_num, var, ROUND(SUM(importance), 3) AS importance FROM {bq_dataset}.{bq_table_name} 
                        WHERE dt >= '{start_date}' AND dt <= '{end_date}' AND y_list = '{y_list}' AND node_num = '{node_num}'
                        GROUP BY node_num, var
                        ORDER BY importance DESC
                        LIMIT {number_of_vars};"""

    else:  # for other markets, node_num is an int in BQ tables
        bq_query = f"""SELECT node_num, var, ROUND(SUM(importance), 3) AS importance FROM {bq_dataset}.{bq_table_name} 
                                WHERE dt >= '{start_date}' AND dt <= '{end_date}' AND y_list = '{y_list}' AND node_num = {node_num}
                                GROUP BY node_num, var
                                ORDER BY importance DESC
                                LIMIT {number_of_vars};"""

    var_imp_df = bigquery_functions.download_df_from_bq(bq_query)
    # cols = [node_num, var, importance]

    # read the latest available csv data file from GCP folder
    # loop through bid_date in reverse to grab the latest available csv
    for bid_date in pd.date_range(start_date, end_date).strftime("%Y-%m-%d")[::-1]:
        try:
            if opexchange == 'ERCOT':
                csv_filename = f'gs://ve_darwin/production/{opexchange}/training/{node_num}__{bid_date}.csv'
            else:
                csv_filename = f'gs://ve_darwin/production/{opexchange}/training/{node_num}_{bid_date}.csv'

            data = pd.read_csv(csv_filename)

            break

        except FileNotFoundError:
            # just continue to next bid_date
            continue

    var_imp_df['opexchange'] = opexchange
    var_imp_df['da_corr'] = 0
    var_imp_df['rt_corr'] = 0
    var_imp_df['da_congestion_corr'] = 0
    var_imp_df['rt_congestion_corr'] = 0
    var_imp_df['id'] = range(len(var_imp_df))  # create "id" column for selecting active cell

    for index, row in var_imp_df.iterrows():
        var_name = row['var']

        if 'total' in y_list:
            corr = data[['da_total', var_name]].corr().values[0][1].round(2)
            var_imp_df.iat[index, var_imp_df.columns.get_loc('da_corr')] = corr

            corr = data[['rt_total', var_name]].corr().values[0][1].round(2)
            var_imp_df.iat[index, var_imp_df.columns.get_loc('rt_corr')] = corr

        elif 'congestion' in y_list:
            corr = data[['da_congestion', var_name]].corr().values[0][1].round(2)
            var_imp_df.iat[index, var_imp_df.columns.get_loc('da_congestion_corr')] = corr

            corr = data[['rt_congestion', var_name]].corr().values[0][1].round(2)
            var_imp_df.iat[index, var_imp_df.columns.get_loc('rt_congestion_corr')] = corr

    # save date-hour level variable data to plot later
    var_plot_df = data[['dt', 'hr'] + list(var_imp_df['var'].values)]

    return var_imp_df.to_dict("records"), var_plot_df.to_dict("records")

    # else:

    #     cols_table_var_importance = [col_name['id'] for col_name in table_columns]

    #     return pd.DataFrame(columns=cols_table_var_importance).to_dict("records"), {}


# get variable plot for a particular variable selection
@callback(
    Output('var_importance_and_plots_variable_fig', 'figure'),
    [
        State('var_importance_and_plots_variable_datePickerRange', 'start_date'),
        Input('var_importance_and_plots_variable_datePickerRange', 'end_date'),
        Input('var_importance_and_plots_gcp_csvfile_data', 'data'),
        Input('var_importance_and_plots_var_importance_table', 'active_cell'),
        Input('var_importance_and_plots_var_importance_table', 'data'),
        Input('var_importance_and_plots_strategy', 'value'),

    ]
)
def callback_function_get_variable_figure(start_date: str, end_date: str, gcp_csvfile_data,
                                          active_cell, var_importance_table_data, strategy: str):
    fig = common_functions.get_fig()

    
    if active_cell is not None:
        try:
            var = var_importance_table_data[int(active_cell["row_id"])]["var"]

            # read the gcp csv-file data
            df = pd.DataFrame(gcp_csvfile_data)

            # filter out data in the relevant date-range
            df = df.query("dt >= @start_date and dt <= @end_date")

            df['dtHr'] = pd.to_datetime(df['dt']) + pd.to_timedelta(df['hr'] - 1, unit='h')

            # plot the trace of the var
            fig.add_trace(go.Scatter(x=df['dtHr'], y=df[var], name=var, showlegend=False))

            # only add the current vertical line when start_dt <= current time <= end_dt
            today = (pd.to_datetime("now", utc=True).tz_convert("US/Central") - pd.Timedelta(hours=1))
            if start_date <= str(today.date()) <= end_date:
                fig.add_vline(x=today.strftime("%Y-%m-%d %H"), line_width=1, line_color="black")

        except IndexError:
            pass

    return fig


# update global date picker on global dropdown change
common_functions.get_callback_for_update_global_datePickerRange(script_name=script_name, dropdown_options='default')

# update local date picker based on global date picker or local date picker dropdown options
common_functions.get_callback_for_update_local_datePickerRange(script_name=script_name, card_name='variable',
                                                               dropdown_options='default')