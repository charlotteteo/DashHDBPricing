from datetime import date
import pandas as pd

import plotly.express as px
import numpy as np
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash import dash_table
import seaborn as sns
import matplotlib.pyplot as plt

current=pd.read_csv('resale-flat-prices/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv')
df=current[['month','flat_model',"flat_type","street_name","resale_price"]]
type_price=current[["flat_type","resale_price"]]

df=pd.read_csv('HDB2017onwardslonglatdata.csv')
df["latitude"]=pd.to_numeric(df["latitude"])
df["longtitude"]=pd.to_numeric(df["longtitude"])


mapfig = px.scatter_mapbox(df,lat='latitude',lon='longtitude', hover_name="town",hover_data= {
                            'resale_price':':$.2f', 
                            "latitude": False,
                            "longtitude": False
                        },
                        color_continuous_scale="sunsetdark",  color=df["resale_price"],
                        zoom=3, height=550,
                        # animation_frame="resale_price", 
                        size=df["resale_price"],size_max=40)
mapfig.update_layout(mapbox_style="open-street-map")
mapfig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#EFEFEF",
)



sns.set_theme(style="ticks", color_codes=True)


tips = sns.load_dataset("tips")
from matplotlib import pyplot
type_price['flat_type'] = pd.Categorical(type_price['flat_type'],
                                   categories=['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM','5 ROOM','EXECUTIVE'],
                                   ordered=True)
a4_dims = (11.7, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)

sns.boxplot(x="flat_type", y="resale_price", data=type_price)


# current['flat_model']=np.where(current['flat_model'].str.contains("Type"),"Type")
typebool=current['flat_model'].str.contains("Type")
mansionettebool=current['flat_model'].str.contains("Maisonette")
modelbool=current['flat_model'].str.contains("Model")
roombool=current['flat_model'].str.contains("room")

p=current['flat_model']
current['flat_model']=np.where(typebool,"Standard",np.where(mansionettebool, 'Maisonette',p.str.replace(" ","")))
current['flat_model']=np.where(modelbool,"Standard",np.where(roombool, 'Standard',p.str.replace(" ","")))
current['flat_model'].unique()



a4_dims = (20, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.boxplot(x="flat_model", y="resale_price", data=current)


# df = px.data.gapminder().query("year == 2007")
fig = px.treemap(current, path=[px.Constant("HDBs"), 'flat_model', 'flat_type'], values='resale_price',
                  color='resale_price', hover_data=['resale_price'],
                  color_continuous_scale='PuBu')
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))


fig1 = px.treemap(current, path=[px.Constant("HDBs"), 'town', 'flat_type'], values='resale_price',
                  color='resale_price', hover_data=['resale_price'],
                  color_continuous_scale='PuBu')
fig1.update_layout(margin = dict(t=50, l=25, r=25, b=25))



########DROP DOWN FEATURE ########################
df=current[['month','flat_model',"flat_type","street_name","resale_price","floor_area_sqm"]]
df["Cost Per SQM"]=df["resale_price"]/df["floor_area_sqm"]
df["Cost Per SQM"]=df["Cost Per SQM"].round(2)

apartmentdata=current[current['flat_model']=='Apartment']
apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
timeseriesdata=apartmentdata.groupby("month").mean()
timeseriesdata
# print(timeseriesdata)


currentgroup=current.groupby(["month","flat_model"]).count()
print(currentgroup)
fig =go.Figure(go.Sunburst(meta=currentgroup,
    labels=currentgroup.index,
    parents=currentgroup.index
))
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))




apartmentdata=current[current['flat_model']=='Apartment']
apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
timeseriesdata=apartmentdata.groupby("month").mean()
timeseriesdata
figApartment = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
figApartment.update_layout(
    title="HDB Apartment Prices",
    xaxis_title="Month/Year",
    yaxis_title="Resale Prices",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)



apartmentdata=current
apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
timeseriesdata=apartmentdata.groupby("month").mean()
timeseriesdata
figOverall = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
figOverall.update_layout(
    title="Overall HDB Prices",
    xaxis_title="Month/Year",
    yaxis_title="Resale Prices",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)



apartmentdata=current[current['flat_model']=='Maisonette']
apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
timeseriesdata=apartmentdata.groupby("month").mean()
timeseriesdata
figMaisonette = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
figMaisonette.update_layout(
    title="HDB Maisonette Prices",
    xaxis_title="Month/Year",
    yaxis_title="Resale Prices",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)




colors = {
    'background': '#EFEFEF',
    'text': '#4D4D4D'
}


app = dash.Dash()

app.layout = html.Div([
    html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='HDB Resale Price Analysis',
        
        style={
            'font-family':'Marker Felt',
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    
    html.Br(),

    
    
    html.H2(children='''HDBs Pricing by Type''',style={
            'font-family':'Marker Felt',
            'textAlign': 'left',
            'color': colors['text']
        }),
    
    html.Br(),
    dcc.Dropdown( id = 'dropdown',
        options = [
            {'label':'Apartment', 'value':'Apartment'},
            {'label': 'Maisonette', 'value':'Maisonette'},
            {'label': 'Terrace', 'value':'Terrace'},
            {'label': 'Multi Generation', 'value':'MultiGeneration'},
            {'label': 'Overall', 'value':'Overall'}
            
            ],
        value = 'Apartment',style={
            'font-family':'Marker Felt',
            'textAlign': 'left',
            'color': colors['text']}),
    html.Br(),
    
    dcc.Graph(id='bar_plot'), html.H2(children='''Singapore HDB Pricing Map''',style={
            'font-family':'Marker Felt',
            'textAlign': 'left',
            'color': colors['text']
        }),
    
    
    dcc.Graph(figure=mapfig),
    html.H2(children='''HDB Data''',style={
            'font-family':'Marker Felt',
            'textAlign': 'left',
            'color': colors['text']
        }),
    html.Div(children=dash_table.DataTable(
        id='datatable-interactivity',
        columns=[

            {"name": i, "id": i, "deletable": False, "selectable": False} for i in df.columns
        ],
        data=df.to_dict('records'),
       
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=False,
        page_action="native",
        page_current= 0,
        page_size= 10,
        style_header={
        'backgroundColor': 'rgb(93, 165, 218)',
        'color': 'black'
    },
    style_data={
        'backgroundColor': '#87CDEE)',
        'color': 'black',
        'textAlign': 'right'
        }
    )),
    html.Div(id='datatable-interactivity-container')

    ])
    ])

        
# ]




    
@app.callback(Output(component_id='bar_plot', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value')])


def graph_update(dropdown_value):
    # print("Here is " + dropdown_value)
    if dropdown_value=='Apartment':
        
        apartmentdata=current[current['flat_model']=='Apartment']
        apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
        timeseriesdata=apartmentdata.groupby("month").mean()
        timeseriesdata
        figApartment = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
        figApartment.update_layout(
            title="HDB Apartment Prices",
            xaxis_title="Month/Year",
            yaxis_title="Average Resale Prices(in SGD)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        figApartment.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        return figApartment
    if dropdown_value=='Terrace':
        
        apartmentdata=current[current['flat_model']=='Terrace']
        apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
        timeseriesdata=apartmentdata.groupby("month").mean()
        timeseriesdata
        figApartment = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
        figApartment.update_layout(
            title="HDB Terrace Prices",
            xaxis_title="Month/Year",
            yaxis_title="Average Resale Prices(in SGD)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        figApartment.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        return figApartment
    if dropdown_value=='MultiGeneration':
        
        apartmentdata=current[current['flat_model']=='MultiGeneration']
        apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
        timeseriesdata=apartmentdata.groupby("month").mean()
        timeseriesdata
        figApartment = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
        figApartment.update_layout(
            title="Multi Generation HDB Prices",
            xaxis_title="Month/Year",
            yaxis_title="Average Resale Prices(in SGD)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        figApartment.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        return figApartment
    elif dropdown_value=='Maisonette':
        
        apartmentdata=current[current['flat_model']=='Maisonette']
        apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
        timeseriesdata=apartmentdata.groupby("month").mean()
        timeseriesdata
        figMaisonette = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
        figMaisonette.update_layout(
            title="HDB Maisonette Prices",
            xaxis_title="Month/Year",
            yaxis_title="Average Resale Prices(in SGD)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        figMaisonette.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        
        
        return figMaisonette
        
    else:   
        apartmentdata=current
        apartmentdata["month"]=pd.to_datetime(apartmentdata["month"])
        timeseriesdata=apartmentdata.groupby("month").mean()
        timeseriesdata
        
        figOverall = go.Figure([go.Scatter(x=timeseriesdata.index, y=timeseriesdata['resale_price'])])
        figOverall.update_layout(
            title="HDB Overall Prices",
            xaxis_title="Month/Year",
            yaxis_title="Average Resale Prices(in SGD)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        figOverall.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
        return figOverall


@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)



def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))


def update_graphs(rows, derived_virtual_selected_rows):
    
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'for i in range(len(dff))]

    return [
        dcc.Graph(
            id=column,
            figure=graph_update_treemap(column,dff),
        )
        
        for column in ["street_name"] if column in dff
    ]

def graph_update_treemap(dropdown_value,data):
    
    
    # fig.show()

    fig1 = px.treemap(data, path=[px.Constant("HDBs"), dropdown_value,"flat_model","flat_type" ], values='resale_price',
                    color='resale_price', hover_data=['resale_price'],
                    color_continuous_scale='Purples')
    fig1.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig1.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)
    
    
    return fig1
    

if __name__ == '__main__': 
    app.run_server()
    
    
    