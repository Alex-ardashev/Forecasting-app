import boto3
import pandas as pd
import numpy as np
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
from pretty_html_table import build_table
from fbprophet import Prophet
import os
from dotenv import load_dotenv
load_dotenv()

# environmental variables



def email_alert(subject, to, dataset_fact, forecast, df_merged, username, api_key, user, password):



    # change URL of the graph per new client
    # make table and pull it to email body

    email_body = build_table(df_merged, 'blue_light')
    # plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset_fact['ds'], y=dataset_fact['y'], name='value_fact', ))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Prediction_lowest', ))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', name='Prediction_highest', ))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend', ))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Yearly', ))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['rain'], name='Rain',))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['temp'], name='Temp',))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['holidays'], name='Holidays', ))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['weekly'], name='Weekly', ))


    # connect to plotly
    username = username # your username
    api_key = api_key # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    # generate URL of the plot
    url = py.plot(fig, auto_open=False, filename='email-report-graph-2')

    template = (''
                #'<a href="{graph_url}" target="_blank">'  # Open the interactive graph when you click on the image
                #'<img src="{graph_url}.png">'  # Use the ".png" magic url so that the latest, most-up-to-date image is included
                #'</a>'
                #'{caption}'  # Optional caption to include below the graph
                '<br>'  # Line break
                '<a href="{graph_url}" style="color: rgb(0,0,0); text-decoration: none; font-weight: 200;" target="_blank">'
                'Click to see the interactive graph'  # Direct readers to Plotly for commenting, interactive graph
                '</a>'
                '<br>'
                '<br>'  # Line break
                '<hr>'  # horizontal line
                '')
    _ = template
    _ = _.format(graph_url=url, caption='')
    email_body += _

    """
    Add template button, if click - then anomaly doesnt count
    """
    # email attributes, connection to gmail
    message = MIMEMultipart()
    message['Subject'] = subject
    user = (user+'@gmail.com')
    password = password
    message['from'] = user
    message['To'] = to

    body_content = email_body
    message.attach(MIMEText(body_content, "html"))
    msg_body = message.as_string()

    server = SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(message['From'], password)
    server.sendmail(message['From'], message['To'], msg_body)
    server.quit()
    return

def prediction(dataset):
    """
    Modeling and prediction making.
    :param dataset: imported dataset
    :return: predicted metrics value for the next period, graph of the model performance
    """
    #obj = s3.Bucket(bucket).Object(forecast_dataset + ".csv").get()
    #forecast = pd.read_csv(obj['Body'])

    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    #model.add_country_holidays(country_name=country_holidays)
    model.fit(dataset)
    future = model.make_future_dataframe(periods=int(forecasting_period), freq=str(forecasting_frequency))
    forecast = model.predict(future)
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    #forecast['yhat'] = forecast['yhat'].fillna(0).astype(np.int64)


    return forecast



def main():


    s3 = boto3.resource(
        service_name='s3',
        region_name='eu-central-1',
        aws_access_key_id=str(aws_access_key_id),
        aws_secret_access_key=str(aws_secret_access_key)
    )

    # dataset_fact: ds; y
    obj = s3.Bucket(bucket).Object(initial_dataset + ".csv").get()
    dataset_fact = pd.read_csv(obj['Body'])
    # forecast: ds; y
    forecast = prediction(dataset_fact)
    # anomaly
    df_merged = pd.merge(dataset_fact, forecast, how='left', left_on='ds', right_on='ds')
    df_merged['percentage'] = abs(df_merged['y'] - df_merged['yhat']) / df_merged['yhat'] * 100
    percentage = df_merged['percentage'].fillna(0).astype(np.int64)

    df_merged['Alert'] = np.where(
        (df_merged['y'] < df_merged['yhat_lower']), f'{percentage.iloc[-1]}% Lower than expected', (np.where (df_merged['y'] > df_merged['yhat_upper'], f'{percentage.iloc[-1]}% Higher than expected', 'No')))
    df_merged = df_merged[['ds', 'yhat_lower', 'yhat', 'yhat_upper', 'y','Alert']].tail(1)
    df_merged.columns = ['Date', 'Lowest expected value', 'Expected prediction', 'Highest expected value',
                         'Actual value', 'Alert']
    print(df_merged)

    if df_merged.iloc[0]['Alert'] != 'No':
        pass
        #email_alert(subject, to, dataset_fact, forecast, df_merged, username, api_key, user, password)
    else:
        pass

if __name__ == "__main__":
    main()
