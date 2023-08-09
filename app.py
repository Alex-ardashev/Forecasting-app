import os
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
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
INITIAL_DATASET = os.getenv('INITIAL_DATASET')
FORECASTING_PERIOD = os.getenv('FORECASTING_PERIOD', 1)
FORECASTING_FREQUENCY = os.getenv('FORECASTING_FREQUENCY', 'D')
SUBJECT = os.getenv('EMAIL_SUBJECT')
TO_ADDRESS = os.getenv('TO_ADDRESS')
USERNAME = os.getenv('PLOTLY_USERNAME')
API_KEY = os.getenv('PLOTLY_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')


def fetch_data_from_s3(bucket_name, dataset_name):
    s3 = boto3.resource(
        service_name='s3',
        region_name='eu-central-1',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    obj = s3.Bucket(bucket_name).Object(dataset_name + ".csv").get()
    return pd.read_csv(obj['Body'])


def prediction(dataset):
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(dataset)
    future = model.make_future_dataframe(periods=int(FORECASTING_PERIOD), freq=FORECASTING_FREQUENCY)
    forecast = model.predict(future)
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    return forecast


def detect_anomalies(dataset_fact, forecast):
    df_merged = pd.merge(dataset_fact, forecast, how='left', left_on='ds', right_on='ds')
    df_merged['percentage'] = abs(df_merged['y'] - df_merged['yhat']) / df_merged['yhat'] * 100
    percentage = df_merged['percentage'].fillna(0).astype(np.int64)

    df_merged['Alert'] = np.where(
        (df_merged['y'] < df_merged['yhat_lower']), f'{percentage.iloc[-1]}% Lower than expected',
        (np.where(df_merged['y'] > df_merged['yhat_upper'], f'{percentage.iloc[-1]}% Higher than expected', 'No')))
    
    return df_merged[['ds', 'yhat_lower', 'yhat', 'yhat_upper', 'y','Alert']].tail(1)


def generate_email_body(df_merged):
    email_body = build_table(df_merged, 'red_light')
    fig = go.Figure(data=[
        go.Scatter(x=df_merged['ds'], y=df_merged['yhat_lower'], name='Lowest expected value'),
        go.Scatter(x=df_merged['ds'], y=df_merged['yhat_upper'], name='Highest expected value', fill='tonexty'),
        go.Scatter(x=df_merged['ds'], y=df_merged['y'], name='Real value'),
        go.Scatter(x=df_merged['ds'], y=df_merged['trend'], name='General trend')
    ])

    chart_studio.tools.set_credentials_file(username=USERNAME, api_key=API_KEY)
    url = py.plot(fig, auto_open=False)

    email_template = 
    f"""
        <a href="{url}" style="color: rgb(0,0,0); text-decoration: none; font-weight: 200;" target="_blank">
            <img src="{url}.png">
        </a>
        <br><br>
        <hr>
        """
    return email_body + email_template


def send_email(subject, to, email_body):
    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = f"{GMAIL_USER}@gmail.com"
    message['To'] = to
    message.attach(MIMEText(email_body, "html"))
    msg_body = message.as_string()

    with SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(message['From'], GMAIL_PASSWORD)
        server.sendmail(message['From'], message['To'], msg_body)


def main():
    dataset_fact = fetch_data_from_s3(BUCKET_NAME, INITIAL_DATASET)
    forecast = prediction(dataset_fact)
    df_merged = detect_anomalies(dataset_fact, forecast)

    if df_merged.iloc[0]['Alert'] != 'No':
        email_body = generate_email_body(df_merged)
        send_email(SUBJECT, TO_ADDRESS, email_body)


if __name__ == "__main__":
    main()
