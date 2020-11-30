
import boto3
import random
import datetime
import json

kinesis = boto3.client('kinesis', region_name='us-east-1')

# Continously write stock data into Kinesis stream
while True:
    now = datetime.datetime.now() 
    str_now = now.isoformat() 
    price = random.random() * 100 
    data = {'event_time': str_now,
            'ticker': 'AAPL',
            'price': round(price, 2)
            }
    kinesis.put_record(StreamName = "stock_stream",
                       Data = json.dumps(data),
                       PartitionKey = "partitionkey"
                      )
    #print("data in")
