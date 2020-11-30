
import boto3
import time
import json

kinesis = boto3.client('kinesis', region_name='us-east-1')
ec2 = boto3.client('ec2', region_name='us-east-1')

shard_it = kinesis.get_shard_iterator(StreamName = "stock_stream",
                                     ShardId = 'shardId-000000000000',
                                     ShardIteratorType = 'LATEST'
                                     )["ShardIterator"]

i = 0
s = 0
    
while True:
    out = kinesis.get_records(ShardIterator = shard_it, Limit = 1)
    for o in out['Records']:
        jdat = json.loads(o['Data'])
        price = jdat['price']
        event_time = jdat['event_time']
        i = i + 1
    
    if i != 0:
        print("Current Stock Price: ", price)
        print("Current Time:", event_time)
        print("\n")
        
        
    shard_it = out['NextShardIterator']
    time.sleep(0.2)
