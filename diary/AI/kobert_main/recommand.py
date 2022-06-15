# -*- coding: utf-8 -*- 
#---- subscriber.py  데이터 받기 
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

# AI Package
import pandas as pd
import torch
from torch.utils.data import DataLoader
import gluonnlp as nlp
import numpy as np
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

from models import BERTClassifier

from data import BERTDatasetInference

# AI package
bert_model, vocab = get_pytorch_kobert_model()
model = BERTClassifier(bert_model, dr_rate=0.5)
model.load_state_dict(torch.load("/home/ubuntu/IOT/restful_server/diary/encorder_data/model1.pt",map_location=torch.device("cpu")))
model.eval()

tokenizer = get_tokenizer()
token = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
device = torch.device("cpu")

def inference_ai(sentence):
    dataset = BERTDatasetInference(sentence, token, 256, True, False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    output = []
    for token_ids, valid_length, segment_ids in loader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        output.append(model(token_ids, valid_length, segment_ids))
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output[0])[0].tolist()
    return output



recommand = [0,0]
color = [0,0,0,0,0,0,0,0]



def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

#서버로부터 publish message를 받을 때 호출되는 콜백
def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))
    
def on_message(client, userdata, msg):

    print(str(msg.payload.decode("utf-8")))
    message = str(msg.payload.decode("utf-8"))
    recommand = message.split(",")
    
    
try:
    client = mqtt.Client() #client 오브젝트 생성
    client.on_connect = on_connect #콜백설정
    client.on_subscribe = on_subscribe
    client.on_message = on_message #콜백설정

    client.connect('172.30.1.35', 1883)  # 라즈베리파이 커넥트  
    client.subscribe('Iot/dairy', 0)  # 토픽 : temp/temp  | qos : 1
    client.loop()
    
    color = ""
    sentiment = [0,0,0,0,0,0,0,0]
    
    content = recommand[1]
    
    temp = pd.read_csv("/home/ubuntu/IOT/restful_server/diary/encorder_data/encoder.csv", encoding="utf-8")
    encoder = temp.sort_values(by=['code']).reset_index(drop=True)
    
    sentence = f"{content}"
    output = inference_ai(sentence)
    
    

    color = (colorize(output, encoder))# RGB 코드 255곱해서 정수로 변환해야 할 수 있다.
    color_hex = "".join([hex(int(255 * x))[-2:] for x in color])

   
    

    
    
   

    while True:
        
        if(recommand[0] == 'recommand'):
            light = f"init,android_mood,{color},on"
            client.publish("Iot/light", color)
            
    


except KeyboardInterrupt:
    print("bye")