from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
from fastapi import FastAPI
import uvicorn


tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", cache_dir='pretrainedmodel/tokenizer')
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", cache_dir='pretrainedmodel/model')


app = FastAPI()

@app.post("/")
def read_root(sentence):

    encoding = tokenizer.encode_plus(sentence, return_tensors='pt')
    outputs = model(**encoding)[0]
    softmax = F.softmax(outputs, dim = 1)

    print(f'***** : {round(float(softmax[0,4]), 2)}')
    print(f'****  : {round(float(softmax[0,3]), 2)}')
    print(f'***   : {round(float(softmax[0,2]), 2)}')
    print(f'**    : {round(float(softmax[0,1]), 2)}')
    print(f'*     : {round(float(softmax[0,0]), 2)}')
    
    
    #return {1 : (round(float(softmax[0,4]), 2)+round(float(softmax[0,3]), 2))/2, 0 : round(float(softmax[0,2]), 2), -1: (round(float(softmax[0,1]), 2)+round(float(softmax[0,0]), 2))/2}
    

    #return {"5" : round(float(softmax[0,4]), 2), "4" : round(float(softmax[0,3]), 2), "3" : round(float(softmax[0,2]), 2), "2" : round(float(softmax[0,1]), 2),"1" : round(float(softmax[0,0]), 2)}
   
    dic = {"5" : round(float(softmax[0,4]), 2), "4" : round(float(softmax[0,3]), 2), "3" : round(float(softmax[0,2]), 2), "2" : round(float(softmax[0,1]), 2),"1" : round(float(softmax[0,0]), 2)}

    return int(max(dic, key=dic.get))

     
   #for sentence in sentences:
   #    encoding = tokenizer.encode_plus(sentence, return_tensors='pt')
   #    outputs = model(**encoding)[0]
   #    softmax = F.softmax(outputs, dim = 1)
   #    dic = {"5" : round(float(softmax[0,4]), 2), "4" : round(float(softmax[0,3]), 2), "3" : round(float(softmax[0,2]), 2), "2" : round(float(softmax[0,1]), 2),"1" : round(float(softmax[0,0]), 2)}

   #    return int(max(dic, key=dic.get))




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
