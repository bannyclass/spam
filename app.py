import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing import sequence
import json



app = Flask(__name__)
#model = pickle.load(open('rf.pkl', 'rb'))
loaded_model = load_model('./spam_model.sav')

@app.route('/')
def hello():
    return 'Hello World'

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    #print("Request:")
    #print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):

    #sessionID=req.get('responseId')
    result = req.get("queryResult")
    #user_says=result.get("queryText")
    #log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    message=parameters.get("msg")

	 
    intent = result.get("intent").get('displayName')
    
    if (intent=='yes'):
        
        tokenizer = None
        with open('./spam_tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        tokenizer

        sen = tokenizer.texts_to_sequences(message)
        sen

        text_matrix = sequence.pad_sequences(sen,maxlen=max_len)
        text_matrix

       

        
        
        
        prediction = loaded_model.predict(text_matrix)
    
        output = round(prediction[0], 2)
    
    	
        if(output<=0.5):
            msg_status = 'Ham'
    
        if(output>=0.5):
             msg_status = 'Spam'
        
            
        fulfillmentText= "The Message appears to be..  {} !".format(msg_status)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }
    #else:
    #    log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)

if __name__ == '__main__':
    app.run()
#if __name__ == '__main__':
#    port = int(os.getenv('PORT', 5000))
#    print("Starting app on port %d" % port)
#    app.run(debug=False, port=port, host='0.0.0.0')
