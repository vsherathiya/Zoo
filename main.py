from flask import jsonify, Flask, render_template, request,send_from_directory
import os
from flask_cors import CORS, cross_origin
import torch
import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
@cross_origin()
def get_response_from_chatbot():
     data = request.get_json()
     user_input = data["message"]
     last_message_data = data["prev_message_data"]
     print(user_input)
     print(last_message_data)
     from src.MitraAI_API.Local_Pack.LocalApi_copy import ProcessResponse
     LocalResponse = ProcessResponse(user_input,None)
     return  jsonify(LocalResponse)

if __name__ == '__main__':
   try:
        app.run(debug=False, port=8000)
   except Exception as e:
        print("Failed in script at: " + str(e))
        