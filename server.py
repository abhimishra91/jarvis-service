from flask import Flask, request, jsonify
from jarvis import Jarvis
import json

jarvis = Jarvis()

app = Flask(__name__)

# Create URL route in our application for "/"
@app.route('/v1')
def home():
    """
    This is the main page. 
    """
    
    return "This is a rest service for Jarvis. Jarvis is a AI text classification agent. He will try to predict the correct queue for the email you pass to it."

@app.route('/v1/predict',methods = ['POST', 'GET'])
def predict():
    """
    This fuction is to create the post and get method for prediction
    """
    if request.method == 'POST':
        data = request.get_json()
        query = data["email"]
        result = jarvis.predict(query)
        return jsonify(result)
    else:
        query = request.args.get('email')
        result = jarvis.predict(query)
        return jsonify(result)

# If we running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)