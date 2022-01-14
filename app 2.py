from flask import Flask,render_template,request,jsonify

from flask_cors import CORS

from chat import chatbot_response

app = Flask(__name__)

CORS(app)

#@app.route("/",methods = ["GET"])
@app.get("/")
def home():

    return render_template("Home_page.html")

@app.post("/response")
def response():

    text = request.get_json().get("message")
    response = chatbot_response(text)
    reply = {"answer" : response}
    return jsonify(reply)

if __name__ == "__main__":

    app.run(debug = True)
