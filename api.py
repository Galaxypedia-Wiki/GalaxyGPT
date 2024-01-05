import flask
from main import answer_question, df, ADCS, dataset, strtobool
import pandas as pd
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from waitress import serve
import traceback
import logging

# Load Flask
def get_proxy_remote_address():
    """
    :return: the ip address for the current request (or 127.0.0.1 if none found)
    """
    if flask.request.headers.get("X-Forwarded-For") != None:
        return str(flask.request.headers.get("X-Forwarded-For"))
    return flask.request.remote_addr or "127.0.0.1"


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
app = flask.Flask(__name__)
limiter = Limiter(
    get_proxy_remote_address,
    app=app,
    storage_uri="memory://",
)

# Check if the dataset is set in the environment variables
if not os.getenv("DATASET"):
    raise Exception("Please set the DATASET environment variable")

adcsservice = ADCS()

# Create the API to answer questions
@app.route("/api/v1/ask", methods=["POST"])
@limiter.limit("10/30 seconds")
def ask():
    data = flask.request.get_json()

    if data is not None:
        question = data["prompt"]

        if question is not None:
            print(question + " - " + get_proxy_remote_address())

            username = data["username"] or None

            # If the username is provided, use it to answer the question
            if username is not None:
                try:
                    answer = answer_question(
                        df, question=question, username=username, debug=True
                    )
                    return flask.jsonify(answer), 200
                except Exception as e:
                    print(traceback.format_exc(), flush=True)
                    return flask.jsonify({"error": str(e)}), 500
            else:
                # Otherwise, answer the question without a username
                try:
                    answer = answer_question(df, question=question, debug=True)
                    return flask.jsonify(answer), 200
                except Exception as e:
                    print(traceback.format_exc(), flush=True)
                    return flask.jsonify({"error": str(e)}), 500


        else:
            return flask.jsonify({"error": "No prompt"}), 400
    else:
        return flask.jsonify({"error": "No data"}), 400


# Create the Web UI
""" @app.route('/<path:path>')
def index(path):
    print(path + "index.html")
    try:
        return flask.send_from_directory('ui', path)
    except werkzeug.exceptions.NotFound as e:
        if path.endswith("/"):
            return flask.send_from_directory('ui', path + "index.html")
        raise e """

# ADCS API
@app.route("/api/ADCS/start", methods=["POST"])
def startADCS():
    adcsservice.start()
    return flask.jsonify({"status": "started"}), 200

@app.route("/api/ADCS/stop", methods=["POST"])
def stopADCS():
    adcsservice.stop()
    return flask.jsonify({"status": "stopped"}), 200

@app.route("/api/ADCS/force-create", methods=["POST"])
def forceCreateADCS():
    reload = strtobool(flask.request.args.get("reload", default="True"))
    noembeddings = strtobool(flask.request.args.get("noembeddings", default="False"))
    
    # For the time being, set noembeddings to True manually
    noembeddings = True
    
    adcsservice.createDataset(reload=reload, noembeddings=noembeddings)
    if reload:
        return flask.jsonify({"status": "created a new dataset & reloaded"}), 200
    elif not reload:
        return flask.jsonify({"status": "created a new dataset"}), 200

@app.route("/api/ADCS/status", methods=["GET"])
def statusADCS():
    return flask.jsonify({"status": adcsservice.status}), 200

if __name__ == "__main__":
    debug = os.environ.get("DEBUG", True)
    use_waitress = os.getenv("USE_WAITRESS", False)
    print("Debug: " + str(debug))

    if debug is not bool:
        debug = strtobool(debug)
        
    if use_waitress is not bool:
        use_waitress = strtobool(use_waitress)

    if debug == True and not use_waitress:
        app.run(host="0.0.0.0", port=3636, debug=True)
        print("Started with flask", flush=True)
    else:
        logger = logging.getLogger("waitress")
        logger.setLevel(logging.INFO)
        serve(app, host="0.0.0.0", port=3636)
        print("Started with waitress", flush=True)
