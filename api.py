import flask
from main import answer_question
import pandas as pd
import numpy as np
import werkzeug.exceptions
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load Flask
def get_proxy_remote_address():
	"""
	:return: the ip address for the current request (or 127.0.0.1 if none found)
	"""
	if flask.request.headers.get('X-Forwarded-For') != None:
		return str(flask.request.headers.get('X-Forwarded-For'))
	return flask.request.remote_addr or '127.0.0.1'

app = flask.Flask(__name__)
limiter = Limiter(
    get_proxy_remote_address,
    app=app,
    storage_uri="memory://",
)


# Load the data
df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()

# Create the API to answer questions
@app.route('/api/v1/ask', methods=['POST'])
@limiter.limit("10/30 seconds")
def ask():
    question = flask.request.json['prompt']
#    print(question + " - " + flask.request.headers['X-Forwarded-For'])
    print(question + " - " + get_proxy_remote_address())
    answer = answer_question(df, question=question, debug=True)
    print(answer)
    return flask.jsonify(answer)
    
# Create the Web UI
@app.route('/<path:path>')
def index(path):
    print(path + "index.html")
    try:
        return flask.send_from_directory('ui', path)
    except werkzeug.exceptions.NotFound as e:
        if path.endswith("/"):
            return flask.send_from_directory('ui', path + "index.html")
        raise e

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3636, debug=True)
