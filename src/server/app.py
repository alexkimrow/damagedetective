from flask import Flask, request, jsonify
from flask_cors import CORS
from message import do_everything

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/upload", methods=["POST"])
def uploadfile():

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    result = processimage(file)

    return result


def processimage(file):
    return do_everything(file)


if __name__ == "__main__":
    app.run(debug=True)
