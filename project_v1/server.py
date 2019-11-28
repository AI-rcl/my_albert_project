from flask import Flask, jsonify, abort, request
import tensorflow as tf
from load_model import Cls_loader
import os

INIT_TEXT='欢迎来到泰康'
cls_loader=Cls_loader()
cls_dict=cls_loader.model_dict

for cls in cls_dict.values():
    cls.predict(INIT_TEXT)

app = Flask(__name__)

@app.route('/AI_model/api/bot/', methods=['GET'])
def predict():
    bot_id=request.args.get('bot_id')
    sentence=request.args.get('sentence')
    res={}
    for name, model in cls_dict.items():
        res.setdefault(name, model.predict(sentence))

    return jsonify(res)


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)