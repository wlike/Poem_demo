# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, abort, jsonify
from jiuge_lvshi import Poem
import json
import logging
from utils import *

model_path = 'model_jl/epoch=0-step=49999.ckpt'
model_config = 'os_model_ch_poem/config.json'
vocab_file = 'os_model_ch_poem/vocab.txt'
genre2name = {0: "五言绝句", 1: "七言绝句", 2: "五言律诗", 3: "七言律诗"}

app = Flask(__name__)

poem = Poem(model_path, model_config, vocab_file)

@app.route("/poem")
def index():
    return render_template("index.html")

@app.route("/poem/generate", methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        genre = int(request.args.get("genre", 1))
        if genre not in set(range(4)):
            return json.dumps({"code": 400, "content": "请设置合法的genre，取值{0, 1, 2, 3}，分别对应五言绝句、七言绝句、五言律诗、七言律诗"}, ensure_ascii=False).encode("utf-8")
        prefix = request.args.get("prefix", None)
        if prefix == "":
            prefix = None
        if prefix is not None:
            if genre in [0, 1]:  # 绝句
                if len(prefix) != 4:
                    return json.dumps({"code": 401, "content": "你选择了{}，它的藏头词只能是4个汉字".format(genre2name[genre])}, ensure_ascii=False).encode("utf-8")
            else:  # 律诗
                if len(prefix) != 8:
                    return json.dumps({"code": 402, "content": "你选择了{}，它的藏头词只能是8个汉字".format(genre2name[genre])}, ensure_ascii=False).encode("utf-8")
        msg = "genre: {}, prefix: {}".format(genre2name[genre], prefix if prefix is not None else "")
        app.logger.info(msg)
        content = poem.generate(genre=genre, prefix=prefix)
        sents = content[:-1].split("，")
        return json.dumps({"code": 200, "content": sents, "prefix": prefix}, ensure_ascii=False).encode("utf-8")
    else:  # request.method == 'POST'
        genre = request.form.get("genre", None)
        if genre is None:
            return render_template("index.html", poem="请选择一种古诗类型")
        else:
            genre = int(genre)
        prefix = request.form.get("prefix", None)
        if prefix == "":
            prefix = None
        if prefix is not None:
            if not is_all_chinese(prefix):
                return render_template("index.html", poem="藏头词只能是汉字")
            if genre in [0, 1]:  # 绝句
                if len(prefix) != 4:
                    res = "你选择了{}，它的藏头词只能是4个汉字".format(genre2name[genre])
                    return render_template("index.html", poem=res)
            else:  # 律诗
                if len(prefix) != 8:
                    res = "你选择了{}，它的藏头词只能是8个汉字".format(genre2name[genre])
                    return render_template("index.html", poem=res)
        msg = "genre: {}, prefix: {}".format(genre2name[genre], prefix if prefix is not None else "")
        app.logger.info(msg)
        content = poem.generate(genre=genre, prefix=prefix)
        res = "\n".join(content[:-1].split("，"))
        return render_template("index.html", prefix=prefix, poem=res)

if __name__ == "__main__":
    handler = logging.FileHandler("poem.log", encoding="UTF-8")
    handler.setLevel(logging.DEBUG)
    logging_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s")
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)
    app.config["JSON_AS_ASCII"] = False
    # 这样设置 host，只能在本机访问
    #app.run(host="127.0.0.1", port="36818")
    # 这样设置 host，可以允许局域网内的其他机器访问
    app.run(host="0.0.0.0", port="36818")
