from flask import Flask, redirect,request,redirect,url_for,render_template
import requests
from werkzeug.utils import secure_filename
import os
import sys
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
import openai
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
import re
import tiktoken
import qdrant_client
from qdrant_client.http import models as rest
import uuid


app = Flask(__name__)
UPLOAD_FOLDER = './static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
os.environ['AZURE_OPENAI_KEY']= "YOUR_AZURE_OPENAI_KEY" 
os.environ['AZURE_OPENAI_ENDPOINT']= "YOUR_AZURE_OPENAI_ENDPOINT" 
os.environ['OPENAI_API_KEY']= "YOUR_OPENAI_API_KEY" 
os.environ['OPENAI_API_ENDPOINT']= "YOUR_OPENAI_API_ENDPOINT" 
openai.api_type = "type"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
openai.api_version = "version"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

def get_completion(prompt, model="LLM_model"): 
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response['choices'][0]['message']['content']

def get_response(img_path):
    f_list = []
    prompt = '''I am a merchant on an ecommerce website. 
    Give me product specifications for this product for listing it on the website. 
    I need the following details: Product Name, Brand, Product Type, Ingredients, Suitable for, Scent, Country of Origin, Product Description, Benefits and Size in Json'''
    for img_file in os.listdir(img_path):
        f = os.path.join(img_path, img_file)
        # checking if it is a file
        if os.path.isfile(f):
            string = pytesseract.image_to_string(f)
            response = get_completion(string + prompt)
            f_list.append(response)
    f_listToStr = ' '.join([str(f) for f in f_list])
    f_listToStr_prmpt = "get maximum details from all the json below and provide the following details in single json: Product Name, Brand, Product Type, Ingredients, Suitable for, Scent, Country of Origin, Product Description, Benefits and Size "+ f_listToStr
    return(get_completion(f_listToStr_prmpt))

def embedding_upload(df_content):
        print("embedding started")
        df_content['concat'] = df_content.astype(str).sum(axis=1)
        def normalize_text(s, sep_token = " \n "):
            s = re.sub(r'\s+',  ' ', s).strip()
            s = re.sub(r". ,","",s)
            s = s.replace("..",".")
            s = s.replace(". .",".")
            s = s.replace("\n", "")
            s = s.strip()
            return s
        df_content['concat']= df_content["concat"].apply(lambda x : normalize_text(x))
        df_count = len(df_content)
        print("normalise text")
        tiktoken_cache_dir = "./static/token"
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
        assert os.path.exists(os.path.join(tiktoken_cache_dir,"tokenfile"))
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print("tokenizer initiated")
        df_content['n_tokens'] = df_content["concat"].apply(lambda x: len(tokenizer.encode(x)))
        print("Text tokenized")
        df_content = df_content[df_content.n_tokens<30000]
        
        print("rows more than 30K token are discarded")
        df_content['ada_v2'] = df_content["concat"].apply(lambda x : get_embedding(x, engine = 'YOUR_ENGINE'))
        client = qdrant_client.QdrantClient(url="YOUR_QDRANT_URL", timeout=3000)
        vector_size = len(df_content["ada_v2"][0])
        print("Vector size ",vector_size)
        
        print("Collection recreation done ")
        for i in range(df_count//300 +1):
            k,j=i*300,(i+1)*300
            if (i+1)*300>df_count:
                j=df_count
            df_content_truncate = df_content.loc[k:j]
            client.upsert(
                collection_name='catalogdb',
                points=[
                    rest.PointStruct(
                        id=k,
                        vector={
                            "title": v["ada_v2"],
                        },
                        payload=v.to_dict(),
                    )
                    for k, v in df_content_truncate.iterrows()
                ],
            )
            print("In loop ", j)
        return "done"

@app.route("/")
def dashboard():
    return render_template('index.html')
@app.route("/about")
def about():
    return render_template('about.html')
@app.route('/details', methods=['POST'])
def upload_file():
   if request.method=="POST":
        uploaded_files = request.files.getlist('uimage')
        print(uploaded_files,file=sys.stderr)
        for file in uploaded_files:
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
        data =get_response("./static/files")
        data = eval(data)
        print(data["Product Name"])
        return render_template('res.html',data=data)
   
@app.route('/review', methods=['POST'])
def review():
   if request.method=="POST":
        data = {}
        data["Product Name"]=request.form.get("pname")
        data["Brand"]=request.form.get("brand")
        data["Product Type"]=request.form.get("ptype")
        data["Ingredients"]=request.form.get("ingredients")
        data["Suitable for"]=request.form.get("suitable")
        data["Scent"]=request.form.get("scent")
        data["Country of Origin"]=request.form.get("coo")
        data["Product Description"]=request.form.get("pdesp")
        data["Benefits"]=request.form.get("benefits")
        print(type(data),file=sys.stderr)
        return render_template('review.html',data=data)
@app.route('/final', methods=['POST'])
def final():
   if request.method=="POST":
        data = {}
        data["Product Name"]=request.form.get("pname1")
        data["Brand"]=request.form.get("brand1")
        data["Product Type"]=request.form.get("ptype1")
        data["Ingredients"]=request.form.get("ingredients1")
        data["Suitable for"]=request.form.get("suitable1")
        data["Scent"]=request.form.get("scent1")
        data["Country of Origin"]=request.form.get("coo1")
        data["Product Description"]=request.form.get("pdesp1")
        data["Benefits"]=request.form.get("benefits1")
        print(data,file=sys.stderr)
        df=pd.json_normalize(data)
        embedding_upload(df)
        return render_template('final.html')
   
if (__name__ == "__main__"):
     app.run(host='0.0.0.0',port=5000,debug=True)