from flask import Flask, redirect, render_template, request, jsonify, Markup, flash

## keep this  commented and run ngrok from a separate terminal it generates multiple errors running it inside the code
#from flask_ngrok import  run_with_ngrok only if you want to use it! For now this is working locally

import subprocess

import os

from flask_cors import CORS
from flask import session

app = Flask(__name__)
# create always a new connection with ngrok using an endpoint in separate file. This file will be read by app.js
app.secret_key = 'secret_key'
CORS(app)
indicator = 0
user_name=''
folding=''
classifiers=[]
features=[]

@app.route("/")
def index():
   return render_template("base.html")

@app.post("/erase")
def erase():
   print('erase')
   session.pop('_flashes', None)
   message = Markup("")
   flash(message)
   return(jsonify("ack"))

@app.route("/user_rec", methods=['POST'])
def user_rec():
    global indicator
    global user_name
    global folding
    global classifiers
    global features
    if indicator==0:
       user_name = request.form['user_input']
       folding = request.form['fold']
       classifiers = request.form.getlist('check')
       features = request.form.getlist('check_features')
    print(user_name,folding,classifiers,features,'data')
    print(indicator,'indicator')
    if (not(user_name) or not(folding) or len(features)==0 or len(classifiers)==0) and indicator==0:
      if indicator==0:
         message = Markup("please fill up all the data!")
      else:
         message = Markup("please wait for the loading..!")
      flash(message)
    else:
       ## select  features and classifiers
       if classifiers[0]=='1':
          classi='Logistic'
       elif classifiers[0]=='2':
          classi='SVM'
       else:
          classi='Neural Network'
       if features[0]=='1':
          Ffeatures='no adding features'
       else:
         Ffeatures='adding features' 
       message = Markup(f"Evaluation of {classi} classifier and {Ffeatures} in {folding}-fold crossvalidation...Want to run the process? If yes wait until it finishes..") 
       flash(message)
       if indicator==1:
          ## wait to finish the subprocess
          if classifiers[0]=='1' and features[0]=='1':
             p = subprocess.Popen(f"python baseline_logistic.py technical_test.csv {folding} 0", stdout=subprocess.PIPE, shell=True)
          if classifiers[0]=='1' and features[0]=='2':
             p = subprocess.Popen(f"python model_evaluation_adding_features.py technical_test.csv {folding} 0", stdout=subprocess.PIPE, shell=True)
          if classifiers[0]=='2' and features[0]=='1':
             p = subprocess.Popen(f"python baseline_svm.py technical_test.csv {folding} 0", stdout=subprocess.PIPE, shell=True)
          if classifiers[0]=='2' and features[0]=='2':
             p = subprocess.Popen(f"python model_evaluation_adding_features_svm.py technical_test.csv {folding} 0", stdout=subprocess.PIPE, shell=True)
          if classifiers[0]=='3' and features[0]=='1':
             p = subprocess.Popen(f"python baseline_nn.py technical_test.csv {folding} 0", stdout=subprocess.PIPE, shell=True)
          if classifiers[0]=='3' and features[0]=='2':
             p = subprocess.Popen(f"python model_evaluation_adding_features_nn.py technical_test.csv {folding} 0", stdout=subprocess.PIPE, shell=True)
          (output, err) = p.communicate() 
          output=str(output.decode('utf-8'))
          output=output.replace(':','<br></br>')
          print(output,'output')
          #p_status = p.wait()
          flash(Markup('<br></br>Results are:<br></br>'+output))
          indicator=0
          user_name = None
          folding = None
          classifiers = []
          features = []
       if (user_name) or (folding) or len(features)==1 or len(classifiers)==1:
          indicator=1

    return redirect('/')


if __name__ == "__main__":
   app.run(debug=True,host='0.0.0.0',port=8080) #use_reloader=False)
