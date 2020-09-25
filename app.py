from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd

app= Flask(__name__)

@app.route('/')
def Index():    
    return render_template('index.html')

@app.route('/add_csv', methods=['POST'])
def Add_CSV():
    if request.method == 'POST':
        #Load file
        try:
            fileLoaded= request.files['file']
            fileLoaded.save(secure_filename(fileLoaded.filename))
            #Get file
            fileSaved= fileLoaded.filename
            data= pd.read_csv(fileSaved)
            #print(data)
            return redirect(url_for('Imput'))
        except:
            return 'Noooooooooooooooo'

@app.route('/imputation')
def Imput():
        return render_template('imputation.html')


if __name__ == '__main__':
    app.run(port = 3000, debug = True)

