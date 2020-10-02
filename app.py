from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd

app= Flask(__name__)

#List with columns almost empty
emptyCols= ''
@app.route('/')
def Index():
    global emptyCols
    #Reset the variable in case of return
    emptyCols=''
    return render_template('index.html')

#Dataset
data= pd.DataFrame()
@app.route('/add_csv', methods=['POST'])
def Add_CSV():
    if request.method == 'POST':
        try:
            #Load file
            fileLoaded= request.files['file']
            fileLoaded.save(secure_filename(fileLoaded.filename))
            #Get file
            fileSaved= fileLoaded.filename
            global data
            data= pd.read_csv(fileSaved)
            #Treeshold if NaN
            df= list(data.loc[:, data.isnull().mean() < .9].columns)
            print(data.head())
            #print('\n')
            print(list(data.columns))
            print(df)
            #If many NaNs, get the columns name
            if(list(data.columns) != df):
                print("Hay columnas vacías")
                global emptyCols
                #Get the columns
                cols= list(set(data) - set(df))
                emptyCols= ', '.join(cols)
                print(emptyCols)
            return redirect(url_for('Imput'))
        except:
            return 'Noooooooooooooooo'


@app.route('/imputation')
def Imput():
    #return template and variable
    return render_template('imputation.html', emptyCols=emptyCols)

@app.route('/add_imput', methods=['POST'])
def Add_Imput():
    if request.method == 'POST':
        #Keep/remove empty columns if exist
        if emptyCols:
            keep= request.form['empty']
            print(keep)
            #If not keep empty columns, change data
            if keep=='0':
                global data
                data= data.loc[:, data.isnull().mean() < .9]
            
        print(list(data.columns))
        #Get apply imputation
        imput= request.form['imputation']
        #print(imput)
        return redirect(url_for('Learning'))

@app.route('/learning')
def Learning():
    return render_template('learning.html')

@app.route('/choose_learn', methods=['POST'])
def ChooseLearning():
    if request.method == 'POST':
        learning= request.form['learning']
        print(learning)
        if learning=='sup':
            return render_template('supfeatures.html')
        else:
            return render_template('unsupfeatures.html')

if __name__ == '__main__':
    app.run(port = 3000, debug = True)

