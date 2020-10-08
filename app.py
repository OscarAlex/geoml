from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor as ETR

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
            data= pd.DataFrame()
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
            
        columns_name= list(data.columns)
        print(columns_name)
        #Get apply imputation
        imput= request.form['imputation']
        print(imput)
        #If apply imputation
        if imput=='1':
            #Separate columns numeric and no numeric
            def splitTypes(dframe):
                objectsN= []
                for i in dframe.columns:
                    #If i column is not  float
                    if(not (dframe[i].dtype == np.float64)):
                        #Get name
                        objectsN.append(i)
                        #Get no numeric column
                        objects= dframe[i]
                        #Drop no numeric column
                        dframe= dframe.drop(columns=objectsN)
                return dframe, objects
            
            x, y= splitTypes(data)
            #Extra Tree Classifier
            impute_est= ETR(n_estimators=10, random_state=0)
            #Iterative imputer
            estimator= IterativeImputer(random_state=0, estimator=impute_est)
            #Fit transform data
            impdf= estimator.fit_transform(x)
            #Join imputed data and class
            data= pd.concat([pd.DataFrame(impdf),y], axis=1)
            #Rename columns
            data.columns= columns_name

        print(data)
        return redirect(url_for('Learning'))

@app.route('/learning')
def Learning():
    return render_template('learning.html')

@app.route('/choose_learn', methods=['POST'])
def Choose_Learning():
    if request.method == 'POST':
        learning= request.form['learning']
        print(learning)
        #Return templates and column names
        if learning=='sup':
            return render_template('supfeatures.html', features=list(data.columns))
        else:
            return render_template('unsupfeatures.html', features=list(data.columns))

@app.route('/add_supfeats', methods=['POST'])
def ADD_SupFeats():
    if request.method == 'POST':
        #Get class selected
        cl= int(request.form['class'])
        print(data.columns[cl])
        clas= data[data.columns[cl]]

        #Get features selected
        fe= request.form.getlist('features')
        fe= [int(i) for i in fe] 
        print(fe)

        final_data= pd.DataFrame()
        for i in fe:
            final_data= pd.concat([final_data, data[data.columns[i]]], axis=1)
        
        print(final_data)
        print(clas)
        return redirect(url_for('Balance'))

@app.route('/balance')
def Balance():
    return render_template('balance.html')



if __name__ == '__main__':
    app.run(port = 3000, debug = True)

"""
@app.route('/add_supfeats', methods=['POST'])
def ADD_SupFeats():
    if request.method == 'POST':
        return redirect(url_for('Balance'))

@app.route('/balance')
def Balance():
    return render_template('balance.html')
"""