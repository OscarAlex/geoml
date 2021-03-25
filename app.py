#Flask
from flask import Flask, render_template, request, redirect, url_for, send_file, make_response, Response
from werkzeug.utils import secure_filename
from flask import g, session
from flask_session import Session
#Dataframe and arrays
import pandas as pd
import numpy as np
#Imputing
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor as ETR
#Processing data
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
#Classifiers
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#Metrics
from sklearn import metrics
from sklearn.metrics import classification_report
import eli5
from eli5.sklearn import PermutationImportance
#Download CSV
import io
from io import StringIO
#PCA and clusters
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.cluster import KMeans
#Plots
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

app= Flask(__name__)
app.secret_key= 'secret'

SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

app.config.update(SESSION_COOKIE_SAMESITE=None, SESSION_COOKIE_SECURE=True)

###############
## Functions ##
###############
#Read the entered csv
def getFile(fileLoaded):
    #fileLoaded.save(secure_filename(fileLoaded.filename))
    #Get file name
    session['Filename']= fileLoaded.filename
    session['Filename']= session['Filename'].replace('.csv', '')
    #Get file
    #fileSaved= fileLoaded.filename
    df= pd.read_csv(fileLoaded)
    #Drop completely empty rows, reset indexes and drop index column
    df= df.dropna(how='all').reset_index().drop('index', axis=1)
    #Drop completely empty columns
    df= df.dropna(axis='columns', how='all')

    return df

#Impute data
def imputData(data):
    #Separate columns numeric and no numeric
    def splitTypes(dframe):
        objectsName= []
        objects= pd.DataFrame()
        #dframeName= []
        for i in dframe.columns:
            #If i column is not  float
            if(not (dframe[i].dtype == np.float64)):
                #Get name
                objectsName.append(i)
                #Get no numeric column
                objects= pd.concat([objects, dframe[i]], axis=1)
                #Drop no numeric column
                dframe= dframe.drop(columns=objectsName[-1])
        #Get numeric columns
        dframeName= list(dframe.columns)

        return dframeName, dframe, objectsName, objects

    w, x, y, z= splitTypes(data)

    #Extra Tree Regressor
    impute_est= ETR(n_estimators=10, random_state=0)
    #Iterative imputer
    estimator= IterativeImputer(random_state=0, estimator=impute_est)
    #Fit transform data
    impdf= estimator.fit_transform(x)
    #Concat imputed data and class
    imp_data= pd.concat([pd.DataFrame(impdf), z], axis=1)
    #Rename columns
    imp_data.columns= w + y
    
    return imp_data

#Normalize dataframe
def normalizeDf(dframe):
    #Original dataframe values
    original_dframe_values= dframe.values
    #Original dataframe columns
    original_dframe_cols= dframe.columns

    #Imput data
    dframe= imputData(dframe)
    #Normalize values
    norm_values= preprocessing.normalize(dframe.to_numpy())
    #Normalized values to dataframe
    norm_df= pd.DataFrame(data=norm_values, columns=original_dframe_cols)

    return norm_df, original_dframe_values, original_dframe_cols

###############
##  Default  ##
###############
@app.errorhandler(Exception)
def all_exception_handler(error):
   return render_template('default_error.html'), 500

###############
##   Info    ##
###############
@app.route('/info')
def Info():
    return render_template('info.html')

###############
##Upload file##
###############
@app.route('/')
def Index():
    return render_template('files/index.html')

@app.route('/add_csv', methods=['POST'])
def ADD_CSV():
    if request.method == 'POST':
        #Load file
        fileLoaded= request.files['file']
        
        try:
            data= getFile(fileLoaded)
        except:
            return render_template('form_error.html')
        
        #Convert to dict
        dict_obj = data.to_dict('list')
        session['Original_data']= dict_obj
        #session.modified = True
        
        #Treeshold if NaN
        df= list(data.loc[:, data.isnull().mean() < .9].columns)
        print('Data')
        print(data)
        session['emptyCols']= ''
        #If many NaNs, get the columns name
        if(list(data.columns) != df):
            #Get the columns
            cols= list(set(data) - set(df))
            session['emptyCols']= ', '.join(cols)
            print(session['emptyCols'])

        return redirect(url_for('Imput'))
        

###############
##Apply imput##
###############
@app.route('/imputation')
def Imput():
    return render_template('files/imputation.html', emptyCols=session['emptyCols'])

@app.route('/add_imput', methods=['POST'])
def ADD_Imput():
    if request.method == 'POST':
        #Copy dataset to use locally
        dict_obj= session['Original_data'] if 'Original_data' in session else ""
        data = pd.DataFrame(dict_obj)

        #Keep/remove empty columns if exist    
        if session['emptyCols']:
            try:
                keep= request.form['empty']
            except:
                return render_template('form_error.html')

            #If not keep empty columns, change data
            if keep == '0':
                data= data.loc[:, data.isnull().mean() < .9]    
        
        try:
            #Impute or remove null values
            imputation= request.form['imputation']
        except:
            return render_template('form_error.html')

        if imputation == '1':
            #Imputate data
            imp_data= imputData(data)
        else:
            #Remove rows with null values
            imp_data= data.dropna().reset_index().drop('index', axis=1)
        
        #Normalize
        #norm_data, dframe_vals, dframe_cols= normalizeDf(imp_data)
        #imp_data= norm_data

        if(len(imp_data) < 50):
            return render_template('samples_error.html', samp_count=len(imp_data), typ='imp')

        session['Imputation']= imputation

        #Convert to dict
        dict_obj = imp_data.to_dict('list')

        session['Imputed_data']= dict_obj
        session['Imputed_data_cols']= list(imp_data.columns)

        return redirect(url_for('Learning'))

################
##Choose learn##
################
@app.route('/learning')
def Learning():
    return render_template('files/learning.html')

@app.route('/choose_learn', methods=['POST'])
def Choose_Learning():
    if request.method == 'POST':
        try:
            learning= request.form['learning']
        except:
            return render_template('form_error.html')

        #Return templates and column names
        if learning=='sup':
            return redirect(url_for('SupFeats'))
        else:
            return redirect(url_for('UnsupFeats'))

######SUP#######
##Select varia##
################
@app.route('/supfeatures')
def SupFeats():
    return render_template('classification/supfeatures.html', variables=session['Imputed_data_cols'])

@app.route('/add_supfeats', methods=['POST'])
def ADD_SupFeats():
    if request.method == 'POST':
        #Copy dataset to use locally
        data = pd.DataFrame(session['Imputed_data'])
        try:
            #Get class selected
            cl= str(request.form['class'])
        except:
            return render_template('form_error.html')

        clas= data[cl[1:]]
        
        #Get features selected
        fe= request.form.getlist('features')
        fe= [int(i) for i in fe]
        if not fe:
            return render_template('form_error.html')

        #Concat features columns
        sel_data= pd.DataFrame()
        for i in fe:
            sel_data= pd.concat([sel_data, data[data.columns[i]]], axis=1)
        #Concat imputed data and class
        sel_data= pd.concat([sel_data, clas], axis=1)

        print(sel_data)
        #Count samples per class and convert to list with indexes
        samples_count= clas.value_counts().reset_index().values.tolist()

        #Sort samples
        samples_count.sort(key=lambda x: -x[1])
        session['Samples_count']= samples_count
        
        #Convert to dict
        dict_obj = sel_data.to_dict('list')
        session['Selected_data']= dict_obj
        session['aux']= session['Selected_data']
        
        return redirect(url_for('Balance'))

######SUP#######
##Balance data##
################
@app.route('/balance')
def Balance():
    return render_template('classification/balance.html', samples=zip(session['Samples_count'], range(len(session['Samples_count']))))

@app.route('/add_balance', methods=['POST'])
def ADD_Balance():
    if request.method == 'POST':
        #Copy dataset to use locally
        data = pd.DataFrame(session['aux'])

        #Get classes selected
        cl= request.form.getlist('classes')
        cl= [int(i) for i in cl]
        if not cl:
            return render_template('form_error.html')

        try:
            #Split dataframe by class
            splits= list(data.groupby(data.columns[-1]))
        except:
            return render_template('form_error.html')

        #Sort according to Samples_Count
        sorted_classes= [i[0] for i in session['Samples_count']]
        splits.sort(key=lambda x: sorted_classes.index(x[0]))
        #Get selected classes
        selected_splits= [splits[i] for i in cl]
        
        session['Selected_classes']= ', '.join(str(classes[0]) for classes in selected_splits)

        try:
            #Get if apply upsampling
            upsamp= request.form['balance']
        except:
            return render_template('form_error.html')
        
        #Upsampling
        final_data= pd.DataFrame()
        if upsamp=='1':
            #Upsample
            for split in selected_splits:
                #Upsample split with the max size of selected_splits
                upsamp= resample(split[1], replace=True, n_samples=len(selected_splits[0][1]))
                #Join upsampled data to final dataframe
                final_data= pd.concat([final_data, upsamp])
            data= final_data
        else:
            #Just join the selected classes
            for split in selected_splits:
                final_data= pd.concat([final_data, split[1]])
            data= final_data

        if(len(data) < 50):
            return render_template('samples_error.html', typ='train')
        
        #Convert to dict
        dict_obj = data.to_dict('list')
        session['Selected_data']= dict_obj#.copy()

        return redirect(url_for('Split'))

######SUP#######
## Split data ##
################
@app.route('/split')
def Split():
    return render_template('classification/split.html')

@app.route('/add_split', methods=['POST'])
def ADD_Split():
    if request.method == 'POST':
        #Copy dataset to use locally
        data = pd.DataFrame(session['Selected_data'])

        #Get percentage of training set
        train= request.form['train']
        train= int(train)/100
        
        #Get class column
        y= data[data.columns[-1]]
        #Drop class column from the dataset
        x= data.drop(data.columns[-1], axis=1)
        
        #Split to train and test sets
        x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, train_size=train)
        #Normalize features
        x_tr = preprocessing.normalize(x_tr)
        x_tst = preprocessing.normalize(x_tst)
        
        #Class to classify
        class Classifications:
            #Function to return 2 dictionaries of accuracies and the other metrics
            #Parameters= training and testing sets
            def results(self, x_tr, y_tr, x_tst, y_tst):
                #List of the models of the classifiers and its name
                kn_name= 'K-Nearest Neighbors'
                lr_name= 'Logistic Regression'
                svm_name= 'Support Vector Machine'
                mlp_name= 'Multi-Layer Perceptron'
                dt_name= 'Decision Trees'
                #rf_name= 'Random Forest'
                knn= [KNeighborsClassifier(), kn_name]
                lr= [LogisticRegression(), lr_name]
                svm= [SVC(probability=True), svm_name]
                mlp= [MLPClassifier(), mlp_name]
                dt= [DecisionTreeClassifier(), dt_name]
                #rf= [RandomForestClassifier(), rf_name]

                #Dictionaries of parameters for GridSearchCV
                kn_params= {
                    'n_neighbors' : [5, 9],
                    'weights': ['uniform', 'distance']
                }
                lr_params = {
                    'solver': ['newton-cg', 'liblinear', 'sag', 'saga'] #'lbfgs'
                }
                svm_params = {
                    'kernel': ['linear', 'rbf', 'sigmoid'], #'poly'
                    'gamma': ['scale', 'auto']
                }
                mlp_params = {
                    'hidden_layer_sizes': [(100,), (60,60,60), (50,100,50)],
                    'activation': ['logistic', 'tanh', 'relu'], 
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, .001, .01],#, 0.05],
                    'learning_rate': ['constant', 'adaptive']
                }
                dt_params = {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [10,15,20]
                }
                #rf_params = {
                #    'criterion': ['gini', 'entropy'],
                #    'max_features': ['sqrt', None]
                #}

                #Dictionaries
                #Metrics except dt and rf
                accs= {}
                reports= {}
                #Models and definitions
                models= {}
                defs= { kn_name:'It stores all available cases and classifies new cases by a majority vote of its k neighbors.',
                        lr_name:'It predicts the probability of occurrence of an event by fitting data to a logit function.',
                        svm_name:'It plots each data point in an n-dimensional space, splitted into groups by a hyperplane.',
                        mlp_name:'It is a feedforward backpropagation artificial neural network that generates a set of outputs from a set of inputs.',
                        dt_name:'It breaks down a dataset into smaller subsets based on most significant attributes that makes the sets distinct.'
                        #rf_name:'It builds multiple decision trees and merges them together to get a more stable prediction. It searches for the best feature among a random subset instead of the most important feature while splitting a node.',
                      }
                #Metrics dt and rf
                accs_trees= {}
                reports_trees= {}
                importances= {}

                #Function to create the GridSearch model
                #Parameters= List of [model, name] and dictionary of parameters
                def createGrid(model, params):
                    grid_model= GridSearchCV(estimator= model[0],
                                            param_grid= params,
                                            scoring= 'accuracy')
                    #Return GSCV model and name of the classifier
                    return grid_model, model[1]
                
                #Function to train the GSCV model
                #Parameter= GSCV model
                def trainModel(grid_model):
                    #Train
                    fitted_model= grid_model[0].fit(x_tr, y_tr)
                    #Add name of the classifier and its model
                    models[grid_model[1]]= fitted_model
                    #Predict
                    predict= fitted_model.predict(x_tst)
                    #Accuracy
                    acc= metrics.accuracy_score(y_tst, predict)
                    #Add name of the classifier and its accuracy
                    accs[grid_model[1]]= acc
                    #Classification report
                    clas_report= classification_report(y_tst, predict, output_dict=True)
                    #Delete accuracy key and value from clas_report
                    del clas_report['accuracy']
                    #Add name of the classifier and its metrics
                    reports[grid_model[1]]= clas_report
                    return fitted_model, grid_model[1]
                
                #Train classifiers
                fitted_knn= trainModel(createGrid(knn, kn_params))
                fitted_lr= trainModel(createGrid(lr, lr_params))
                fitted_svm= trainModel(createGrid(svm, svm_params))
                fitted_mlp= trainModel(createGrid(mlp, mlp_params))
                fitted_dt= trainModel(createGrid(dt, dt_params))
                #fitted_rf= trainModel(createGrid(rf, rf_params))
                
                #Sort accuracies
                sorted_accs= sorted(accs.items(), key=lambda x: -x[1])
                
                #Get best model name
                best_model_name= sorted_accs[0][0]
                #Get best model
                session['Best_model']= []
                session['Best_model'].append(models.get(best_model_name))

                #Feature importance
                #Importance of dataframe
                def impDf(column_names, importances):
                    df= pd.DataFrame({'feature': column_names,
                                      'feature_importance': importances}) \
                        .sort_values('feature_importance', ascending=False) \
                        .reset_index(drop=True)
                    return df
                
                #Calculate feature importance
                def calcFeatImp(grid_model):
                    perm= PermutationImportance(grid_model[0], \
                                                cv=None, \
                                                refit=False, \
                                                n_iter=50) \
                                                .fit(x_tr, y_tr)
                    perm_imp_eli5= impDf(x.columns, perm.feature_importances_)
                    #print("Feature importance DT ELI5:\n", perm_imp_eli5)
                    importances[grid_model[1]]= perm_imp_eli5

                #Remove dt metrics
                accs_trees[dt_name]= accs.get(dt_name)
                accs.pop(dt_name)
                reports_trees[dt_name]= reports.get(dt_name)
                reports.pop(dt_name)
                #Remove rf metrics
                #accs_trees[rf_name]= accs.get(rf_name)
                #accs.pop(rf_name)
                #reports_trees[rf_name]= reports.get(rf_name)
                #reports.pop(rf_name)
                #Calculate feature importances
                calcFeatImp(fitted_dt)
                #calcFeatImp(fitted_rf)
                
                session['Best_import']= []
                #If best model is a tree
                if (best_model_name == dt_name): #or (best_model_name == rf_name):
                    session['Best_acc']= accs_trees.get(best_model_name)
                    accs_trees.pop(best_model_name)

                    session['Best_metric']= reports_trees.get(best_model_name)
                    reports_trees.pop(best_model_name)

                    session['Best_import']= importances.get(best_model_name)
                    importances.pop(best_model_name)
                
                #If best model is not a tree
                else:
                    session['Best_acc']= accs.get(best_model_name)
                    accs.pop(best_model_name)

                    session['Best_metric']= reports.get(best_model_name)
                    reports.pop(best_model_name)

                session['Best_model_name']= best_model_name
                
                return accs, reports, accs_trees, reports_trees, importances, defs#, best_model_name
        
        #Create Classifications object
        class_model= Classifications()
        #Get the results of train and evaluate the classifiers
        models_metrics= class_model.results(x_tr, y_tr, x_tst, y_tst)
        
        #Get accuracies
        session['Accuracies']= models_metrics[0]
        #Get the metrics of the classifiers
        session['Metrics']= models_metrics[1]

        #Get accuracies trees
        session['Accuracies_trees']= models_metrics[2]
        #Get the metrics of the classifiers
        session['Metrics_trees']= models_metrics[3]
        #Get the importance of the features
        session['Importances']= models_metrics[4]
        #Get the definitions of the classifiers
        session['Definitions']= models_metrics[5]

        return redirect(url_for('Classification'))

######SUP#######
##Class report##
################
@app.route('/class_report')
def Classification():
    return render_template('classification/class_report.html', accs=session['Accuracies'], accsList= list(session['Accuracies'].keys()), mets=session['Metrics'], 
                                                accs_trees=session['Accuracies_trees'], accsTList= list(session['Accuracies_trees'].keys()), mets_trees=session['Metrics_trees'], 
                                                imps=session['Importances'], defs=session['Definitions'],
                                                best_acc=session['Best_acc'], best_met=session['Best_metric'], best_imp=session['Best_import'],
                                                best_name=session['Best_model_name'])

@app.route('/add_report', methods=['POST'])
def ADD_Report():
    if request.method == 'POST':
        #Selected features
        dict_obj= session['Selected_data']
        session['Selected_feats']= ', '.join(pd.DataFrame(dict_obj).columns[:-1])
        
        return redirect(url_for('Classify'))

######SUP#######
##  Classify  ##
################
@app.route('/classify')
def Classify():
    return render_template('classification/classify.html', feats=session['Selected_feats'], 
                            classes=session['Selected_classes'], model=session['Best_model_name'])

@app.route('/add_classify', methods=['POST'])
def ADD_Classify():
    if request.method == 'POST':
        #Load file
        fileLoaded= request.files['file']
        
        #Get file name
        session['Filename']= fileLoaded.filename
        session['Filename']= session['Filename'].replace('.csv', '')

        try:
        #File to csv
            data= getFile(fileLoaded)
        except:
            return render_template('form_error.html')

        #Order columns
        dict_obj= session['Selected_data']
        sel_data= pd.DataFrame(dict_obj)
        try:
            data= data[sel_data.columns[:-1]]
        except:
            return render_template('samples_error.html', typ='clas', feats=session['Selected_feats'])

        #Imputation
        if session['Imputation'] == '1':
            #Imputate data
            data= imputData(data)
        else:
            #Remove rows with null values
            data= data.dropna().reset_index().drop('index', axis=1)
        #Normalize
        #norm_data= data.copy()
        norm_data, dframe_vals, dframe_cols= normalizeDf(data)

        #Add new column with the predictions
        data[sel_data.columns[-1]]= session['Best_model'][0].predict(norm_data)
        #Probabilities of classification
        probabilities= session['Best_model'][0].predict_proba(norm_data[norm_data.columns])
        
        #Add new column with the probabilities of classifications
        data['proba']= np.array([max(x) for x in probabilities])
        #Index start with 1
        data.index+= 1 
        #Round 2 decimals
        data= data.round(decimals=2)

        #Convert to dict
        dict_obj = data.to_dict('list')
        session['New_data']= dict_obj
        return redirect(url_for('Results'))

######SUP#######
##  Results   ##
################
@app.route('/results')
def Results():
    n_data= pd.DataFrame(session['New_data'])
    return render_template('classification/results.html', table=[n_data.replace(np.nan, '', regex=True).to_html(classes='other', header="true")], 
                                                          model=session['Best_model_name'])

@app.route('/add_results', methods=['GET'])
def ADD_Results():
    #Create StringIO
    execel_file= StringIO()
    #Name of the file
    filename= "%s.csv" % (session['Filename'] + ' - report')
    
    #Dataframe to csv
    n_data= pd.DataFrame(session['New_data'])
    n_data.to_csv(execel_file, index=False, encoding='utf-8')
    #Get dataframe data
    csv_output= execel_file.getvalue()
    #Close
    execel_file.close()

    resp= make_response(csv_output)
    #Filename
    resp.headers["Content-Disposition"]= ("attachment; filename=%s" % filename)
    #Csv
    resp.headers["Content-Type"]= "text/csv"

    return resp

#####UNSUP######
##Select varia##
################
@app.route('/unsupfeatures')
def UnsupFeats():
    imp= pd.DataFrame(session['Imputed_data'])
    return render_template('clustering/unsupfeatures.html', features=list(imp.columns))

Variance= []
Final_Variance= ()
PCA_Transpose= []
@app.route('/add_unsup', methods=['POST'])
def ADD_UnsupFeats():
    if request.method == 'POST':
        #Copy dataset to use locally
        data= pd.DataFrame(session['Imputed_data'])

        #Get features selected
        fe= request.form.getlist('features')
        
        fe= [int(i) for i in fe] 
        if not fe:
            return render_template('form_error.html')
        
        #Concat features columns
        sel_data= pd.DataFrame()
        for i in fe:
            sel_data= pd.concat([sel_data, data[data.columns[i]]], axis=1)

        #Scale
        scale_df= scale(sel_data)
        #Covariance matrix
        covar_matrix= np.cov(scale_df, rowvar=False)

        #Eingenvalues and eigenvectors
        eigen_values, eigen_vectors= np.linalg.eig(covar_matrix)
        #Eigenvalues greather than one
        #greater_one= sum(1 for i in eigen_values if i > 1)

        #SUMMARY TABLE
        #Index of eigen values max to min
        ix_ei_vals= np.argsort(eigen_values)[::-1]
        #Sorted variances
        sort_ei= eigen_values[ix_ei_vals]
        #Percentage of variance
        variance_explained= []
        for i in sort_ei:
            variance_explained.append((i/sum(sort_ei))*100)
        #Cumulative percentage
        cumulative_variance_explained= np.cumsum(variance_explained)

        #Create summary dataframe
        summary= pd.DataFrame(list(zip(sort_ei, variance_explained, cumulative_variance_explained)),
                            columns=['SS Loadings', 'Percentage of variance', 'Cumulative percentage']).round(decimals=2)                            
    
        #Start index from 1
        summary.index+= 1
        #Convert to dict
        dict_obj = summary.to_dict('list')
        session['Summary']= dict_obj
        session['ComponentsNo']= len(summary)

        #COMPONENTS TABLE
        #Columns
        cols= sel_data.columns
        #PCA
        pca_spon= PCA(n_components=session['ComponentsNo'])
        #New variables
        prin_spon= pca_spon.fit_transform(scale_df)
        #Component loadings
        comp_loadings= pca_spon.components_.T*np.sqrt(pca_spon.explained_variance_)
        
        #Create dataframe with loadings
        components= pd.DataFrame(data=comp_loadings, index=cols).round(decimals=2)
        #Rename columns
        components= components.rename(columns=lambda x: x+1)

        #Components
        dict_obj = components.to_dict('list')
        session['Components']= dict_obj
        session['Comp_dataset']= prin_spon

        dict_obj = sel_data.to_dict('list')
        session['Selected_data']= dict_obj
        #Array transpose
        session['PCA_transpose']= np.transpose(pca_spon.components_)
        #Variance explained
        session['Variance']= variance_explained

        return redirect(url_for('PCAnalysis'))

def create_bar_plot():
    fig, ax= plt.subplots(1, 1)
    #Bar plot 
    ax.bar(range(1, len(session['Variance'])+1), session['Variance']) 
    #Remove axes splines 
    for s in ['top', 'bottom', 'left', 'right']: 
        ax.spines[s].set_visible(False) 
    #Remove x, y ticks 
    #ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    #Add padding between axes and labels 
    ax.xaxis.set_tick_params(pad = 5) 
    ax.yaxis.set_tick_params(pad = 10) 
    #Add annotation to bars
    rects= ax.patches
    for i, v in zip(rects, session['Variance']):
        height= i.get_height()
        ax.text(i.get_x()+i.get_width()/2, height+2, (str(round(v,2))+'%'),
                ha='center', va='bottom', fontsize=7, rotation=90)
    #Add Plot Title 
    ax.set_title('Components variances\n') 
    ax.set_xlabel('Components') 
    ax.set_ylabel('Data percentage')

    return fig

@app.route('/bar.png')
def bar_png():
    fig= create_bar_plot()
    output= io.BytesIO()
    FigureCanvas(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')

#####UNSUP######
##    PCA     ##
################
@app.route('/pca')
def PCAnalysis():
    summ= pd.DataFrame(session['Summary'])
    comp= pd.DataFrame(session['Components'])
    return render_template('clustering/pca.html', summary=[summ.to_html(classes='other', header="true")], #cum_var= Final_Variance,
                                                  components=[comp.to_html(classes='other', header="true")],
                                                  no_components=session['ComponentsNo'])

@app.route('/add_pca', methods=['POST'])
def ADD_PCA():
    if request.method == 'POST':
        try:
            #Get number of components
            components= int(request.form['pcnumber'])
        except:
            return render_template('form_error.html')
        
        session['Sel_components']= components

        ks= range(1, 10)
        inertias= []
        comp_data= session['Comp_dataset']

        for k in ks:
            #Create a KMeans instance with k clusters: model
            model= KMeans(n_clusters=k)
            #Fit model to samples
            model.fit(comp_data[:,0:components])
            #Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        kl= KneeLocator(ks, inertias, curve="convex", direction="decreasing")
        kmeans= KMeans(n_clusters=kl.elbow).fit(comp_data)
        labels= kmeans.predict(comp_data)

        imp_data= pd.DataFrame(session['Imputed_data'])
        sel_data= pd.DataFrame(session['Selected_data'])

        data_labeled= imp_data[sel_data.columns]
        print(data_labeled)
        data_labeled['cluster']= [x+1 for x in labels]
        
        dict_obj = data_labeled.to_dict('list')
        session['PCA_labeled']= dict_obj
        session['No_clusters']= kl.elbow

        data_labeled= data_labeled.sort_values('cluster')
        session['New_data'] = data_labeled.to_dict('list')

        return redirect(url_for('ClusterReport'))


def create_pc_plot():
    comp_data= pd.DataFrame(session['Comp_dataset'])
    trans= session['PCA_transpose'][:,0:session['Sel_components']]
    data_labeled= pd.DataFrame(session['PCA_labeled'])
    n= trans.shape[0]
    df= pd.DataFrame(data=comp_data)
    
    fig, ax = plt.subplots(nrows=session['Sel_components'], ncols=session['Sel_components'], figsize=(session['Sel_components']*5, session['Sel_components']*5))#, constrained_layout=True)
    plt.subplots_adjust(hspace=session['Sel_components']*0.075, wspace=session['Sel_components']*0.075, left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    cmap= plt.cm.get_cmap('viridis')
    my_legend= []
    num_clusters= session['No_clusters']
    for i in range(num_clusters):
        my_legend.append(Line2D([0], [0], lw=0, marker="o", color=cmap(1/(num_clusters-1)*i), label="Cluster {0}".format(i+1)))

    sel_data= pd.DataFrame(session['Selected_data'])

    for i in range(session['Sel_components']):
        for j in range(session['Sel_components']):
            #if(i != j):
            #ax.subplot(Sel_Components, Sel_Components, i*Sel_Components+j+1)
            #df= pd.DataFrame(data=Comp_Dataset, columns=[i, j])
            xs= df[i]
            ys= df[j]
            
            scalex= 1.0/(xs.max()-xs.min())
            scaley= 1.0/(ys.max()-ys.min())

            #fig, ax= plt.subplots(1, 1)
            ax[i,j].scatter(xs*scalex, ys*scaley, c=data_labeled['cluster'], label=data_labeled['cluster'], alpha=0.5)
            for k in range(n):
                    ax[i,j].arrow(0, 0, trans[k,i], trans[k,j], head_width=0.02, head_length=0.04, color='r', alpha= 0.5)
                    ax[i,j].text(trans[k,i]*1.15, trans[k,j]*1.15, sel_data.columns[k], color='r', ha='center', va='center')
                    #Remove x, y ticks 
                    ax[i,j].xaxis.set_ticks_position('none') 
                    ax[i,j].yaxis.set_ticks_position('none')
                    #ax.set_xlim(-1,1)
                    #ax.set_ylim(-1, 1)
            fig.suptitle('Principal Components and Clusters\n', fontsize=session['Sel_components']*7)
            ax[i,j].set_xlabel("PC "+str(i+1))
            ax[i,j].set_ylabel("PC "+str(j+1))
            ax[i,j].legend(handles=my_legend)
            ax[i,j].grid()

    return fig

@app.route('/plot.png')
def plot_png():
    fig= create_pc_plot()
    output= io.BytesIO()
    FigureCanvas(fig).print_png(output)
        
    return Response(output.getvalue(), mimetype='image/png')

#####UNSUP######
##Clust report##
################
@app.route('/cluster_report')
def ClusterReport():
    n_data= pd.DataFrame(session['New_data'])
    return render_template('clustering/cluster_report.html', data=[n_data.replace(np.nan, '', regex=True).to_html(classes='other', header="true")],
                                                             no_components=session['Sel_components'], no_clusters=session['No_clusters'])

################
##  Something ##
################
if __name__ == '__main__':
    app.run(port = 3000, debug = False, threaded=True)
    #app.run(port = 3000, debug = True, use_reloader = True, threaded=True)

"""
@app.route('/route')
def Name():
    return render_template('template.html')

@app.route('/work_route', methods=['POST'])
def Work_Function():
    if request.method == 'POST':
        return redirect(url_for('Next_Function_Name'))
"""