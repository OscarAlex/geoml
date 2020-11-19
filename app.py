#Flask
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, make_response
from werkzeug.utils import secure_filename
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
from sklearn.ensemble import RandomForestClassifier
#Metrics
from sklearn import metrics
from sklearn.metrics import classification_report
import eli5
from eli5.sklearn import PermutationImportance
#Download CSV
from io import StringIO
#PCA and clusters
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.cluster import KMeans

app= Flask(__name__)
app.secret_key= 'secret'

###############
## Variables ##
###############


###############
## Functions ##
###############


###############
##Upload file##
###############
#List with columns almost empty
emptyCols= ''
@app.route('/')
def Index():
    global emptyCols
    #Reset the variable in case of return
    emptyCols= ''
    return render_template('files/index.html')

Filename= ''
#Function to read the entered csv
def getFile(fileLoaded):
    #fileLoaded.save(secure_filename(fileLoaded.filename))
    #Get file name
    global Filename
    Filename= fileLoaded.filename
    Filename= Filename.replace('.csv', '')
    #Get file
    #fileSaved= fileLoaded.filename
    df= pd.read_csv(fileLoaded)
    #Drop completely empty rows, reset indexes and drop index column
    df= df.dropna(how='all').reset_index().drop('index', axis=1)
    #Drop completely empty columns
    df= df.dropna(axis='columns', how='all')
    return df

#Dataset file
Data= pd.DataFrame()
@app.route('/add_csv', methods=['POST'])
def Add_CSV():
    if request.method == 'POST':
        try:
            #Load file
            fileLoaded= request.files['file']
            
            global Data
            Data= getFile(fileLoaded)

            #Treeshold if NaN
            df= list(Data.loc[:, Data.isnull().mean() < .9].columns)
            print(Data)
            print(list(Data.columns))
            print(df)
            #If many NaNs, get the columns name
            if(list(Data.columns) != df):
                print("Hay columnas vacías")
                global emptyCols
                #Get the columns
                cols= list(set(Data) - set(df))
                emptyCols= ', '.join(cols)
                print(emptyCols)
            return redirect(url_for('Imput'))
        except:
            flash('This message will not show')
            return redirect(url_for('Index'))

###############
##Apply imput##
###############
@app.route('/imputation')
def Imput():
    #Copy dataset when refresh page in case of return
    Imput.data= Data.copy()
    return render_template('files/imputation.html', emptyCols=emptyCols)

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
            print(w)
            print(x)
            print(y)
            print(z)
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
            #Return imputed data
            return imp_data

#Imputed dataset
Imputed_Data= pd.DataFrame()
#Dataset columns
Imputed_Data_Cols= []
@app.route('/add_imput', methods=['POST'])
def Add_Imput():
    if request.method == 'POST':
        #Copy dataset to use locally
        Imput()
        data= Imput.data.copy()

        #Keep/remove empty columns if exist    
        if emptyCols:
            keep= request.form['empty']
            print(keep)
            #If not keep empty columns, change data
            if keep=='0':
                data= data.loc[:, data.isnull().mean() < .9]    
        
        #Imputate data
        imp_data= imputData(data)
        
        global Imputed_Data
        global Imputed_Data_Cols
        Imputed_Data= imp_data.copy()
        Imputed_Data_Cols= list(Imputed_Data.columns)
        print(Imputed_Data)
        return redirect(url_for('Learning'))

################
##Choose learn##
################
@app.route('/learning')
def Learning():
    #Copy dataset when refresh page in case of return
    Learning.data= Imputed_Data.copy()
    return render_template('files/learning.html')

@app.route('/choose_learn', methods=['POST'])
def Choose_Learning():
    if request.method == 'POST':
        learning= request.form['learning']
        #Return templates and column names
        if learning=='sup':
            #return render_template('supfeatures.html', variables=imputed_data_cols)
            return redirect(url_for('SupFeats'))
        else:
            #return render_template('unsupfeatures.html', features=Imputed_Data_Cols)
            return redirect(url_for('UnsupFeats'))

######SUP#######
##Select varia##
################
@app.route('/supfeatures')
def SupFeats():
    #Copy dataset when refresh page in case of return
    SupFeats.data= Imputed_Data.copy()
    return render_template('classification/supfeatures.html', variables=Imputed_Data_Cols)

#Selected features
Selected_Data= pd.DataFrame()
#Samples per class
Samples_Count= []
@app.route('/add_supfeats', methods=['POST'])
def Add_SupFeats():
    if request.method == 'POST':
        #Copy dataset to use locally
        SupFeats()
        data= SupFeats.data.copy()
        
        #Get class selected
        cl= int(request.form['class'])
        clas= data[data.columns[cl]]
        #Get features selected
        fe= request.form.getlist('features')
        fe= [int(i) for i in fe] 
        print(fe)
        #Concat features columns
        sel_data= pd.DataFrame()
        for i in fe:
            sel_data= pd.concat([sel_data, data[data.columns[i]]], axis=1)
        #Concat imputed data and class
        sel_data= pd.concat([sel_data, clas], axis=1)

        print(sel_data)
        global Samples_Count
        #Count samples per class and convert to list with indexes
        Samples_Count= clas.value_counts().reset_index().values.tolist()
        
        #Sort samples
        Samples_Count.sort(key=lambda x: -x[1])
        print(Samples_Count)
        """
        samples= [sample[1] for sample in Samples_Count]
        Q1 = np.percentile(samples, 25, interpolation = 'midpoint')  
        Q2 = np.percentile(samples, 50, interpolation = 'midpoint')  
        Q3 = np.percentile(samples, 75, interpolation = 'midpoint')  
        IQR = Q3 - Q1  
        low_lim = Q1 - 1.5 * IQR 
        up_lim = Q3 + 1.15 * IQR 

        outlier =[] 
        for x in samples: 
            if ((x> up_lim) or (x<low_lim)): 
                outlier.append(x) 
        print('Outlier in the dataset is', outlier)
        """
        global Selected_Data
        Selected_Data= sel_data.copy()
        print(Selected_Data)
        #print(Samples_Count)
        return redirect(url_for('Balance'))

######SUP#######
##Balance data##
################
@app.route('/balance')
def Balance():
    #Copy dataset when refresh page in case of return
    Balance.data= Selected_Data.copy()
    return render_template('classification/balance.html', samples=zip(Samples_Count, range(len(Samples_Count))))

Selected_Classes= ''
@app.route('/add_balance', methods=['POST'])
def Add_Balance():
    if request.method == 'POST':
        #Copy dataset to use locally
        Balance()
        data= Balance.data.copy()

        #Get classes selected
        cl= request.form.getlist('classes')
        cl= [int(i) for i in cl] 
        #Split dataframe by class
        splits= list(data.groupby(data.columns[-1]))
        #Sort according to Samples_Count
        sorted_classes= [i[0] for i in Samples_Count]
        splits.sort(key=lambda x: sorted_classes.index(x[0]))
        #Get selected classes
        selected_splits= [splits[i] for i in cl]
        
        global Selected_Classes
        Selected_Classes= ', '.join(classes[0] for classes in selected_splits)

        #Get if apply upsampling
        upsamp= request.form['balance']
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

        global Selected_Data
        Selected_Data= data.copy()

        print(data[data.columns[-1]].value_counts())
        print(data)
        return redirect(url_for('Split'))

######SUP#######
## Split data ##
################
@app.route('/split')
def Split():
    #Copy dataset when refresh page in case of return
    Split.data= Selected_Data.copy()
    return render_template('classification/split.html')

#Dictionaries
Accuracies= {}
Metrics= {}
Accuracies_Trees= {}
Metrics_Trees= {}
Importances= {}
Definitions= {}
Best_Model= []
Best_Model_Name= ''

Best_Acc= []
Best_Metric= []
Best_Import= []
@app.route('/add_split', methods=['POST'])
def ADD_Split():
    if request.method == 'POST':
        #Copy dataset to use locally
        Split()
        data= Split.data.copy()

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
                rf_name= 'Random Forest'
                knn= [KNeighborsClassifier(), kn_name]
                lr= [LogisticRegression(), lr_name]
                svm= [SVC(probability=True), svm_name]
                mlp= [MLPClassifier(), mlp_name]
                dt= [DecisionTreeClassifier(), dt_name]
                rf= [RandomForestClassifier(), rf_name]

                #Dictionaries of parameters for GridSearchCV
                kn_params= {
                    'n_neighbors' : [5, 20],
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
                    'hidden_layer_sizes': [(100,), (60,60,60)],
                    'activation': ['logistic', 'tanh', 'relu'], 
                    'solver': ['sgd', 'adam'], #'lbfgs'
                    'alpha': [.001, .01],
                    'learning_rate': ['constant', 'adaptive']
                }
                dt_params = {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random']
                }
                rf_params = {
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', None]
                }

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
                        dt_name:'It breaks down a dataset into smaller subsets based on most significant attributes that makes the sets distinct.',
                        rf_name:'It builds multiple decision trees and merges them together to get a more stable prediction. It searches for the best feature among a random subset instead of the most important feature while splitting a node.',
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
                fitted_rf= trainModel(createGrid(rf, rf_params))
                
                #Sort accuracies
                sorted_accs= sorted(accs.items(), key=lambda x: -x[1])
                
                global Best_Model_Name
                global Best_Model
                #Get best model name
                best_model_name= sorted_accs[0][0]
                Best_Model_Name= best_model_name
                #Get best model                
                Best_Model.append(models.get(best_model_name))

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
                accs_trees[rf_name]= accs.get(rf_name)
                accs.pop(rf_name)
                reports_trees[rf_name]= reports.get(rf_name)
                reports.pop(rf_name)
                #Calculate feature importances
                calcFeatImp(fitted_dt)
                calcFeatImp(fitted_rf)
                
                global Best_Acc
                global Best_Metric
                global Best_Import
                #If best model is a tree
                if (best_model_name == dt_name) or (best_model_name == rf_name):
                    Best_Acc= accs_trees.get(best_model_name)
                    accs_trees.pop(best_model_name)

                    Best_Metric= reports_trees.get(best_model_name)
                    reports_trees.pop(best_model_name)

                    Best_Import= importances.get(best_model_name)
                    importances.pop(best_model_name)
                #If best model is not a tree
                else:
                    Best_Acc= accs.get(best_model_name)
                    accs.pop(best_model_name)

                    Best_Metric= reports.get(best_model_name)
                    reports.pop(best_model_name)

                #Return dictionaries
                return accs, reports, accs_trees, reports_trees, importances, defs#, best_model_name
        
        #Create Classifications object
        class_model= Classifications()
        #Get the results of train and evaluate the classifiers
        models_metrics= class_model.results(x_tr, y_tr, x_tst, y_tst)
        
        global Accuracies
        global Metrics
        global Accuracies_Trees
        global Metrics_Trees
        global Importances
        global Definitions

        #Get accuracies
        Accuracies= models_metrics[0]
        #Get the metrics of the classifiers
        Metrics= models_metrics[1]
        print(Metrics)

        #Get accuracies trees
        Accuracies_Trees= models_metrics[2]
        #Get the metrics of the classifiers
        Metrics_Trees= models_metrics[3]
        print(Metrics)
        #Get the importance of the features
        Importances= models_metrics[4]

        #Get the definitions of the classifiers
        Definitions= models_metrics[5]

        return redirect(url_for('Classification'))

######SUP#######
##Class report##
################
@app.route('/class_report')
def Classification():
    return render_template('classification/class_report.html', accs=Accuracies, accsList= list(Accuracies.keys()), mets=Metrics, 
                                                accs_trees=Accuracies_Trees, accsTList= list(Accuracies_Trees.keys()), mets_trees=Metrics_Trees, 
                                                imps=Importances, defs=Definitions,
                                                best_acc=Best_Acc, best_met=Best_Metric, best_imp=Best_Import,
                                                best_name=Best_Model_Name)

#Selected features
Selected_Feats= ''
@app.route('/add_report', methods=['POST'])
def ADD_Report():
    if request.method == 'POST':
        #Selected features
        global Selected_Data
        global Selected_Feats
        Selected_Feats= ', '.join(Selected_Data.columns[:-1])
        return redirect(url_for('Classify'))

######SUP#######
##  Classify  ##
################
@app.route('/classify')
def Classify():
    #Copy dataset when refresh page in case of return
    #Balance.data= Imputed_Data.copy()
    return render_template('classification/classify.html', feats=Selected_Feats, classes=Selected_Classes, model=Best_Model_Name)

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

New_Data= pd.DataFrame()
@app.route('/add_classify', methods=['POST'])
def ADD_Classify():
    if request.method == 'POST':
        #Load file
        fileLoaded= request.files['file']
        #Get file name
        global Filename
        Filename= fileLoaded.filename
        Filename= Filename.replace('.csv', '')

        #File to csv
        data= getFile(fileLoaded)
        #Normalize
        norm_data, dframe_vals, dframe_cols= normalizeDf(data)

        global Selected_Data
        global Best_Model
        #Add new column with the predictions
        data[Selected_Data.columns[-1]]= Best_Model[0].predict(norm_data)
        #Probabilities of classification
        probabilities= Best_Model[0].predict_proba(norm_data[norm_data.columns])
        #print(probabilities)
        #Add new column with the probabilities of classifications
        data['proba']= np.array([max(x) for x in probabilities])
        #Index start with 1
        data.index+= 1 
        #Round 2 decimals
        data= data.round(decimals=2)

        global New_Data
        New_Data= data.copy()
        #print(New_Data)
        return redirect(url_for('Results'))

######SUP#######
##  Results   ##
################
@app.route('/results')
def Results():
    #Copy dataset when refresh page in case of return
    Balance.data= Imputed_Data.copy()
    return render_template('classification/results.html', table=[New_Data.replace(np.nan, '', regex=True).to_html(classes='data', header="true")], model=Best_Model_Name)

@app.route('/add_results', methods=['GET'])
def ADD_Results():
    #Create StringIO
    execel_file= StringIO()
    global Filename
    #Name of the file
    filename= "%s.csv" % (Filename + ' - report')
    
    global New_Data
    #Dataframe to csv
    New_Data.to_csv(execel_file, index=False, encoding='utf-8')
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
    #Copy dataset when refresh page in case of return
    UnsupFeats.data= Imputed_Data.copy()
    return render_template('clustering/unsupfeatures.html', features=Imputed_Data_Cols)

#Summary dataframe
Summary= pd.DataFrame()
#Components dataframe
Components= pd.DataFrame()
#Number of components
ComponentsNo= ()
#Components dataset
Comp_Dataset= []
@app.route('/add_unsup', methods=['POST'])
def ADD_SupFeats():
    if request.method == 'POST':
        #Copy dataset to use locally
        UnsupFeats()
        data= UnsupFeats.data.copy()

        #Get features selected
        fe= request.form.getlist('features')
        fe= [int(i) for i in fe] 
        #print(fe)
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
        greater_one= sum(1 for i in eigen_values if i > 1)
        
        global ComponentsNo
        ComponentsNo= greater_one

        ##############SUMMARY TABLE###############
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
        
        #Rename index
        #summary= summary.rename_axis(('Components'))
        summary.index+= 1
        #Drop rows
        drop_rows= len(summary)-greater_one
        summary= summary.drop(summary.tail(drop_rows).index)

        global Summary
        Summary= summary.copy()

        ############COMPONENTS TABLE############
        #Columns
        cols= sel_data.columns
        #PCA
        pca_spon= PCA(n_components=greater_one)
        #New variables
        prin_spon = pca_spon.fit_transform(scale_df)
        #Component loadings
        comp_loadings= pca_spon.components_.T * np.sqrt(pca_spon.explained_variance_)
        #pca_df = pd.DataFrame(data = prin_spon, columns = ['PC1', 'PC2','PC3','PC4','PC5'])
        
        #Create dataframe with loadings
        components= pd.DataFrame(data=comp_loadings, index=cols).round(decimals=2)
        #Rename columns
        components= components.rename(columns=lambda x: x+1)

        global Components
        Components= components.copy()
        global Comp_Dataset
        Comp_Dataset= prin_spon.copy()
        global Selected_Data
        Selected_Data= sel_data.copy()
        #global Selected_Data
        #Selected_Data= sel_data.copy()
        #print(Selected_Data)

        return redirect(url_for('PCAnalysis'))

#####UNSUP######
##    PCA     ##
################
@app.route('/pca')
def PCAnalysis():
    #Copy dataset when refresh page in case of return
    Balance.data= Imputed_Data.copy()
    return render_template('clustering/pca.html', summary=[Summary.to_html(header="true")],
                                       components=[Components.to_html(header="true")],
                                       no_components=ComponentsNo)

#Labeled data
Cluster_Data= pd.DataFrame()
@app.route('/add_pca', methods=['POST'])
def ADD_PCA():
    if request.method == 'POST':
        ks = range(1, 10)
        inertias = []
        global Comp_Dataset
        for k in ks:
            # Create a KMeans instance with k clusters: model
            model = KMeans(n_clusters=k)
            # Fit model to samples
            model.fit(Comp_Dataset)
            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
        kmeans = KMeans(n_clusters=kl.elbow).fit(Comp_Dataset)
        labels = kmeans.predict(Comp_Dataset)

        #data= pd.DataFrame()
        
        global Data
        global Selected_Data
        
        data_labeled= Data[Selected_Data.columns]
        data_labeled['clusters']= [x+1 for x in labels]
        
        data_labeled= data_labeled.sort_values('labels')
        #centroids = kmeans.cluster_centers_
        
        global New_Data
        New_Data= data_labeled.copy()

        return redirect(url_for('ClusterReport'))

#####UNSUP######
##Clust report##
################
@app.route('/cluster_report')
def ClusterReport():
    #Copy dataset when refresh page in case of return
    #Balance.data= Imputed_Data.copy()
    return render_template('clustering/cluster_report.html', data=[New_Data.replace(np.nan, '', regex=True).to_html(header="true")])

################
##  Something ##
################
if __name__ == '__main__':
    app.run(port = 3000, debug = True)

"""
@app.route('/balance')
def Balance():
    #Copy dataset when refresh page in case of return
    Balance.data= Imputed_Data.copy()
    return render_template('balance.html')

@app.route('/add_supfeats', methods=['POST'])
def ADD_SupFeats():
    if request.method == 'POST':
        return redirect(url_for('Balance'))
"""