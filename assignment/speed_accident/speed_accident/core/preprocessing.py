import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from joblib import dump,load
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report


def read_csv(name,sep=","):
    data = pd.read_csv(name,sep=sep)
    return data

def read_from_file(typ_,name,sep=None):
    if typ_ == "csv":
        data = read_csv(name,sep)
    else:
        data = None
    return data
    
def read_from_data(typ_,data):
    pass

def read_from_file_or_data(typ_,sect=None,name=None,data=None,sep=None):
    if sect == "file":
        data = read_from_file(typ_,name,sep)
    else:
        read_from_data(typ_,data)
    return data

def write_dataframe_csv(dataframe,name):
    dataframe.to_csv(name)
    
def write_data(typ_,data,name):
    if typ_ == "csv":
        write_dataframe_csv(data,name)
    else:
        pass

def load_one(data):
    return data

def load_two(data1,data2,axis=0,drop_index=True):
    combined = pd.concat([data1,data2],axis=axis)
    combined = combined.sample(frac=1,random_state=2).reset_index(drop=drop_index)
    return combined

def loader(typ_,data,axis=0,drop_index=True):
    if typ_ == "one":
        data = load_one(data)
    elif typ_ == "two":
        data = load_two(data[0],data[1],axis=0,drop_index=True)
    else:
        data = None
    return data

def data_spliter(train,label,random_state=2,test_size=0.3):
    train_x,test_x,train_y,test_y = train_test_split(train,label,random_state=random_state,test_size=test_size)
    return train_x,test_x,train_y,test_y

def set_ss(data):
    ss = StandardScaler()
    scaled = ss.fit_transform(data)
    return scaled

def scalers(typ_,data):
    scaled = None
    if typ_ == "ss":
        scaled = set_ss(data)
    else:
        pass
    return scaled


def select_train_columns(data,n):
    train = data.iloc[:,:n]
    return train,train.columns
    
def select_label_column(data,column_name):
    labels = np.array(data[column_name])
    return labels

def dcl_train(train,label,model_name):
    dcl = DecisionTreeClassifier()
    dcl.fit(train,label)
    dump(dcl,model_name) 
    return dcl

def train(typ_,train,label,model_name):
    if typ_ == "dcl":
        return dcl_train(train,label,model_name)
    else:
        return None


def predict(model_name,data):
    model = load(model_name) 
    label = model.predict(data)
    return label

def retrain_with_new_data(model_name,train,label):
    model = load(model_name)
    model.fit(train,label)
    dump(model,model_name) 
    return model
    
    
def retrain_with_new_and_old_data(model_name,train,label):
    c_train = np.vstack([train[0], train[1]])
    clabel = np.concatenate([label[0], label[1]])
    model = load(model_name)
    model.fit(c_train,clabel)
    dump(model,model_name)  
    return model
    
def retrain(typ_,model_name,train,label):
    if typ_ == "only_new":
        model = retrain_with_new_data(model_name,train,label)
    elif typ_ == "new_and_old":
        model = retrain_with_new_and_old_data(model_name,train,label)
    else:
        pass
    return model
    
def get_metrics(true_labels, predicted_labels):
    metrics_ = {}
    
    metrics_['Accuracy:'] = np.round(accuracy_score(true_labels,predicted_labels),4)
    metrics_['Precision:'] =  np.round(precision_score(true_labels,predicted_labels,average='weighted'),4)
    metrics_['Recall:'] = np.round(recall_score(true_labels,predicted_labels,average='weighted'),4)
    metrics_['F1 Score:'] = np.round(f1_score(true_labels,predicted_labels,average='weighted'),4)
    return metrics_                     
   
def display_classification_report(true_labels, predicted_labels, classes=[1,0]):

    report = classification_report(y_true=true_labels, 
                                           y_pred=predicted_labels, 
                                           labels=classes) 
    return report
    
    
    
def display_model_performance_metrics(true_labels, predicted_labels, classes=[1,0]):
    stats = {}
    metrics = get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    report = display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels,classes=classes)
    stats["metrics"] = metrics
    stats["report"] = report
    return stats

def model_analysis(true_labels,predicted_labels,classes,typ_="default"):
    if typ_ == "default":
        metrics = display_model_performance_metrics(true_labels, predicted_labels,classes)
    else:
        pass
    return metrics
    
    
def covert_objects_to_int(data):
    columns = data.columns
    for i in columns:
        if data[i].dtype == object:
            data[i] = label_encoder(data[i])
    return data

def label_encoder(data):
    l = LabelEncoder()
    data = l.fit_transform(data)
    return data

    

