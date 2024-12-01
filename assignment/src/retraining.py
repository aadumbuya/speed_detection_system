from preprocessing import * 

#retraining
def retrained(train,label,test,test_label,classes,typ_,model_name,analysis_type):
    model = retrain(typ_=typ_,model_name=model_name,train=train,label=label)
    preds = model.predict(test)
    #evaluation
    metrics = model_analysis(true_labels=test_label,predicted_labels=preds,classes=classes,typ_=analysis_type) 
    return metrics

def process(loader_type,train,label,train_column_limit,scaler_type,data_type,test_ratio=0.3):
    train = loader(loader_type,train,axis=0,drop_index=True)
    
    #convert objects to int
    train = covert_objects_to_int(train)
    
    #column and label selection
    train_,_ = select_train_columns(train,train_column_limit)
    label = select_label_column(train,label)
    
    #spilting
    train_x,test_x,label_x,test_y = data_spliter(train_,label,random_state=2,test_size=test_ratio)

    #saving training and testing data
    write_data(data_type,train_x,"..\\data\\train\\training_data")
    write_data(data_type,test_x,"..\\data\\test\\testing_data")

    #scaling
    scaled_train_x = scalers(scaler_type,train_x)
    scaled_test_x = scalers(scaler_type,test_x)
    return scaled_train_x,scaled_test_x,label_x,test_y







