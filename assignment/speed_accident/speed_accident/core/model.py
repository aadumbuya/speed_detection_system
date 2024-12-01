from .preprocessing import * 


#model
def model(loader_type,data,data_type,scaler_type,model_type,model_name,analysis_type,train_column_limit,classes,label,test_ratio=0.3):
    data = loader(loader_type,data,axis=0,drop_index=True)
    
    #convert objects to int
    rta_ = covert_objects_to_int(data)
    
    #column and label selection
    rta_t,columns = select_train_columns(rta_,train_column_limit)
    rta_label = select_label_column(rta_,label)

    #spilting
    rta_t_train_x1,rta_t_test_x1,rta_label_train_y1,rta_label_test_y1 = data_spliter(rta_t,rta_label,random_state=2,test_size=test_ratio)

    #saving training and testing data
    write_data(data_type,rta_t_train_x1,"training_data")
    write_data(data_type,rta_t_test_x1,"testing_data")

    #scaling
    ss1_fit = scalers(scaler_type,rta_t_train_x1)
    ss1_test = scalers(scaler_type,rta_t_test_x1)


    #training of the model
    model = train(model_type,rta_t_train_x1,rta_label_train_y1,model_name)

    #predicting
    preds = predict(model_name,ss1_test)

    #evaluation
    metrics = model_analysis(rta_label_test_y1,preds,classes,typ_=analysis_type)
    return metrics



