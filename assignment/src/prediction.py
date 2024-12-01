from preprocessing import * 


#prediction
def prediction(data,model_name,scaler_type):
    data = pd.DataFrame(data)
    data = covert_objects_to_int(data)
    data = scalers(scaler_type,data)
    preds = predict(model_name,data)[0]
    return preds




