import numpy as np
import pandas as pd
import sys
import json
from tensorflow.keras.models import load_model

def load_collaborative_model():
    model = load_model('/model/collaborative_filtering.h5')
    return model

if __name__ == "__main__":
    
    #pinjam untuk testing:
    df_phone_dataset = pd.read_csv('phone_dataset.csv')
    #end
    
    input_data = sys.stdin.read()
    params = json.loads(input_data)
    PHONE_COUNT = params.get('phone_count', 96)
    
    
    model = load_collaborative_model()
    user_click = params.get('user_click', [0 for _ in range(PHONE_COUNT)])
    user_click_input = np.reshape(user_click, (1,PHONE_COUNT))

    user_rating = params.get('user_rating', [0 for _ in range(PHONE_COUNT)])
    user_rating_input = np.reshape(user_rating, (1,PHONE_COUNT))
    user_rating_input = user_rating_input / 5
    result = model.predict([user_rating_input,user_click_input])
    result = result[0]    
    
    #Post processing
    for i in range(PHONE_COUNT):
        if user_click[i] != 0 or user_rating[i] != 0:
            result[i] = 0
    result = result / max(result)
    probability = np.array((result))
    df_result = pd.DataFrame()
    df_result['probability'] = np.reshape(probability, 96).astype(float)
    df_result['name'] = df_phone_dataset['name']
    sorted_df_result = df_result.sort_values(by='probability', ascending=False)
    sorted_df_result['probability'] = sorted_df_result['probability'] / sorted_df_result['probability'].max()
    datas = sorted_df_result.to_numpy()
    for i in range(10):
        prob = datas[i][0]
        print(f'{i + 1}. Recommending clicking {prob:0.2f} for phone { (datas[i][1]) }')