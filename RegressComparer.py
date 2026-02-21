import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class EvalPipe:
    def __init__(self, x, y, y_pred, first_model):
        self.x = x
        self.y = y
        self.y_mean = np.mean(y)
        if isinstance(y_pred, (list, np.ndarray)) and not isinstance(y_pred[0], (list, np.ndarray)):
            self.pred_list = [y_pred]
            self.pred_title = [first_model]
        else:
            self.pred_list = list(y_pred)
            self.pred_title = list(first_model)

    def MSE(self, p):
        return mean_squared_error(p, self.y)

    def MAE(self, p):
        return mean_absolute_error(p, self.y)

    def RMSE(self, p):
        return np.sqrt(mean_squared_error(p, self.y))
    
    def R_SQRT(self, p):
        y_true = np.array(self.y)
        y_pred = np.array(p)
        
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - self.y_mean)**2)
        
        return 1 - (ss_res / ss_tot)


    def add(self, model_name, new_preds):
        if not isinstance(model_name, str):
            raise ValueError(f"Model name must be a string. Received: {type(model_name)}")
        
        if isinstance(new_preds, str):
            raise TypeError("The prediction data cannot be a string! Please provide a list or array of numbers.")
        
        if len(new_preds) != len(self.y):
            raise ValueError(f"Input size mismatch! Expected {len(self.y)} values, but got {len(new_preds)}.")

        self.pred_title.append(model_name)
        self.pred_list.append(new_preds)
        return f"Model '{model_name}' added successfully."

    def vision(self):
        results = []
        
        for i, p in enumerate(self.pred_list):
            mse = self.MSE(p)
            mae = self.MAE(p)
            rmse = self.RMSE(p)
            r_sqrt = self.R_SQRT(p)
            
            results.append({
                'Model': self.pred_title[i],
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R_SQRT': r_sqrt,
            })
        
        df = pd.DataFrame(results).set_index('Model').T
        return df
    
    @classmethod
    def from_demo(cls, demo_obj):
        x, y, p_list, p_titles = demo_obj.fit()
        instance = cls(x, y, p_list, p_titles)
        return instance

class demo:
    def __init__(self):
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([1, 2, 3, 4, 5])
        self.pred_list = [np.array([0.8, 2.2, 1.1, 3.7, 5.5]), np.array([0.1, 2.9, 3.1, 4.8, 5.8])]
        self.pred_title = ["preds_0", "preds_1"]

    def fit(self):
        return self.x, self.y, self.pred_list, self.pred_title