import math
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier


#平衡+委员会加权
class qbc_bcw:
    
    def __init__(self,balance_scale=0.3,bigger_par=1.5,each_size=30):
        assert balance_scale>=0 and balance_scale<=1,"balance_scale must in [0,1]"
        assert bigger_par>=0,"bigger_par must be positive"
        assert distance_scope>=2,"distance_scope must greater than 2"
        assert each_size>0,"each_size must be positive"
        self.balance_scale = balance_scale
        self.bigger_par = bigger_par
        self.each_size = each_size
        self._X_train = None
        self._y_train = None
        
    def fit(self,X_train,y_train):
        assert X_train.shape[0]==y_train.shape[0],"the size of X_train must be equal to the size of y_train"
        
        self._X_train = X_train
        self._y_train = y_train

        choice_list = [i for i in range(len(self._X_train))]
        random.shuffle(choice_list)
    
        num1 = int(len(self._X_train)/3)
        num2 = int(len(self._X_train)*2/3)
    
        x_init1 = pd.concat([self._X_train.iloc[choice_list[:num1]]])
        x_init2 = pd.concat([self._X_train.iloc[choice_list[num1:num2]]])
        x_init3 = pd.concat([self._X_train.iloc[choice_list[num2:]]])
    
        y_init1 = pd.concat([self._y_train.iloc[choice_list[:num1]]])
        y_init2 = pd.concat([self._y_train.iloc[choice_list[num1:num2]]])
        y_init3 = pd.concat([self._y_train.iloc[choice_list[num2:]]])
    
        self.gb_clf1 = GradientBoostingClassifier()
        self.gb_clf2 = GradientBoostingClassifier()
        self.gb_clf3 = GradientBoostingClassifier()

        self.gb_clf1.fit(x_init1,y_init1)
        self.gb_clf2.fit(x_init2,y_init2)
        self.gb_clf3.fit(x_init3,y_init3)
        
        return self
    
    def predict(self,X_predict):
        assert self._X_train is not None and self._y_train is not None,"must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1],"the feature number of X_predict must be equal to X_train"
        
        scores_sort = self.__predict(X_predict)
        scores_sort = np.array(scores_sort)
        scores_sorted = np.argsort(scores_sort)
        
        X_output = X_predict.iloc[scores_sorted[-self.each_size:]]
        
        return X_output
                                    
    def __scores_func(self,proba):
        scores_sort = []
        #proba形如[[0.1,0.9],[0.4,0.6],[0.7,0.3]]
        for sc in proba:
            col = 0
            for p in sc:
                #避免出现log0
                if p in [0,1]:
                    col += 0
                else:
                    col += -p*math.log(p,math.e)
            scores_sort.append(col)
        return scores_sort
    
    def __predict(self,x_choice):
                                
        proba1 = self.gb_clf1.predict_proba(x_choice)
        proba2 = self.gb_clf2.predict_proba(x_choice)
        proba3 = self.gb_clf3.predict_proba(x_choice)
        
        scores1_sort = self.__scores_func(proba1)  
        scores2_sort = self.__scores_func(proba2)
        scores3_sort = self.__scores_func(proba3)
                                          
        distribution_normal = self._y_train.value_counts()[1]
        distribution_suspicious = self._y_train.value_counts()[2]
        distribution = distribution_suspicious/(distribution_normal+distribution_suspicious)
        
        score_weight1 = self.gb_clf1.score(self._X_train,self._y_train)
        score_weight2 = self.gb_clf2.score(self._X_train,self._y_train)
        score_weight3 = self.gb_clf3.score(self._X_train,self._y_train)
        
        scores_sort = []
        if distribution >= self.balance_scale:
            for i in range(len(scores1_sort)):
                col = max(scores1_sort[i]*score_weight1,scores2_sort[i]*score_weight2,scores3_sort[i]*score_weight3)
                scores_sort.append(col)
        else:
            for i in range(len(scores1_sort)):
                if (proba1[i][0]+proba2[i][0]+proba3[i][0])/3 < 0.5:
                    col = max(scores1_sort[i]*score_weight1,scores2_sort[i]*score_weight2,scores3_sort[i]*score_weight3)*self.bigger_par
                else:
                    col = max(scores1_sort[i]*score_weight1,scores2_sort[i]*score_weight2,scores3_sort[i]*score_weight3)
                scores_sort.append(col)
        return scores_sort
    
    def __repr__(self):
        return "qbc_ddbcw(balance_scale=%d,bigger_par=%d,each_size=%d)"%(self.balance_scale,self.bigger_par,self.each_size)
