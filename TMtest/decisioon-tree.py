import numpy as np
import pandas as pd
import math
import copy
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
import sys


"""class Node:
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None
        self.isLeef= False
"""
def fuzzification_age(age):
    fage={
        'input':df['age'],
        'young':[],
        'mid':[],
        'old':[],
        'result':[]
    }
    x_age=np.arange(0,101,1)   
    age_young=fuzz.trapmf(x_age,[-30,-5,30,40])
    age_mid=fuzz.trapmf(x_age,[30,40,50,60])
    age_old=fuzz.trapmf(x_age,[50,60,100,120])

    fage['young']=fuzz.interp_membership(x_age,age_young,age)
    fage['mid']=fuzz.interp_membership(x_age,age_mid,age)
    fage['old']=fuzz.interp_membership(x_age,age_old,age)
    #print(age['young'])
    for i in range(len(age)):
        if fage['young'][i] > fage['mid'][i]:
            fage['result'].append('young')
        elif  fage['mid'][i] > fage['old'][i]:
            fage['result'].append('mid')
        else:
            fage['result'].append('old')
    return fage
def fuzzification_bmi(bmi):
    fbmi={
        'input': bmi,
        'underweight':[],
        'normal':[],
        'overweight':[],
        'fat':[],
        'result':[],
    }
    x_bmi=np.arange(0,50,0.1)
    bmi_underweight=fuzz.smf(x_bmi,0,18)
    bmi_normal=fuzz.smf(x_bmi,18,24.9)
    bmi_overweight=fuzz.smf(x_bmi,25,30)
    bmi_fat=fuzz.smf(x_bmi,30,50)
    fbmi['underweight']=fuzz.interp_membership(x_bmi,bmi_underweight,bmi)
    fbmi['normal']=fuzz.interp_membership(x_bmi,bmi_normal,bmi)
    fbmi['overweight']=fuzz.interp_membership(x_bmi,bmi_overweight,bmi)
    fbmi['fat']=fuzz.interp_membership(x_bmi,bmi_fat,df['bmi'])
    for i in range(len(bmi)):
        if fbmi['underweight'][i] < 1 and fbmi['underweight'][i] > 0:
            fbmi['result'].append('underweight')
        elif fbmi['normal'][i] < 1 and fbmi['normal'][i] > 0:
            fbmi['result'].append('normal')
        elif fbmi['overweight'][i] < 1 and fbmi['overweight'][i] > 0 :
            fbmi['result'].append('overweight')
        else:
            fbmi['result'].append('fat')
    #print(df['bmi'])
    #print(bmi['result'])
    return fbmi
def entropy(target):
    elements, counts = np.unique(target, return_counts=True)
    #print (elements)
    entropy = np.sum( [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(df,split_attribute_name,target_name):
    total_entropy=entropy(df[target_name])
    #print(total_entropy)
    #print(df[split_attribute_name].astype(str))
    vals, counts = np.unique(df[split_attribute_name].astype(str), return_counts=True)
    #print("===========================")
    #print(split_attribute_name)
    #print(vals)
    #print(counts)
    weighted_entropy = np.sum( [(counts[i] / np.sum(counts)) * entropy(df.where(df[split_attribute_name] == vals[i]).dropna()[target_name])for i in range(len(vals))])
    Information_Gain = np.sum((total_entropy - weighted_entropy))
    return Information_Gain

def ID3(data,df,features,target_name,parent_node=None):
    
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]

    elif len(data) == 0:
        return np.unique(df[target_name])[np.argmax(np.unique(df[target_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node

    else:

        parent_node=np.unique(df[target_name])[np.argmax(np.unique(df[target_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_name) for feature in features]
        #print(item_values)
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        #print(features)
        #print(best_feature)

        tree = {best_feature: {}}

        features =[i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data,df,features,target_name,parent_node)
            tree[best_feature][value] = subtree

        return tree

def predict(query, tree,training_data, target_name):
    max_target = np.unique(training_data[target_name])[np.argmax(np.unique(training_data[target_name], return_counts=True)[1])]
    for key in list(query.keys()):
        if key in list(tree.keys()):
            
            try:
                result = tree[key][query[key]]
            except:
                pred = max_target
                return pred
    
            result = tree[key][query[key]]

            if isinstance(result, dict):
                return predict(query, result,training_data, target_name)

            else:
                return result

def plot_confusion_matrix(y_true, y_pred):
    # unique classes
    conf_mat = {}
    classes = np.unique(y_true)
    # C is positive class while True class is y_true or temp_true
    for c in classes:
        temp_true = y_true[y_true == c]
        temp_pred = y_pred[y_true == c]
        conf_mat[c] = {pred: np.sum(temp_pred == pred) for pred in classes}
    print("Confusion Matrix: \n", pd.DataFrame(conf_mat))

    # plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(data=pd.DataFrame(conf_mat), annot=True, cmap=plt.get_cmap("Blues"), fmt='d')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def calculate_metrics(y_true, y_pred):
    # convert to integer numpy array
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pre_list = []
    rec_list = []
    f1_list = []
    # loop over unique classes
    for c in np.unique(y_true):
        # copy arrays
        temp_true = y_true.copy()
        temp_pred = y_pred.copy()

        # positive class
        temp_true[y_true == c] = '1'
        temp_pred[y_pred == c] = '1'

        # negative class
        temp_true[y_true != c] = '0'
        temp_pred[y_pred != c] = '0'

        # tp, fp and fn
        tp = np.sum(temp_pred[temp_pred == '1'] == temp_true[temp_pred == '1'])
        tn = np.sum(temp_pred[temp_pred == '0'] == temp_true[temp_pred == '0'])
        fp = np.sum(temp_pred[temp_pred == '1'] != temp_true[temp_pred == '1'])
        fn = np.sum(temp_pred[temp_pred == '0'] != temp_true[temp_pred == '0'])

        precision = tp / (tp + fp) * 100
        recall = tp / (tp + fn) * 100
        f1 = 2 * (precision * recall) / (precision + recall)

        pre_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        print(
            "Class {}: Precision = {:0.3f}    Recall = {:0.3f}    F1-Score = {:0.3f}".format(c, precision, recall, f1))

    print("Average: Precision = {:0.3f}    Recall = {:0.3f}    F1-Score = {:0.3f}   Accuracy = {:0.3f}".
          format(np.mean(pre_list),
                 np.mean(rec_list),
                 np.mean(f1_list),
                 np.sum(y_pred == y_true)/y_pred.shape[0]*100))

def test(data, tree,training_data, target_name):
    
    queries = data.iloc[:, :-1].to_dict(orient="records")

    
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree,training_data, target_name)

    return predicted["predicted"]

df = pd.read_csv("Temperament_Survey.csv")

# print(df)
# print ("Dataset Length: ", len(df))
# print ("Dataset Shape: ", df.shape)
      
# Printing the dataset obseravtions
# print ("Dataset: ",df.head())
age=fuzzification_age(df['age'])
bmi=fuzzification_bmi(df['bmi'])
df['age']=age['result']
df['bmi']=bmi['result']
#print ("Dataset: ",df.head())
print(df.shape[0])
all_true = []
all_pred = []
# Build Tree for dry or wet

with open('result1.txt', 'w') as f:
    sys.stdout = f 
    df_dry=df.drop('A1',axis='columns')
    df_cold=df.drop('your feeling', axis='columns')



    train_dry,test_dry=train_test_split(df_dry,test_size=0.2,random_state=0)

    tree_dry=ID3(train_dry,train_dry,df_dry.columns[:-1],'A2')
    print('-------tree for dry or wet--------')
    pprint(tree_dry)

    y_pred1=test(test_dry,tree_dry,train_dry, 'A2')
    y_orig1=test_dry['A2']

    #print(y_pred1)
    #print(y_orig1)

    # Build Tree for cold or hot

    df_cold=df.drop('A2', axis='columns')
    df_cold=df.drop('bmi', axis='columns')


    train_cold,test_cold=train_test_split(df_cold,test_size=0.33,random_state=0)

    tree_cold=ID3(train_cold,train_cold,df_cold.columns[:-1],'A1')

    print('-------tree for cold or hot--------')
    pprint(tree_cold)

    y_pred2=test(test_cold,tree_cold,train_cold,'A1')
    y_orig2=test_cold['A1']

    y_pred1=np.array(y_pred1).astype(str)
    y_orig1=np.array(y_orig1).astype(str)

    y_pred2=np.array(y_pred2).astype(str)
    y_orig2=np.array(y_orig2).astype(str)


    acc = (np.sum(y_pred1 == y_orig1) / y_orig1.shape[0])*100
    print(" Accuracy: {:.3f}".format(acc))

    acc = (np.sum(y_pred2 == y_orig2) / y_orig2.shape[0])*100
    print(" Accuracy: {:.3f}".format(acc))

    calculate_metrics(y_orig1, y_pred1)
    plot_confusion_matrix(y_orig1, y_pred1)

    calculate_metrics(y_orig2, y_pred2)
    plot_confusion_matrix(y_orig2, y_pred2)
