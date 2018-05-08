import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
#import scikitplot as skplt
import collections as c
from sklearn.utils import resample

def fitOneHotEncodings(nparray,dim1):

    nls = ['CHN', 'PAK', 'IDN', 'JPN', 'THA', 'PHL', 'SIN', 'TWN', 'KOR', 'HKG']
    le = LabelEncoder()

    le.fit(nls)
    new1 = le.transform(nparray[:,dim1])
    nparray[:,dim1] = new1

    encoder = OneHotEncoder(categorical_features=[dim1])
    encoder.fit(nparray)
    return encoder.transform(nparray).toarray()

def resampledata(input_path,output_path):
    df = pd.read_csv(input_path, header=0)
    #print(list(df.columns.values)) #Prints column names.
    #print(df['post_category'].value_counts()) #710, 295, 137, 40
   # print(df['post_category']=="crisis")
    df_a2 = df[df['CEFR']=="A2_0"]
    df_b1_1 = df[df['CEFR']=="B1_1"]
    df_b1_2 =  df[df['CEFR']=="B1_2"] #This is the one with largest examples in training data.
    df_b2_0 =  df[df['CEFR']=="B2_0"]
    df_a2_upsampled = resample(df_a2,replace=True, n_samples=len(df_b1_2), random_state=123) # reproducible results
    df_b11_upsampled = resample(df_b1_1,replace=True, n_samples=len(df_b1_2), random_state=123) # reproducible results
    df_b20_upsampled = resample(df_b2_0,replace=True, n_samples=len(df_b1_2), random_state=123) # reproducible results
    print(len(df_a2_upsampled))
    print(len(df_b11_upsampled))
    print(len(df_b1_2))
    print(len(df_b20_upsampled))
    df_full_upsampled = pd.concat([df_a2_upsampled,df_b11_upsampled,df_b1_2,df_b20_upsampled])
    df_full_upsampled.to_csv(output_path)
    return df_full_upsampled

def traincv(all_data,all_labels):
        random_seed = 1234
        #classifiers = [LogisticRegression()]
        classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed, class_weight="balanced"),RandomForestClassifier(random_state=random_seed, class_weight="balanced" , n_estimators=50),
                       LinearSVC(random_state=random_seed, class_weight="balanced"),GradientBoostingClassifier(random_state=random_seed, n_estimators=20, max_depth=5, max_features=20)]

        #,XGBClassifier(),LogisticRegression(random_state=random_seed),LinearSVC(random_state=random_seed)] #, , RandomForestClassifier() LinearSVC()
        #class_weight={"A2_0":0.4,"B1_1":0.2,"B2_0":0.3,"B1_2":0.1}
        k_fold = StratifiedKFold(10,shuffle=True)

        for classifier in classifiers:
                print("********", "\n", "10 fold CV Results with: ", str(classifier))
                cross_val = cross_val_score(classifier, all_data, all_labels, cv=k_fold, n_jobs=1)
                predicted = cross_val_predict(classifier, all_data, all_labels, cv=k_fold)
                print(cross_val)
                print(sum(cross_val)/float(len(cross_val)))
                print(confusion_matrix(all_labels, predicted))


def plots(all_data,all_labels):
        clf = RandomForestClassifier()
        clf.fit(all_data,all_labels)
        probas = clf.predict_proba(all_data)
        #skplt.metrics.plot_precision_recall_curve(y_true=all_labels, y_probas=probas)
        skplt.estimators.plot_learning_curve(clf, all_data, all_labels)
        plt.savefig(".png")

def predict_on_test(train_data,train_labels,test_data,test_labels):
    random_seed = 1234
    classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),RandomForestClassifier(random_state=random_seed, n_estimators=50),
                       LinearSVC(random_state=random_seed),GradientBoostingClassifier(random_state=random_seed, n_estimators=20, max_depth=5, max_features=20)]
 
    for classifier in classifiers:
        print("********", "\n", "Results on Test data with : ", str(classifier))
        classifier.fit(train_data,train_labels)
        #dot_data = export_graphviz(classifier, out_file="temp.dot")
        predictions = classifier.predict(test_data)
        print(np.mean(predictions == test_labels,dtype=float))
        print(confusion_matrix(test_labels, predictions))

"""
        # converted to one hot.
       # nls = numpy_array[:,-3] #native language column. Need to convert to one hot vector.
         #last column is the number. We don't need if we do classification
"""
def processData(input_file):
     df = pd.read_csv(input_file, header = 0)
     numpy_array = df.ix[:,:].as_matrix()
     all_data = numpy_array[:,1:-2]#First and last columns are id, and CEFR_NUM. We don't need here.
     print(len(all_data[0]))
     new_data = fitOneHotEncodings(all_data,-1) #Last column needs to be one hot vectorized.
     print(len(new_data[0]))
     all_labels = numpy_array[:,-2]
     return new_data,all_labels

def processSMOTEFile(input_file):
     df = pd.read_csv(input_file, header = 0)
     numpy_array = df.ix[:,:].as_matrix()
     all_data = numpy_array[:,1:-1]
     print(len(all_data[0]))
     new_data = fitOneHotEncodings(all_data,-1) #Last column needs to be one hot vectorized.
     print(len(new_data[0]))
     all_labels = numpy_array[:,-1]
     return new_data,all_labels

def main():
        train_file = 'PTJ1_ALL_LARC_2Feb.csv'
        test_file = 'SMK1_ALL_LARC_2Feb.csv'
        train_data,train_labels = processData(train_file) #processSMOTEFile(train_file) #if smote resampled
        test_data, test_labels = processData(test_file)
        print(c.Counter(train_labels))
        print(len(test_data),len(test_data[0]))
        traincv(train_data,train_labels)
        predict_on_test(train_data,train_labels,test_data,test_labels)
        #resampledata(train_file, '../data/larc/SMK1_ALL_LARC_2Feb_resampled.csv')
if __name__ == '__main__':
    main()


