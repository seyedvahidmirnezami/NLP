
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
training_original = pd.read_csv('SMK1_ALL_LARC_2Feb_SMOTE.csv')
#print(training_original.head())
size = 400        # sample size
replace = True  # with replacement
fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
training = training_original.groupby('CEFR_NUM', as_index=False).apply(fn).reset_index()
training.drop(['level_0', 'level_1','id','Country','CEFR'], axis=1, inplace=True)
#print(training.head())
training_x = training[training.columns[0:-1]].as_matrix()
training_y = training[training.columns[-1]].as_matrix()
from keras.utils.np_utils import to_categorical
training_y = to_categorical(training_y)
print(training_y)
print(training_x.shape)


# In[2]:


testing = pd.read_csv('PTJ1_ALL_LARC_2Feb_SMOTE.csv')
testing.drop(['id','Country','CEFR'], axis=1, inplace=True)
print(testing.shape)
testing_x = testing[testing.columns[0:-1]].as_matrix()
testing_y = testing[testing.columns[-1]].as_matrix()
from keras.utils.np_utils import to_categorical
testing_y = to_categorical(testing_y)
#print(testing_y)


# In[3]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
numpy.random.seed(7)
from keras.callbacks import EarlyStopping


# In[4]:


learning_rate=[0.5,0.05,0.01,0.000001];
hidden_units=[100,200,300];
momentum_rate=[0.2,0.4,0.6,0.9];
layer=3;
learning_rate_output=[];
hidden_layers_output=[];
hidden_units_output=[];
momentum_rate_output=[];
optimizer_list_output=[];
loss_list_output=[];
optimizer_list = ['SGD', 'Adam']
loss_list = ['mean_squared_error','categorical_crossentropy']
acc=[];
t=[];
loss_report=[];
val_acc_report=[];
val_loss_report=[];

for rate in learning_rate:
    for momentum in momentum_rate:
        for unit in hidden_units:
            for opt in optimizer_list:
                for ls in loss_list:

                    model = Sequential()
                    model.add(Dense(units=unit, activation='relu', input_dim=134))
                    #model.add(Dropout(0.25))
                    model.add(Dense(units=unit, activation='relu', input_dim=100))
                    #model.add(Dense(units=unit, activation='relu', input_dim=100))
                    model.add(Dense(units=4, activation='softmax'))
                    # compiling model\n",
                    sgd = SGD(lr=rate, decay=1e-6, momentum=momentum, nesterov=True)
                    model.compile(loss=ls, optimizer=opt, metrics=['accuracy'])

                    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
                    callbacks_list = [earlystop]   

                    

                    import time
                    start=time.time();
                    train_history = model.fit(training_x, training_y, epochs=500,validation_split=0.2,callbacks=callbacks_list)


                    end=time.time();
                    run_time=end-start;
                    print(run_time)
                    loss = train_history.history['loss']
                    val_loss = train_history.history['val_loss']
                    val_acc = train_history.history['val_acc']
                    acc_value = train_history.history['acc']

                    testPred = model.predict(testing_x)

                    labelPredicted = pd.DataFrame(testPred).idxmax(axis=1)
                    labelTrue = pd.DataFrame(testing_y).idxmax(axis=1)


                    learning_rate_output.append(rate);
                    hidden_layers_output.append(layer);
                    hidden_units_output.append(unit);
                    momentum_rate_output.append(momentum);
                    optimizer_list_output.append(optimizer_list);
                    loss_list_output.append(loss_list);
                    acc.append(acc_value[-1]);
                    t.append(run_time);
                    loss_report.append(loss[-1]);
                    val_loss_report.append(val_loss[-1]);
                    val_acc_report.append(val_acc[-1]);

       

dic_original_equal={'optimizer':optimizer_list_output,'loss':loss_list_output,'hidden_layer':hidden_layers_output,'learning_rate':learning_rate_output,'momentum':momentum_rate_output,
     'hidden_unit':hidden_units_output,'accuracy':acc,'time':t,'loss':loss_report,'val_loss':val_loss_report,
    'val_acc':val_acc_report};

print(pd.DataFrame(dic_original_equal))

from pandas import ExcelWriter
dic_over_equal_df = pd.DataFrame(dic_original_equal)

writer = pd.ExcelWriter('over_equal.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
dic_over_equal_df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

