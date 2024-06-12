#CMP510 - Engineering Resilient Systems coursework
# Classifer to categorise network packets according to attack category
# Random Forest chosen & visualisation using : Confusion Matrix, Precision Recall Curve, ROC

#importing required libraries - some might not be used due to unused code in program
from sklearn.calibration import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#reading the testing and training data 
training_data = pd.read_csv("CMP510_training_dataset1.csv")
testing_data = pd.read_csv("CMP510_testing_dataset1.csv")


#edit the head value to display more or less data from the dataset
print ("Loading training Data:")
print (training_data.head())
print ("")
print ("Loading testing Data:")
print (testing_data.head())


#data preprocessing
'''
accounting for the entires of '-' within the data , creating a new category unknown so we can account for missing values in service
#the below gives a 'future warning' for chaning together assignment, so implimenting another method to try remove this, should be 
#fine regarldess but was giving a cluttered output 
#training_data['service'].replace('-', 'Unknown', inplace=True)
#testing_data['service'].replace('-', 'Uknown', inplace=True)
'''
#assigning missing values in the service colun with 'unknown' 
training_data.loc[training_data['service'] == '-', 'service'] = 'Unknown'
testing_data.loc[testing_data['service'] == '-', 'service'] = 'Unknown'

#transforming categorical features to numerical values
label_encoder = LabelEncoder()
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    training_data[col] = label_encoder.fit_transform(training_data[col])
    testing_data[col] = label_encoder.fit_transform(testing_data[col])

# #transforming target variable to numerical values 
target_label_encoder = LabelEncoder()
training_data['attack_cat'] = target_label_encoder.fit_transform(training_data['attack_cat'])
testing_data['attack_cat'] = target_label_encoder.fit_transform(testing_data['attack_cat'])

# Split the data
X_training_data = training_data.drop('attack_cat', axis=1)
y_training_data = training_data['attack_cat']

X_testing_data = testing_data.drop('attack_cat', axis=1)
y_testing_data = testing_data['attack_cat']

#trying to increase the accuracy of the RF using gridsearch CV
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],

# }
# clf = RandomForestClassifier()
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)


# initilising and training random forest classifier 
clf = RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=102, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True)
trained_model = clf.fit(X_training_data, y_training_data)
print ("")
# Printing the accuracy of the model on the training data 
print ("Score: ", trained_model.score(X_training_data, y_training_data))

# Predict on the testing data
y_predicting_data = clf.predict(X_testing_data)

# Evaluate the model
# Then ccuracy of the model on the testing data 
print('Classification Report:')
print(classification_report(y_testing_data, y_predicting_data))
print('Accuracy Score:', accuracy_score(y_testing_data, y_predicting_data))




# Compute confusion matrix
compute_matrix = confusion_matrix(y_testing_data, y_predicting_data)

#creating a visualisation for the confusion matrix to remove the basic text one
figure_plot, axes_plot = plt.subplots()
object_on_axes = axes_plot.matshow(compute_matrix, cmap=plt.cm.Greens)
figure_plot.colorbar(object_on_axes)
plt.xlabel('Predicted Attack Category')
plt.ylabel('Actual Attack Category')

#annotating the plot
for (i, j), z in np.ndenumerate(compute_matrix):
           axes_plot.text(j, i, str(z), ha='center', va='center')

plt.show()

### ROC IMPLIMENTATION #### 

""" REFERENCES -> PLEASE READ
The following code below to impliment the ROC 
has been pulled together from multiple sources online, some of it i found quite difficult to 
implement therefore i have just moved over the structure and adjusted for this project. 
Therefore  I wanted to ensure i correctly referenced them, further references will be given in the report,
however I also wanted to ensure i referenced here. This ends around line 163 for the ROC implimentation
https://stats.stackexchange.com/questions/2151/how-to-plot-roc-curves-in-multiclass-classification
https://stackoverflow.com/questions/50941223/plotting-roc-curve-with-multiple-classes
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
https://scikit-learn.org/stable/modules/multiclass.html#multiclass
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-binarization
https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a

"""

### IMPORTANT -> For this to work it is important to undo the commenting for the label encoder on line 48 as this
# will enable the labels to be taken as numerical values 
#binarising the values for ROC and precision recall curve implementation & visualisation
y_testing_data_binarized = label_binarize(y_testing_data, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = y_testing_data_binarized.shape[1]

#initistialising the classifier and predicting probabilities 
classifier = OneVsRestClassifier(clf)
y_score = classifier.fit(X_training_data, y_training_data).predict_proba(X_testing_data)

#initialising dictionaries to store false positive, true postive, ROC and AUC scores 
false_positive_rates = dict()
true_positive_rates = dict()
roc_auc = dict()


for i in range(n_classes):
   false_positive_rates[i], true_positive_rates[i], _ = roc_curve(y_testing_data_binarized[:, i], y_score[:, i])
   roc_auc[i] = auc(false_positive_rates[i], true_positive_rates[i])


# Compute micro-average true positive rate
fpr_micro, tpr_micro, _ = roc_curve(y_testing_data_binarized.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# plotting ROC for each class 
plt.figure()
plt.plot(fpr_micro, tpr_micro,
        label='micro-average ROC curve (area = {0:0.2f})'
              ''.format(roc_auc_micro))
for i in range(n_classes):
   plt.plot(false_positive_rates[i], true_positive_rates[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                  ''.format(i, roc_auc[i]))

#formating the labels, legend and look of the ROC 
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic to Multi-Class')
plt.legend(loc="lower right")
plt.show()



### PRECISION RECALL CURVE IMPLEMENTATION 
""" REFERENCES -> Please read
 Much like above, difficult implementaiton albeit was easier after understanding how to implement ROC
 Despite this though, code heavily referenced from the following links: 
 https://stackoverflow.com/questions/50941223/plotting-roc-curve-with-multiple-classes
 https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision%2Drecall%20curve%20shows,a%20low%20false%20negative%20rate.
 https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier
 https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
 https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
 """

#initilisaing dictionares to store preciison, recall and average precision scores
precision = dict()
recall = dict()
average_precision = dict()

#computing precision recall and average precision scores 
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_testing_data_binarized[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_testing_data_binarized[:, i], y_score[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(y_testing_data_binarized.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_testing_data_binarized, y_score, average="micro")

#plotting and formatting precision recall curve simiarly to above 
plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: Average Precision ={0:0.2f}'.format(average_precision["micro"]))
plt.show()
