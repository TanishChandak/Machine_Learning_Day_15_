import pandas as pd
from sklearn import svm, datasets
iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
print(df[47:52])

# We are using here the Hyper-parameter tuning in which GridSearchCV will be used:
# In this we have to apply some operation to optimize the solution or accuracy:
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)
# Training the model:
clf.fit(iris.data, iris.target)
print(clf.cv_results_)

# Conveting the training data into dataframe:
df = pd.DataFrame(clf.cv_results_)
print(df.head())
print(df[['param_C', 'param_kernel', 'mean_test_score']])

# elements of the clf:
print(dir(clf))
print(clf.best_score_) # it will provide the best score from all
print(clf.best_params_) # it will give you the best C and KERNEL


# We are using here the Hyper-parameter tuning in which RandomizedSearchCV will be used:
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False, n_iter=2)
# Training the data:
rs.fit(iris.data, iris.target)
df1 = pd.DataFrame(rs.cv_results_)[['param_C', 'param_kernel', 'mean_test_score']]
print(df1)

# Now applied to the different models like svm, randomforestclassifier, logisticregression:
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# for SVM:
from sklearn.model_selection import GridSearchCV
clf_svm = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)
clf_svm.fit(iris.data, iris.target)
df = pd.DataFrame(clf_svm.cv_results_)
print("For svm:",df)
print(df[['model', 'best_scores', 'best_params']])

# for RandomForestClassifier:
clf_rf = GridSearchCV(RandomForestClassifier(), {
    'n_estimator': [1, 5, 10]
}, cv=5, return_train_score=False)
clf_rf.fit(iris.data, iris.target)
df = pd.DataFrame(clf_rf.cv_results_)
print("For Rn:",df)
print(df[['model', 'best_scores', 'best_params']])

# for LogisticRegression:
clf_Lr = GridSearchCV(LogisticRegression(solver='liblinear', multi_class='auto'), {
    'C': [1, 5, 10]
}, cv=5, return_train_score=False)
clf_Lr.fit(iris.data, iris.target)
df = pd.DataFrame(clf_Lr.cv_results_)
print(df)
print(df[['model', 'best_scores', 'best_params']])

'''
# Now applied to the different models like svm, randomforestclassifier, logisticregression:
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimator': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 10]
        }
    }
}
scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores, columns=['model', 'best_scores', 'best_params'])
print(df)'''
