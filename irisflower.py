# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)

data=pd.read_csv('iris.csv')

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

#with default PRAMETER = to predict and accuracy
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)


# Predict the dependent variable for the test set
pred=clf.predict(X_test)
acc=accuracy_score(y_test,pred)
print('prediction',pred)
print('accuracy',acc*100)

#another with criterion='entropy',max_depth=3 - to predict and accuracy
clf1=DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf1.fit(X_train,y_train)



pred1=clf1.predict(X_test)
acc1=accuracy_score(y_test,pred1)
print('prediction',pred1)
print('accuracy',acc1*100)


plt.figure(figsize=(20,16))
tree.plot_tree(clf,fontsize=14,rounded=True,filled=True,max_depth=True)
plt.show()
     























