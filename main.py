import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#setting the option of pandas to view all features in csv files
pd.set_option("display.max_columns",None)
#reading the csv file
malData= pd.read_csv("MalwareData.csv",sep="|")

#reading legitimate files
legit= malData[0:41323].drop(["legitimate"],axis=1)
#reading malware files
mal=malData[41323::].drop(["legitimate"],axis=1)

#printing samples
#
#print("The shape of the legit dataset is: %s samples, %s features"%(legit.shape[0],legit.shape[1]))

#print("The shape of the mal dataset is: %s samples, %s features"%(mal.shape[0],mal.shape[1]))

#print("Showing all faetures:",malData.columns)

#print(legit.head(1))


# taking only important data
data_in = malData.drop(['Name','md5','legitimate'],axis=1).values

labels = malData['legitimate'].values

extratrees = ExtraTreesClassifier().fit(data_in,labels)

select = SelectFromModel(extratrees,prefit=True)

data_in_new = select.transform(data_in)

#print(data_in_new.columns)

#spliting data to 20% only for test

legit_train,legit_test,mal_train,mal_test = train_test_split(data_in_new,labels,test_size=0.2)

classif=RandomForestClassifier(n_estimators=50)

classif.fit(legit_train,mal_train)

print("score is:",classif.score(legit_test,mal_test)*100)

result = classif.predict(legit_test)

conf_mat = confusion_matrix(mal_test,result)

#print(conf_mat.shape)

#print(conf_mat)

print("False positives:",conf_mat[0][1]/sum(conf_mat[0])*100)
print("False negatives:",conf_mat[1][0]/sum(conf_mat[1])*100)
