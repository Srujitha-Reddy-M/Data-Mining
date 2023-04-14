import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
np.set_printoptions(precision=2)


#Loading the dataset
df = pd.read_csv('Training_data.dat', header = None, sep="\t", names = ["sentiment", "review"])
df_testData = pd.read_fwf('Test_data.dat', header= None,sep = '\n', skip_blank_lines=False)
df_testData = df_testData[[0]]
df_testData.rename(columns = {0:'review'}, inplace = True)
print("Train data shape", df.shape)
print("Test data shape", df_testData.shape)
print(df.head())
print(df_testData.head())


#Handling null values
print("Shape before handling rows that have null values", df.shape)
print("No.of null values in each column\n", df.isnull().sum())

df.dropna(inplace=True) # since the no.of null values are smaller i.e just 9, it is wise to remove those rows instead of filling them
print("Shape after handling rows that have null values", df.shape)
print("No.of null values in each column\n", df.isnull().sum())


#Pre-processing
stop = stopwords.words('english')
def preprocessor(text):
    text = str(text).replace('[^\w\s]','')
    text = ' '.join([word for word in 
                     text.split() if word not in (stop)])
    return text.lower()


#Tokenization of documents
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


#Word relevancy using term frequency-inverse document frequency

# Transform text data into tf-idf vectors
from sklearn.feature_extraction.text import TfidfVectorizer
def getTfidVectors(column):
  tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)
  return tfidf.fit_transform(column)

combinedDf = df.review.append(df_testData.review)
combinedDf = combinedDf.apply(preprocessor)
combinedDf_vectors = getTfidVectors(combinedDf)

X = combinedDf_vectors[:df.review.shape[0]]
y = df.sentiment.values
X_testData = combinedDf_vectors[df.review.shape[0]:]


#Training a logistic regression model
# Splitting train data to obtain known test data to validate how the model and Tfidfvectorizer are performing
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1,test_size=0.2,shuffle=True)


clf = LogisticRegressionCV(cv=5,
                          scoring='accuracy',
                          random_state=0,
                          n_jobs=-1,class_weight=None,
                          verbose=3).fit(X_train,y_train)

print("Training accuracy in train data is ", clf.score(X_train, y_train))
print("Testing accuracy in test data is ", clf.score(X_test, y_test))


#Predicting sentiments for Unknown Test data
# Now we can use entire train data to train the model as new unknown test data is available
clf_full = LogisticRegressionCV(cv=5,
                          scoring='accuracy',
                          random_state=0,
                          n_jobs=-1,
                          verbose=3).fit(X,y)

y_testData = clf_full.predict(X_testData)
pd.DataFrame(y_testData).to_csv("Test_data_ans.csv", index=False, header=False)
