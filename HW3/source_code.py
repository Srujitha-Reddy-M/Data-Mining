import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Helper functions
def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):
    return np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))



# Define AdaBoost class
class AdaBoost:
    
    def __init__(self):
        # self.w_i = None
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.training_errors_unweighted = []
        self.cumulative_training_errors_unweighted = []
        self.minerTest = pd.read_csv("/content/1646102777_591544_test35-nolabels.txt", names=list(range(256)))

    def fit(self, X, y, M = 100):             #M is the no of rounds for boosting     
        self.alphas = []                      
        self.training_errors = []
        self.M = M

        for m in range(0, M):                         #looping over all weak classifiers
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)      # when m = 0, initialize weights 1 / N
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred)

            G_m = DecisionTreeClassifier(max_depth = 1, min_samples_split=500)
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m)

            #calculating the error 
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)
            self.training_errors_unweighted.append(accuracy_score(y, y_pred))

            #calculating the alpha value 
            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)
            if (m%10 == 0):
              y_pred_test = self.predict(self.minerTest).to_frame(name="Label")
              y_pred_test["Label"] = y_pred_test["Label"].replace(to_replace = [1,-1], value=[3,5])
              y_pred_test["Label"].to_csv('output' + str(m) + ".txt", index=False, header=False)
            y_pred_cum = self.predict(X)
            self.cumulative_training_errors_unweighted.append(accuracy_score(y, y_pred_cum))
    
        assert len(self.G_M) == len(self.alphas)


    def predict(self, X):
        #initialize the dataframe
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        #predicting the label
        for m in range(len(self.G_M)):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        #Final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
      
    def error_rates(self, X, y):
        self.prediction_errors = [] 
        
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X)          
            error_m = compute_error(y = y, y_pred = y_pred_m, w_i = np.ones(len(y)))
            self.prediction_errors.append(error_m)

#loading the data
names = ["Number"]
pixels = list(range(256))
names.extend(pixels)
miner = pd.read_csv("/content/1646102777_5857556_train35.txt", names=names)  
miner['Label'] = miner['Number'].replace(to_replace = [3,5], value=[1,-1])
miner = miner.drop('Number', axis = 1)
miner.head()

X = miner.drop(columns = "Label").values
Y = miner["Label"].values

numOfRounds = 200

ab = AdaBoost()
ab.fit(X, Y, M = numOfRounds)

#prediction on test set

pixels = list(range(256))
minerTest = pd.read_csv("/content/1646102777_591544_test35-nolabels.txt", names=pixels)
y_pred_test = ab.predict(minerTest).to_frame(name="Label")
y_pred_test["Label"] = y_pred_test["Label"].replace(to_replace = [1,-1], value=[3,5])
y_pred_test["Label"].to_csv('output.txt', index=False, header=False)

#using single decision tree
ab_sk_dt = DecisionTreeClassifier(criterion="gini")
ab_sk_dt.fit(X, Y)
y_pred_test = ab_sk_dt.predict(minerTest)
y_pred_test = pd.DataFrame(y_pred_test, columns = ["Label"])
y_pred_test = y_pred_test.replace(to_replace = [1,-1], value=[3,5])
y_pred_test.to_csv('output_single_dt.txt', index=False, header=False)

#Plots 

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(list(range(200)), ab.training_errors)
plt.xlabel("Rounds")
plt.ylabel("Training error Weighted")
plt.show()

rounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
test_errors_from_miner = [0.85,0.91,0.92,0.91,0.94,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93]
test_error_single_dt_from_miner = 0.91

training_errors_unweighted = [1- ab.training_errors_unweighted[m] for m in range(numOfRounds) if m%10==0]
cumulative_training_errors_unweighted = [1 - ab.cumulative_training_errors_unweighted[m] for m in range(numOfRounds) if m%10==0]
test_errors = [1-x for x in test_errors_from_miner]
single_dt_error = [1-test_error_single_dt_from_miner]*20
ticks = [0]
while(True):
  if(ticks[-1] >= 0.75):
    break
  ticks.append(round(ticks[-1]+0.05,2))

df = pd.DataFrame({
    "rounds" : rounds, 
    "training_errors_unweighted" : training_errors_unweighted, 
    "test_errors" : test_errors, 
    "single_decision_tree_test_error" : single_dt_error,
    "cumulative_training_errors_unweighted" : cumulative_training_errors_unweighted})


plt.plot(df.rounds, df.training_errors_unweighted)
plt.xlabel("Rounds")
plt.yticks(ticks)
plt.ylabel("training_errors_unweighted")
plt.show()

plt.plot(df.rounds, df.cumulative_training_errors_unweighted)
plt.xlabel("Rounds")
plt.yticks(ticks)
plt.ylabel("cumulative_training_errors_unweighted")
plt.show()

plt.plot(df.rounds, df.test_errors)
plt.xlabel("Rounds")
plt.yticks(ticks)
plt.ylabel("test_errors")
plt.show()

ax = df.plot(x="rounds", y="training_errors_unweighted", legend=False)
plt.yticks(ticks)
ax4 = ax.twinx()
df.plot(x="rounds", y="cumulative_training_errors_unweighted", ax=ax4, legend=False, color="y")
plt.yticks(ticks)
ax3 = ax.twinx()
df.plot(x="rounds", y="single_decision_tree_test_error", ax=ax3, legend=False, color="g")
plt.yticks(ticks)
ax2 = ax.twinx()
df.plot(x="rounds", y="test_errors", ax=ax2, legend=False, color="r")
plt.yticks(ticks)

ax.figure.legend()
plt.show()