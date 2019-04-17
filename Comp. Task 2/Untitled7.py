
# coding: utf-8

# In[454]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import math as math
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def dataset():
    #SELECT THE DATA 
    EL_df = pd.read_csv('/Users/glebvorobcuk/Desktop/ACNM 2018/Term 2/Data Mining/Comp. Task 2/EL_df.csv')
    #print(EL_df)
    #select random values from each class and remove it from training set
    TRAINING_df=EL_df
    TEST_df=pd.DataFrame()
    P_rand_tmp=pd.DataFrame()
    O_rand_tmp=pd.DataFrame()
    P_rand=pd.DataFrame()
    O_rand=pd.DataFrame()
    for i in range(0,3):
        P_rand_tmp = pd.DataFrame(EL_df.loc[EL_df['Number'].isin(["P"])].sample(n=1))
        P_rand=P_rand.append(P_rand_tmp)
        TRAINING_df.drop(P_rand_tmp.index,inplace=True)
    for j in range(0,2):    
        O_rand_tmp = pd.DataFrame(EL_df.loc[EL_df['Number'].isin(["O"])].sample(n=1))
        O_rand = O_rand.append(O_rand_tmp)
        TRAINING_df.drop(O_rand_tmp.index,inplace=True)

    TEST_df=TEST_df.append(P_rand)
    TEST_df=TEST_df.append(O_rand)
    RETURN=[TRAINING_df,TEST_df]
    RETURN_df=pd.concat(RETURN)

    return RETURN_df

#m=2 by default
def DT(dataset, m):
    #print(dataset)
    TRAINING_df=dataset.iloc[0:25,:]
    TEST_df=dataset.iloc[26:,:]


    #target attribute
    X_train = TRAINING_df.loc[:,'Q1':'Q12']
    y_train = TRAINING_df.Number
    year_train=TRAINING_df.Year
    X_test = TEST_df.loc[:,'Q1':'Q12']
    y_test = TEST_df.Number
    year_test=TEST_df.Year
    
    #copy dataset to prevent changes
    X_train_knn = X_train
    X_test_knn = X_test
    y_train_knn = y_train
    y_test_knn = y_test
    
   # print(TRAINING_df)
    #print(TEST_df)
    #create decision tree
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,min_samples_split=m)
    clf_entropy.fit(X_train, y_train)
    dot_data = tree.export_graphviz(clf_entropy, out_file=None,filled = True) 
    graph = graphviz.Source(dot_data) 
    graph
    
    #create 1-nn and 3-nn classifiers
    
    
    #predict
        #test
    pred_test = clf_entropy.predict(X_test)
    pred_test = pd.DataFrame(pred_test,index=year_test.index)
    pred_test_ = pd.concat([year_test,y_test,pred_test], axis=1)
    pred_test_.columns=['Year','Exact','Predicted']
    print('m:',m)
    print(pred_test_)
    err_test =1 - accuracy_score(y_test, pred_test)
    print('test error:',err_test)
        #train
    pred_tr = clf_entropy.predict(X_train)
    pred_tr = pd.DataFrame(pred_tr,index=year_train.index)
    pred_tr_ = pd.concat([year_train,y_train,pred_tr], axis=1)
    pred_tr_.columns=['Year','Exact','Predicted']
    err_tr =1 - accuracy_score(y_train, pred_tr)
    print('train error:',err_tr)
    
    

    
    
    #return trees with predicrions and accuracy scores accuracy scores
    return [graph,
            pred_test_,
            err_test,
            pred_tr_,
            err_tr,
            m]



def knn(dataset):
    TRAINING_df=dataset.iloc[0:25,:]
    TEST_df=dataset.iloc[26:,:]
    #target attribute
    X_train = TRAINING_df.loc[:,'Q1':'Q12']
    y_train = TRAINING_df.Number
    year_train=TRAINING_df.Year
    X_test = TEST_df.loc[:,'Q1':'Q12']
    y_test = TEST_df.Number
    year_test=TEST_df.Year
    
    #copy dataset to prevent changes
    X_train_knn = X_train
    X_test_knn = X_test
    y_train_knn = y_train
    y_test_knn = y_test
    
    
    
    #knn
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(X_train,y_train)
    pred_test = knn1.predict(X_test)
    pred_test = pd.DataFrame(pred_test,index=year_test.index)
    pred_test_ = pd.concat([year_test,y_test,pred_test], axis=1)
    pred_test_.columns=['Year','Exact','Predicted']
    err_test1 =1 - accuracy_score(y_test, pred_test)

    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(X_train,y_train)
    pred_test2 = knn3.predict(X_test)
    pred_test2 = pd.DataFrame(pred_test,index=year_test.index)
    pred_test2_ = pd.concat([pred_test_,pred_test], axis=1)
    pred_test2_.columns=['Year','Exact','1nn Predicted','3nn Predicted']
    print(pred_test2_)
    err_test2 =1 - accuracy_score(y_test, pred_test2)
    print('1nn error:',err_test1)
    print('3nn error:',err_test2)
    
    
    return [pred_test2_,err_test1,err_test2]
    
    
    


#random sampling
dataset_1=dataset()
dataset_2=dataset()
dataset_3=dataset()
dataset_4=dataset()
dataset_5=dataset()
dataset_6=dataset()
dataset_7=dataset()
#preparing trees




#knn for best m for 7 datasets?


# In[438]:


m=2
A=DT(dataset_1,m)
B=DT(dataset_2,m)
C=DT(dataset_3,m)
D=DT(dataset_4,m)
E=DT(dataset_5,m)
F=DT(dataset_6,m)
G=DT(dataset_7,m)

m=3
H=DT(dataset_1,m)
I=DT(dataset_2,m)
J=DT(dataset_3,m)
K=DT(dataset_4,m)
L=DT(dataset_5,m)
M=DT(dataset_6,m)
N=DT(dataset_7,m)

m=4
O=DT(dataset_1,m)
P=DT(dataset_2,m)
Q=DT(dataset_3,m)
R=DT(dataset_4,m)
S=DT(dataset_5,m)
T=DT(dataset_6,m)
U=DT(dataset_7,m)

#calculate average errors for different m
AVG_TE_m2=np.mean([A[2],B[2],C[2],D[2],E[2],F[2],G[2]])
AVG_TE_m3=np.mean([H[2],I[2],J[2],K[2],L[2],M[2],N[2]])
AVG_TE_m4=np.mean([O[2],P[2],Q[2],R[2],S[2],T[2],U[2]])

AVG_LE_m2=np.mean([A[4],B[4],C[4],D[4],E[4],F[4],G[4]])
AVG_LE_m3=np.mean([H[4],I[4],J[4],K[4],L[4],M[4],N[4]])
AVG_LE_m4=np.mean([O[4],P[4],Q[4],R[4],S[4],T[4],U[4]])



# In[455]:


G[0]


# In[467]:



knn1=knn(dataset_1)
knn2=knn(dataset_2)
knn3=knn(dataset_3)
knn4=knn(dataset_4)
knn5=knn(dataset_5)
knn6=knn(dataset_6)
knn7=knn(dataset_7)
AVG_1NN=np.mean([knn1[1],knn2[1],knn3[1],knn4[1],knn5[1],knn6[1],knn7[1]])
AVG_3NN=np.mean([knn1[2],knn2[2],knn3[2],knn4[2],knn5[2],knn6[2],knn7[2]])
print(AVG_1NN,AVG_3NN)

AVG_1NN=np.mean([0.2,0.2,0.2,0.2,0.6,0.0,0.4])
AVG_3NN=np.mean([0.2,0.2,0.2,0.2,0.6,0.0,0.4])
print(AVG_1NN,AVG_3NN)


# In[465]:


#learning curves and best m selection
plt.plot([AVG_TE_m2,AVG_TE_m3,AVG_TE_m4],[AVG_LE_m2,AVG_LE_m3,AVG_LE_m4])
plt.ylabel("Learning error")
plt.xlabel("Test error")
plt.show
print([AVG_LE_m2,AVG_LE_m3,AVG_LE_m4])
print([AVG_TE_m2,AVG_TE_m3,AVG_TE_m4])


# In[466]:


plt.plot([2,3,4],[AVG_LE_m2,AVG_LE_m3,AVG_LE_m4])
plt.plot([2,3,4],[AVG_TE_m2,AVG_TE_m3,AVG_TE_m4])
plt.xlabel("M")
plt.ylabel("Error")
plt.legend(['Learning error','Test error'])
plt.show
print('le',[AVG_LE_m2,AVG_LE_m3,AVG_LE_m4])
print('te',[AVG_TE_m2,AVG_TE_m3,AVG_TE_m4])


# In[484]:


#lda #perceptron


EL_df = pd.read_csv('/Users/glebvorobcuk/Desktop/ACNM 2018/Term 2/Data Mining/Comp. Task 2/EL_df.csv')
X = EL_df.loc[:,'Q1':'Q12']
y = EL_df.Number
Fisher = LDA()
Fisher.fit(x, y)
print('accuracy:',Fisher.score(x, y, sample_weight=None)*100,'%','\nerror:',(1-Fisher.score(x, y, sample_weight=None))*100,'%')

Perceptron = PCP()
Perceptron.fit(x, y)
print('accuracy:',Perceptron.score(x, y, sample_weight=None)*100,'%','\nerror:',(1-Perceptron.score(x, y, sample_weight=None))*100,'%')



# In[398]:





# In[399]:





# In[474]:


#k-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Separating out the features
X = dataset_1.loc[:,'Q1':'Q12']
# Separating out the target
y = dataset_1.Number
# Standardizing the features
x = StandardScaler().fit_transform(X)

#reduce dimetionality
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, dataset_1.Number], axis = 1)
print(finalDf)
print(pca.explained_variance_ratio_)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['P', 'O']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Number'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[475]:


print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

data = dataset_1.loc[:,'Q1':'Q12']
#scaling
data = StandardScaler().fit_transform(X)
from sklearn.datasets import load_digits
digits = load_digits()
labels = dataset_1.Number
n_samples=31
n_features = 12
n_digits = len(np.unique(y))
sample_size = 11

print("n_classes: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(x)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=5,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced and scaled data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


# In[452]:





# In[483]:


#trump election
EL_df = pd.read_csv('/Users/glebvorobcuk/Desktop/ACNM 2018/Term 2/Data Mining/Comp. Task 2/EL_df.csv')
EL_y=pd.DataFrame(EL_df.Number)
EL_x=pd.DataFrame(EL_df.loc[:,'Q1':'Q12'])
EL_y.Number.replace(['P'], 1,inplace=True)
EL_y.Number.replace(['O'], 0,inplace=True)
DT_EL=DecisionTreeClassifier(criterion = "entropy", random_state = 100)
Perceptron = PCP()
Fisher = LDA()
knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
print(EL_y)
DT_EL.fit(EL_x, EL_y)
knn1.fit(X_train,y_train)
knn3.fit(X_train,y_train)
Perceptron.fit(EL_x, EL_y)
Fisher.fit(EL_x, EL_y)
XX=[[1,1,0,1,0,1,1,1,1,1,1,0]]
Election_DT = DT_EL.predict(XX)
Election_knn1 = knn1.predict(XX)
Election_knn3 = knn1.predict(XX)
Election_Perceptron = Perceptron.predict(XX)
Election_PFisher =Fisher.predict(XX)
print(Election_DT)
print(Election_knn1)
print(Election_knn3)
print(Election_Perceptron)
print(Election_PFisher)

#answer question 

