from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import random
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
header = ['Sample code number','Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
df = pd.read_csv('http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = ['Sample code number','Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])
#df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
lst = df.values.tolist()

t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
leaf_depth = {}
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
    leaf_depth[leaf.depth] = 0
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
innerNodeID = []
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
    innerNodeID.append(inner.id)


trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy

# list containing all the Inner node IDs
print("innerNodeID "+str(innerNodeID))
#selecting 5 random node IDs to prune 
random_samples = random.sample(innerNodeID, 5)
print("random_samples "+str(random_samples))
t_pruned = prune_tree(t, random_samples)

print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))
