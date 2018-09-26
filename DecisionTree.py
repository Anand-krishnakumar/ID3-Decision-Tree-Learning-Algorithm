from __future__ import print_function
import  math

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

#######
# unique_vals(training_data, 0)
# unique_vals(training_data, 1)
#######


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

#######
# class_counts(training_data)
#######




def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

#######
# is_numeric(7)
# is_numeric("Red")
#######


class Question:
    """A Question is used to partition a dataset.
    """

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))


def partition(rows, question):
    

    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def entropy(rows):
    counts = class_counts(rows)
    impurity = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity += -prob_of_lbl*math.log(prob_of_lbl,2)
    return impurity


def info_gain(left, right, current_uncertainty):
   
    p = float(len(left)) / (len(left) + len(right))

   
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_best_split(rows, header):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val, header)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

  
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.
    """

    def __init__(self, rows, id, depth):
        self.predictions = class_counts(rows)
        self.id = id
        self.depth = depth



class Decision_Node:
    

    def __init__(self,
                 question,
                 true_branch,
                 false_branch,
                 depth,
                 id,
                 rows,
                 pruned=0):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth  = depth
        self.id = id
        self.rows = rows





def build_tree(rows, header, depth=0, id=0):
   
   
    gain, question = find_best_split(rows, header)

   
    if gain == 0:
        return Leaf(rows, id, depth)

    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, header, depth + 1, 2 * id + 2 )

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, header, depth + 1, 2 * id + 1)

    return Decision_Node(question, true_branch, false_branch, depth, id, rows)

def prune_tree(node, prunedList):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node
    # If we reach a pruned node, make that node a leaf node and return. Since it becomes a leaf node, the nodes
    # below it are automatically not considered
    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth)

    # Call this function recursively on the true branch
    node.true_branch = prune_tree(node.true_branch, prunedList)

    # Call this function recursively on the false branch
    node.false_branch = prune_tree(node.false_branch, prunedList)

    return node


def classify(row, node):


    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_tree(node, spacing=""):


    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + 'Node ID: '+str(node.id))
    print (spacing + 'Node Depth: '+str(node.depth))
    # Print the question at this node
    print (spacing + str(node.question))
    

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

   



def print_leaf(counts):
    
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs





def getLeafNodes(node, leafNodes =[]):

    # Returns a list of all leaf nodes of a tree
    if node:
        if isinstance(node, Leaf):
            leafNodes.append(node)
        else:
            getLeafNodes(node.true_branch, leafNodes)
            getLeafNodes(node.false_branch, leafNodes)   

    return leafNodes


def getInnerNodes(node, innerNodes =[]):
    if node:
        if not isinstance(node, Leaf):
            innerNodes.append(node)
            getInnerNodes(node.true_branch, innerNodes)
            getInnerNodes(node.false_branch, innerNodes)

    # Returns a list of all non-leaf nodes of a tree


    return innerNodes



def computeAccuracy(rows, node):

    totalRows = len(rows)
    numAccurate = 0
    for row in rows :
        prediction = classify(row, node)
        if row[-1] in prediction:
            numAccurate = numAccurate + 1
        
    return round(numAccurate/totalRows , 2)


