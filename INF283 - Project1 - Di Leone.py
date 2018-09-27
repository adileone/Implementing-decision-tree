
# coding: utf-8

# In[114]:


import numpy as np
from random import shuffle


class DecisionTree:
    
    #Constructor
    def __init__(self):
        self.featureNames = []
        self.classes = []

    
    def predict(self, tree, data):
        """
        Name: predict
        Parameters: 
        •   self
        •   tree (a dictionary)
        •   data (the whole dataset)
        Return:
        •   results (an array)
        Description:
        This function creates an array with the predictions (one prediction for each row in the data);
        this is possible thanks to the for cycle in which there is the recursion that allows to classify a single datapoint.
        """
        
        if type(tree) == type("string"):
            return tree
        
        results = []
        
        for datapoint in data:
            
            a = list(tree.keys())
            for i in range(len(self.featureNames)):
                if self.featureNames[i] == a:
                    break

            try:
                t = tree[a][datapoint[i]]
                results.append(self.predict(t, datapoint))
            
            except:
                results.append('p')
                
                
        return results 
    
    
    def learn(self, data, classes, feature_names, imp_measure="entropy", pruning=False):
        """
        Name: learn
        Parameters: 
        •   self
        •   data (the whole dataset)
        •   classes (targets)
        •   feature_names (the names of the features used)
        •   imp_measure [default = entropy] (this parameter is added in order to choose between the impurity measures (entropy or gini)
        •   pruning [default=false] (this parameter is added in order to let the user decide if he wants a pruned tree or not) 
        Return:
        •   tree (a dictionary)
        Description:
        The learn function is the most important and it is the function that create the dictionary that we will use as a tree. 
        As required in the exercise, if learn is called without any setting for the imp_measure and for the pruning parameters, it will compute the tree with entropy as impurity measure and it will prune the tree in the end.
        In the first lines, we just calculate the values that are needed during the computation, then we have added some if cycles because we needed to check for the impurity measure to use (as requested in exercise 2) but also because, due to the fact that this is a recursive function, we needed some stopping criteria in order to end the recursion once the computation is completed. 
        After that, we checked again if the impurity measure is calculated as entropy or as gini index in order to obtain the information gain for each feature and to find out which one is the best one for the split. 
        Then, we enumerate the possible values for best feature and, for each possible value, we build a subtree that will be attached to the tree that we already have and then we did the recursion in order to make the construction going.
        In the end, we checked if the pruning parameter is set to True/False to understand if the tree has to be pruned or not.

        """
        dataLength = len(data)
        featuresLength = len(feature_names)
        unique_classes = list(set(classes))
        frequency = np.zeros(len(unique_classes))

        total_disorder = 0
        index = 0
        
        if imp_measure=="entropy":
            for a_class in unique_classes:
                frequency[index] = classes.count(a_class)
                total_disorder += self.calc_entropy(float(frequency[index])/dataLength)
                index += 1
                
        elif imp_measure=="gini":
            for a_class in unique_classes:
                frequency[index] = classes.count(a_class)
                total_disorder += (float(frequency[index])/dataLength)
                index += 1

        default = unique_classes[int(np.argmax(frequency, axis=0))]

        # If less than 90% sure about edible, always pick poisonous
        if len(unique_classes) == 2:
            if (frequency[unique_classes.index('e')] / dataLength < 0.90):
                default = 'p'

        if dataLength == 0:
            return default
        
        elif featuresLength == 0:
            return defalut
        
        elif max(frequency) == dataLength:
            return default
        
        else:
            
            gain = np.zeros(featuresLength)
            for feat in range(featuresLength):
                feature_gain = self.informationGainGini(data, classes, feat)
                gain[feat] = total_disorder - feature_gain
            
            if (imp_measure=="entropy"):
                for feature in range(len(feature_names)):
                    feature_gain = self.informationGainEntropy(data, classes, feature)
                    gain[feature] = total_disorder - feature_gain
                    
            elif (imp_measure=="gini"):
                for feature in range(len(feature_names)):
                    feature_gain = self.informationGainGini(data, classes, feature)
                    gain[feature] = total_disorder - feature_gain
                    
            best_feature = np.argmax(gain)
            tree = {feature_names[best_feature]: {}}

            possible_values = []
            
            for data_point in data:
                value = data_point[best_feature]
                if value not in possible_values:
                    possible_values.append(value)

            for value in possible_values:
                subtree_data = []
                subtree_classes = []
                subtree_feature_names = feature_names[:best_feature] + feature_names[best_feature+1:]
                index = 0
                
                for data_point in data:
                    if data_point[best_feature] == value:
                        subtree_classes.append(classes[index])
                        subtree_data.append(data_point[:best_feature] + data_point[best_feature+1:])
                        
                    index += 1
    
                subtree = self.learn(subtree_data, subtree_classes, subtree_feature_names, imp_measure, pruning)

                tree[feature_names[best_feature]][value] = subtree
    
            if (pruning == True):
                return DecisionTree.prune(self, tree, data)
            else:
                return tree
        
      
    def informationGainEntropy(self, data, classes, feature):
        """
        Name: information Gain Entropy
        Parameters:
        •	self
        •	data (the whole dataset)
        •	classes (targets)
        •	feature (the feature of which we want to calculate the information gain)
        Return:
        •	gain (a numeric value)
        Description:
        This function is needed to calculate the entropy value of a feature. This function uses 4 support functions that are “valueList”, “findClasses”, “getClassIndex” and “calc_entropy” that will be described later.
        Actually, while doing the first exercise, we did not create the first three functions and every calculation was done inside the information Gain Entropy but then, when we had to do the second exercise,
        in order not to have duplicate code (that means that there are two function that are basically the same except for some lines of code)
        we decided to create these functions that would have been used by both information Gain Entropy and information Gain Gini.
        """
        
        gain = 0
        dataLength = len(data)
        values = DecisionTree.valuesList(data,feature)
        
        featureLength = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0

        for value in values:
            
            new_classes = DecisionTree.findClasses(data,feature,value,featureLength,valueIndex,classes)
            class_values = list(set(new_classes))
            class_counts = np.zeros(len(class_values))
            class_index = DecisionTree.getClassIndex(class_values, class_counts, new_classes)
            
            for class_index in range(len(class_values)):
                entropy[valueIndex] += DecisionTree.calc_entropy(float(class_counts[class_index]) / sum(class_counts))

            gain += float(featureLength[valueIndex]) / dataLength * entropy[valueIndex]
            valueIndex += 1
            
        return gain
    
    
    def informationGainGini(self, data, classes, feature):
        """
        Name: information Gain Gini
        Parameters:
        •	self
        •	data (the whole dataset)
        •	classes (targets)
        •	feature (the feature of which we want to calculate the information gain)
        Return:
        •	final_gain (a numeric value)
        Description:
        In general, Gini index is another way to calculate the impurity of a dataset.
        This function is needed to calculate the gini value of a feature it can be used in substitution of the entropy impurity measure to create the dictionary used as decision tree.
        As said for the information Gain Entropy, this function uses 4 support functions that are “valueList”, “findClasses”, “getClassIndex” and “calc_entropy” that will be described later.
        """

        gain = 0
        dataLength = len(data)
        values = DecisionTree.valuesList(data,feature)

        featureLength = np.zeros(len(values))
        gini = np.zeros(len(values))
        valueIndex = 0

        
        for value in values:
            new_classes = DecisionTree.findClasses(data,feature,value,featureLength,valueIndex,classes)
            class_values = list(set(new_classes))
            classLength = np.zeros(len(class_values))
            class_index = DecisionTree.getClassIndex(class_values, classLength, new_classes)

            for index in range(len(class_values)):
                gini[valueIndex] += (float(classLength[index])/sum(classLength)**2)
                
            gain = gain + (float(featureLength[valueIndex])/dataLength * gini[valueIndex])
            valueIndex += 1
        
        final_gain = (1-gain)

        return final_gain

    
    @staticmethod
    def calc_entropy(p):
        """
        Name: calc entropy
        Parameters:
        •	p (a probability)
        Return:
        •	pp (a numeric value, if the probability is not 0)
        •	0 (otherwise)
        Description:
        This function will do the calculations needed for computing of the entropy.
        """
        if p != 0:
            pp = (-p * np.log2(p))
            return pp
        else:
            return 0

    
    def valuesList(data,feature):
        """
        Name: valuesList
        Parameters:
        •	data (the whole dataset)
        •	feature (the feature of which we want to calculate the information gain)
        Return:
        •	values (an array of strings)
        Description:
        Creates an array with all the values that a feature can assume.
        """

        values = []
        for data_point in data:
            if data_point[feature] not in values:
                values.append(data_point[feature])
        return values
    
    
    def findClasses(data,feature,value,featureLength,valueIndex,classes):
        """
        Name: findClasses
        Parameters:
        •	data (the whole dataset)
        •	feature (the feature of which we want to calculate the information gain)
        •	value (is the index used in the for cycle in the information gain method)
        •	featureLength (is the length of the target array)
        •	valueIndex (is an index used to get a position of featureLength)
        •	classes (targets)
        Return:
        •	new classes (an array of strings)
        Description:
        For each value that the feature can assume, the function looks for where those values appear in the data and the corresponding class.

        """
        new_classes = []
        data_index = 0
        for data_point in data:
            if data_point[feature] == value:
                featureLength[valueIndex] += 1
                new_classes.append(classes[data_index])
            data_index += 1
        
        return new_classes
    
    
    def getClassIndex(class_values, class_counts, new_classes):
        """
        Name: getClassIndex
        Parameters:
        •	class_values (a list obtained by removing duplicates from new_classes)
        •	class_counts (an array of 0s that has the same length of class_values)
        •	new_classes (an array obtained from the previous method)
        Return:
        •	class_index (an integer value)
        Description:
        This function returns an index that will be used to calculate the entropy of the feature that the information gain method is considering.
        """
        class_index = 0
        for class_value in class_values:
                for a_class in new_classes:
                    if a_class == class_value:
                        class_counts[class_index] += 1
                class_index += 1
        return class_index
    
    
    def prune(self, tree, test_data): 
        """
        Name: prune 
        Parameters:
        •	self
        •	tree (a dictionary)
        •	test_data (a partition of the whole dataset)
        Return/Description:
        This function does not have a proper return,
        it just calls for another method that will compare the accuracy of the three trees;
        in order to do that we created four other functions that allowed us to create the two new trees.
        """
        result_pruneAllP = DecisionTree.pruneAllP(self, tree)
        treeAllP = result_pruneAllP[0]
        result_pruneAllE = DecisionTree.pruneAllE(self, tree)
        treeAllE = result_pruneAllE[0]
        
        DecisionTree.compare_trees_for_pruning(self, tree, treeP, treeE, test_data)
        
        
    def pruneAllP(self, tree):
        """
        Name: pruneAllP
        Parameters:
        •	self
        •	tree (a dictionary)
        Return:
        •	(tree, list_leaves) (a dictionary, a list of pairs)
        Description:
        Given the original tree, this function will call for pruneListSubstitution that has “e” as the label to get a list of leaves and then it will recursively build the treeAllP.
        The list of leaves is returned as well because we needed it for some runs that we have done during the creation of this project but actually,
        to do the pruning, just the returned tree will be used.
        """
        list_leaves = DecisionTree.pruneListSubstitution(self, tree, label = "e")
        
        for key in list(tree.keys()):
            if(tree[key]=="e"):
                tree[key]=="p"
            if not isinstance(tree[key], str):
                DecisionTree.pruneAllP(self,tree[key])
        
        return (tree, list_leaves)
    
    
    def pruneAllE(self, tree):
        """
        Name: pruneAllE
        Parameters:
        •	self
        •	tree (a dictionary)
        Return:
        •	(tree, list_leaves) (a list of pairs)
        Description:
        The description is the same as the one for the pruneAllP function but the label for pruneListSubstitution is “p” and not “e”.
        """
        list_leaves = DecisionTree.pruneListSubstitution(self, tree, label = "p")
        
        for key in list(tree.keys()):
            if(tree[key]=="p"):
                tree[key]=="e"
            if not isinstance(tree[key], str):
                DecisionTree.pruneAllE(self,tree[key])
        
        return (tree, list_leaves)

    
    def pruneListSubstitution(self, tree, label):
        """
        Name: pruneListSubstitution
        Parameters:
        •	self
        •	tree (a dictionary)
        •	label (a string)
        Return:
        •	lista (a list)
        Description:
        This function substitutes “p”s with “e”s if the label is “p” and vice versa if the label is “e”. 
        """
        keysAnDvalues = DecisionTree.key_values(self, tree,[]) 
        
        lista = []
        if(label == "p"):
            for (elemento1,elemento2) in keysAnDvalues:
                if (elemento2=="p"):
                    lista.append((elemento1,"e"))
                    
        elif (label == "e"):
            for (elemento1,elemento2) in keysAnDvalues:
                if (elemento2=="e"):
                    lista.append((elemento1,"p"))
        
        return(lista) 
    
    
    def key_values(self, tree, keysAnDvaluesPar): 
        """
        Name: key values
        Parameters:
        •	self
        •	tree (a dictionary)
        •	keysAnDvaluesPar (a list of pairs)
        Return:
        •	keysAnDvalues (a list of pairs)
        Description:
        This function will create a list of pairs where the first element of the pair is the key of the dictionary and the second element is its value;
        at the beginning of this list we have a pair like (tree root, all the other nodes of the tree) while at its end
        we have pairs like (single node, leaf) and this is why we need to return the reverse list so that at the beginning of the list itself
        we will find pairs that are easy to compute and overall that represent the last level of the tree (the one with only the leaves).
        """
        keysAnDvalues = keysAnDvaluesPar #these are the values of the keys
        for key in list(tree.keys()):
            keysAnDvalues.append((key,tree[key])) # i create a list of pairs (key, value of the key)
            if not (isinstance(tree[key], str)):
                DecisionTree.key_values(self, tree[key], keysAnDvalues)
        
            else:
                keysAnDvalues.reverse()
        
        return keysAnDvalues
    
    
    def compare_trees_for_pruning(self, tree, treeAllP, treeAllE, test_data):
        """
        Name: compare trees for pruning
        Parameters:
        •	self
        •	tree (a dictionary)
        •	treeAllP (a dictionary)
        •	treeAllE (a dictionary)
        •	test_data (a partition of the whole data)
        Description:
        This function compares the three trees mentioned above and their accuracy values.
        At the beginning, some if cycles are used to check if one (or all) of the dictionaries is empty because the tree have to have the same size;
        then we compared the accuracy values to define which one is the most accurate so that we can use that specific tree to keep going with the pruning.
        """
        if (len(tree.keys()) != 0):
            if (len(treeAllP.keys()) != 0):            
                if (len(treeAllE.keys()) != 0):
                    print("ERROR") #the trees have to have the same size
           
        if (acc_tree >= acc_treeAllP):
            if(acc_tree>=acc_treeAllE):
                DecisionTree.prune_definitivo(self, tree, test_data)
                
        if (acc_treeAllP > acc_tree):
            if(acc_treeAllP>acc_treeAllE):
                DecisionTree.prune_definitivo(self, treeAllP, test_data)
                
        if (acc_treeAllE > acc_tree):
            if(acc_treeAllE>acc_treeAllP):
                DecisionTree.prune_definitivo(self, treeAllE,test_data)
    
    
    def accuracy(self, predictions, test_data):
        """
        Name: accuracy
        Parameters:
        •	self
        •	predictions (the values obtained by performing the algorithm)
        •	test_data (a partition of the whole dataset)
        Return:
        •	accuracy (a numeric value)
        Description:
        This function calculates the value of the accuracy. It takes in input the tree that is created by the learn function and, using the test_data, it gives the percentage of the right predicted values.
        The accuracy function is based on the actual number of the correctly classified and misclassified samples that will be used to calculate the percentage itself.
        """
            
        num_correct = 0
        false_positives = 0

        for index in range(len(test_targets)):
            if predictions[index] == test_targets[index]:
                num_correct += 1
            if predictions[index] == 'e':
                if test_targets[index] == 'p':
                    false_positives += 1
        
        accuracy = (num_correct*100)/(len(test_data)) 
        
        return accuracy

        
    def print_tree(tree, name):
        """
        Name: print tree
        Parameters:
        •	tree (a dictionary)
        •	name
        Description:
        This function prints the tree given as a parameter
        """
        
        if type(tree) == dict:
            print (name, tree.keys()[0])
            
            for item in (tree.values()[0].keys()):
                print (name,  item)
                DecisionTree.print_tree(tree.values()[0][item], name)
        
        else:
            print (name)
            
    
    def read_data(self, filename):
        """
        Name: read data
        Parameters:
        •	self
        •	filename (a string that has the same name of the dataset)
        Return:
        •	(data, self.classes, self.featureNames) (the whole dataset, targets, feature names)
        Description:
        This function allows to read the file with the dataset and it returns the whole dataset, the targets and the names that the features can assume.
        """
        fid = open(filename, "r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            # if "?" not in d1: --- could have chosen to not include the missing values
            data.append(d1.split(","))
        fid.close()

        shuffle(data)

        self.featureNames = ['Cap-shape', 'Cap-surface', 'Cap-color', 'Bruises', 'Gill-attachment',
                             'Gill-spacing', 'Gill-size','Gill-color', 'Stalk-shape', 'Stalk-root',
                             'Stalk-surface-above-ring', 'Stalk-surface-below-ring','Stalk-color-above-ring',
                             'Stalk-color-below-ring', 'Veil-type', 'Veil-color', 'Ring-number','Ring-type',
                             'Spore-print-color', 'Population', 'Habitat']
        
        self.classes = []
        
        for d in range(len(data)):
            self.classes.append(data[d][0])
            data[d] = data[d][1:]

        return data, self.classes, self.featureNames
        

if __name__ == '__main__':
    t = DecisionTree()
    all_data, targets, feat_names = t.read_data("mush.data")

    # Split the data into training data (67%) and test data (33%)
    training_data = all_data[:int(len(all_data)*0.67)]
    training_targets = targets[:int(len(targets)*0.67)]
    test_data = all_data[int(len(all_data)*0.33):]
    test_targets = targets[int(len(targets)*0.33):]


    #Ex 5
    from sklearn import tree
    from sklearn import preprocessing
    from sklearn.metrics import accuracy_score

    
    training_data_encoded=[]
    
    for item in training_data:
        le = preprocessing.LabelEncoder()
        le.fit(item)
        training_data_encoded.append(le.transform(item))
    
    test_data_encoded=[]

    for item1 in test_data:
        le1 = preprocessing.LabelEncoder()
        le1.fit(item1)
        test_data_encoded.append(le1.transform(item1))

    clf = tree.DecisionTreeClassifier()
    clf.fit(training_data_encoded, training_targets)
    predicted_values_clf=clf.predict(test_data_encoded)

    tree = t.learn(training_data, training_targets, feat_names)
    predictions = t.predict(tree,test_data)
    
    
    print("Decision tree: \n")
    t.print_tree(tree)
    
    print("\n Accuracy of the classifier that we implemented (using entropy as default): ", 
            t.accuracy(predictions,test_data))
          
    print("\n Accuracy of the sklearn classifier:" , 
          (accuracy_score(test_targets, predicted_values_clf))*100)


# In[119]:


#Comparing between entropy//gini

tree_e = t.learn(training_data, training_targets, feat_names)
tree_g = t.learn(training_data, training_targets, feat_names, imp_measure="gini")

predictions_e = t.predict(tree_e,test_data)
predictions_g = t.predict(tree_g,test_data)

print(t.accuracy(predictions_e,test_data))

print(t.accuracy(predictions_g,test_data))

