import csv
import random
import numpy as np
from math import floor


class Data(object):

    def __init__(self, classNames=['class'], categoricalVars=[]):
        self.keys = []
        self.instances = []
        self.classNames = classNames
        self.classesSplitted = False

        if isinstance(categoricalVars, list):
            self.categoricalAttr = categoricalVars
        elif isinstance(categoricalVars, str):
            self.categoricalAttr = [categoricalVars]
        else:
            raise ValueError("Attribute 'categorical' should be a list or str")

    def __repr__(self):
        return "<Data {} instances>".format(len(self.instances))

    def addInstance(self, newInstance):

        if len(self.instances) == 0:
            # First instance
            [self.keys.append(x) for x in newInstance.keys()]

        elif self.instances[-1].keys() != newInstance.keys():
            # Validate keys
            raise ValueError("Keys for new instance '{}' don`t match previous instances".format(newInstance))

        self.instances.append(newInstance)

    def listAttributeValues(self, attr):
        """
        Returns a list contaning all the possible value labels for a given
        attribute.
        """
        valueList = []
        for row in self.instances:
            if row[attr] not in valueList:
                valueList.append(row[attr])
        return valueList

    def listClassValues(self):
        """
        Returns a list containing all the possible class values
        """
        values = []
        # print(self.classNames)
        for instance in self.instances:
            for className in self.classNames:
                if instance[className] not in values:
                    values.append(instance[className])

        return values

    def splitClasses(self):
        """
        Splits categorical classes into binary classes for each possible value
        """
        if self.classesSplitted:
            print("Classes have already been splitted, skipping...")
            return
        else:
            self.classesSplitted = True

        initialClasses = self.categoricalAttr.copy()
        values = []
        for className in self.classNames:
            if className in self.categoricalAttr:
                values = self.listAttributeValues(className)
                # print("Class {} has values {}".format(className, values))

                for instance in self.instances:
                    for value in values:
                        if instance[className] == value:
                            instance[className+"_"+value] = 1
                        else:
                            instance[className+"_"+value] = 0

        if len(values) == 0:
            print("ERROR, no categorical classes defined to split")
            return

        for x in values:
            self.classNames.append(x)
            self.categoricalAttr.append(x)
        
        for name in initialClasses:
            self.categoricalAttr.remove(name)
            self.classNames.remove(name)

    def parseFromFile(self, filename, delimiter=',', quotechar='"'):

        reader = csv.DictReader(open(filename, mode='r'), delimiter=delimiter, 
                                quotechar=quotechar)
        for row in reader:
            # row is an OrderedDict by default
            for key in row.keys():
                if key not in self.categoricalAttr:
                    row[key] = float(row[key])

            self.addInstance(dict(row))

    def calculateMean(self, attrName):
        """
        Calculates the mean value for a numeric attribute.
        """
        if not self.isNumeric(attrName):
            raise SystemError("Attribute '{}' is not numeric.".format(attrName))
            
        # Calculate the mean value
        sum = 0
        for entry in self.instances:
            sum += float(entry[attrName])
        mean = sum/float(len(self.instances))
        return mean

    def generateFolds(self, k=1, discardExtras=False):
        '''
        Generates 'k' sets of instances without repetition.
        WARNING: setting 'discardExtras' to False will cause folds to have different sizes if the number of instances isn't divisible by 'k'.
        Returns the list of folds.
        '''

        instancesCopy = self.instances.copy()
        folds = []
        for i in range(k):
            folds.append([])    #New fold

            #Adds 'instances/k' (rounded down) random instances to each fold
            for j in range(floor(len(self.instances)/k)):
                folds[i].append(instancesCopy.pop(random.randint(0, len(instancesCopy)-1)))

        #Evenly distributes the remaining instances. Will cause some folds to have 1 more instance than others if the number of instances isn't divisible by 'k'.
        if discardExtras == False:
            while len(instancesCopy) != 0:
                for i in range(k):
                    if(len(instancesCopy) != 0):
                        folds[i].append(instancesCopy.pop(random.randint(0, len(instancesCopy)-1)))    

        return folds

    # generateStratifiedFolds was removed

    def generateBootstraps(self, k=1):
        '''
        Randomly generates 'k' sets of instances with repetition for the training set and sets of instances that aren't in the training set for the testing set.
        Returns the list of bootstraps.
        '''
        bootstraps = []
        for i in range(k):
            bootstraps.append( ([], []) )   #Adds a new bootstrap. Each bootstrap is a tuple with a list of training instances (index 0) and a list of testing instances (index 1)

            for j in range(len(self.instances)):    #Bootstraps have the same number of instances as the original data set
                bootstraps[i][0].append(self.instances[random.randint(0, len(self.instances)-1)]) #Adds a random instance to the current's bootstrap training list

            for j in range(len(self.instances)): #Goes through every instance again
                if self.instances[j] not in bootstraps[i][0]:    #Checks for instances that weren't picked for the training list
                    bootstraps[i][1].append(self.instances[j])   #Adds the instance to the testing list

        return bootstraps

    # generateStratifiedBootstraps() was removed

    @staticmethod
    def getAttrMatrix(instance):
        values = [[1.0]]
        for key, value in zip(instance.keys(), 
                              instance.values()):
            if 'class' not in key:
                values.append([value])
        return np.matrix(values)

    @staticmethod
    def getResultMatrix(instance):

        isSplitted = False
        for key in instance.keys():
            if 'class_' in key:
                isSplitted = True
                
        if isSplitted:
            substring = 'class_'
        else:
            substring = 'class'

        values = []
        for key, value in zip(instance.keys(), 
                              instance.values()):
            if substring in key:
                values.append([value])
        return np.matrix(values)

    def isEmpty(self):

        if len(self.instances) == 0:
            return True
        else:
            return False

    def isNumeric(self, attrName):
        
        if attrName in self.categoricalAttr:
            return False
        else:
            return True

    def normalize(self, maxValue=1, normalizeClass=False):
        """
        Normalizes all attributes to be a value between 0 an 'maxValue'.
        """
        #Goes through every instance and finds the highest value for each attribute
        maxValues = {}
        minValues = {}
        for key in self.keys:
            #Gets the max and min values of specified attribute
            if not normalizeClass and 'class' in key or\
                    key in self.categoricalAttr:
                continue
            else:
                maxValues[key] = max(instance[key] for instance in self.instances)  
                minValues[key] = min(instance[key] for instance in self.instances)

        for instance in self.instances:
            for key in self.keys:
                if not normalizeClass and 'class' in key or\
                        key in self.categoricalAttr:
                    continue
                else:
                    if maxValues[key] - minValues[key] == 0:
                        instance[key] = 0
                    else:
                        instance[key] = ((instance[key] - minValues[key])/(maxValues[key] - minValues[key])) * maxValue
