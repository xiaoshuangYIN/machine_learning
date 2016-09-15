from random import choice
import numpy

## list file names
test_label_file = "/Users/Sharon/Desktop/2016_fall/machine-learning/hw1/data_set/mnist/mnist_test_labels.txt"

test_sample_file = "/Users/Sharon/Desktop/2016_fall/machine-learning/hw1/data_set/mnist/mnist_test.txt"

train_label_file = "/Users/Sharon/Desktop/2016_fall/machine-learning/hw1/data_set/mnist/mnist_train_labels.txt"

train_sample_file = "/Users/Sharon/Desktop/2016_fall/machine-learning/hw1/data_set/mnist/mnist_train.txt"

## functions
'''
function: get the index of 1 and 0
arguments: label list
return index_tuple
''' 
def get_index_tuple(label_list):
    index_list = [] 
    i = 0
    for label in label_list:
        if label == '1\n':
            index_list = index_list + [i,]
        elif label == '0\n':
            index_list = index_list + [i,]
        i = i + 1
    return index_list

'''
function: fill sample tuple
arguments: index_tuple, samples
return sample_tuple
'''
def fill_sample_list(index_tuple, samples):
    sample_list= []
    for i in index_tuple:
        result = [int(j) for j in samples[i].split()]
        sample_list = sample_list + [result,]
    return sample_list
'''
function: fill label tuple
arguments: index_tuple, labels
return label_tuple
'''
def fill_label_list(index_list, labels):
    label_list = []
    for i in index_list:
        if labels[i][0] == '0':
            label_list = label_list+ [-1,]
        elif labels[i][0] == '1':
            label_list = label_list + [1,]
    return label_list

'''
function: reduce features
arguments: train+ test sample_list, total sample_size, total feature_num, train sample size 
return new train sample list, new test sample list, number of collumn deleted
'''
def reduce_features(sample_list, num):
    
    del_indexes = []
    arr = numpy.array(sample_list)
    std_array = numpy.std(arr, axis = 0)
    for j in xrange(len(std_array)):
        if std_array[j] <= num:
            del_indexes += [j,]
    new_arr = numpy.delete(arr, numpy.array(del_indexes), axis = 1)
    return new_arr.tolist(), del_indexes  


'''
function: reduce features in the test sample according to deleted-column-index
arguments: test_sample_list, column_indexes
return: featuer-reduced test sample list
'''
def reduce_test_sample_feature(test_sample_list, column_indexes):
   arr = numpy.array(test_sample_list)
   new_arr = numpy.delete(arr, numpy.array(column_indexes), axis = 1)
   return new_arr.tolist()

'''
function: train
arguments:train_sample_tuple, train_label_tuple, num_sample, num_feature
return : w
'''
def train(train_sample_list, train_label_list):
    num_feature = len(train_sample_list[0])
    num_sample = len(train_sample_list)
    w = [0,] * num_feature
    w_list = []
    convergent = False
    while not convergent:
        global_error = 0
        for i in xrange(num_sample):
            x = train_sample_list[i]
            y = train_label_list[i]
            predicted_label = numpy.dot(w,x)
            result = numpy.dot(y,predicted_label)
        
            if result <= 0:
                w = w + numpy.dot(y,x)
                w_list += [w,]
                global_error += 1 

        if global_error == 0:
            convergent = True
    return w
'''
function: test
arguments:
return error/i
'''
def test(w, test_sample_list, test_label_list):
    num_test_sample = len(test_sample_list)
    error = 0
    
    for i in xrange(num_test_sample):
        x = test_sample_list[i]
        y = test_label_list[i]
        predicted_label = numpy.dot(w,x)
        result = numpy.dot(y,predicted_label)
        if result <= 0:
            error += 1
        
    print "error = ", error
    print "total case = ", num_test_sample
    return float(float(error)/float(num_test_sample))

## read files
test_label = (open(test_label_file, "r")).readlines()
test_sample = (open(test_sample_file, "r")).readlines()
train_label = (open(train_label_file, "r")).readlines()
train_sample = (open(train_sample_file, "r")).readlines()

## get index_tuple of '0' and '1'
train_index_list = get_index_tuple(train_label)
test_index_list = get_index_tuple(test_label)

## get sample and label tuples
train_sample_list = fill_sample_list(train_index_list, train_sample)
test_sample_list = fill_sample_list(test_index_list, test_sample)
train_label_list = fill_label_list(train_index_list, train_label)
test_label_list = fill_label_list(test_index_list, test_label)

## reduce featues of train data+test data
std = 40
print "standatd deviation threshold: ", std
less_train_sample,column_indexes = reduce_features(train_sample_list, std)
print len(column_indexes),"featuers reduced"
'''
## without feature reduce
w = train(train_sample_list, train_label_list)
misc_rate = test(w, test_sample_list, test_label_list)
print misc_rate
'''
## train
w_fr = train(less_train_sample, train_label_list)

## test
less_test_sample = reduce_test_sample_feature(test_sample_list, column_indexes)
misc_rate = test(w_fr,less_test_sample, test_label_list)
print "missclassification rate: ", misc_rate
print "accuracy : ", 1 - misc_rate
