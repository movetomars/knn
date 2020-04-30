#!/usr/bin/env python

## Example of execution: knn.py ../data/train.dat ../data/test.dat

import sys, logging, os
from optparse import OptionParser
import collections

## Reads corpus and creates the appropiate data structures:
def read_corpus(file_name):
    f = open(file_name, 'r')

    ## first line contains the list of attributes
    attr = {}
    ind = 0
    for att in f.readline().strip().split("\t"):
        attr[att] = {'ind': int(ind)}
        ind += 1

    ## the rest of the file contains the instances
    instances = []
    ind = 0
    for inst in f.readlines():
        inst = inst.strip()

        elems = inst.split("\t")
        if len(elems) < 3: continue

        instances.append({'values': [int(elem) for elem in elems[0:-1]],
                          'class': int(elems[-1]),
                          'index': int(ind),
                          })
        ind += 1

    return attr, instances


def get_prediction(inst, instances):
    # Initializing list of instances
    closest_instances = []

    # Looping through all instances
    for instance in instances:
        distance = 0

        # Aggregating distance
        for value_0, value_1 in zip(inst['values'], instance['values']):
            if value_0 != value_1:
                distance += 1

        # For now, adding the class and distance of each instance to a list
        closest_instances.append((instance['class'], distance))

    # Sorting all instances by distance values and isolating a subset of the closest 3
    closest_instances.sort(key=lambda x: float(x[1]))
    nearest_k = closest_instances[0:2]

    # Creating a Counter object that can yield the majority class
    c = collections.Counter(item[0] for item in nearest_k)

    return max(c)


def calculate_accuracy(instances, predictions):
    predictions_ok = 0
    for i in range(len(instances)):
        if instances[i]["class"] == predictions[i]:
            predictions_ok += 1

    return 100 * predictions_ok / float(len(instances))


if __name__ == '__main__':
    usage = "usage: %prog [options] TRAINING_FILE TEST_FILE"

    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--debug", action='store_true',
                      help="Turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Incorrect number of arguments")
    if not os.path.isfile(args[0]):
        parser.error("Training file does not exist\n\t%s" % args[0])
    if not os.path.isfile(args[1]):
        parser.error("Training file does not exist\n\t%s" % args[1])

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    file_tr = args[0]
    file_te = args[1]
    logging.info("Training: " + file_tr)
    logging.info("Testing: " + file_te)

    ## (I)  Training: read instances
    attr_tr, instances_tr = read_corpus(file_tr)

    ## (II) Testing: read instances and
    ##      predict the class of the closest instance in training
    attr_te, instances_te = read_corpus(file_te)
    predictions = []
    ## for each test instance
    for i_te in instances_te:
        ## get the closest one and store the prediction
        prediction = get_prediction(i_te, instances_tr)
        predictions.append(prediction)

    if options.debug:
        print(predictions)

    accuracy_te = calculate_accuracy(instances_te, predictions)
    print(f"Accuracy on test set ({len(instances_te)}  instances): {accuracy_te:.2f}")
