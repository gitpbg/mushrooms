import os
import sys
import pandas as pd
from dataset import make_dataframe
import numpy as np
import tensorflow as tf

def loadData():
    datafile = "data.npy"
    if os.path.exists(datafile):
        data = np.load(datafile)
#        labels = np.load(labelsfile)
    else:
        frame = make_dataframe()
        print(frame.head())
#        labels = np.array(frame[['class_edible', 'class_poisonous']])
#        del(frame['class_edible'])
#        del(frame['class_poisonous'])
        data = np.array(frame)
        np.save(datafile, data)
 #       np.save(labelsfile, labels)
    np.random.shuffle(data)
    labels = data[:, 0:2]
    data = data[:, 2:]
    print(len(data), len(labels))
    print(labels[0:2])
    print(data[0:2])
    return (data, labels)

class DataFeeder:
    """Data Feeder class, chop the data into batches"""
    def __init__(self, theData, theLabels, batchSize):
        if len(theData) != len(theLabels):
            print("Data size (%d) does not match label size (%d)" %(len(theData), len(theLabels)))
            sys.exit(1)
        self.data = theData
        self.labels = theLabels
        self.start = 0
        self.end = 0
        self.batchsize = batchSize
        self.reset()

    def moveforward(self):
        """move to the next batch"""
        self.start = min(self.start+self.batchsize, len(self.data))
        self.end = min(self.end + self.batchsize, len(self.data))
        #print("Start %d, end %d"%(self.start, self.end))

    def getbatch(self):
        """return a tuple for the current batch's data and label set"""
        return (self.data[self.start:self.end], self.labels[self.start:self.end])

    def hasmoredata(self):
        """predicate to indicate if there are more batches in the dataset"""
        return not (self.start == self.end)

    def reset(self):
        """reset the feeder to start from the beginning"""
        self.start = 0
        self.end = min(self.batchsize, len(self.data))

class Network:
    """Class to create the tensorflow neural network"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self, theData, theLabels, batchsize, layers):
        dataitems = len(theData)
        self.data = theData
        self.labels = theLabels
        self.layer_sizes = [len(self.data[0]), len(self.labels[0])]
        self.layer_sizes[1:1] = layers
        print("Will create Layers", self.layer_sizes)
        #split data into training and testing
        trainsplit = int(dataitems * 0.80)
        print("Training/Testing Split %s"%(trainsplit))
        self.train_ds = {
            "data" : self.data[0:trainsplit],
            "labels":  self.labels[0:trainsplit]
        }
        self.test_ds = {
            "data": self.data[trainsplit:],
            "labels": self.labels[trainsplit:]
        }
        print("Testing Data Size %s"%(len(self.test_ds['data'])))
        self.traindf = DataFeeder(self.train_ds['data'], self.train_ds['labels'], batchsize)
        self.testdf = DataFeeder(self.test_ds['data'], self.test_ds['labels'], batchsize)
        self.input = None
        self.expected_output = None
        self.layers = []
        self.finaloutput = None
        self.loss = None
        self.optimizer = None
        self.session = None
        self.init = None
        self.saver = None

    def print_info(self):
        """print information about the generated network"""
        print("Layers %s"%(self.layer_sizes))
        print("Input: %s"%(self.input))
        for layer_index in range(0, len(self.layers)):
            layer = self.layers[layer_index]
            print("HiddenLayer %s"%(layer_index))
            print("  Weight %s"%(layer['weight']))
            print("  Bias: %s"%(layer['bias']))
            print("  Output: %s"%(layer['output']))
        print("Output: %s"%(self.expected_output))

    def get_train_feeder(self):
        """get the feeder dictionary for the training set"""
        (td, tl) = self.traindf.getbatch()
        return {self.input: td, self.expected_output: tl}

    def get_test_feeder(self):
        """get the feeder dictionary for the test set"""
        (td, tl) = self.testdf.getbatch()
        return {self.input: td, self.expected_output: tl}

    def BuildNetwork(self):
        """Build the actual network"""
        # build input
        self.input = tf.placeholder(tf.float32, shape=[None, self.layer_sizes[0]], name="Inputs")
        self.expected_output = tf.placeholder(tf.float32, shape=[None, self.layer_sizes[-1]], 
            name="ExpectedOutput")
        #now start appending the interim layers
        for ls in range(1, len(self.layer_sizes)-1):
            self.layers.append({
                'weight': tf.Variable(tf.random_normal([self.layer_sizes[ls-1], self.layer_sizes[ls]]), name="L%sW"%(ls)),
                'bias' : tf.Variable(tf.random_normal([self.layer_sizes[ls]]), name="L%sB"%(ls)),
                'output' : None,
            })
        # need the final output layer as well
        self.layers.append({
            'weight': tf.Variable(tf.random_normal([self.layer_sizes[-2], self.layer_sizes[-1]], name="LastLayerW")),
            'bias': tf.Variable(tf.random_normal([self.layer_sizes[-1]]), name="LastLayerB"),
            'output': None
        })

        for ls in range(0, len(self.layers)):
            if ls==0:
                lprev = None
                prev_output = self.input
            else:
                lprev = self.layers[ls-1]
                prev_output = lprev['output']
            lcur = self.layers[ls]
            lcur['output'] = tf.nn.relu(tf.add(tf.matmul(prev_output, lcur['weight']), lcur['bias']))
        self.finaloutput = self.layers[-1]['output']
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.finaloutput, labels=self.expected_output))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        saver_vars = []
        for layer in self.layers:
            saver_vars.append(layer['weight'])
            saver_vars.append(layer['bias'])
        self.session = tf.Session()
        self.session.run(self.init, self.get_train_feeder())
        self.saver = tf.train.Saver(saver_vars)


    def train(self, maxiter, loss_limit):
        """run the training loops"""
        self.traindf.reset()
        for epoch in range(0, maxiter):
            eloss = 0
            while(self.traindf.hasmoredata()):
                (_, loss) = self.session.run([self.optimizer, self.loss], self.get_train_feeder())
#                print("loss %s"%(loss))
                eloss = eloss + loss
                self.traindf.moveforward()
            self.traindf.reset()
            if epoch % 100==0:
                print("Epoch %d Loss %s" % (epoch, eloss))
            if eloss < loss_limit:
                print("Epoch %d Loss %s" % (epoch, eloss))
                self.saver.save(self.session, "./mymodel")
                break

    def test(self):
        """test the network with the testing dataset"""
        self.testdf.reset()
        numpass = 0
        numfail = 0
        failp = 0
        faile = 0
        while(self.testdf.hasmoredata()):
            (_, labels) = self.testdf.getbatch()
            feeder = self.get_test_feeder()
            loss, output = self.session.run([self.loss, self.finaloutput], feeder)
            output = np.array(output)
            for tl in range(0, len(output)):
                oam = np.argmax(output[tl])
                lam = np.argmax(labels[tl])
                if oam==lam:
                    numpass = numpass+1
                else:
                    print("Loss %s %s %s"%(loss, output[tl], labels[tl]))
                    numfail = numfail+1
                    if oam==1:
                        faile = faile+1
                    else:
                        failp = failp+1
            self.testdf.moveforward()

        print("Total %d PASS %d FAIL %d - %3.2f%%"%(numpass+numfail, 
            numpass, numfail, (100.0*numpass/(numpass+numfail))))
        print("Fail - Edible As Poisonous %d Poisionous as Edible %d"%(faile, failp))

def main():
    """Main entry point"""
    (data, labels) = loadData()
    #nn = Network(data, labels, 1000, [90, 45, 15])
    edible, poisonous = 0, 0
    for i in labels:
        edible = edible + i[0]
        poisonous = poisonous + i[1]
    the_network = Network(data, labels, 1000, [30, 30])
    the_network.BuildNetwork()
    if len(sys.argv) == 1:
        try:
            the_network.train(100000, 1)
        except KeyboardInterrupt:
            print("Interrupted")
        the_network.test()
    else:
        print("Loading %s"%(sys.argv[1]))
        the_network.saver.restore(the_network.session, sys.argv[1])
        the_network.test()
    print("%s edible and %s poisonous mushrooms"%(edible, poisonous))
    the_network.print_info()

if __name__ == "__main__":
    main()