import os
import sys
import pandas as pd
from dataset import make_dataframe
import numpy as np
import tensorflow as tf

def loadData():
    datafile = "data.npy"
    labelsfile = "labels.npy"
    if (os.path.exists(datafile)) and (os.path.exists(labelsfile)):
        data = np.load(datafile)
        labels = np.load(labelsfile)
    else:
        frame = make_dataframe()
        print(frame.head())
        labels = np.array(frame[['class_edible', 'class_poisonous']])
        del(frame['class_edible'])
        del(frame['class_poisonous'])
        data = np.array(frame)
        np.save(datafile, data)
        np.save(labelsfile, labels)

    print(len(data), len(labels))
    print(labels[0:2])
    print(data[0:2])
    return (data, labels)

class DataFeeder:
    def __init__(self, theData, theLabels, batchSize):
        if len(theData)!=len(theLabels):
            print("Data size (%d) does not match label size (%d)" %(len(theData), len(theLabels)))
            sys.exit(1)
        self.data = theData
        self.labels = theLabels
        self.start = 0
        self.end = 0
        self.batchsize = batchSize
        self.reset()
    
    def moveforward(self):
        self.start = min(self.start+self.batchsize, len(self.data))
        self.end = min(self.end + self.batchsize, len(self.data))
        #print("Start %d, end %d"%(self.start, self.end))

    def getbatch(self):
        return (self.data[self.start:self.end], self.labels[self.start:self.end])
    
    def hasmoredata(self):
        return not (self.start == self.end)

    def reset(self):
        self.start = 0
        self.end = min(self.batchsize, len(self.data))

class Network:
    def __init__(self, theData, theLabels, batchsize, layers):
        dataitems = len(theData)
        self.data = theData
        self.labels = theLabels
        self.layerSizes = [len(self.data[0]), len(self.labels[0])]
        self.layerSizes[1:1] = layers
        print("Will create Layers", self.layerSizes)
        #split data into training and testing
        trainsplit = int(dataitems * 0.75)
        print("Training/Testing Split %s"%(trainsplit))
        self.trainDS = {
            "data" : self.data[0:trainsplit],
            "labels":  self.labels[0:trainsplit]
        }
        self.testDS = {
            "data": self.data[trainsplit:],
            "labels": self.labels[trainsplit:]
        }
        print("Testing Data Size %s"%(len(self.testDS['data'])))
        self.traindf = DataFeeder(self.trainDS['data'], self.trainDS['labels'], batchsize)
        self.testdf = DataFeeder(self.testDS['data'], self.testDS['labels'], batchsize)
        self.input = None
        self.expected_output = None
        self.layers = []
        self.finaloutput = None
        self.loss = None
        self.optimizer = None
        self.session = None
        self.init = None 
        self.saver = None 

    def get_train_feeder(self):
        (td, tl) = self.traindf.getbatch()
        return {self.input: td, self.expected_output: tl}

    def get_test_feeder(self):
        (td, tl) = self.testdf.getbatch()
        return {self.input: td, self.expected_output: tl}

    def BuildNetwork(self):
        # build input
        self.input = tf.placeholder(tf.float32, shape=[None, self.layerSizes[0]], name="Inputs")
        self.expected_output = tf.placeholder(tf.float32, shape=[None, self.layerSizes[-1]], name="ExpectedOutput")
        #now start appending the interim layers
        for ls in range(1, len(self.layerSizes)-1):
            self.layers.append({
                'weight': tf.Variable(tf.random_normal([self.layerSizes[ls-1], self.layerSizes[ls]]), name="L%sW"%(ls)),
                'bias' : tf.Variable(tf.random_normal([self.layerSizes[ls]]), name="L%sB"%(ls)),
                'output' : None,
            })
        # need the final output layer as well
        self.layers.append({
            'weight': tf.Variable(tf.random_normal([self.layerSizes[-2], self.layerSizes[-1]], name="LastLayerW")),
            'bias': tf.Variable(tf.random_normal([self.layerSizes[-1]]), name="LastLayerB"),
            'output': None
        })

        for ls in range(0, len(self.layers)):
            if ls==0:
                lprev = None
                prevOutput = self.input
            else:
                lprev = self.layers[ls-1]
                prevOutput = lprev['output']
            lcur = self.layers[ls]
            lcur['output'] = tf.nn.relu(tf.add(tf.matmul(prevOutput, lcur['weight']), lcur['bias']))
        self.finaloutput = self.layers[-1]['output']
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.finaloutput, labels=self.expected_output))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.init = tf.global_variables_initializer()
        vars = []
        for layer in self.layers:
            vars.append(layer['weight'])
            vars.append(layer['bias'])
        self.session = tf.Session()
        self.session.run(self.init, self.get_train_feeder())
        self.saver = tf.train.Saver(vars)


    def train(self, maxiter):
        self.traindf.reset()
        for i in range(0, maxiter):
            eloss = 0
            while(self.traindf.hasmoredata()):
                (_, loss) = self.session.run([self.optimizer, self.loss], self.get_train_feeder())
#                print("loss %s"%(loss))
                eloss = eloss + loss
                self.traindf.moveforward()
            self.traindf.reset()
            print("Epoch %d Loss %s" % (i, eloss))
            if eloss < 1:
                self.saver.save(self.session, "./mymodel")
                break

    def test(self):
        self.testdf.reset()
        numpass = 0
        numfail = 0
        while(self.testdf.hasmoredata()):
            (_, labels) = self.testdf.getbatch()
            feeder = self.get_test_feeder()
            output = np.array(self.session.run(self.finaloutput, feeder))
            print(len(output))
            for tl in range(0, len(output)):
                oam = np.argmax(output[tl])
                lam = np.argmax(labels[tl])
                if oam==lam:
                    numpass = numpass+1
                else:
                    numfail = numfail+1
            self.testdf.moveforward()

        print("Total %d PASS %d FAIL %d - %3.2f"%(numpass+numfail, numpass, numfail, (100.0*numpass/(numpass+numfail))))
        

def main():
    (data, labels) = loadData()
    #nn = Network(data, labels, 1000, [90, 45, 15])
    nn = Network(data, labels, 1000, [40, 40, 12])
    nn.BuildNetwork()
    if len(sys.argv) == 1:
        try:
            nn.train(10000)
        except KeyboardInterrupt:
            print("Interrupted")
        nn.test()
    else:
        print("Loading %s"%(sys.argv[1]))
        nn.saver.restore(nn.session, sys.argv[1])
        nn.test()


if __name__ == "__main__":
    main()