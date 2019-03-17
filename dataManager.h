#pragma once
#include <vector>
#include "perceptron.h"
#include <boost/iterator/counting_iterator.hpp>
#include <string>
#include <utility>
#include <math.h>


class dataManager
{
private:
	std::vector<perceptron> outputLayer;
	std::vector<perceptron> hiddenLayer;
	
	std::vector<double> trainingRepresentation;
	std::vector<double> testRepresentation;
	std::vector<double> fractionData;

	std::vector<int> confusionMatrix;

	std::vector<std::vector<double>> trainingData;
	std::vector<std::vector<double>> testData;

	int bias;
	int colCount;
	int rowCount;
	float learningRate;
	float momentum;
	std::pair<int, int> trainingDim;
	std::pair<int, int> testingDim;

	//make pair for row cols for data and test?
	 
	bool loadData(std::string inFile, std::vector<std::vector<double>> &data, std::vector<double> &repHolder, int rowCount, int colCount);
	//void setBias(std::vector<std::vector<double>> &data, std::vector<double> &repHolder, int bias);
	double determineTarget(int outputIndex, int rowRep)
	{
		if (rowRep == outputIndex)
			return 0.9;
		else
			return 0.1;
	}

public:
	dataManager(int outputCount, int hiddenCount, int trainingRep, int cCount, int rCount, int b):colCount(cCount), 
		rowCount(rCount), bias(b) //fix this
	{
		learningRate = 0.1;
		momentum = 0.9;

		//parallelize this 
		for (int perc = 0; perc < outputCount; ++perc)
		{
			//The outer layer only has the hidden layers connecting to them
			perceptron p(perc, hiddenCount);
			outputLayer.push_back(p);
		}
		for(int perc = 0; perc < hiddenCount; ++perc)
		{
			//The hidden layer is connected to all inputs
			perceptron p(perc, colCount);
			hiddenLayer.push_back(p);
		}
		
	}

	double computeActivation(double sum)
	{
		//exp() returns e^-sum
		return 1 / (1 + exp(-sum));
	}
	bool loadWrapper(std::string inFile_training, std::string inFile_test);
	bool saveData(std::string outFile);
	std::vector<double> getHiddenActivations();
	void learn();
	void calculateActivation(std::vector<perceptron> &node, std::vector<double> &inputData, int offset);
	std::vector<double> getTestingActivations(std::vector<perceptron> & node, std::vector<double> &inputData, int offset);
	void calculateOutputError(double rowRepresentation);
	void calculateHiddenError();
	void updateWeights(int learningRate, int momentum, std::vector<double> data);
	void testWrapper();
	int test(std::vector<double> &testRow);
};

