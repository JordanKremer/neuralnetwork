#pragma once
#include <vector>
#include "perceptron.h"
#include <boost/iterator/counting_iterator.hpp>
#include <string>
class dataManager
{
private:
	std::vector<perceptron> outputLayer;
	std::vector<perceptron> hiddenLayer;
	std::vector<int> confusionMatrix;
	std::vector<double> trainingRepresentation;
	std::vector<double> testRepresentation;
	std::vector<double> fractionData;

	std::vector<std::vector<double>> trainingData;
	std::vector<std::vector<double>> testData;

	int colCount;
	int rowCount;
	float learningRate;
	float momentum;

	bool loadData(std::string inFile, std::vector<std::vector<double>> &data, int rowCount);
	void setBias(std::vector<std::vector<double>> &data, std::vector<double> &repHolder, int bias);


public:
	dataManager(int outputCount, int hiddenCount, int trainingRep, int cCount, int rCount):colCount(cCount), rowCount(rCount)
	{
		//outputLayer(boost::counting_iterator<int>(0), boost::counting_iterator<int>(10));
		//std::vector<perceptron> v(boost::counting_iterator<int>(0), boost::counting_iterator<int>(10));
		
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

	bool loadWrapper(std::string inFile_training, std::string inFile_test);
	bool saveData(std::string outFile);
	void setBiasWrapper(int bias);

	//~dataManager();
};

