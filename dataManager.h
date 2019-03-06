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

	std::vector<double> trainingData;
	std::vector<double> testData;

	int colCount;
	int rowCount;
	float learningRate;
	float momentum;

	//can use for test vector and training vector, and confusion matrix -- we are using a 
	//variable cCount so we can pass it 785 or 10 depending
	int getDataIndex(std::vector<perceptron> &vec, int row, int col, int cCount);

public:
	dataManager(int outputCount, int hiddenCount, int trainingRep, int cCount, int rCount):colCount(cCount), rowCount(rCount)
	{
		//outputLayer(boost::counting_iterator<int>(0), boost::counting_iterator<int>(10));
		//std::vector<perceptron> v(boost::counting_iterator<int>(0), boost::counting_iterator<int>(10));
		
		//parallelize this 
		for (int perc = 0; perc < outputCount; ++perc)
		{
			perceptron p(perc, colCount);
			outputLayer.push_back(p);
		}
		for(int perc = 0; perc < hiddenCount; ++perc)
		{
			perceptron p(perc, colCount);
			hiddenLayer.push_back(p);
		}
		
	}

	bool loadData(std::string inFile);
	bool saveData(std::string outFile);
	

	~dataManager();
};

