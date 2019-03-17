

//*************
//SPEED IS AT 02, IF THERE ARE ISSUES TURN THIS OFF**************888
//***********

#include "dataManager.h"
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <omp.h>
#include <algorithm>
#include <ppl.h>
#include <execution>
#include <numeric>
#include <iterator>
#include "parser.hpp"
#include <array>
#include <memory>

bool dataManager::loadWrapper(std::string inFile_training, std::string inFile_test)
{
	std::cout << "\nLoading...";
	loadData(inFile_training, trainingData, trainingRepresentation, 60000, 785);
	loadData(inFile_test, testData, testRepresentation, 10000, 785);
	return true;
}


bool dataManager::loadData(std::string inFile, std::vector<std::vector<double>> &data, 
								std::vector<double> &repHolder, int rowCount, int colCount)
{
	//make sure to handle exceptions
	if(inFile.empty())
	{
		return false;
	}
	else
	{
		//Cuts down on resize time.
		data.reserve(rowCount);
		//repHolder.reserve(rowCount);

		std::ifstream input(inFile);
		aria::csv::CsvParser parser(input);
		bool first = true;
		for (auto& row : parser) {
			std::vector<double> tmpRow;
			tmpRow.reserve(colCount);
			for (auto& field : row) {
				if (first)
				{
					//repHolder.push_back(stod(field));
					//first is what the row represents
					tmpRow.push_back(stod(field)); //do not convert to /255 I think
					tmpRow.push_back(bias);
					first = false;
				}
				else
				{
					tmpRow.push_back(stod(field) / 255);
				}
			}
			data.push_back(tmpRow);
			first = true;
		}

		std::srand(unsigned(std::time(0)));
		auto rng = std::default_random_engine{};
		std::shuffle(std::begin(data), std::end(data), rng);

		return true;
	}

}


//remember compilation tags for exception handling, etc
void dataManager::learn()
{
	int rowRep = 0;
	//This CANNOT be parallelized. 1 row at a time.
	for (auto &row : trainingData)
	{
		std::cout << "\nRow: " << rowRep;
		calculateActivation(hiddenLayer, row, 1);
		calculateActivation(outputLayer, getHiddenActivations(), 0);
		calculateOutputError(row[0]); //should be the row rep
		calculateHiddenError();
		//first updates output, then hidden
		updateWeights(learningRate, momentum, row);

		++rowRep;
	}
}



//make sure to include compilation tags
void dataManager::calculateActivation(std::vector<perceptron> &nodes, std::vector<double> & inputData, int offset) 
{
//#pragma omp parallel for
	for(int idx = 0; idx < nodes.size(); ++idx)
	{
		std::shared_ptr<std::vector<double>> weights = nodes[idx].getWeights();
		double sum = std::transform_reduce(std::execution::par, weights->begin(), weights->end(), inputData.begin()+offset,  //offset of 1 for the row rep
											0.0, std::plus<double>(), std::multiplies<double>());
		nodes[idx].setActivation(computeActivation(sum));
	}
}


std::vector<double> dataManager::getTestingActivations(std::vector<perceptron> & nodes, std::vector<double> &inputData, int offset)
{
	double* tmpAct = new double(nodes.size());

#pragma omp parallel for
	for (int idx = 0; idx < nodes.size(); ++idx)
	{
		std::shared_ptr<std::vector<double>> weights = nodes[idx].getWeights();
		double sum = std::transform_reduce(std::execution::par, weights->begin(), weights->end(), inputData.begin() + offset,  //offset of 1 for the row rep
			0.0, std::plus<double>(), std::multiplies<double>());

		tmpAct[idx] = computeActivation(sum);
	}

	std::vector<double> activations(tmpAct, tmpAct + nodes.size());

	return activations;
}



void dataManager::calculateOutputError(double rowRepresentation) 
{
	//we can use machinenum for target because there are only 10 in the ouput layer, 1 for each number to represent
	std::for_each(std::execution::par, outputLayer.begin(), outputLayer.end(), [rowRepresentation](perceptron &p) 
	{
		double target = (p.getMachineNum() == rowRepresentation) ? 0.9 : 0.1;
		p.setError(p.getActivation() * (1 - p.getActivation()) * (target - p.getActivation()));
	});
}



void dataManager::calculateHiddenError()
{
	std::vector<perceptron> * out = &outputLayer;
	std::for_each(std::execution::par, hiddenLayer.begin(), hiddenLayer.end(), [out](perceptron &hidden) 
	{
		int machNum = hidden.getMachineNum();
		double activationWeightDotProduct = std::transform_reduce(std::execution::par, out->begin(), out->end(), 0.0, std::plus<double>(), 
			[machNum](perceptron &outNode) 
		{
			std::shared_ptr<std::vector<double>> outWeights = outNode.getWeights();
			//machnum is the index of the hidden node, machnum can be used for the dot product
			return outNode.getError() * outWeights->at(machNum);
		});
		hidden.setError(hidden.getActivation() * (1 -hidden.getActivation()) * activationWeightDotProduct);
	});
}



//learning rate and momentum need to be passed as args for the lambdas
void dataManager::updateWeights(int learningRate, int momentum, std::vector<double> data) 
{
	std::vector<double> hiddenActivations = getHiddenActivations();

	std::for_each(std::execution::par, outputLayer.begin(),
		outputLayer.end(), [learningRate, momentum, hiddenActivations](perceptron & p) 
								{p.updateWeights(learningRate, momentum, hiddenActivations);});

	std::for_each(std::execution::par, hiddenLayer.begin(),
		hiddenLayer.end(), [learningRate, momentum, data](perceptron & p) 
								{p.updateWeights(learningRate, momentum, data);});
}



std::vector<double>& dataManager::getHiddenActivations()
{
	std::vector<double> activations; 
	activations.reserve(hiddenLayer.size());

	//do not parallelize, this needs to be in correct order
	for (auto& h : hiddenLayer)
	{
		activations.push_back(h.getActivation());
	}

	return activations;
}


void dataManager::testWrapper()
{
	//row rep is now in the row itself
	//because we are done training this can be done in parallel
	//make sure that nothing is saved to mem in shared data in test

	//this may not be thread safe to update count like this
	int count = 0;
	std::for_each(std::execution::par, testData.begin(), testData.end(), [&count, this](std::vector<double> &row) 
	{
		count += this->test(row);
	});
}


int dataManager::test(std::vector<double> &testRow)
{
	//calculate activations for the given row
	//returns array so we can use OMP parallel directive
	std::vector<double> hiddenActivations = getTestingActivations(hiddenLayer, testRow, 1);  //offset of 1 because 0 is the row rep
	std::vector<double> outputActivations = getTestingActivations(outputLayer, hiddenActivations, 0); //no need for offset heree
	
	double guess = *std::max_element(outputActivations.begin(), outputActivations.end());
	return (guess == testRow[0]) ? 1 : 0;


	//if we update the confusion matrix here, make sure to use a mutex
}