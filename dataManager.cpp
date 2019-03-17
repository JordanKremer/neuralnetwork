

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
//#include <experimental>
//#include <experimental/numeric>
//#include <experimental/execution_policy>
//#include <thread>  can use to create tasks
#include <numeric>
#include <iterator>
#include "parser.hpp"
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
		repHolder.reserve(rowCount);

		std::ifstream input(inFile);
		aria::csv::CsvParser parser(input);
		bool first = true;
		for (auto& row : parser) {
			std::vector<double> tmpRow;
			tmpRow.reserve(colCount);
			for (auto& field : row) {
				if (first)
				{
					repHolder.push_back(stod(field) / 255);
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

		//shuffle data rows
		std::srand(unsigned(std::time(0)));
		auto rng = std::default_random_engine{};
		std::shuffle(std::begin(data), std::end(data), rng);


		return true;
	}

}

void dataManager::learn()
{
	//for each row in the
	//for each row
	/*
		calculateActivation hidden
		calculateActivation Output
		calculateOutputError
		calculateHiddenError

		thread task both of these
			updateWeights-- can use same function
			use transform?
				parallel version?
					vector of inputs & vector of weights
						should be same length
						zipsum

						Can use reduce for this
							openMP
		

	*/

}



//make sure to include compilation tags
void dataManager::calculateActivation(std::vector<perceptron> &nodes, std::vector<double> & inputData) 
{
	//maybe make this parallel also
	//for (auto n : nodes)
	//std::for_each(std::execution::par, begin(nodes), end(nodes), [&])
#pragma omp parallel for
	for(int idx = 0; idx < nodes.size(); ++idx)
	{
		std::shared_ptr<std::vector<double>> weights = nodes[idx].getWeights();
		//may need to swap plus and multiplies, one will multiple the elements together, the other
		//will apply the reduction e.g. element 1 + element 2...= sum
		double sum = std::transform_reduce(std::execution::par, weights->begin(), weights->end(), inputData.begin(),
											0.0, std::plus<double>(), std::multiplies<double>());
		nodes[idx].setActivation(computeActivation(sum));
	}
}
void dataManager::calculateOutputError() 
{

}
void dataManager::calculateHiddenError()
{

}
void dataManager::updateWeights() 
{

}
