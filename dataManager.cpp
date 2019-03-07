





//*************
//SPEED IS AT 02, IF THERE ARE ISSUES TURN THIS OFF**************888
//***********

#include "dataManager.h"
//#include <boost/tokenizer.hpp>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <omp.h>
//#include <thread>  can use to create tasks
#include <iterator>
#include "parser.hpp"

bool dataManager::loadWrapper(std::string inFile_training, std::string inFile_test)
{
	std::cout << "\nLoading..";
	loadData(inFile_training, trainingData, 60000);
	loadData(inFile_test, testData, 10000);
	return true;
}
bool dataManager::loadData(std::string inFile, std::vector<std::vector<double>> &data, int rowCount)
{
	//make sure to handle exceptions
	if(inFile.empty())
	{
		return false;
	}
	else
	{
		std::ifstream input(inFile);

		aria::csv::CsvParser parser(input);
		std::vector<std::vector<double>> tmpData;


		//Including loading percentage based on count?
		for (auto& row : parser) {
			std::vector<double> tmpRow;
			for (auto& field : row) {
				tmpRow.push_back(stod(field)/255);
			}
			tmpData.push_back(tmpRow);
		}
		return true;
	}

}


void dataManager::setBiasWrapper(int bias)
{
//	omp_set_num_threads(2);
#pragma omp parallel
	{
	setBias(trainingData, trainingRepresentation, bias);
	setBias(testData, testRepresentation, bias);
	}
}

void dataManager::setBias(std::vector<std::vector<double>> &data, std::vector<double> &repHolder, int bias)
{
	/*
#pragma omp parallel for
		for (int idx = 0; idx < data.size; ++idx)
		{

		}
		*/
}