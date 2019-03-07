





//*************
//SPEED IS AT 02, IF THERE ARE ISSUES TURN THIS OFF**************888
//***********

#include "dataManager.h"
//#include <boost/tokenizer.hpp>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <omp.h>
//#include "csv.h"


bool dataManager::loadWrapper(std::string inFile_training, std::string inFile_test)
{
	//std::vector<std::string> results;
	loadData(inFile_training, trainingData, 60000);
	//loadData(inFile_test, testData);

	//manage this..maybe make void and just use exceptions?
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
	{ //https://www.boost.org/doc/libs/1_42_0/libs/tokenizer/introduc.htm
		//std::filebuf fb;
		std::ifstream input(inFile);
		//if (fb.open(inFile, std::ios::in)) {
		//std::istream in(&fb);
		std::string line;
		std::vector<std::string> test;
		test.reserve(rowCount);
		std::string test3;
		while (std::getline(input, line))
		{
			test.push_back(line);
			//The problem here is also, we then have to convert
			//transfer the entire data table
			std::vector<std::string> results;
			//std::vector<double> resultsD(785);
			//boost::split(results, line, [](char c) {return c == ','; });
			/*std::transform(results.begin(), results.end(), resultsD.begin(), [](const std::string& val)
			{
				return stod(val);
			});
			data.insert(
				data.end(),
				std::make_move_iterator(results.begin()),
				std::make_move_iterator(results.end())
			);*/
			/*boost::tokenizer<> tok(line);
			for (boost::tokenizer<>::iterator beg = tok.begin(); beg != tok.end(); ++beg)
			{
				//string to double
				data.push_back(stod(*beg));
			}*/
		}
		std::string st = test[1];
		std::vector<double> d = convertToDouble(st);
		
		//do this rather than resizing each time4
		//it's in the slides he just went over
		data.reserve(60000);
		std::vector<std::vector<std::string>> test2;
#pragma omp parallel for 
		for (int i = 0; i < test.size(); ++i)
		{	
#pragma omp critical
			data.push_back(convertToDouble(test[i]));
		}
		
		return true;
	}

}

std::vector<double> dataManager::convertToDouble(std::string to_split)
{
	std::vector<std::string> myStr;
	std::vector<double> d;
	myStr.reserve(785);
	d.reserve(785);
	boost::split(myStr, to_split, [](char c) {return c == ','; });
	
	for (auto s : myStr)
	{
		d.push_back(std::stod(s)/255);
	}
	return d;
}

void dataManager::setBiasWrapper(int bias)
{
//	omp_set_num_threads(2);
#pragma omp parallel{
	setBias(trainingData, trainingRepresentation, bias);
	setBias(testData, testRepresentation, bias);
	}
}

void dataManager::setBias(std::vector<std::vector<double>> &data, std::vector<double> &repHolder, int bias)
{

}