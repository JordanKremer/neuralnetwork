#include "dataManager.h"
#include <boost/tokenizer.hpp>
#include <iostream>
#include <fstream>



bool dataManager::loadWrapper(std::string inFile_training, std::string inFile_test)
{
	loadData(inFile_training, trainingData);
	loadData(inFile_test, testData);

	//manage this..maybe make void and just use exceptions?
	return true;
}
bool dataManager::loadData(std::string inFile, std::vector<double> &data)
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
		while (std::getline(input, line))
		{
			string dataString[] = line.split(',');
			/*boost::tokenizer<> tok(line);
			for (boost::tokenizer<>::iterator beg = tok.begin(); beg != tok.end(); ++beg)
			{
				//string to double
				data.push_back(stod(*beg));
			}*/
		}
		return true;
	}



}