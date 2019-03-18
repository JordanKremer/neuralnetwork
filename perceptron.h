/*
perceptron.h
Jordan Kremer
3/17/2019

Provides a container and functionaltiy for perceptron values. Header only. The functionality including mostly
reflects setter and getter however weight update functionality is implemented within this file. 

*/

#pragma once
#include <vector>
#include <random>
#include <execution>
class perceptron
{
private:
	int machineNum;
	double error;
	double activation;
	std::vector<double> weights;
	std::vector<double> previousDeltaWeights;

public:
	perceptron(int mNum, int weightCount):machineNum(mNum), error(0), activation(0), previousDeltaWeights(weightCount, 0)
	{

		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-0.05, 0.05);

		weights.reserve(weightCount);
		for (int col = 0; col < weightCount; ++col)
		{
			weights.push_back(dis(gen));
		}

	};


	void updateWeights(float learningRate, float momentum, std::vector<double> data, int offset)
	{
		//must start at index of 1 for hiddennodes because 0th is row representation and not a weight
		//start at 0 for output layer
//#pragma omp parallel for  //do not use, much slower to use, do not use with top level for each parallel either, too many threads
							//
		for (int index = 0; index < weights.size(); ++index)
		{
			double deltaWeight = (learningRate * error * data[index + offset]) + (momentum * previousDeltaWeights[index]);
			previousDeltaWeights[index] = deltaWeight;
			weights[index] += deltaWeight;
		}
	}



	void updateWeightsParallel(float learningRate, float momentum, std::vector<double> data, int offset)
	{
		int idx = 0;
		double errorTmp = error;
		auto prevDeltaTmp = std::make_shared<std::vector<double>>(previousDeltaWeights);
		std::for_each(std::execution::par, weights.begin(), weights.end(),
			[data, errorTmp, &prevDeltaTmp, &idx, learningRate, momentum, offset](double & weight)
		{
			double deltaWeight = (learningRate * errorTmp * data[idx + offset]) + (momentum * prevDeltaTmp->at(idx));
			prevDeltaTmp->at(idx) = deltaWeight;
			weight += deltaWeight;
			++idx;
		});
	}

	const std::shared_ptr <std::vector<double>> getWeights() { return  std::make_shared<std::vector<double>>(weights); }
	const std::shared_ptr <std::vector<double>> getPreviousWeights(){ std::make_shared<std::vector<double>>(previousDeltaWeights); }
	void setActivation(double act) { activation = act; }
	void setError(double er) { error = er; };
	const double getError() { return error; };
	const double getActivation() { return activation; };
	const int getMachineNum() { return machineNum; }

};



