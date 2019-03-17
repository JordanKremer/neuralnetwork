#pragma once

//Make inline?

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
	perceptron(int mNum, int weightCount):machineNum(mNum), error(0), activation(0)
	{

		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-0.05, 0.05);

		//parallelize this
		//https://thrust.github.io/doc/classthrust_1_1random_1_1uniform__real__distribution.html
		/*
			  thrust::minstd_rand rng;
			  thrust::uniform_real_distribution<float> dist(-0.05,0.05);

			  Or, parallelize the creation of perceptrons, no need to
			  use rng in the device unless there is an easy way to do it
		
			or
			https://stackoverflow.com/questions/12614164/generating-random-numbers-with-uniform-distribution-using-thrust
			try this!^^^
		*/
		for (int col = 0; col < weightCount; ++col)
		{
			weights.push_back(dis(gen));
		}
	};

	inline bool setWeights(std::vector<double> newWeights)
	{ 
		weights = newWeights;
		//https://thrust.github.io/doc/group__copying.html
		/*
			thrust::copy(newWeights.begin(), newWeights.end(),
			 weights.begin());
		*/
	};


	void updateWeights(int learningRate, int momentum, std::vector<double> data)
	{
#pragma omp parallel for
		for (int index = 0; index < weights.size(); ++index)
		{
			double deltaWeight = (learningRate * error * data[index]) + (momentum * previousDeltaWeights[index]);
			previousDeltaWeights[index] = deltaWeight;
			weights[index] += deltaWeight;
		}
	}

	const std::shared_ptr <std::vector<double>> getWeights() { return  std::make_shared<std::vector<double>>(weights); }
	const std::shared_ptr <std::vector<double>> getPreviousWeights(){ std::make_shared<std::vector<double>>(previousDeltaWeights); }
	void setActivation(double act) {}
	const double getError() { return error; };
	const double getActivation() { return activation; };

};



