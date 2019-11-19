#include <iostream>
#include <fstream>
#include <string>

#include <ctime>
#include <ratio>
#include <chrono>

#include "perceptron.h"
#include "dataManager.h"

//run with input / output file as args?
int main(int argc, char** argv) 
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	dataManager manager(10,10,0,785,0, 1);
	manager.loadWrapper("mnist_train.csv", "mnist_test.csv");
	std::cout << "\nFiles loaded. Training...";

	//Get initial test
	manager.testWrapper();
	manager.printMatrix();

	int epochCount = 10;
	bool isFinalEpoch = false;
	for (int epoch = 0; epoch < epochCount; ++epoch)
	{
		std::cout << "\nEpoch : " << epoch;
		manager.learn();
		manager.testWrapper();
		manager.printMatrix();
	}


	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

	std::cout << "It took " << time_span.count() << " seconds.";
	std::cout << std::endl;

}