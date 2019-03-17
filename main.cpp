#include <iostream>
#include <fstream>
#include <string>
#include "perceptron.h"
#include "dataManager.h"

//run with input / output file as args?
int main(int argc, char** argv)
{

	dataManager manager(10,100,0,785,0, 1);
	manager.loadWrapper("mnist_train.csv", "mnist_test.csv");
	std::cout << "\nFiles loaded. Training...";
	int epochCount = 2;
	for (int epoch = 0; epoch < epochCount; ++epoch)
	{
		std::cout << "\nEpoch : " << epoch;
		manager.learn();
		manager.testWrapper();
	}
	manager.printMatrix();

}