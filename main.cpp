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
	manager.learn();
	//manager.setBiasWrapper(1);



	//load data
	//get initial correct count

	//loop learning through epochs


}