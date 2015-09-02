//predefined classes
#include "ncmf_class_tree.h"
#include "ncmf_forest.h"
//c++ libraries
#include <iostream>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

void format_test_data(std::string test_sets, std::vector<cv::Mat>& test_samples, cv::Mat& labels){	
	
	//iterate through all the feauture spaces
	bool flag_get_labels = false;
	std::stringstream parser;
	int ssize = 255; 						//size of char buffer
	char buf[ssize];						//buffer to store words
	char delimiter = ' ';
	parser << test_sets;
	
	while(parser.getline(buf,ssize,delimiter)){
		
		std::string test_file(buf);
		//std::cout << test_file << std::endl;
		cv::FileStorage fstore_test_params(test_file, cv::FileStorage::READ);
		
		if(!flag_get_labels){
			fstore_test_params["labels"] >> labels;
			if(labels.type() >= 5 ){
				labels.convertTo(labels, CV_32SC1);
			}
		}
		cv::Mat tmp_data;
		fstore_test_params["matDescriptors"] >> tmp_data;
		test_samples.push_back(tmp_data);
		tmp_data.release();			

	}
	
}

double eval_set(NCMF_forest* ncmf, std::string test_sets, const cv::Mat classes){
		
	double accuracy = 0.;
	std::vector<cv::Mat> test_samples;
	cv::Mat test_labels;
	format_test_data(test_sets, test_samples, test_labels);
	
	int no_samples = (int)test_samples.at(0).rows;
	int no_features = ncmf->get_no_features();
	//std::cout << no_features << std::endl;
	//int a;
	
	for(int ex = 0; ex < no_samples; ex++){
		cv::Mat sample;
		//std::cout << "sample" << std::endl;
		for(int f = 0; f < no_features; f++){
			sample.push_back(test_samples.at(f).row(ex));
			//std::cout << sample.row(f) << std::endl;
		}
		//std::cout << "***" << std::endl;
		//std::cin >> a;
		
		int prediction;
		ncmf->predict_with_hist(sample, &prediction);
		if(prediction == test_labels.at<int>(ex))
			accuracy += 1;
		//int prediction = ncmf->predict(sample);
		//if(prediction == test_labels.at<int>(ex))
			//accuracy += 1;
		
		sample.release();
	}
	accuracy = accuracy/no_samples;
	test_samples.clear();
	test_labels.release();
	
	return accuracy;
}

int main ( int argc, char *argv[] ){
	
	//Read from file the user requirements (detector, descriptors and correspondant parameters)
	std::cout << "READING USER REQUIREMENTS" << std::endl;
	std::string model_params_file = argv[1];//"mdncmf_parameters.yml";
	
	//variables
	std::string model_name, model_file;
	std::string training_sets, test_sets;
	int max_depth, min_samples, active_classes, no_trees;
	bool save_model, test_trainSet, test_testSet, eval_per_feature;
	cv::Mat classes;
	
	//read values
	cv::FileStorage fstore_model_params(model_params_file, cv::FileStorage::READ);
	fstore_model_params["model_name"] >> model_name;
	fstore_model_params["model_file"] >> model_file;
	fstore_model_params["save_model"] >> save_model;
	fstore_model_params["training_sets"] >> training_sets;
	fstore_model_params["test_sets"] >> test_sets;
	fstore_model_params["max_depth"] >> max_depth;
	fstore_model_params["min_samples"] >> min_samples;
	fstore_model_params["active_classes"] >> active_classes;
	fstore_model_params["no_trees"] >> no_trees;
	fstore_model_params["test_trainSet"] >> test_trainSet;
	fstore_model_params["test_testSet"] >> test_testSet;
	fstore_model_params["eval_per_feature"] >> eval_per_feature;
	fstore_model_params.release();
	std::cout << training_sets << std::endl;
	
	/// CREATE RANDOM FOREST
	NCMF_forest* ncmf = new NCMF_forest();
	std::cout << "TRAINING RANDOM FOREST" << std::endl;
	ncmf->train(max_depth, min_samples, active_classes, no_trees, training_sets);	
	
	/// Save model
	if(save_model){
		std::cout << "Test saving the model to YAML" << std::endl;
		ncmf->save_model(model_file, model_name, training_sets);
		std::cout << "Model saved" << std::endl;
	}
	
	/// Test accuracy in training set
	double good_classif = 0.;
	cv::Mat prediction_mat;
	
	if(test_trainSet){
		good_classif = eval_set(ncmf,  training_sets, classes);
		std::cout << "Training accuracy: " << good_classif << std::endl;
		
	}
	if(test_testSet){
		good_classif = eval_set(ncmf, test_sets, classes);
		std::cout << "Testing accuracy: " << good_classif << std::endl;
	}
	
	//>[DEBUGGING]
	///// Test if the loading the model works
	//NCMF_forest* ncmf_copy = new NCMF_forest();
	//std::cout << "LOADING MODEL FROM FILE" << std::endl;
	//ncmf_copy->load_model(model_file, model_name);
	
	//good_classif = 0;
	//if(test_trainSet){
		//good_classif = eval_set(ncmf_copy, training_sets, classes);
		//std::cout << "Training accuracy: " << good_classif << std::endl;
	//}
	//if(test_testSet){
		//good_classif = eval_set(ncmf_copy, test_sets, classes);
		//std::cout << "Testing accuracy: " << good_classif << std::endl;
	//}
	//<[DEBUGGING]
	
}
