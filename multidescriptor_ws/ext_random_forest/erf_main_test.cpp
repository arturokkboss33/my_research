//predefined classes
#include "dectree_class.h"
#include "erf_class.h"
//c++ libraries
#include <iostream>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

double eval_set(ERF_class* erf, const cv::Mat& test_set, const cv::Mat& test_labels, const cv::Mat& features_per_image, 
	const cv::Mat classes, bool eval_per_feature){
		
	double accuracy = 0.;
	
	if(eval_per_feature){
		for(int ex = 0; ex < test_set.rows; ex++)
		{
			int prediction = erf->predict(test_set.row(ex));
			if(prediction == test_labels.at<int>(ex))
				accuracy += 1;
		}
			accuracy = accuracy/test_set.rows;

		//std::cout << "Training accuracy: " << accuracy << std::endl;
			
	}else{
		std::map<int,int> predictions_count;			
		
		for(int c = 0; c < classes.rows; c++)
			predictions_count.insert(std::pair<int,int>((int)classes.at<float>(c),0));
		
		int last_feat = 0;
		for(int ex = 0; ex < features_per_image.rows; ex++)
		{
			int limit_feat = last_feat+features_per_image.at<int>(ex);
			//std::cout << last_feat << " " << limit_feat << std::endl;
			for(int f = last_feat; f < limit_feat; f++){
				int prediction = erf->predict(test_set.row(f));
				predictions_count[prediction] += 1;
			}
			
			//std::map<int,int>::iterator iterator;
			//for(iterator = predictions_count.begin(); iterator != predictions_count.end(); iterator++) {
				//std::cout << iterator->first << " " << iterator->second << std::endl;
			//}
			
			int max_count = 0;
			int best_label;
			for(int p = 0; p < classes.rows; p++){
				if(predictions_count[(int)classes.at<float>(p)] > max_count){
					max_count = predictions_count[(int)classes.at<float>(p)];
					best_label = (int)classes.at<float>(p);
				}
			}
			//std::cout << "***" << best_label << std::endl;
			//std::cout << test_labels.at<int>(last_feat) << std::endl;
			if(best_label == test_labels.at<int>(last_feat))
				accuracy += 1;
			
			last_feat += features_per_image.at<int>(ex);
			
			for(int p = 0; p < classes.rows; p++){
				predictions_count[(int)classes.at<float>(p)] = 0;
			}

		}
		accuracy = accuracy/features_per_image.rows;

		//std::cout << "Training accuracy: " << accuracy << std::endl;
	}

	
	
	return accuracy;
}

int main ( int argc, char *argv[] ){
	
	//Read from file the user requirements (detector, descriptors and correspondant parameters)
	std::cout << "READING USER REQUIREMENTS" << std::endl;
	std::string model_params_file = "erf_parameters.yml";
	
	//variables
	std::string model_name, model_file;
	std::string training_set, test_set;
	int max_depth, min_samples, active_vars, no_trees;
	bool save_model, test_trainSet, test_testSet, eval_per_feature;
	cv::Mat classes;
	
	//read values
	cv::FileStorage fstore_model_params(model_params_file, cv::FileStorage::READ);
	fstore_model_params["model_name"] >> model_name;
	fstore_model_params["model_file"] >> model_file;
	fstore_model_params["save_model"] >> save_model;
	fstore_model_params["training_set"] >> training_set;
	fstore_model_params["test_set"] >> test_set;
	fstore_model_params["max_depth"] >> max_depth;
	fstore_model_params["min_samples"] >> min_samples;
	fstore_model_params["active_vars"] >> active_vars;
	fstore_model_params["no_trees"] >> no_trees;
	fstore_model_params["test_trainSet"] >> test_trainSet;
	fstore_model_params["test_testSet"] >> test_testSet;
	fstore_model_params["eval_per_feature"] >> eval_per_feature;
	fstore_model_params.release();
	//std::cout << training_set << std::endl;
	
	//save data to matrixes
	cv::Mat train_samples, train_labels;
	cv::Mat test_samples, test_labels;
	cv::Mat train_features_per_image, test_features_per_image;
	cv::FileStorage fstore_train_params(training_set, cv::FileStorage::READ);
	fstore_train_params["matDescriptors"] >> train_samples;
	fstore_train_params["labels"] >> train_labels;
	fstore_train_params["classes"] >> classes;
	fstore_train_params["features_per_image"] >> train_features_per_image;
	fstore_train_params.release();
	cv::FileStorage fstore_test_params(test_set, cv::FileStorage::READ);
	fstore_test_params["matDescriptors"] >> test_samples;
	fstore_test_params["labels"] >> test_labels;
	fstore_test_params["features_per_image"] >> test_features_per_image;
	fstore_test_params.release();
	if(train_labels.type() >= 5 ){
		train_labels.convertTo(train_labels, CV_32SC1);
	}
	if(test_labels.type() >= 5 ){
		test_labels.convertTo(test_labels, CV_32SC1);
	}
	
	/// CREATE RANDOM FOREST
	ERF_class* erf = new ERF_class();
	std::cout << "TRAINING RANDOM FOREST" << std::endl;
	erf->train(train_samples, train_labels, max_depth, min_samples, active_vars, no_trees);
	
	/// Save model
	if(save_model){
		std::cout << "Test saving the model to YAML" << std::endl;
		erf->save_model(model_file, model_name);
		std::cout << "Model saved" << std::endl;
	}
	
	/// Test accuracy in training set
	double good_classif = 0.;
	
	if(test_trainSet){
		good_classif = eval_set(erf, train_samples, train_labels, train_features_per_image, classes, eval_per_feature);
		std::cout << "Training accuracy: " << good_classif << std::endl;
	}
	if(test_testSet){
		good_classif = eval_set(erf, test_samples, test_labels, test_features_per_image, classes, eval_per_feature);
		std::cout << "Testing accuracy: " << good_classif << std::endl;
	}
	
	
}
