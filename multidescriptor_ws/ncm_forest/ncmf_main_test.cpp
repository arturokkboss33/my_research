//predefined classes
#include "ncmf_class_tree.h"
#include "ncmf_forest.h"
//c++ libraries
#include <iostream>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

void get_mult_sets_data(cv::Mat& samples, cv::Mat& feat_per_sample, cv::Mat& labels, cv::Mat& classes, 
	std::string data_file, bool concatenate){
	
	if(!concatenate){
		cv::FileStorage fstore_train_params(data_file, cv::FileStorage::READ);
		fstore_train_params["matDescriptors"] >> samples;
		fstore_train_params["labels"] >> labels;
		fstore_train_params["classes"] >> classes;
		fstore_train_params["features_per_image"] >> feat_per_sample;
		fstore_train_params.release();
		if(labels.type() >= 5 ){
			labels.convertTo(labels, CV_32SC1);
		}
	}else{
		
		//iterate through all the feauture spaces
		bool flag_get_labels = false;
		std::stringstream parser;
		int ssize = 255; 						//size of char buffer
		char buf[ssize];						//buffer to store words
		char delimiter = ' ';
		parser << data_file;
		
		while(parser.getline(buf,ssize,delimiter)){
			
			std::string training_file(buf);
			//std::cout << training_file << std::endl;
			cv::FileStorage fstore_train_params(training_file, cv::FileStorage::READ);
			
			if(!flag_get_labels){
				fstore_train_params["labels"] >> labels;
				fstore_train_params["classes"] >> classes;
				fstore_train_params["features_per_image"] >> feat_per_sample;
				if(labels.type() >= 5 ){
					labels.convertTo(labels, CV_32SC1);
				}
			}
			cv::Mat tmp_data;
			fstore_train_params["matDescriptors"] >> tmp_data;
			cv::transpose(tmp_data, tmp_data);
			samples.push_back(tmp_data);
			tmp_data.release();	
		}
		
		cv::transpose(samples, samples);
		
		//>[DEBUGGING]
		//std::cout << samples.rows << " " << samples.cols << std::endl;
		//std::cout << samples.row(0) << std::endl;
		//<[DEBUGGING]
		
	}
}


double eval_set(NCMF_forest* ncmf, const cv::Mat& test_set, const cv::Mat& test_labels, const cv::Mat& features_per_image, 
	const cv::Mat classes, bool eval_per_feature){
		
	double accuracy = 0.;
	
	if(eval_per_feature){
		for(int ex = 0; ex < test_set.rows; ex++)
		{
			//int prediction = ncmf->predict_wi(test_set.row(ex));
			int prediction;
			ncmf->predict_with_hist(test_set.row(ex), &prediction);
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
				int prediction = ncmf->predict(test_set.row(f));
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
	std::string model_params_file = argv[1];//"ncmf_parameters.yml";
	
	//variables
	std::string model_name, model_file;
	std::string training_set, test_set;
	int max_depth, min_samples, active_classes, no_trees;
	bool save_model, test_trainSet, test_testSet, eval_per_feature;
	bool concatenate_sets;
	cv::Mat classes;
	
	//read values
	cv::FileStorage fstore_model_params(model_params_file, cv::FileStorage::READ);
	fstore_model_params["model_name"] >> model_name;
	fstore_model_params["model_file"] >> model_file;
	fstore_model_params["save_model"] >> save_model;
	fstore_model_params["concatenate_sets"] >> concatenate_sets;
	fstore_model_params["training_set"] >> training_set;
	fstore_model_params["test_set"] >> test_set;
	fstore_model_params["max_depth"] >> max_depth;
	fstore_model_params["min_samples"] >> min_samples;
	fstore_model_params["active_classes"] >> active_classes;
	fstore_model_params["no_trees"] >> no_trees;
	fstore_model_params["test_trainSet"] >> test_trainSet;
	fstore_model_params["test_testSet"] >> test_testSet;
	fstore_model_params["eval_per_feature"] >> eval_per_feature;
	fstore_model_params.release();
	std::cout << training_set << std::endl;
	
	
	//save data to matrixes
	cv::Mat train_samples, train_labels;
	cv::Mat test_samples, test_labels;
	cv::Mat train_features_per_image, test_features_per_image;
	
	get_mult_sets_data(train_samples, train_features_per_image, train_labels, classes, training_set, concatenate_sets);
	
	/*
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
	*/
	
	/// CREATE RANDOM FOREST
	NCMF_forest* ncmf = new NCMF_forest();
	std::cout << "TRAINING RANDOM FOREST" << std::endl;
	ncmf->train(train_samples, train_labels, max_depth, min_samples, active_classes, no_trees);
	
	/// Save model
	if(save_model){
		std::cout << "Test saving the model to YAML" << std::endl;
		ncmf->save_model(model_file, model_name, training_set);
		std::cout << "Model saved" << std::endl;
	}

	/// Test accuracy in training set
	double good_classif = 0.;
	
	if(test_trainSet){
		good_classif = eval_set(ncmf, train_samples, train_labels, train_features_per_image, classes, eval_per_feature);
		std::cout << "Training accuracy: " << good_classif << std::endl;
	}
	if(test_testSet){
		get_mult_sets_data(test_samples,test_features_per_image, test_labels, classes, test_set, concatenate_sets);		
		good_classif = eval_set(ncmf, test_samples, test_labels, test_features_per_image, classes, eval_per_feature);
		std::cout << "Testing accuracy: " << good_classif << std::endl;
	}

	//>[DEBUGGING]	
	///// Test if the loading the model works
	//NCMF_forest* ncmf_copy = new NCMF_forest();
	//std::cout << "LOADING MODEL FROM FILE" << std::endl;
	//ncmf_copy->load_model(model_file, model_name);
	
	//good_classif = 0;
	//if(test_trainSet){
		//good_classif = eval_set(ncmf_copy, train_samples, train_labels, train_features_per_image, classes, eval_per_feature);
		//std::cout << "Training accuracy: " << good_classif << std::endl;
	//}
	//if(test_testSet){
		//good_classif = eval_set(ncmf_copy, test_samples, test_labels, test_features_per_image, classes, eval_per_feature);
		//std::cout << "Testing accuracy: " << good_classif << std::endl;
	//}
	//<[DEBUGGING]

	
}
