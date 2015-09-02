/*
 * =====================================================================================
 *
 *       Filename:  dectree_test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09/17/2014 01:28:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

//predefined classes
#include "ncmf_class_tree.h"
#include "ncmf_forest.h"
//c++ libraries
#include <iostream>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

//define
#define TRAIN_PATH "training_sets/"
#define TEST_PATH "test_sets/"

int main ( int argc, char *argv[] )
{
	
	//+++ LOAD TRAINING AND TEST SET +++//
	std::cout << "LOADING TRAINING AND TEST SET..." << std::endl;
	//variables to parse the console input and search te trainig and test files
	//std::string train_path = TRAIN_PATH;
	//train_path += argv[1];	
	//std::string test_path = TEST_PATH;
	//test_path += argv[2];
	std::string train_path = argv[1];
	std::string test_path = argv[2];	
	//for debugging
	std::cout << "Train path: " << train_path.c_str() << std::endl;
	std::cout << "Test path: " << test_path.c_str() << std::endl;
	//saving the data in to opencv matrix
	cv::Mat tmp1, tmp2;
	cv::Mat train_samples, train_labels;
	cv::Mat test_samples, test_labels;
	cv::FileStorage fstore_train_data(train_path.c_str(), cv::FileStorage::READ);
	//fstore_train_data["matDescriptors"] >> train_samples;
	fstore_train_data["train_samples"] >> train_samples;
	//fstore_train_data["train_labels"] >> train_labels;
	fstore_train_data["train_labels"] >> tmp1;
	//fstore_train_data["labels"] >> tmp1;
	fstore_train_data.release();
	cv::FileStorage fstore_test_data(test_path.c_str(), cv::FileStorage::READ);
	//fstore_test_data["matDescriptors"] >> test_samples;
	fstore_test_data["test_samples"] >> test_samples;
	//fstore_test_data["test_labels"] >> test_labels;
	fstore_test_data["test_labels"] >> tmp2;
	//fstore_test_data["labels"] >> tmp2;
	fstore_train_data.release();

	//uncomment if the labels are stored as floats
	
	for(int row = 0; row < tmp1.rows; row++)
	{
		for(int col = 0; col < tmp1.cols; col++)
		{
			train_labels.push_back((int)(tmp1.at<float>(row,col)));
		}
	}
	for(int row = 0; row < tmp2.rows; row++)
	{
		for(int col = 0; col < tmp2.cols; col++)
		{
			test_labels.push_back((int)(tmp2.at<float>(row,col)));
		}
	}
	
	//train_labels = tmp1.clone();
	//test_labels = tmp2.clone();

	//for debugging
	
	std::cout << train_labels.type() << std::endl;
	std::cout << test_labels.type() << std::endl;
	//std::cout << train_samples << std::endl;
	//std::cout << train_labels << std::endl;
	//std::cout << test_samples << std::endl;
	//std::cout << test_labels << std::endl;
	//std::cout << test_labels.type() << std::endl;
	
	
	//++++++++++++++++++++++++++++++++++//

	
	//+++ CREATE A DECISION TREE +++//
	NCMF_class_tree* dectree;	
	dectree = new NCMF_class_tree();
	dectree->set_dectree_idx(3); //give an id to the tree
	//std::cout << dectree->get_dectree_idx() << std::endl;
	std::cout << "TRAINING DECISION TREE" << std::endl;
	int max_depth = -1;
	int min_samples = 1;
	int active_classes = 3;
	dectree->train(train_samples, train_labels, max_depth, min_samples, active_classes);
	//++++++++++++++++++++++++++++++++++//	
	//+++ PRINT TREE STRUCTURE +++//
	std::cout << "Max depth: " << dectree->get_maxDepth() << std::endl;
	std::cout << "No. Leaves: " << dectree->get_noLeaves() << std::endl;
	std::cout << "No. Nodes: " << dectree->get_noNodes() << std::endl;
	std::cout << "\nInOrder traversal: " << std::endl;
	//dectree->inOrder_tree();
	std::cout << std::endl;
	std::cout << "\nPostOrder traversal: " << std::endl;
	//dectree->postOrder_tree();
	std::cout << std::endl;


	
	//+++ TESTING DECISION TREE +++//
	double good_classif = 0;
	for(int ex = 0; ex < train_samples.rows; ex++)
	{
		int prediction = dectree->predict(train_samples.row(ex));
		//std::cout << "+++ " << prediction << " ";
		if(prediction == train_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/train_samples.rows;

	std::cout << std::endl;
	std::cout << "Training accuracy: " << good_classif << std::endl;

	good_classif = 0;
	cv::Mat used_leaves(0,2,CV_32SC1);
	//cv::Mat a;
	for(int ex = 0; ex < test_samples.rows; ex++)
	{
		int prediction = dectree->predict(test_samples.row(ex));
		//used_leaves.push_back(dectree->predict_with_idx(test_samples.row(ex)));
		//std::cout << "+++ " << prediction << std::endl;
		if(prediction == test_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/test_samples.rows;

	std::cout << "Testing accuracy: " << good_classif << std::endl;
	//std::cout << used_leaves.rows << " " << used_leaves.cols << std::endl;
	//cv::Mat idxs = used_leaves.col(1).clone();
	//idxs = idxs.reshape(0,1);
	//std::cout << "Used leaves in testing:\n" << idxs << std::endl;
	//++++++++++++++++++++++++++++++++++//	
	
	
	/*
	//+++ PRINT TREE STRUCTURE +++//
	std::cout << "Max depth: " << dectree->get_maxDepth() << std::endl;
	std::cout << "No. Leaves: " << dectree->get_noLeaves() << std::endl;
	std::cout << "No. Nodes: " << dectree->get_noNodes() << std::endl;
	std::cout << "\nInOrder traversal: " << std::endl;
	dectree->inOrder_tree();
	std::cout << std::endl;
	std::cout << "\nPostOrder traversal: " << std::endl;
	dectree->postOrder_tree();
	std::cout << std::endl;
	*/	

	
	//+++ CREATE RANDOM FOREST +++//
	NCMF_forest* ncmf = new NCMF_forest();
	std::cout << "TRAINING RANDOM FOREST" << std::endl;
	int no_trees = 10;
	ncmf->train(train_samples, train_labels, max_depth, min_samples, active_classes, no_trees);

	//+++ TESTING DECISION TREE +++//
	good_classif = 0;
	for(int ex = 0; ex < train_samples.rows; ex++)
	{
		int prediction = ncmf->predict(train_samples.row(ex));
		//std::cout << "+++ " << prediction << std::endl;
		if(prediction == train_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/train_samples.rows;

	std::cout << "Training accuracy: " << good_classif << std::endl;

	good_classif = 0;
	cv::Mat pred_per_tree(0,2,CV_32SC1);
	for(int ex = 0; ex < test_samples.rows; ex++)
	{
		int prediction = ncmf->predict(test_samples.row(ex));
		//std::cout << "+++ " << prediction << std::endl;
		//pred_per_tree = ncmf->predict_with_idx(test_samples.row(ex));
		//std::cout << pred_per_tree << std::endl;
		if(prediction == test_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/test_samples.rows;

	std::cout << "Testing accuracy: " << good_classif << std::endl;
	
	//+++ TEST SAVING THE MODEL +++//
	std::cout << "Test saving the model to YAML" << std::endl;
	std::string model_file= "ncmf_model2.yml";
	std::string model_name = "ncmf_model2";
	ncmf->save_model(model_file, model_name);
	std::cout << "Model saved" << std::endl;
	
	//+++++++++++
	//+++ TEST LOADING THE MODEL +++//
	std::cout << "Test loading the model" << std::endl;
	NCMF_forest* ncmf_copy = new NCMF_forest();
	ncmf_copy->load_model(model_file, model_name);
	std::cout << "Model loaded" << std::endl;
	
	//+++ TEST SAVING THE MODEL +++//
	std::cout << "Test saving the model to YAML" << std::endl;
	std::string model_file_copy= "ncmf_model2_copy.yml";
	std::string model_name_copy = "ncmf_model2_copy";
	ncmf_copy->save_model(model_file_copy, model_name_copy);
	std::cout << "Model saved" << std::endl;
	
	//+++ TESTING NEW DECISION TREE +++//
	good_classif = 0;
	for(int ex = 0; ex < train_samples.rows; ex++)
	{
		int prediction = ncmf_copy->predict(train_samples.row(ex));
		//std::cout << "+++ " << prediction << std::endl;
		if(prediction == train_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/train_samples.rows;

	std::cout << "Training accuracy: " << good_classif << std::endl;

	good_classif = 0;
	cv::Mat pred_per_tree_copy(0,2,CV_32SC1);
	for(int ex = 0; ex < test_samples.rows; ex++)
	{
		int prediction = ncmf_copy->predict(test_samples.row(ex));
		//std::cout << "+++ " << prediction << std::endl;
		//pred_per_tree = erf->predict_with_idx(test_samples.row(ex));
		//std::cout << pred_per_tree << std::endl;
		if(prediction == test_labels.at<int>(ex))
			good_classif += 1;
	}
		good_classif = good_classif/test_samples.rows;

	std::cout << "Testing accuracy: " << good_classif << std::endl;
	
	cv::Mat hist = ncmf_copy->predict_with_hist(test_samples.row(10));
	std::cout << hist << std::endl;

	


	return 0;
}				/* ----------  end of function main  ---------- */
