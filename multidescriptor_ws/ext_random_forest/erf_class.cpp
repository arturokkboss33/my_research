/* =============================================================*/
/* --- DECISION TREES- DECISION TREE CLASS SOURCE FILE       ---*/
/* FILENAME: erf_class.cpp 
 *
 * DESCRIPTION: source file for the struct object which implements
 * a extremely random forest learning algorithm.
 *
 * VERSION: 1.0
 *
 * CREATED: 09/26/2013
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

//headers
#include "erf_class.h"
//c++ libraries
#include <iostream>
#include <stdlib.h>
#include <string>

//*********************** CONSTRUCTOR *****************************//
ERF_class::ERF_class()
{
	max_trees = 1;
	tree_idx = 0;
	depth_limit = 1;
	min_samples = 1;
	active_vars = 1;
}

//*********************** PUBLIC FUNCTIONS *************************//
//training algorithm
void ERF_class::train(const cv::Mat& training_data, const cv::Mat& labels, int depth_thresh, unsigned int samples_thresh, 
	int vars_per_node, int no_trees)
{

	//initialize random generator
	rng = cv::RNG(time(NULL));
	max_trees = no_trees;
	depth_limit = depth_thresh;
	min_samples = samples_thresh;
	active_vars = vars_per_node;

	for(int tree = 0; tree < max_trees; tree++)
	{
		Dectree_class* dectree = new Dectree_class(rng);
		dectree->set_dectree_idx(tree);
		dectree->train(training_data, labels, depth_thresh, samples_thresh, vars_per_node);
		rng = dectree->get_rng();	//get the last state of the random generator
		forest.push_back(dectree);		
	}

}

//prediction done by majority voting, returns a label
int ERF_class::predict(const cv::Mat& sample)
{
	Dectree_class* dectree_tmp = forest.at(0);
	cv::Mat classes = dectree_tmp->get_classes();
	std::map<int, unsigned int> class_count;

	//create a hash map with the available classes to keep a vote count
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		class_count.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );
	
	//compute the prediciton in each tree
	for(int tree = 0; tree < max_trees; tree++)
	{
		Dectree_class* dectree = forest.at(tree);
		int prediction = dectree->predict(sample);
		class_count[prediction] += 1;
	}

	//see which class has more votes in the hash map
	bool flag_compare = false;
	int best_class;
	unsigned int max_votes;
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
	{
		if(flag_compare)
		{
			if(class_count[classes.at<int>(no_classes)] > max_votes)
			{
				best_class = classes.at<int>(no_classes);			
				max_votes = class_count[best_class];
			}
		}	
		else
		{
			best_class = classes.at<int>(no_classes);			
			max_votes = class_count[best_class];
			flag_compare = true;
		}		
	}

	return best_class;
}

//prediction done by majority voting
//this method returns a matrix where each row hs information about each tree (rows=no trees)
//the first column has the predcited labels from each tree
//the second column has the index of the leaf which decided the classification (relative to its tree)
//the last row of the matrix has first the decided label by majority voting, and then the number of votes it received
cv::Mat ERF_class::predict_with_idx(const cv::Mat& sample)
{
	Dectree_class* dectree_tmp = forest.at(0);
	cv::Mat classes = dectree_tmp->get_classes();
	std::map<int, unsigned int> class_count;

	//create a hash map with the available classes to keep a vote count
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		class_count.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );
	
	//compute the predictino in each tree and store the indexes
	cv::Mat used_leaves(0,2,CV_32SC1);
	for(int tree = 0; tree < max_trees; tree++)
	{
		Dectree_class* dectree = forest.at(tree);
		cv::Mat prediction = dectree->predict_with_idx(sample);
		used_leaves.push_back(prediction);
		class_count[prediction.at<int>(0)] += 1;
		prediction.release();
	}

	//see which class has more votes in the hash map
	bool flag_compare = false;
	int best_class;
	unsigned int max_votes;
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
	{
		if(flag_compare)
		{
			if(class_count[classes.at<int>(no_classes)] > max_votes)
			{
				best_class = classes.at<int>(no_classes);			
				max_votes = class_count[best_class];
			}
		}	
		else
		{
			best_class = classes.at<int>(no_classes);			
			max_votes = class_count[best_class];
			flag_compare = true;
		}		
	}

	//add an extra row to the output matrix indicating the predicted label
	//and the number of votes it received
	cv::Mat final_prediction = (cv::Mat_<int>(1,2) << best_class, max_votes);
	used_leaves.push_back(final_prediction); 
	
	return used_leaves;
}

void ERF_class::save_model(std::string filename, std::string name)
{
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
	Dectree_class* dectree_tmp = forest.at(0);
	cv::Mat classes = dectree_tmp->get_classes();
	
	fs << name << "{";
	
	fs << "trainingParams" << "{";
	fs << "noTrees" << max_trees;
	fs << "depthLimit" << depth_limit;
	fs << "minSamples" << min_samples;
	fs << "activeVars" << active_vars;
	fs << "classes" << classes;
	fs << "}";
	
	fs << "trees" << "[";
	for(int tree = 0; tree < max_trees; tree++)
		forest.at(tree)->save_model(fs);

	fs << "]";
	
	fs << "}";	
	fs.release();
	
}

void ERF_class::load_model(std::string filename, std::string name)
{
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
	
	cv::FileNode erf_obj = fs[name.c_str()];
	
	cv::FileNode training_params = erf_obj["trainingParams"];
	max_trees = (int)training_params["noTrees"];
	depth_limit = (int)training_params["depthLimit"];
	min_samples = (int)training_params["minSamples"];
	active_vars = (int)training_params["activeVars"];
	
	cv::FileNode trees = erf_obj["trees"];
	cv::FileNodeIterator tree_it = trees.begin(), it_end = trees.end();
	
	for(; tree_it != it_end; ++tree_it)
	{
		Dectree_class* dectree = new Dectree_class();
		cv::FileNode tmp_tree = *tree_it;
		dectree->load_model(tmp_tree, training_params);	
		forest.push_back(dectree);	
	}
	
	fs.release();
		
}




