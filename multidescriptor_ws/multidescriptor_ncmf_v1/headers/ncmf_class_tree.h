/* =============================================================*/
/* --- DECISION TREES- DECISION TREE CLASS HEADER FILE       ---*/
/* FILENAME: dectree_class.h 
 *
 * DESCRIPTION: header file for the class object which implements
 * a decision tree learning algorithm to be used in a random 
 * forest
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


#ifndef NCMF_CLASS_TREE_H
#define NCMF_CLASS_TREE_H

//headers
#include "ncmf_bst.h"
//c++ libraries
#include <string>
#include <vector>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

//structure used to represent a split in a decision tree
//where the split cut-point and the subset of samples are saved
struct dectree_split 
{
	int attr_name;
	int attr_idx;
	double cut_point;
	cv::Mat neg_attr_data;		//samples with 'negative' classification
	cv::Mat pos_attr_data;		//samples with 'positive' classification
	cv::Mat neg_attr_labels;	//negative samples' labels
	cv::Mat pos_attr_labels;	//positive samples' labels
};

//decision tree class
class NCMF_class_tree
{
	public:
		//---constructors---//
		NCMF_class_tree();		//use when the decision tree is not part of a random forest
		NCMF_class_tree(cv::RNG);	//use when the tree is part of a forest
		
		//---get and set methods---//
		int get_dectree_idx();
		void set_dectree_idx(int idx);
		NCMF_node* get_root();
		cv::Mat get_classes();
		cv::RNG get_rng();
		int get_maxDepth();
		unsigned int get_noNodes();
		unsigned int get_noLeaves();
		int get_no_features();
		
		//---auxiliary functions---//
		//use next fcn to train the decision tree
		//void train(const cv::Mat& training_data, const cv::Mat& labels, 
		//	int depth_thresh, unsigned int samples_thresh, int classes_per_node);
		void train(const std::vector<cv::Mat>* ptr_training_data, const cv::Mat& labels, int depth_thresh, 
			unsigned int samples_thresh, int classes_per_node);
		// training data     --> a sample per row, where each column represents an attribute
		// labels            --> column vector with the label for each sample (integers is the data type)
		// depth_thresh      --> maximum depth of the tree
		// samples_thresh    --> minimum number of samples a node should have to make a split; if number
		//			 of samples is less, a majority voting is made
		// vars_per_node     --> number of variables/attributes selected randomly at each node to choose the best split 
		
		int predict(const cv::Mat& sample);					//prediction - returns a class label
		cv::Mat predict_with_idx(const cv::Mat& sample);	//prediction which returns the leaves indexes reached
															//it returns a row vector with the class output and the 
															//leaf index reached in the decision process
		cv::Mat predict_with_hist(const cv::Mat& sample);	//prediction - returns class distribution

		void inOrder_tree();		//to print the structure of the tree
		void postOrder_tree();		//to print the structure of the tree
		
		void save_model(cv::FileStorage& fs);	//save model into YAML file
		void load_model(cv::FileNode& fnode_tree, cv::FileNode& fnode_params); //load model from YAML file


	private:

		//---data members - REQUIREMENTS TO CREATE A TREE---//
		cv::Mat classes;		//column vector with the label classes
		//cv::Mat init_samples;		//all the samples received for training
		const std::vector<cv::Mat>* init_samples_vec; //all the samples described with different features
		cv::Mat init_labels; 		//all the labels received for training
		//std::vector<int> attributes;	//vector with the index for each attribute
		NCMF_BST dbst;			//binary search tree class (with methods for printing, searching)
		bool is_ext_rng;		//flag to know if an external number generator is given
		cv::RNG rng; 			//random number generator
		int no_features;

		//---data members - CONSTRAINTS---//
		int dectree_idx;			//tree index
		int active_classes;			//number of classes chosen randomly to create a split
		unsigned int min_samples;	//minimum number of samples in a node to make a split
		int depth_limit;		//maximum depth the tree can reach
		int max_split_fcns;		//maximum number of splitting fcns generated per node
		int max_split_fail;		//counter for when a split fails to partition the input samples
						//it's necessary because the random selection of attributes can 
						//cause 

		//---data members - PARAMETERS OF THE CREATED TREE---//
		int max_depth;			//maximum depth of the created tree
		int min_depth;			//minimum depth of the created tree
		bool more_split_fcns;		//flag to know if the number of classes can generate more splits than allowed
		unsigned int split_nodes_idx;	//index count of all the split nodes
		unsigned int terminal_nodes_idx;//index count of all the leaves/terminal nodes

		//---auxiliary functions---//
		//methods called before learning
		void set_classes(const cv::Mat& labels);
		//void set_attributes(const cv::Mat& samples);
		//learning decision algorithm
		NCMF_node* learn_dectree(const cv::Mat& p_samples, const cv::Mat& samples, const cv::Mat& samples_data, 
						int depth, int no_split_fail);
		//methods called by the learning algorithm
		//base cases
		int plurality(const cv::Mat& samples);//majority count/vote
		bool check_classif(const cv::Mat& samples);//check if all the examples have the same classification
		//methods to create splits
		NCMF_node* erf_split(const cv::Mat& samples, const cv::Mat& labels); //see which attribute has more information gain
		cv::Mat pick_classes(const cv::Mat curr_labels); //pick k random classes
		//std::vector<double> gen_ran_splits(std::vector<int> attr, const cv::Mat& samples); //generate random splits
		double compute_erf_entropy(const cv::Mat& labels, const cv::Mat& neg_labels, const cv::Mat& pos_labels); //shannon entropy
		double compute_entropy(const cv::Mat& labels); //standard entropy
		cv::Mat get_classes_hist(const cv::Mat& sample_labels); // get the class distribution at a leaf node

		//extra methods
		void find_depth(NCMF_node*); 

};

#endif 
