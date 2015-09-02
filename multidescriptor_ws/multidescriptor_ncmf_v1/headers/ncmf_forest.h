/* =============================================================*/
/* --- DECISION TREES- DECISION TREE CLASS HEADER FILE       ---*/
/* FILENAME: ed_class.cpp 
 *
 * DESCRIPTION: header file for the struct object which implements
 * a extremely random forest learning algorithm.
 *
 * VERSION: 1.0
 *
 * CREATED: 26/18/2014
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */


#ifndef NCMF_FOREST_H
#define NCMF_FOREST_H

//header files
#include "ncmf_class_tree.h"
//c++ libraries
#include <vector>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

class NCMF_forest
{
	public:
		//---constructor---
		NCMF_forest();
		//get and set methods
		int get_no_features(void);
		
		//---auxiliary methods---
		//void train(const cv::Mat& training_data, const cv::Mat& labels, 
		//	int depth_thresh, unsigned int samples_thresh, int classes_per_node, int no_trees);
		void train(int depth_thresh, unsigned int samples_thresh, int classes_per_node, int no_trees, std::string training_sets);
		int predict(const cv::Mat& sample);
		//cv::Mat predict(std::string test_sets);
		cv::Mat predict_with_idx(const cv::Mat& sample);
		cv::Mat predict_with_hist(const cv::Mat& sample, int* predicted_label);
		
		void save_model(std::string filename, std::string name, std::string training_file);
		void load_model(std::string filename, std::string name);

	private:
		//data members
		int max_trees;
		int tree_idx;
		int depth_limit;
		int min_samples; 
		int active_classes;
		int no_features;
		float training_time;
		cv::RNG rng; 
		std::vector<NCMF_class_tree*> forest;
		std::vector<cv::Mat> training_samples;
		//functions
		void format_training_data(std::string training_sets, cv::Mat& labels);
		void format_test_data(std::string test_sets, std::vector<cv::Mat>& test_samples);
};

#endif 
