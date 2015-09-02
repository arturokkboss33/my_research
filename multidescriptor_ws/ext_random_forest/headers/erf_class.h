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


#ifndef ERF_CLASS_H
#define ERF_CLASS_H

//header files
#include "dectree_class.h"
//c++ libraries
#include <vector>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

class ERF_class
{
	public:
		//---constructor---
		ERF_class();
		//get and set methods
		
		//---auxiliary methods---
		void train(const cv::Mat& training_data, const cv::Mat& labels, int depth_thresh, unsigned int samples_thresh, int vars_per_node, int no_trees);
		int predict(const cv::Mat& sample);
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
		int active_vars;
		float training_time;
		cv::RNG rng; 
		std::vector<Dectree_class*> forest;
};

#endif 
