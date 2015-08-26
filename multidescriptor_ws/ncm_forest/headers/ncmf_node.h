/* =============================================================*/
/* --------      NCMF - NCMF NODE  HEADER FILE             -----*/
/* FILENAME: ncmf_node.h 
 *
 * DESCRIPTION: header file for the struct object of a node
 * in a nrearest class mean forest. 
 *
 * VERSION: 1.0
 *
 * CREATED: 09/30/2014
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

//guard header file against multiple inclusions
#ifndef NCMF_NODE_H
#define NCMF_NODE_H
//c++ libraries
#include <vector>
#include <string>
#include <map>
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

struct ncmf_node 
{
	//information about the node
	unsigned int node_idx;
	int depth;					//depth of the node in the tree
	unsigned long long cardin;	//cardinality of the node's subtree;
	std::string type; 			//terminal (states a classification), 
								//split node (it only states an attribute and splits the tree)
								
	//information for classification
	int output_id; 							//classification id
	std::map<int, cv::Mat> left_centroids;	//feature centroids assigned to the left child
	std::map<int, cv::Mat> right_centroids;	//feature centroids assigned to the right child
	cv::Mat left_data;						//samples assigned to the left child
	cv::Mat right_data;						//samples assigned to the right child
	cv::Mat left_labels;					//left samples' labels
	cv::Mat right_labels;					//right samples' labels
	ncmf_node* f; 							//pointer to left branch
	ncmf_node* t; 							//pointer to right branch
};

typedef ncmf_node NCMF_node;


#endif

