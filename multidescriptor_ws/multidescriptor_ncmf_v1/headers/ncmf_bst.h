/* =============================================================*/
/* --- DECISION TREES-BINARY TREE SEARCH CLASS HEADER   FILE ---*/
/* FILENAME: dectree_bst.cpp 
 *
 * DESCRIPTION: header file for the struct object of a binary
 * search tree, modified for its use in a decision tree 
 * learning algorithm.
 *
 * VERSION: 1.0
 *
 * CREATED: 03/16/2013
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

#ifndef NCMF_BST_H
#define NCMF_BST_H

#include "ncmf_node.h"

#include <string>
#include <map>

class NCMF_BST
{

	public:
		NCMF_BST(NCMF_node* rootPtr = NULL);
		//get and set methods
		NCMF_node* get_root();
		void set_root(NCMF_node* rootPtr);
		//insert a node and complete its fields
		//based on it's type: terminal or splitting
		void insert_node(NCMF_node** rootPtr, std::string type, unsigned int idx, int depth, int classification, 
					std::map<int, cv::Mat> left_centr, std::map<int, cv::Mat> right_centr, const cv::Mat& left_data, 
					const cv::Mat& right_data, const cv::Mat& left_labels, const cv::Mat& right_labels, const cv::Mat& hist, 
					std::string feature_type, int feature_idx); 
		void insert_node(NCMF_node** rootPtr, NCMF_node* node);
		//common methods to traverse a binary tree
		void inOrder(NCMF_node* root);
		void postOrder(NCMF_node* root);
		//save or load model from YAML file
		void postOrder_save(NCMF_node* ptr, cv::FileStorage& fs);

	private:
		//member data
		NCMF_node* root;
		//auxiliary method to insert a node
		NCMF_node* create_node(std::string type, unsigned int idx, int depth, int classification, 
					std::map<int, cv::Mat> left_centr, std::map<int, cv::Mat> right_centr, const cv::Mat& left_data,
					const cv::Mat& right_data, const cv::Mat& left_labels, const cv::Mat& right_labels, const cv::Mat& hist,
					std::string feature_type, int feature_idx);
};


#endif
