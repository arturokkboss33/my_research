/* =============================================================*/
/* --- DECISION TREES - BINARY TREE NODE  SOURCE FILE        ---*/
/* FILENAME: NCMF_bst.h 
 *
 * DESCRIPTION: source file for the struct object of a binary
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

//headers
#include "ncmf_bst.h"
//c++ libraries
#include <iostream>
#include <sstream>

//constructor
NCMF_BST::NCMF_BST(NCMF_node* rootPtr)
{
	root = rootPtr;
}

//get and set methods
NCMF_node* NCMF_BST::get_root() { return root;}

void NCMF_BST::set_root(NCMF_node* rootPtr)
{ root = rootPtr;}

//insert is not the same as in a common binary balance tree
//the logic to insert the node in a branch of the tree is handled in the decision tree learning algorithm
void NCMF_BST::insert_node(NCMF_node** rootPtr, std::string type, unsigned int idx, int depth, int classification, 
	std::map<int, cv::Mat> left_centr, std::map<int, cv::Mat> right_centr, const cv::Mat& left_data, const cv::Mat& right_data, 
	const cv::Mat& left_labels, const cv::Mat& right_labels, const cv::Mat& hist, std::string feature_type, int feature_idx)
{
	//check if the tree is empty
	if(*rootPtr == NULL)
	{
		*rootPtr = create_node(type, idx, depth, classification, left_centr, right_centr, left_data, 
			right_data, left_labels, right_labels, hist, feature_type, feature_idx);
	}
	else
	{
		//normally logic to insert the node in the 'appropriate branch'
		//would be here, but as described this is handled by another program 
		std::cout << "\n Node could not be created" << std::endl;
	}

}

void NCMF_BST::insert_node(NCMF_node** rootPtr, NCMF_node* node)
{
	//check if the tree is empty
	if(*rootPtr == NULL)
	{
		*rootPtr = node;
	}
	else
	{
		//normally logic to insert the node in the 'appropriate branch'
		//would be here, but as described this is handled by another program 
		std::cout << "\n Node could not be created" << std::endl;
	}
}

//the next methods are used to print the decision tree
//the inOrder traversal and postOrder traversal are necessary
//to retrieve a unique strucuture of the tree
void NCMF_BST::inOrder(NCMF_node* ptr)
{
	if(ptr != NULL)
	{
		inOrder(ptr->f);

		if(!((ptr->type).compare("terminal")))
			std::cout << "Terminal " << ptr->node_idx << ": " << ptr->output_id << " ";
		else
			std::cout << "Split " << ptr->node_idx <<  " ";
	
		inOrder(ptr->t);
	}
}

void NCMF_BST::postOrder(NCMF_node* ptr)
{
	if(ptr != NULL)
	{
		postOrder(ptr->f);
		postOrder(ptr->t);

		if(!((ptr->type).compare("terminal")))
			std::cout << "Terminal " << ptr->node_idx << ": " << ptr->output_id << " ";
		else
			std::cout << "Split " << ptr->node_idx << " ";
	}
}

//create a node for the decision tree
//depending if it is a leaf or a split node, the node's fields are
//filled out differently
NCMF_node* NCMF_BST::create_node(std::string type, unsigned int idx, int depth, int classification, const std::map<int, cv::Mat> left_centr, 
	std::map<int, cv::Mat> right_centr, const cv::Mat& left_data, const cv::Mat& right_data, 
	const cv::Mat& left_labels, const cv::Mat& right_labels, const cv::Mat& hist, std::string feature_type, int feature_idx)
{
	NCMF_node* new_node = new NCMF_node();
	new_node->type = type;
	new_node->node_idx = idx;
	new_node->depth = depth;
	new_node->feature_type = feature_type;
	new_node->feature_idx = feature_idx;

	//if the node is a split, the attribute or feature that caused the
	//split is stored
	if(!(type.compare("split")))
	{
		
		new_node->output_id = -1;
		new_node->left_centroids = left_centr;
		new_node->right_centroids = right_centr;
		new_node->left_data = left_data.clone();
		new_node->right_data = right_data.clone();
		new_node->left_labels = left_labels.clone();
		new_node->right_labels = right_labels.clone();
		new_node->classes_dist = cv::Mat();

		//std::cout << "Node split inserted" << std::endl;
	}
	
	//if the node is a leaf, an output is stored which contains the 
	//classification of particular case
	else
	{
		
		new_node->output_id = classification;
		//if it is a terminal node, we store all the remaining examples in the left_data/label  matrix
		new_node->left_centroids = left_centr;
		new_node->right_centroids = right_centr;
		new_node->left_data = left_data.clone();
		new_node->right_data = cv::Mat();
		new_node->left_labels = left_labels.clone();
		new_node->right_labels = cv::Mat();
		new_node->classes_dist = hist.clone();


		//std::cout << "Node terminal inserted" << std::endl;
	}
	new_node->f = new_node->t = NULL;

	return new_node;
}

void NCMF_BST::postOrder_save(NCMF_node* ptr, cv::FileStorage& fs)
{
	std::string ter = "terminal";
	std::string spl = "split";
	if(ptr != NULL)
	{
		postOrder_save(ptr->f, fs);
		postOrder_save(ptr->t, fs);

		fs << "{";
		
		if(!((ptr->type).compare("terminal")))
		{
			fs << "Type" << ter;
		}
		else
		{
			fs << "Type" << spl;
		}
		fs << "Idx" << (int)ptr->node_idx;
		fs << "Depth" << (int)ptr->depth;
		fs << "FeatureType" << ptr->feature_type;
		fs << "FeatureIdx" << (int)ptr->feature_idx;
		fs << "Output" << (int)ptr->output_id;
		
		cv::Mat tmp_centroids, tmp_centroid_labels;
		for(std::map<int,cv::Mat>::iterator it = ptr->left_centroids.begin(); it != ptr->left_centroids.end(); ++it ){
			tmp_centroids.push_back(it->second);
			tmp_centroid_labels.push_back(it->first);
		}
		fs << "LeftCentroids" << tmp_centroids;
		fs << "LeftCentroidLabels" << tmp_centroid_labels;
		fs << "LeftSamples" << ptr->left_data;
		fs << "LeftSampleLabels" << ptr->left_labels;
		tmp_centroids.release(); tmp_centroid_labels.release();
		
		for(std::map<int,cv::Mat>::iterator it = ptr->right_centroids.begin(); it != ptr->right_centroids.end(); ++it ){
			tmp_centroids.push_back(it->second);
			tmp_centroid_labels.push_back(it->first);
		}
		fs << "RightCentroids" << tmp_centroids;
		fs << "RightCentroidLabels" << tmp_centroid_labels;
		fs << "RightSamples" << ptr->right_data;
		fs << "RightSampleLabels" << ptr->right_labels;
		tmp_centroids.release(); tmp_centroid_labels.release();
		
		fs << "ClassesHist" << ptr->classes_dist;
		
		fs << "}";
	}
}
