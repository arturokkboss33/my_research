/* =============================================================*/
/* --- DECISION TREES - BINARY TREE NODE  SOURCE FILE        ---*/
/* FILENAME: dectree_bst.h 
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
#include "dectree_bst.h"
//c++ libraries
#include <iostream>
#include <sstream>

//constructor
Dectree_BST::Dectree_BST(dectree_node* rootPtr)
{
	root = rootPtr;
}

//get and set methods
dectree_node* Dectree_BST::get_root() { return root;}

void Dectree_BST::set_root(dectree_node* rootPtr)
{ root = rootPtr;}

//insert is not the same as in a common binary balance tree
//the logic to insert the node in a branch of the tree is handled in the decision tree learning algorithm
void Dectree_BST::insert_node(dectree_node** rootPtr, std::string type, unsigned int idx, int depth, int attribute, 
	double cut_point, int classification, const cv::Mat& hist)
{
	//check if the tree is empty
	if(*rootPtr == NULL)
	{
		*rootPtr = create_node(type, idx, depth, attribute, cut_point, classification, hist);
	}
	else
	{
		//normally logic to insert the node in the 'appropriate branch'
		//would be here, but as described this is handled by another program 
		std::cout << "\n Node could not be created" << std::endl;
	}

}

//ths method is not used in the decision tree learning process
//but is added because it can be used for certain inquiries
//for example, to check which attributes form part of the decision tree 
//and which not
bool Dectree_BST::search_node(dectree_node* root, int attribute_id)
{
	
	if(root != NULL)
	{
		if(!((root->type).compare("split")))
			if (attribute_id == root->attribute_id)
				return true;
		
		search_node(root->f, attribute_id);
		search_node(root->t, attribute_id);
				
	}
	else
		return false;

	//added just to make clear that something is "return" for the compiler
	return false;

}

//the next methods are used to print the decision tree
//the inOrder traversal and postOrder traversal are necessary
//to retrieve a unique strucuture of the tree
void Dectree_BST::inOrder(dectree_node* ptr)
{
	if(ptr != NULL)
	{
		inOrder(ptr->f);

		if(!((ptr->type).compare("terminal")))
			std::cout << "Terminal " << ptr->node_idx << ": " << ptr->output_id << " ";
		else
			std::cout << "Split " << ptr->node_idx << ": " << ptr->attribute_id << " ";
	
		inOrder(ptr->t);
	}
}

void Dectree_BST::postOrder(dectree_node* ptr)
{
	if(ptr != NULL)
	{
		postOrder(ptr->f);
		postOrder(ptr->t);

		if(!((ptr->type).compare("terminal")))
			std::cout << "Terminal " << ptr->node_idx << ": " << ptr->output_id << " ";
		else
			std::cout << "Split " << ptr->node_idx << ": " << ptr->attribute_id << " ";
	}
}

//create a node for the decision tree
//depending if it is a leaf or a split node, the node's fields are
//filled out differently
dectree_node* Dectree_BST::create_node(std::string type, unsigned int idx, int depth, int attribute, 
	double cut_point, int classification, const cv::Mat& hist)
{
	dectree_node* new_node = new dectree_node();
	new_node->type = type;
	new_node->node_idx = idx;
	new_node->cut_point = cut_point;
	new_node->depth = depth;
	std::stringstream ss;

	//if the node is a split, the attribute or feature that caused the
	//split is stored
	if(!(type.compare("split")))
	{
		ss << attribute;
		new_node->attribute = ss.str(); //this can be modified to receive more literal names 
						//and assign them to label each node
		new_node->attribute_id = attribute;
		new_node->output = "none";
		new_node->output_id = -1;
		new_node->classes_dist = cv::Mat();

		ss << "";
		ss.clear();

		//std::cout << "Node split inserted" << std::endl;
	}
	
	//if the node is a leaf, an output is stored which contains the 
	//classification of particular case
	else
	{
		
		ss << classification;
		new_node->attribute = "none";
		new_node->attribute_id = -1;
		new_node->output = ss.str();
		new_node->output_id = classification;
		new_node->classes_dist = hist;

		ss << "";
		ss.clear();

		//std::cout << "Node terminal inserted" << std::endl;
	}
	new_node->f = new_node->t = NULL;

	return new_node;
}


void Dectree_BST::postOrder_save(dectree_node* ptr, cv::FileStorage& fs)
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
		fs << "Attribute" << (int)ptr->attribute_id;
		fs << "CutPoint" << (float)ptr->cut_point;
		fs << "Output" << (int)ptr->output_id;
		fs << "ClassesHist" << ptr->classes_dist;
		
		fs << "}";
	}
}


