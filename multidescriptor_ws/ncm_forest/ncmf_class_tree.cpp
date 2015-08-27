/* =============================================================*/
/* --- DECISION TREES- DECISION TREE CLASS SOURCE FILE       ---*/
/* FILENAME: NCMF_class_tree.cpp 
 *
 * DESCRIPTION: source file for the struct object which implements
 * a decision tree learning algorithm.
 *
 * VERSION: 1.0
 *
 * CREATED: 03/18/2013
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

//headers
#include "ncmf_class_tree.h"
//libraries
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <map>
#include <limits>
#include <bitset>
#include <queue>
#include <boost/dynamic_bitset.hpp>

//*********************** CONSTRUCTORS *****************************//
NCMF_class_tree::NCMF_class_tree()
{
	dbst = NULL;
	classes = cv::Mat(0,1,CV_32SC1);
	dectree_idx = 0;
	max_depth = 0;
	min_depth = 1000;
	split_nodes_idx = 0;
	terminal_nodes_idx = 0;
	depth_limit = 0;
	max_split_fcns = 10; //this is represented as number of bits, so the "true" value is 2^10=1024
	more_split_fcns = false;
	max_split_fail = std::numeric_limits<int>::max();
	is_ext_rng = false;
}

NCMF_class_tree::NCMF_class_tree(cv::RNG ext_rng)
{
	dbst = NULL;
	classes = cv::Mat(0,1,CV_32SC1);
	dectree_idx = 0;
	max_depth = 0;
	min_depth = 1000;
	split_nodes_idx = 0;
	terminal_nodes_idx = 0;
	depth_limit = 0;
	max_split_fcns = 10; //this is represented as number of bits, so the "true" value is 2^10=1024
	more_split_fcns = false;
	max_split_fail = std::numeric_limits<int>::max();
	is_ext_rng = true;
	rng = ext_rng;
}
//******************************************************************//

//*********************** GET-SET METHODS *************************//
int NCMF_class_tree::get_dectree_idx() { return dectree_idx;}

void NCMF_class_tree::set_dectree_idx(int idx)
{ dectree_idx = idx;}

NCMF_node* NCMF_class_tree::get_root(){ return dbst.get_root();}

cv::Mat NCMF_class_tree::get_classes() { return classes;}

cv::RNG NCMF_class_tree::get_rng() {return rng;}

int NCMF_class_tree::get_maxDepth() { return max_depth;}

unsigned int NCMF_class_tree::get_noNodes() { return (split_nodes_idx+terminal_nodes_idx);}

unsigned int NCMF_class_tree::get_noLeaves() { return terminal_nodes_idx;}

//******************************************************************//

//*********************** PUBLIC FUNCTIONS *************************//

//main method to train the decision treee
void NCMF_class_tree::train(const cv::Mat& training_data, const cv::Mat& labels, int depth_thresh, unsigned int samples_thresh, int classes_per_node)
{
	//---(1)initialize random generator if no external is given---//
	if(!is_ext_rng)
	{
		rng = cv::RNG(time(NULL));
		//std::cout << "No ext rng" << std::endl;
	}

	//---(2)determine the number of classes based on the training data---//
	set_classes(labels);
	//std::cout << "Classes:\n" << classes << std::endl;
	//---(3)create a copy of the training data and indexes to it---//
	cv::Mat samples_data_idx(0,1,CV_32SC1);
	for(int row = 0; row < training_data.rows; row++)
		samples_data_idx.push_back(row);
	//we want to have a copy of the data for retraining
	init_samples = training_data.clone();
	init_labels = labels.clone();

	//---(3)make a vector giving an id to each attribute---//
	//set_attributes(training_data);
	//for debbugging
	/*
	for(std::vector<int>::iterator it = attributes.begin(); it != attributes.end(); ++it)
		std::cout << *it << " ";
	std::cout << std::endl;
	*/

	//---(4)verify constraints---//
	//maximum depth
	if(depth_thresh < 0)
		depth_limit = std::numeric_limits<int>::max();
	else
		depth_limit = depth_thresh;
	//minimum samples
	min_samples = samples_thresh;
	//active variables
	if(classes.rows < classes_per_node)
		active_classes = classes.rows;
	else
		active_classes = classes_per_node;
	//max number of bits used to generate splits
	if(active_classes > max_split_fcns)
		more_split_fcns = true;
	//maximum number of split fails
	max_split_fail = 5;

	//---(5)train the tree---//
	int depth = 1;
	int no_split_fail = 0;
	dbst.set_root(learn_dectree(cv::Mat(),init_labels, samples_data_idx, depth, no_split_fail));
	find_depth(dbst.get_root()); //to find the real-true depth of the created tree

}


//prediction method --- only returns the predicted label
int NCMF_class_tree::predict(const cv::Mat& sample)
{
	NCMF_node* tmp_node = dbst.get_root();
	int prediction = -1;

	while(tmp_node != NULL)
	{
		if( !(tmp_node->type.compare("terminal")) )
			return tmp_node->output_id;
		else
		{
			bool flag_compare = false;
			double min_dist_left = 0;
			for(std::map<int,cv::Mat>::iterator it = tmp_node->left_centroids.begin(); 
				it != tmp_node->left_centroids.end(); ++it)
			{
				double dist = cv::norm(sample,it->second);
				if(!flag_compare)
				{
					min_dist_left = dist;
					flag_compare = true;
				}
				else
				{
					if(dist < min_dist_left)
						min_dist_left = dist;
				}
			}
			flag_compare = false;
			double min_dist_right = 0.;
			for(std::map<int,cv::Mat>::iterator it = tmp_node->right_centroids.begin(); 
				it != tmp_node->right_centroids.end(); ++it)
			{
				double dist = cv::norm(sample,it->second);
				if(!flag_compare)
				{
					min_dist_right = dist;
					flag_compare = true;
				}
				else
				{
					if(dist < min_dist_right)
						min_dist_right = dist;
				}
			}

			if( min_dist_left < min_dist_right)
				tmp_node = tmp_node->f;
			else
				tmp_node = tmp_node->t;
		}
	}

	return prediction;


}

//prediction method --- returns the predicted label and leaf index which output the prediction
cv::Mat NCMF_class_tree::predict_with_idx(const cv::Mat& sample)
{
	NCMF_node* tmp_node = dbst.get_root();
	cv::Mat prediction_mat = cv::Mat(0,1,CV_32SC1);
	int prediction = -1;
	int id = -1;

	while(tmp_node != NULL)
	{
		if( !(tmp_node->type.compare("terminal")) )
		{
			prediction = tmp_node->output_id;
			id = tmp_node->node_idx;
			prediction_mat.push_back(prediction);
			prediction_mat.push_back(id);
			prediction_mat = prediction_mat.reshape(0,1);
			return prediction_mat;
		}
		else
		{
			bool flag_compare = false;
			double min_dist_left = 0;
			for(std::map<int,cv::Mat>::iterator it = tmp_node->left_centroids.begin(); 
				it != tmp_node->left_centroids.end(); ++it)
			{
				double dist = cv::norm(sample,it->second);
				if(!flag_compare)
				{
					min_dist_left = dist;
					flag_compare = true;
				}
				else
				{
					if(dist < min_dist_left)
						min_dist_left = dist;
				}
			}
			flag_compare = false;
			double min_dist_right = 0.;
			for(std::map<int,cv::Mat>::iterator it = tmp_node->right_centroids.begin(); 
				it != tmp_node->right_centroids.end(); ++it)
			{
				double dist = cv::norm(sample,it->second);
				if(!flag_compare)
				{
					min_dist_right = dist;
					flag_compare = true;
				}
				else
				{
					if(dist < min_dist_right)
						min_dist_right = dist;
				}
			}

			if( min_dist_left < min_dist_right)
				tmp_node = tmp_node->f;
			else
				tmp_node = tmp_node->t;
		}
	}

	return prediction_mat;

}

cv::Mat NCMF_class_tree::predict_with_hist(const cv::Mat& sample){
	NCMF_node* tmp_node = dbst.get_root();
	//int prediction = -1;
	cv::Mat prediction_mat;

	while(tmp_node != NULL)
	{
		if( !(tmp_node->type.compare("terminal")) )
			return tmp_node->classes_dist;
		else
		{
			bool flag_compare = false;
			double min_dist_left = 0;
			for(std::map<int,cv::Mat>::iterator it = tmp_node->left_centroids.begin(); 
				it != tmp_node->left_centroids.end(); ++it)
			{
				double dist = cv::norm(sample,it->second);
				if(!flag_compare)
				{
					min_dist_left = dist;
					flag_compare = true;
				}
				else
				{
					if(dist < min_dist_left)
						min_dist_left = dist;
				}
			}
			flag_compare = false;
			double min_dist_right = 0.;
			for(std::map<int,cv::Mat>::iterator it = tmp_node->right_centroids.begin(); 
				it != tmp_node->right_centroids.end(); ++it)
			{
				double dist = cv::norm(sample,it->second);
				if(!flag_compare)
				{
					min_dist_right = dist;
					flag_compare = true;
				}
				else
				{
					if(dist < min_dist_right)
						min_dist_right = dist;
				}
			}

			if( min_dist_left < min_dist_right)
				tmp_node = tmp_node->f;
			else
				tmp_node = tmp_node->t;
		}
	}

	return prediction_mat;

}

//methods that invoke the fucntions to print the structure of the tree
void NCMF_class_tree::inOrder_tree()
{
	dbst.inOrder(dbst.get_root());
}
void NCMF_class_tree::postOrder_tree()
{
	dbst.postOrder(dbst.get_root());
}

//methods to save and load the built model into memory (YAML file)
//methods to save and load the model into a YAML file for reuse
void NCMF_class_tree::save_model(cv::FileStorage& fs)
{
	
	fs << "{";
	
	fs << "treeIndex" << dectree_idx;
	fs << "noLeaves" << (int)terminal_nodes_idx;
	fs << "noNodes" << (int)(split_nodes_idx+terminal_nodes_idx);
	fs << "maxDepth" << max_depth;
	//fs << "classes" << classes;
	
	fs << "nodes" << "[";
	dbst.postOrder_save(dbst.get_root(),fs);	
	fs << "]";
	
	fs << "}";	
	
}

void NCMF_class_tree::load_model(cv::FileNode& fnode_tree, cv::FileNode& fnode_params)
{
	std::queue<NCMF_node*> dummy_queue;
	//cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
		
	dectree_idx = (int)fnode_tree["treeIndex"];
	max_depth = (int)fnode_tree["maxDepth"];
	terminal_nodes_idx = (int)fnode_tree["noLeaves"];
	split_nodes_idx = (int)fnode_tree["noNodes"] - terminal_nodes_idx;
	
	depth_limit = (int)fnode_params["depthLimit"];
	min_samples = (int)fnode_params["minSamples"];
	active_classes = (int)fnode_params["activeClasses"];
	fnode_params["classes"] >> classes;		
	
	cv::FileNode tree_nodes = fnode_tree["nodes"];
	cv::FileNodeIterator node_it = tree_nodes.begin(), it_end = tree_nodes.end();
	std::vector< std::queue<NCMF_node*> > tree_queue(max_depth,dummy_queue);
	//std::cout << "***" << tree_queue.size() << std::endl;
	
	for(; node_it != it_end; ++node_it)
	{	
		//===For debugging===//
		//std::cout << (std::string)(*node_it)["Type"] << " " << (int)(*node_it)["Idx"] << std::endl;
		
		NCMF_node* nod = new NCMF_node();
		nod->node_idx = (int)(*node_it)["Idx"];
		nod->type = (std::string)(*node_it)["Type"];
		nod->depth = (int)(*node_it)["Depth"];
		nod->output_id = (int)(*node_it)["Output"];
		(*node_it)["LeftSamples"] >> nod->left_data;
		(*node_it)["LeftSampleLabels"] >> nod->left_labels;
		(*node_it)["RightSamples"] >> nod->right_data;
		(*node_it)["RightSampleLabels"] >> nod->right_labels;
		cv::Mat tmp_labels, tmp_vals;
		std::map<int,cv::Mat> tmp_centroids;
		(*node_it)["LeftCentroidLabels"] >> tmp_labels;
		(*node_it)["LeftCentroids"] >> tmp_vals;
		for(int no_centroid = 0; no_centroid < tmp_vals.rows; no_centroid++){
			tmp_centroids.insert( std::pair<int, cv::Mat>( tmp_labels.at<int>(no_centroid,1),
					tmp_vals.row(no_centroid).clone() ) );
		}
		nod->left_centroids.swap(tmp_centroids);
		tmp_centroids.clear();
		tmp_labels.release(); tmp_vals.release();
		
		(*node_it)["RightCentroidLabels"] >> tmp_labels;
		(*node_it)["RightCentroids"] >> tmp_vals;
		for(int no_centroid = 0; no_centroid < tmp_vals.rows; no_centroid++){
			tmp_centroids.insert( std::pair<int, cv::Mat>( tmp_labels.at<int>(no_centroid,1),
					tmp_vals.row(no_centroid).clone() ) );
		}
		nod->right_centroids.swap(tmp_centroids);
		tmp_centroids.clear();
		tmp_labels.release(); tmp_vals.release();	
		
		(*node_it)["ClassesHist"] >> nod->classes_dist;;
		
		
		if(nod->depth < max_depth)
		{
			if( tree_queue.at((nod->depth)).size() == 2)
			{
				NCMF_node* tmpnod = tree_queue.at(nod->depth).front();
				tree_queue.at(nod->depth).pop();
				nod->f = tmpnod;
				tmpnod = tree_queue.at(nod->depth).front();
				tree_queue.at(nod->depth).pop();
				nod->t = tmpnod;
			}
		}
		tree_queue.at((nod->depth)-1).push(nod);
		
	}
	
	//std::cout << "ooo" << std::endl;
	dbst.set_root(tree_queue.at(0).front());
	tree_queue.at(0).pop();
	//fs.release();
		
}

//******************************************************************//

//*********************** PRIVATE FUNCTIONS *************************//

//+++ METHODS USED BEFORE THE LEARNING PROCESS +++//
//check number of classes; labels will be represented as a column vector
void NCMF_class_tree::set_classes(const cv::Mat& labels)
{
	bool flag_new_class = true;
	int k = 0;
	
	classes.push_back(labels.at<int>(0));
	for(int e = 1; e < labels.rows; e++)
	{
		flag_new_class = true;
		k = 0;
		while(flag_new_class == true && k < classes.rows)
		{
			if(labels.at<int>(e) == classes.at<int>(k))
				flag_new_class = false;
			k++;
		}	
		if(flag_new_class)
			classes.push_back(labels.at<int>(e));
	}

}



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

//algorithm to learn a decision tree
//inputs: parent labels, current labels, current data samples, current tree depth, current split fails
//outputs: it returns the root of the decision tree (or one of its subtrees in the recursion) 
NCMF_node* NCMF_class_tree::learn_dectree(const cv::Mat& p_samples_labels, const cv::Mat& samples_labels, 
	const cv::Mat& samples_data, int depth, int no_split_fail)
{
	//for debugging
	//std::cout << "Depth: " << depth << std::endl;
	//std::cout << "t: " << terminal_nodes_idx << std::endl;
	//std::cout << "n: " << split_nodes_idx << std::endl;

	//dectree_split* split = new dectree_split();	
	NCMF_BST dectree;
	NCMF_node* dectree_root = dectree.get_root();
	std::map<int, cv::Mat> dummy_table;
	
	//void insert_node(NCMF_node** rootPtr, std::string type, unsigned int idx, int depth, int classification, 
	//				cv::Mat& left_centr,  cv::Mat& right_centr, cv::Mat left_data, cv::Mat right_data, 
	//				cv::Mat left_labels, cv::Mat right_labels);
	//check the base cases to get out of the recursion
	
	//if there are no more examples	
	if(samples_labels.empty())
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, plurality(p_samples_labels), 
			dummy_table, dummy_table, samples_data, cv::Mat(), samples_labels, cv::Mat(), get_classes_hist(p_samples_labels));
		return dectree_root;
	}
	//if all examples have the same classification
	else if(check_classif(samples_labels))
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, samples_labels.at<int>(0), 
			dummy_table, dummy_table, samples_data, cv::Mat(), samples_labels, cv::Mat(), get_classes_hist(samples_labels));
		return dectree_root;
	}
	//if this case is hit, there are attributes and samples to analyze
	//it checks the maximum depth and minimum sample constraints
	else if(depth >= depth_limit || (unsigned)samples_labels.rows < min_samples)
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, plurality(samples_labels), 
			dummy_table, dummy_table, samples_data, cv::Mat(), samples_labels, cv::Mat(), get_classes_hist(samples_labels));
		return dectree_root;
	}
	
	//else call the method recursively after finding the best split
	else
	{
		//find the attrribute with the highest information gain	
		NCMF_node* split = erf_split(samples_data, samples_labels);
		//split = erf_split(samples_data, samples_labels);

		//for debugging
		//double h_curr = compute_entropy(samples_labels);
		//std::cout << "Current entropy: " << h_curr << std::endl;
		//std::cout << "Best attribute: " << split->attr_name << std::endl;
		//std::cout << "Cut point: " << split->cut_point << std::endl;

		//if the samples were not further separated, it's a bad split
		if (split->left_labels.rows == 0 || split->right_labels.rows == 0 )
			no_split_fail++;
		//if they were separated, reset the counter
		else
			no_split_fail = 0;
		//if the maximum number of split fails is reached, convert the node into a leaf/terminal node
		if(no_split_fail >= max_split_fail)
		{
			//std::cout << "FAIL" << std::endl;
			dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, plurality(samples_labels), 
				dummy_table, dummy_table, samples_data, cv::Mat(), samples_labels, cv::Mat(), get_classes_hist(samples_labels));
			no_split_fail = 0;
			return dectree_root;
		}	
		//if there is no split fail, insert a split node and call the method recursively
		else
		{
			//create a node split with the best attribute as generator of a split
			//dectree.insert_node(&dectree_root,"split", (++split_nodes_idx), depth, split->attr_name, split->cut_point, -1);
			split->output_id = -1;
			split_nodes_idx+=1;
			split->node_idx = split_nodes_idx;
			split->depth = depth;
			split->type = "split";	
			dectree.insert_node(&dectree_root, split);

			//call the fcn recursively
			//Here, no attributes are erased
			(dectree_root)->f = learn_dectree(samples_labels,split->left_labels,split->left_data,(depth+1), no_split_fail);	
			(dectree_root)->t = learn_dectree(samples_labels,split->right_labels,split->right_data,(depth+1), no_split_fail);

			return dectree_root;
		}	
	}
	
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

//+++ METHODS CALLED BY THE LEARNING FUNCTION +++//

//Acoording to the basic cases of the learning algorithm
//To decide a classification output it's necessary to do a majority vote
//according to the remaining training examples
//Reminder: give samples matrix as a column vector
int NCMF_class_tree::plurality(const cv::Mat& labels)
{

	std::map<int, unsigned int> samples_per_class;

	//create a hash table with the classes names as keys, they save the number
	//of samples that have that particular label
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		samples_per_class.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );

	for(int ex = 0; ex < labels.rows; ex++)
		samples_per_class[labels.at<int>(ex)] += 1;

	//iterate through the hash table to see which class has more votes
	//tied classes are kept in a vector
	std::map<int,unsigned int>::iterator it = samples_per_class.begin();
	int best_class = it->first;
	unsigned int max_votes = it->second;
	++it;
	cv::Mat tied_classes(0,1, CV_32SC1);
	tied_classes.push_back(best_class);
	
	for(; it != samples_per_class.end(); ++it)
	{
		if(it->second > max_votes)
		{
			best_class = it->first;
			max_votes = it->second;
			tied_classes.release();
			tied_classes.push_back(best_class);
		}
		else if(it->second == max_votes)
			tied_classes.push_back(it->first);
	}

	//if there are tied classes, pick one of them randomly
	//cv::RNG rng(time(NULL));
	if(tied_classes.rows > 1)
	{
		//std::cout << "Tie" << std::endl;
		int random_class = (int)(rng.uniform(0.,(double)tied_classes.rows));
		//Note: rng.uniform does not accept int values
		return tied_classes.at<int>(random_class);
	}
	else
		return best_class;	
}

//check if the remaining examples have the same classification
//if true, there is no necessity to continue the learning process
//through the respective branch
//Reminder: samples is a column vector
bool NCMF_class_tree::check_classif(const cv::Mat& labels)
{

	bool flag_dif_class = false;
	int ex = 1;

	while(flag_dif_class == false && ex < labels.rows)
	{
		if(labels.at<int>(0) != labels.at<int>(ex))
			flag_dif_class = true;
		ex++;
	}

	return !flag_dif_class;

}

//main method to decide which is the best split possible among the randomly selected variables
NCMF_node* NCMF_class_tree::erf_split(const cv::Mat& samples, const cv::Mat& labels)
{
	//---select K classes randomly---//
	cv::Mat selected_classes = pick_classes(labels);

	//---select the samples that correspond to these classes (and their labels)---//
	cv::Mat selected_samples(0,1,CV_32SC1);
	cv::Mat selected_labels(0,1,CV_32SC1);
	bool sample_belongs = false;
	int idx_class = 0;
	for(int s = 0; s < samples.rows; s++)
	{	
		idx_class = 0;
		sample_belongs = false;
		while( !sample_belongs && idx_class < selected_classes.rows)
		{
			if(labels.at<int>(s) == selected_classes.at<int>(idx_class))
			{
				sample_belongs = true;
				selected_samples.push_back(samples.at<int>(s));
				selected_labels.push_back(labels.at<int>(s));
			}
			idx_class++;
		}
	}
	//for debugging
	//std::cout << "Selected: " << std::endl;
	//std::cout << selected_samples << std::endl;
	//std::cout << selected_labels << std::endl;

	//---compute the centroids of the selected samples---//
	//create hash table
	std::map<int, cv::Mat> centroids;
	std::map<int, unsigned int> cc_centroids; 
	for(int idx = 0; idx < selected_classes.rows; idx++)
	{
		centroids.insert( std::pair<int, cv::Mat>( selected_classes.at<int>(idx),
					cv::Mat(1, init_samples.cols, CV_32FC1, cv::Scalar::all(0)) ) );
		cc_centroids.insert( std::pair<int, unsigned int>( selected_classes.at<int>(idx), 0) );
	}
	//compute the mean of the samples per class
	for(int s = 0; s < selected_samples.rows; s++)
	{
		centroids[selected_labels.at<int>(s)] += init_samples.row(selected_samples.at<int>(s));
		cc_centroids[selected_labels.at<int>(s)] +=1;
	}

	//for debugging
	/*
	std::cout << "Centroids: " << std::endl;
	for(std::map<int,cv::Mat>::iterator it = centroids.begin(); it != centroids.end(); ++it)
	{
		std::cout << it->first << ": \n" << it->second << std::endl;
	}
	std::cout << "Counters: " << std::endl;
	for(std::map<int,unsigned int>::iterator it = cc_centroids.begin(); it != cc_centroids.end(); ++it)
	{
		std::cout << it->first << ": \n" << it->second << std::endl;
	}
	*/

	for(int idx = 0; idx < selected_classes.rows; idx++)
	{
		double scale = (1./cc_centroids[selected_classes.at<int>(idx)]);
		centroids[selected_classes.at<int>(idx)] *= scale;
	}

	//for debugging
	/*
	std::cout << "Centroids: " << std::endl;
	for(std::map<int,cv::Mat>::iterator it = centroids.begin(); it != centroids.end(); ++it)
	{
		std::cout << it->first << ": \n" << it->second << std::endl;
	}
	*/
	//---create M split fcns at random---//

	//first compute which is the closest centroid to each sample
	cv::Mat closest_centr_pos(0,1,CV_32SC1);
	for(int s = 0; s < selected_samples.rows; s++)
	{
		//find closest centroid for sample ex
		cv::Mat ex = init_samples.row(selected_samples.at<int>(s));
		int class_pos = 0;
		double min_dist = 0.;
		bool flag_compare_centroids = false;
		for(int idx = 0; idx < selected_classes.rows; idx++)
		{
			double dist = cv::norm(ex,centroids[selected_classes.at<int>(idx)]);
			if(!flag_compare_centroids)
			{
				class_pos = idx;
				min_dist = dist;
				flag_compare_centroids = true;
			}
			else
			{
				if(dist < min_dist)
				{
					class_pos = idx;
					min_dist = dist;
				}
			}
		}
		closest_centr_pos.push_back(class_pos);
	}

	//for debugging
	//std::cout << "Closest centroids pos: " << std::endl;
	//std::cout << closest_centr_pos << std::endl;

	//variables to create partitions
	boost::dynamic_bitset<> partition(selected_classes.rows);
	//std::bitset<active_classes> partition;
	partition.reset();
	partition.set(0,1);
	boost::dynamic_bitset<> stop_partition(selected_classes.rows);
	//std::bitset<active_classes> stop_partition;
	stop_partition.reset();
	stop_partition.set(selected_classes.rows-1,1);
	int cc_partitions = 1;
	//others
	bool flag_compare_info_gain = false;
	double max_split_score = 0.;
	boost::dynamic_bitset<> best_partition(selected_classes.rows);
	//std::bitset<active_classes> best_partition;

	//split the samples according to their closest centroid
	if(!more_split_fcns)
	{
		//split sampples
		while(stop_partition != partition)
		{
			//std::cout << partition << std::endl;
			cv::Mat right_labels(0,1,CV_32SC1);
			cv::Mat left_labels(0,1,CV_32SC1);
			for(int ex = 0; ex < closest_centr_pos.rows; ex++)
			{
				//see to which branch the closest centroid was assigned and store the sample there
				//std::cout << "&&& " << partition[ex] << std::endl;
				if(partition[closest_centr_pos.at<int>(ex)])
					right_labels.push_back(selected_labels.at<int>(ex));
				else
					left_labels.push_back(selected_labels.at<int>(ex));
			}

			//compute the shannon entropy of the obtained split
			double split_score = compute_erf_entropy(selected_labels, left_labels, right_labels);
			//std::cout << split_score << std::endl;

			//compare the score with the previous ones, and save the best
			if(flag_compare_info_gain)
			{
				if(split_score > max_split_score)
				{
					max_split_score = split_score;
					best_partition = partition;
				}
			}
			else
			{
				max_split_score = split_score;
				best_partition = partition;
				flag_compare_info_gain = true;
			}

			//create the bitset that will generate the next partition
			cc_partitions++;
			for(int no_bit = 0; no_bit < active_classes; no_bit++)
			{
				int d = (int)(pow(2.,(double)no_bit));
				if(cc_partitions%d == 0)
					partition.flip(no_bit);
			}

			right_labels.release();
			left_labels.release();
		}
	}

	//std::cout << "Best partition: " << best_partition << std::endl;
	//with the best partition found, divide the data and labels and create a node
	cv::Mat final_right_labels(0,1,CV_32SC1);
	cv::Mat final_left_labels(0,1,CV_32SC1);
	cv::Mat final_right_data(0,1,CV_32FC1);
	cv::Mat final_left_data(0,1,CV_32FC1);
	std::map<int, cv::Mat> right_centroids;
	std::map<int, cv::Mat> left_centroids;
	for(int ex = 0; ex < selected_samples.rows; ex++)
	{
		//see to which branch the closest centroid was assigned and store the sample there
		if(best_partition[closest_centr_pos.at<int>(ex)])
		{
			final_right_labels.push_back(selected_labels.at<int>(ex));
			final_right_data.push_back(selected_samples.at<int>(ex));			
		}
		else
		{
			final_left_labels.push_back(selected_labels.at<int>(ex));
			final_left_data.push_back(selected_samples.at<int>(ex));
		}
	}
	for(int no_bit = 0; no_bit < selected_classes.rows; no_bit++)
	{
		int curr_class = selected_classes.at<int>(no_bit);
		if(best_partition[no_bit])
			right_centroids.insert( std::pair<int,cv::Mat>( curr_class, centroids[curr_class]) );
		else
			left_centroids.insert( std::pair<int,cv::Mat>( curr_class, centroids[curr_class]) );
	}

	//for debugging
	/*
	std::cout << "Left centroids: " << std::endl;
	for(std::map<int,cv::Mat>::iterator it = left_centroids.begin(); it != left_centroids.end(); ++it)
	{
		std::cout << it->first << ": \n" << it->second << std::endl;
	}
	std::cout << "Right centroids: " << std::endl;
	for(std::map<int,cv::Mat>::iterator it = right_centroids.begin(); it != right_centroids.end(); ++it)
	{
		std::cout << it->first << ": \n" << it->second << std::endl;
	}
	*/
	NCMF_node* node = new NCMF_node();
	node->left_centroids = left_centroids;
	node->right_centroids = right_centroids;
	node->left_data = final_left_data.clone();
	node->left_labels = final_left_labels.clone();
	node->right_data = final_right_data.clone();
	node->right_labels = final_right_labels.clone();
	node->f = NULL;
	node->t = NULL;

	return node;

}



//method to randomly pick k distinct attributes
cv::Mat NCMF_class_tree::pick_classes(const cv::Mat curr_labels)
{

	//check which classes are available
	cv::Mat picked_classes(0,1,CV_32SC1);
	cv::Mat curr_classes(0,1,CV_32SC1);
	bool flag_new_class = true;
	int k = 0;
	
	curr_classes.push_back(curr_labels.at<int>(0));
	for(int e = 1; e < curr_labels.rows; e++)
	{
		flag_new_class = true;
		k = 0;
		while(flag_new_class == true && k < curr_classes.rows)
		{
			if(curr_labels.at<int>(e) == curr_classes.at<int>(k))
				flag_new_class = false;
			k++;
		}	
		if(flag_new_class)
			curr_classes.push_back(curr_labels.at<int>(e));
	}

	//std::map<unsigned int, short> table_classes;
	//cv::Mat picked_classes(0,1,CV_32SC1);
	//int no_picked_classes = 0;

	//create a hash table, to quickly check if the random generated attribute has already been selected
	//for(int no_class = 0; no_class < curr_classes.rows; no_class++)
	//	table_classes.insert( std::pair<int, short>(curr_classes.at<int>(no_class),0) );

	//pick k classes randomly
	if(curr_classes.rows <= active_classes)
		return curr_classes;
	else
	{
		std::map<unsigned int, short> table_classes;
		int no_picked_classes = 0;

		//create a hash table, to quickly check if the random generated attribute has already been selected
		for(int no_class = 0; no_class < curr_classes.rows; no_class++)
			table_classes.insert( std::pair<int, short>(curr_classes.at<int>(no_class),0) );

		while(no_picked_classes < active_classes)
		{
			int r = (int)(rng.uniform(0., (double)curr_classes.rows ));
			if(table_classes[curr_classes.at<int>(r)] == 0)
			{
				picked_classes.push_back(curr_classes.at<int>(r));
				table_classes[curr_classes.at<int>(r)] += 1;
				no_picked_classes++;
			}
		}

		//for debugging
		/*
		std::cout << "Selected classes:" << std::endl;
		for ( std::vector<int>::iterator it=picked_classes.begin(); it != picked_classes.end(); ++it ) 
		{
			std::cout << ' ' << *it;
		}
		std::cout << std::endl;
		*/

		return picked_classes;
	}
	
}

//compute the shannon entropy of a given split
double NCMF_class_tree::compute_erf_entropy(const cv::Mat& labels, const cv::Mat& neg_labels, const cv::Mat& pos_labels)
{
	//class entropy or entropy before split
	double class_entropy = compute_entropy(labels);
	//mutual information (information gain)
	double imp_neg_labels = ((double)neg_labels.rows/labels.rows)*compute_entropy(neg_labels);
	double imp_pos_labels = ((double)pos_labels.rows/labels.rows)*compute_entropy(pos_labels);
	double imp_attr = imp_neg_labels + imp_pos_labels;
	double info_gain = class_entropy - imp_attr;
	//split entropy
	double split_entropy = 0;
	double neg_prob = ((double)neg_labels.rows/labels.rows); //because of the base cases in the learning algorithm, we are sure
								 //labels.rows is greater than 0
	double pos_prob = ((double)pos_labels.rows/labels.rows);
	if( fabs(neg_prob-0.) > FLT_EPSILON && fabs(neg_prob-1.) > FLT_EPSILON )
		split_entropy += neg_prob*log2(neg_prob);
	if( fabs(pos_prob-0.) > FLT_EPSILON && fabs(pos_prob-1.) > FLT_EPSILON )
		split_entropy += pos_prob*log2(pos_prob);
	split_entropy *= -1;
	//shannon entropy
	double shannon_entropy = (2*info_gain)/(class_entropy+split_entropy);

	//for debugging
	/*
	std::cout << "Class entropy: " << class_entropy << std::endl;
	std::cout << "Info gain: " << info_gain << std::endl;
	std::cout << "Split entropy: " << split_entropy << std::endl;
	std::cout << "Shannon entropy: " << shannon_entropy << std::endl;
	*/

	return shannon_entropy;
}


//entropy computation of a given set
double NCMF_class_tree::compute_entropy(const cv::Mat& labels)
{
	double entropy = 0.;
	std::map<int, unsigned int> samples_per_class;

	//create a hash table with the classes names as keys, they save the number
	//of samples that have that particular label
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
		samples_per_class.insert( std::pair<int, unsigned int>(classes.at<int>(no_classes),0) );

	for(int ex = 0; ex < labels.rows; ex++)
		samples_per_class[labels.at<int>(ex)] += 1;

	//compute entropy based on the previous counting
	for(int no_classes = 0; no_classes < classes.rows; no_classes++)
	{
		double label_prob = (double)samples_per_class[classes.at<int>(no_classes)]/labels.rows;
		//std::cout << label_prob << std::endl;

		if( fabs(label_prob-0.) > FLT_EPSILON && fabs(label_prob-1.) > FLT_EPSILON )
			entropy += label_prob*log2(label_prob);
	}

	entropy *= -1.;

	return entropy;
}

cv::Mat NCMF_class_tree::get_classes_hist(const cv::Mat& sample_labels){
	cv::Mat classes_hist = cv::Mat(classes.rows,2, CV_32FC1);
	std::map<int,float> classes_count;
	
	for(int c = 0; c < classes.rows; c++){
		classes_count.insert(std::pair<int,int>(classes.at<int>(c),0.));
	}
	
	for(int l = 0; l < sample_labels.rows; l++){
		classes_count[sample_labels.at<int>(l)] +=1;
	}
	
	for(int c = 0; c < classes.rows; c++){
		classes_count[classes.at<int>(c)] /= sample_labels.rows;
		classes_hist.at<float>(c,0) = classes.at<int>(c);
		classes_hist.at<float>(c,1) = classes_count[classes.at<int>(c)];
	}
	
	
	return classes_hist;
	
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

//++++++++++++++ EXTRA METHODS TO FIND INFORMATION ABOUT THE TREE +++++++++++//

void NCMF_class_tree::find_depth(NCMF_node* ptr)
{
	if(ptr != NULL)
	{
		find_depth(ptr->f);

		if(!((ptr->type).compare("terminal")))
		{
			if(ptr->depth > max_depth)
				max_depth = ptr->depth;
			if(ptr->depth < min_depth)
				min_depth = ptr->depth;
		}
		
		find_depth(ptr->t);
	}
	
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

