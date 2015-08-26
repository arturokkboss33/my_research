/* =============================================================*/
/* --- DECISION TREES- DECISION TREE CLASS SOURCE FILE       ---*/
/* FILENAME: dectree_class.cpp 
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
#include "dectree_class.h"
//libraries
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <map>
#include <limits>
#include <queue>

//*********************** CONSTRUCTORS *****************************//
Dectree_class::Dectree_class()
{
	dbst = NULL;
	classes = cv::Mat(0,1,CV_32SC1);
	dectree_idx = 0;
	max_depth = 0;
	min_depth = 1000;
	split_nodes_idx = 0;
	terminal_nodes_idx = 0;
	depth_limit = 0;
	max_split_fail = std::numeric_limits<int>::max();
	is_ext_rng = false;
}

Dectree_class::Dectree_class(cv::RNG ext_rng)
{
	dbst = NULL;
	classes = cv::Mat(0,1,CV_32SC1);
	dectree_idx = 0;
	max_depth = 0;
	min_depth = 1000;
	split_nodes_idx = 0;
	terminal_nodes_idx = 0;
	depth_limit = 0;
	max_split_fail = std::numeric_limits<int>::max();
	is_ext_rng = true;
	rng = ext_rng;
}
//******************************************************************//

//*********************** GET-SET METHODS *************************//
int Dectree_class::get_dectree_idx() { return dectree_idx;}

void Dectree_class::set_dectree_idx(int idx)
{ dectree_idx = idx;}

dectree_node* Dectree_class::get_root(){ return dbst.get_root();}

cv::Mat Dectree_class::get_classes() { return classes;}

cv::RNG Dectree_class::get_rng() {return rng;}

int Dectree_class::get_maxDepth() { return max_depth;}

unsigned int Dectree_class::get_noNodes() { return (split_nodes_idx+terminal_nodes_idx);}

unsigned int Dectree_class::get_noLeaves() { return terminal_nodes_idx;}

//******************************************************************//

//*********************** PUBLIC FUNCTIONS *************************//

//main method to train the decision treee
void Dectree_class::train(const cv::Mat& training_data, const cv::Mat& labels, int depth_thresh, unsigned int samples_thresh, int vars_per_node)
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

	//---(3)make a vector giving an id to each attribute---//
	set_attributes(training_data);
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
	if(attributes.size() < (unsigned)vars_per_node)
		active_vars = attributes.size();
	else
		active_vars = vars_per_node;
	//maximum number of split fails
	max_split_fail = attributes.size();

	//---(5)train the tree---//
	int depth = 1;
	int no_split_fail = 0;
	dbst.set_root(learn_dectree(cv::Mat(),labels, training_data, depth, no_split_fail));
	find_depth(dbst.get_root()); //to find the real-true depth of the created tree

}

//prediction mehtod --- only returns the predicted label
int Dectree_class::predict(const cv::Mat& sample)
{
	dectree_node* tmp_node = dbst.get_root();
	int prediction = -1;

	while(tmp_node != NULL)
	{
		if( !(tmp_node->type.compare("terminal")) )
			return tmp_node->output_id;
		else
		{
			if( sample.at<float>(tmp_node->attribute_id) < tmp_node->cut_point)
				tmp_node = tmp_node->f;
			else
				tmp_node = tmp_node->t;
		}
	}

	return prediction;


}
//prediction method --- returns the predicted label and leaf index which output the prediction
cv::Mat Dectree_class::predict_with_idx(const cv::Mat& sample)
{
	dectree_node* tmp_node = dbst.get_root();
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
			if( sample.at<float>(tmp_node->attribute_id) < tmp_node->cut_point)
				tmp_node = tmp_node->f;
			else
				tmp_node = tmp_node->t;
		}
	}

	return prediction_mat;


}

//methods to save and load the model into a YAML file for reuse
void Dectree_class::save_model(cv::FileStorage& fs)
{
	/*
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
	
	fs << name << "{";
	
	fs << "trainingParams" << "{";
	fs << "depthLimit" << depth_limit;
	fs << "min_samples" << (int)min_samples;
	fs << "}";
	*/
	
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

void Dectree_class::load_model(cv::FileNode& fnode_tree, cv::FileNode& fnode_params)
{
	std::queue<dectree_node*> dummy_queue;
	//cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
		
	dectree_idx = (int)fnode_tree["treeIndex"];
	max_depth = (int)fnode_tree["maxDepth"];
	terminal_nodes_idx = (int)fnode_tree["noLeaves"];
	split_nodes_idx = (int)fnode_tree["noNodes"] - terminal_nodes_idx;
	
	depth_limit = (int)fnode_params["depthLimit"];
	min_samples = (int)fnode_params["minSamples"];
	active_vars = (int)fnode_params["activeVars"];
	fnode_params["classes"] >> classes;		
	
	cv::FileNode tree_nodes = fnode_tree["nodes"];
	cv::FileNodeIterator node_it = tree_nodes.begin(), it_end = tree_nodes.end();
	std::vector< std::queue<dectree_node*> > tree_queue(max_depth,dummy_queue);
	std::cout << "***" << tree_queue.size() << std::endl;
	
	for(; node_it != it_end; ++node_it)
	{	
		//===For debugging===//
		//std::cout << (std::string)(*node_it)["Type"] << " " << (int)(*node_it)["Idx"] << std::endl;
		
		dectree_node* nod = new dectree_node();
		nod->node_idx = (int)(*node_it)["Idx"];
		nod->type = (std::string)(*node_it)["Type"];
		nod->depth = (int)(*node_it)["Depth"];
		nod->attribute_id = (int)(*node_it)["Attribute"];
		nod->output_id = (int)(*node_it)["Output"];
		nod->cut_point = (float)(*node_it)["CutPoint"];
		
		if(nod->depth < max_depth)
		{
			if( tree_queue.at((nod->depth)).size() == 2)
			{
				dectree_node* tmpnod = tree_queue.at(nod->depth).front();
				tree_queue.at(nod->depth).pop();
				nod->f = tmpnod;
				tmpnod = tree_queue.at(nod->depth).front();
				tree_queue.at(nod->depth).pop();
				nod->t = tmpnod;
			}
		}
		tree_queue.at((nod->depth)-1).push(nod);
		
	}
	
	std::cout << "ooo" << std::endl;
	dbst.set_root(tree_queue.at(0).front());
	tree_queue.at(0).pop();
	//fs.release();
		
}

//methods that invoke the fucntions to print the structure of the tree
void Dectree_class::inOrder_tree()
{
	dbst.inOrder(dbst.get_root());
}
void Dectree_class::postOrder_tree()
{
	dbst.postOrder(dbst.get_root());
}

//******************************************************************//

//*********************** PRIVATE FUNCTIONS *************************//

//+++ METHODS USED BEFORE THE LEARNING PROCESS +++//
//check number of classes; labels will be represented as a column vector
void Dectree_class::set_classes(const cv::Mat& labels)
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

//set a vector of ints identifying each attribute
void Dectree_class::set_attributes(const cv::Mat& training_data)
{
	for(int attr = 0; attr < training_data.cols; attr++)
		attributes.push_back(attr);

}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

//algorithm to learn a decision tree
//inputs: parent labels, current labels, current data samples, current tree depth, current split fails
//outputs: it returns the root of the decision tree (or one of its subtrees in the recursion) 
dectree_node* Dectree_class::learn_dectree(const cv::Mat& p_samples_labels, const cv::Mat& samples_labels, const cv::Mat& samples_data, int depth, int no_split_fail)
{
	//for debugging
	//std::cout << "Depth: " << depth << std::endl;
	//std::cout << "t: " << terminal_nodes_idx << std::endl;
	//std::cout << "n: " << split_nodes_idx << std::endl;

	dectree_split* split = new dectree_split();	
	Dectree_BST dectree;
	dectree_node* dectree_root = dectree.get_root();

	//check the base cases to get out of the recursion
	//if there are no more examples		
	if(samples_labels.empty())
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, -1, -1, plurality(p_samples_labels));
		return dectree_root;
	}
	//if all examples have the same classification
	else if(check_classif(samples_labels))
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, -1, -1, samples_labels.at<int>(0));
		return dectree_root;
	}
	//if this case is hit, there are attributes and samples to analyze
	//it checks the maximum depth and minimum sample constraints
	else if(depth >= depth_limit || (unsigned)samples_labels.rows < min_samples)
	{
		dectree.insert_node(&dectree_root, "terminal", (++terminal_nodes_idx), depth, -1, -1, plurality(samples_labels));
		return dectree_root;
	}
	//else call the method recursively after finding the best split
	else
	{
		//find the attrribute with the highest information gain	
		split = erf_split(samples_data, samples_labels);

		//for debugging
		//double h_curr = compute_entropy(samples_labels);
		//std::cout << "Current entropy: " << h_curr << std::endl;
		//std::cout << "Best attribute: " << split->attr_name << std::endl;
		//std::cout << "Cut point: " << split->cut_point << std::endl;

		//if the samples were not further separated, it's a bad split
		if (split->neg_attr_labels.rows == 0 || split->pos_attr_labels.rows == 0 )
			no_split_fail++;
		//if they were separated, reset the counter
		else
			no_split_fail = 0;
		//if the maximum number of split fails is reached, convert the node into a leaf/terminal node
		if(no_split_fail >= max_split_fail)
		{
			//std::cout << "FAIL" << std::endl;
			dectree.insert_node(&dectree_root,"terminal", (++terminal_nodes_idx), depth, -1, -1, plurality(samples_labels));
			no_split_fail = 0;
			return dectree_root;
		}	
		//if there is no split fail, insert a split node and call the method recursively
		else
		{
			//create a node split with the best attribute as generator of a split
			dectree.insert_node(&dectree_root,"split", (++split_nodes_idx), depth, split->attr_name, split->cut_point, -1);	

			//call the fcn recursively
			//Here, no attributes are erased
			(dectree_root)->f = learn_dectree(samples_labels,split->neg_attr_labels,split->neg_attr_data,(depth+1), no_split_fail);	
			(dectree_root)->t = learn_dectree(samples_labels,split->pos_attr_labels,split->pos_attr_data,(depth+1), no_split_fail);

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
int Dectree_class::plurality(const cv::Mat& labels)
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
bool Dectree_class::check_classif(const cv::Mat& labels)
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
dectree_split* Dectree_class::erf_split(const cv::Mat& samples, const cv::Mat& labels)
{
	//select K attributes randomly
	std::vector<int> attr = pick_variables();
	//generate K random splits
	std::vector<double> attr_splits = gen_ran_splits(attr, samples);

	//Split the samples according the generated splits 
	std::vector<int> best_attr_info(2,0);
	int true_attr_pos = 0;
	int attr_idx;
	double max_split_score = -1.;
	double best_cut_point = 0.;
	bool flag_compare_info_gain = false;
	cv::Mat final_neg_attr_data(0,1,CV_32FC1);
	cv::Mat final_pos_attr_data(0,1,CV_32FC1);
	cv::Mat final_neg_attr_labels(0,1,CV_32SC1);
	cv::Mat final_pos_attr_labels(0,1,CV_32SC1);

	for(std::vector<int>::size_type it_attr = 0; it_attr != attr.size(); it_attr++)
	{
		cv::Mat neg_attr_labels(0,1,CV_32SC1);
		cv::Mat pos_attr_labels(0,1,CV_32SC1);
		attr_idx = attr[it_attr];
		//partition the examples according to the selected attribute 
		for(int ex = 0; ex < samples.rows; ex++)
		{
			if( samples.at<float>(ex,attr_idx) < attr_splits[it_attr] )
				neg_attr_labels.push_back(labels.at<int>(ex));
			else
				pos_attr_labels.push_back(labels.at<int>(ex));
		}
		//compute the shannon entropy of the computed split
		double split_score = compute_erf_entropy(labels, neg_attr_labels, pos_attr_labels);

		//compare the score with the previous ones, and save the best
		if(flag_compare_info_gain)
		{
			if(split_score > max_split_score)
			{
				best_attr_info.at(0) = attr_idx;
				best_attr_info.at(1) = true_attr_pos;
				best_cut_point = attr_splits[it_attr];
				max_split_score = split_score;
			}
		}
		else
		{
			flag_compare_info_gain = true;
			best_attr_info.at(0) = attr_idx;
			best_attr_info.at(1) = true_attr_pos;
			best_cut_point = attr_splits[it_attr];
			max_split_score = split_score;
		}
		
		true_attr_pos++;
		neg_attr_labels.release();
		pos_attr_labels.release();

	}

	//for debugging
	//std::cout << std::endl;
	//std::cout << "max: " << max_info_gain << std::endl;

	//with the found best attribute; fill the split structure
	//it's prefere to separate the samples again because otherwise a lot of memory reallocations take place
	for(int ex = 0; ex < samples.rows; ex++)
	{
		if( samples.at<float>(ex,best_attr_info.at(0)) < best_cut_point)
		{
			final_neg_attr_data.push_back(samples.row(ex));
			final_neg_attr_labels.push_back(labels.at<int>(ex));
		}
		else
		{
			final_pos_attr_data.push_back(samples.row(ex));
			final_pos_attr_labels.push_back(labels.at<int>(ex));
		}
	}

	//declare and define the split structure
	dectree_split* split = new dectree_split();	//split structure
	split->attr_name = best_attr_info.at(0); 	//global idx of the attribute - to be use later for prediction
	split->attr_idx = best_attr_info.at(1);		//local idx of the attribute
	split->cut_point = best_cut_point;		//random cut point selected
	split->neg_attr_data = final_neg_attr_data;	//sample information
	split->pos_attr_data = final_pos_attr_data;
	split->neg_attr_labels = final_neg_attr_labels;
	split->pos_attr_labels = final_pos_attr_labels;

	return split;

}

//method to randomly pick k distinct attributes
std::vector<int> Dectree_class::pick_variables()
{
	std::map<unsigned int, short> chosen_attr;
	std::vector<int> picked_vars;
	int no_picked_vars = 0;

	//create a hash table, to quickly check if the random generated attribute has already been selected
	for(unsigned int no_attr = 0; no_attr < attributes.size(); no_attr++)
		chosen_attr.insert( std::pair<int, short>(no_attr,0) );

	//pick k attributes randomly
	while(no_picked_vars < active_vars)
	{
		int r = (int)(rng.uniform(0., (double)attributes.size()));
		if(chosen_attr[r] == 0)
		{
			picked_vars.push_back(r);
			chosen_attr[r] += 1;
			no_picked_vars++;
		}
	}

	//for debugging
	/*
	std::cout << "Selected variables:" << std::endl;
	for ( std::vector<int>::iterator it=picked_vars.begin(); it != picked_vars.end(); ++it ) 
	{
		std::cout << ' ' << *it;
	}
	std::cout << std::endl;
	*/

	return picked_vars;
	
}

//method to generate a split in the range of the min and max values of a fix attribute 
std::vector<double> Dectree_class::gen_ran_splits(std::vector<int> attr, const cv::Mat& samples)
{
	std::vector<double> ran_splits;

	for ( std::vector<int>::iterator it=attr.begin(); it != attr.end(); ++it ) 
	{
		double* max_val = new double();
		double* min_val = new double();
		cv::Mat attr_values = samples.col(*it).clone();
		cv::minMaxIdx(attr_values, min_val, max_val, NULL, NULL);	//find min and max values
		double cut = rng.uniform(*min_val, *max_val);
		ran_splits.push_back(cut);
		//for debugging	
		//std::cout << "min: " << *min_val << " max: " << *max_val << " cut: " << cut << std::endl;	
	}

	return ran_splits;
	
}

//compute the shannon entropy of a given split
double Dectree_class::compute_erf_entropy(const cv::Mat& labels, const cv::Mat& neg_labels, const cv::Mat& pos_labels)
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
double Dectree_class::compute_entropy(const cv::Mat& labels)
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
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

//++++++++++++++ EXTRA METHODS TO FIND INFORMATION ABOUT THE TREE +++++++++++//
void Dectree_class::find_depth(dectree_node* ptr)
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

