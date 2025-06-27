/*
Copyright (c) 2017 Theodoros Chondrogiannis
*/

#include "graph.h"

RoadNetwork::RoadNetwork(const char *filename) {
    FILE *fp;
    NodeID lnode, rnode, tmp;
    int w;
    char c;
	char str[80];

    fp = fopen(filename, "r");
    fscanf(fp, "%[^\n]%*c", str);
    fscanf(fp, "%[^\n]%*c", str);
    fscanf(fp, "%[^\n]%*c", str);
    fscanf(fp, "%[^\n]%*c", str);
	//cout << str << endl;
    fscanf(fp, "%c %2c %u %u\n", &c, &str, &this->numNodes, &this->numEdges);
    //cout << this->numEdges << endl; 
    fscanf(fp, "%[^\n]%*c", str);
    fscanf(fp, "%[^\n]%*c", str);
	//cout << str << endl;
    this->adjListOut = vector<EdgeList>(this->numNodes);
    this->adjListInc = vector<EdgeList>(this->numNodes);
    
    while (fscanf(fp, "%c %u %u %d\n", &tmp, &lnode, &rnode, &w) != EOF) {
/*		cout << lnode << "\t" << rnode << endl;
		if(lnode == 236219)
		{
			for(auto& it : this->adjListOut[lnode])
				cout << "out:" << it.first << "\t" << it.second << endl;
			for(auto& it : this->adjListInc[rnode])
				cout << "In:" << it.first << "\t" << it.second << endl;
		}*/
		this->adjListOut[lnode-1].insert(make_pair(rnode-1, w));
        this->adjListInc[rnode-1].insert(make_pair(lnode-1, w));
		//cout << this->adjListInc[rnode][1] << endl;
    }
	//cout << lnode-1 << endl;
    //cout << filename << endl;
    fclose(fp);
}

int RoadNetwork::getEdgeWeight(NodeID lnode, NodeID rnode) {
    return this->adjListOut[lnode][rnode];
}

RoadNetwork::~RoadNetwork() {
    this->adjListOut.clear();
   	this->adjListInc.clear();
}

bool operator==(const EdgeNew& le, const EdgeNew& re) {
    return (le.first == re.first && le.second == re.second) || (le.second == re.first && le.first == re.second);
}

bool Path::containsEdge(EdgeNew e) {
    bool res = false;
    
	for(unsigned int i=0;i<this->nodes.size()-1;i++) {
		if(this->nodes[i] == e.first && this->nodes[i+1] == e.second) {
			res = true;
			break;
		}
	}
	
    return res;
}

double Path::overlap_ratio(RoadNetwork *rN, Path &path2) {
	double sharedLength = 0;
	
	for(unsigned int i=0;i<path2.nodes.size()-1;i++) {
        EdgeNew e = make_pair(path2.nodes[i],path2.nodes[i+1]);
		if(this->containsEdge(e))
			sharedLength += rN->getEdgeWeight(path2.nodes[i],path2.nodes[i+1]);
	}
    int maxLength;
    if(path2.length > this->length)
        maxLength = path2.length;
    else
        maxLength = this->length;

    //s1
	return sharedLength/(this->length+path2.length-sharedLength);
    //s2
//    return sharedLength/(2*path2.length) + sharedLength/(2*this->length);
    //s3
//    return sqrt((sharedLength*sharedLength) / ((double)path2.length*(double)this->length));
    //s4
//	return sharedLength / maxLength;
    //s5
//    return sharedLength/path2.length;
}

bool operator==(const Path& lp, const Path& rp) {
	if(lp.length != rp.length || lp.nodes.size() != rp.nodes.size())
		return false;
	for(int i=0;i<lp.nodes.size();i++) {
		if(lp.nodes[i] != rp.nodes[i])
			return false;
	}
    return true;
}
