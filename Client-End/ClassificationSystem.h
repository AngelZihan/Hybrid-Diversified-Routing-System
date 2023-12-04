#ifndef GRIDCOHERENCE_H
#define GRIDCOHERENCE_H

#include "graph.h"
#include <set>
#include <map>
#include <vector>
#include <queue>

class Classification
{
public:
	int level;	//from 0 to level, so totally level+1
	Graph* g;
	double minX, minY, maxX, maxY;

	Classification(){}

    Classification(Graph* _g, int _level)
	{
		g = _g;
		level = _level;
		minX = g->minX;
		minY = g->minY;
		maxX = g->maxX+0.01;
		maxY = g->maxY+0.01;

		initCoherence();

	}

	//level, x, y, coherence value / Road Number
	vector<vector<vector<pair<double, int> > > > vvvCoherence;
	vector<int> vBase;
	vector<pair<double, double> > vpXYBase;
    vector<vector<vector<double> > > vvGridDistance;
    vector<vector<vector<int> > > vvGridNumber;
    vector<vector<vector<pair<double, int> > > > KSPNumber;
    vector<vector<vector<pair<int, double> > > > KSPPercent;


	void initCoherence();
	void createCoherence(int mm, int nn, int& xLevel, int& yLevel);
	void loadCoherence();
	void saveCoherence(); 


    vector<int> EulerSeq;
    vector<int> rEulerSeq;
    vector<int> toRMQ;
    vector<vector<int>> RMQIndex;
    vector<int> makeRMQDFS(Graph* _g, int p, vector<vector<int> >& vSPT, vector<int>& vSPTParent);
    vector<vector<int>> makeRMQ(Graph* _g, int p, vector<vector<int> >& vSPT, vector<int>& vSPTHeight, vector<int>& vSPTParent);
    int LCAQuery(int _p, int _q, vector<vector<int> >& vSPT, vector<int>& vSPTHeight, vector<int>& vSPTParent);
    void SPT(int root, int ID2, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<vector<int> >& vSPT);
    int FirstPath(Graph* _g, int ID1, int ID2, vector<pair<string,string>>& edge_attribute, vector<pair<string,string>>& edge_index, vector<pair<string,string>>& graph_data,vector<vector<int> >& staMatrix, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<int>& vSPTChildren, vector<vector<int> >& vSPT, vector<vector<int> >& vvPathCandidate, vector<vector<int> >& vvPathCandidateEdge, vector<int>& kResults, vector<vector<int> >& vkPath);
    int eKSPCompare(Graph* _g, int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, double t, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<int>& vSPTChildren, vector<vector<int> >& vSPT, vector<vector<int> >& vvPathCandidate, vector<vector<int> >& vvPathCandidateEdge, float& maxSim);
    int DynamicSimilarity(Graph* _g, int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, double t, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<int>& vSPTChildren, vector<vector<int> >& vSPT, vector<vector<int> >& vvPathCandidate, vector<vector<int> >& vvPathCandidateEdge, float& maxSim);
};

#endif
