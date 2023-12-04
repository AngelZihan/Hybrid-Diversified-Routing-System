#ifndef COHERENCEDECOMPOSITION_H
#define COHERENCEDECOMPOSITION_H

#include "graph.h"
#include "ClassificationSystem.h"
#include <set>
#include <cmath>

using namespace std;

struct cmp
{
	template<typename T>	
	bool operator()(const T& l, const T& r) const	
	{		
		if (l.first == r.first)				
			return l.second < r.second;			
		return l.first < r.first;	
	}
};

typedef struct QUERY
{
	int qID;
	int ID1;
	int ID2; 
	int h;
	int m;
	int td;
	double direction;
	int x1, y1;	//grid Coordinate of ID1
	int x2, y2; //grid Coordinate of ID2
	int d;
    int xLevel;
    int yLevel;
	int ClusterID;
	
}Query;

typedef struct CLUSTER
{
public:
	int clusterID;
	set<pair<int, int> > sQuerySorted;
	set<int> sQuery;
	bool bF; 
	int d;
	double direction;

	//The grids this ellipse covers
	set<pair<int, int> > sGrid;
	
	bool operator < (const CLUSTER c) const
	{
		if (sQuery.size() == c.sQuery.size())
			return clusterID < c.clusterID;
		return sQuery.size() < c.sQuery.size();
	}
}Cluster;

class CoherenceDecomposition
{
public:
	Classification gc;
	int level;
	Graph* g;
	
	vector<Query> vQuery;
	vector<Cluster> vCluster;

	vector<vector<set<int> > > vvsGridF;
	vector<vector<set<int> > > vvsGridB;

	int base;
	double xBase;
	double yBase;

	CoherenceDecomposition()
	{
	}

	CoherenceDecomposition(Graph* _g, int _level, int mm, int nn, int& xLevel, int& yLevel)
	{
        //vector<vector<vector<int> > > vvGridNode;
		g = _g;
		level = _level;
		gc = Classification(g, level);
		gc.createCoherence(mm, nn, xLevel, yLevel);
	}

	void loadQuery(string filename);
	void decompose();
	void writeCluster(string filename);
	void writeClusterSorted(string filename);
};
#endif
