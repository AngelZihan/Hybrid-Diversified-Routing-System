//
// Created by Angel on 18/9/2023.
//

#include "graph.h"
#include "ClassificationSystem.h"

using namespace std;
typedef pair<int, pair<int,int>> pd;
struct myComp {
    constexpr bool operator()(
            pair<int, pair<int,int> > const& a,
            pair<int, pair<int,int> > const& b)
    const noexcept
    {
        return a.first > b.first;
    }
};
void Classification::initCoherence()
{
    vvvCoherence.reserve(level+1);
    for(int i = 0; i <= level; i++)
    {
        int baseTmp = pow(2, i);
        vBase.push_back(baseTmp);
        pair<double, int> pTmp;
        vector<pair<double, int> > vTmp;
        vector<vector<pair<double, int> > > vvTmp;
        vTmp.assign(baseTmp, pTmp);
        vvTmp.assign(baseTmp, vTmp);
        vvvCoherence.push_back(vvTmp);
        //cout << "baseTmp: " << baseTmp << " step: " << (maxX - minX) /baseTmp << "3 step: " << ((maxY - minY)/baseTmp)*3<< endl;
        //vpXYBase.push_back(make_pair((double)(maxX - minX) / baseTmp, (double)(maxY - minY)/baseTmp));
        vpXYBase.push_back(make_pair((double)(maxX - minX) / baseTmp, (double)((maxY - minY)/baseTmp)));
    }
}



void Classification::createCoherence(int mm, int nn, int& xLevel, int& yLevel) {
    Classification obj1;
    vector<int> vTmp;
    vector<vector<int> > vvTmp;
    vvTmp.assign(vBase[level], vTmp);
    g->vvGridNode.assign(vBase[level], vvTmp);

    g->vvNodeGrid.assign(g->n, -1);
    vector<double> vTmp2;
    vector<vector<double> > vvTmp2;
    vvTmp2.assign(vBase[level] * vBase[level], vTmp2);
    vvGridDistance.assign(vBase[level] * vBase[level], vvTmp2);
    vector<pair<double, int> > vTmp3;
    vector<vector<pair<double, int> > > vvTmp3;
    vvTmp3.assign(vBase[level] * vBase[level], vTmp3);
    KSPNumber.assign(vBase[level] * vBase[level], vvTmp3);
    vector<pair<int, double> > vTmp111;
    vector<vector<pair<int, double> > > vvTmp111;
    vvTmp111.assign(vBase[level] * vBase[level], vTmp111);
    KSPPercent.assign(vBase[level] * vBase[level], vvTmp111);
    double xBase = vpXYBase[level].first;
    double yBase = vpXYBase[level].second;
    int x, y;
    for (int i = 0; i < g->n; i++) {
        x = int((g->Locat[i].second - g->minX) / xBase);
        y = int((g->Locat[i].first - g->minY) / yBase);
        g->vvGridNode[x][y].push_back(i);
        g->vvNodeGrid[i] = vBase[level] * x + y;
    }
    xLevel = g->vvNodeGrid[mm];
    yLevel = g->vvNodeGrid[nn];
}




int Classification::FirstPath(Graph* _g, int ID1, int ID2, vector<pair<string,string>>& edge_attribute, vector<pair<string,string>>& edge_index, vector<pair<string,string>>& graph_data, vector<vector<int> >& staMatrix, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<int>& vSPTChildren, vector<vector<int> >& vSPT, vector<vector<int> >& vvPathCandidate, vector<vector<int> >& vvPathCandidateEdge, vector<int>& kResults, vector<vector<int> >& vkPath){
    g = _g;
    SPT(ID1, ID2, vSPTDistance, vSPTParent,vSPTHeight, vSPTParentEdge, vSPT);

    vector<int> vEdgeReducedLength(g->vEdge.size(), INF);
    vector<double> vEdgeReducedLengthPercent(g->vEdge.size(), INF);
    for(int i = 0; i < (int)g->vEdge.size(); i++) {
        vEdgeReducedLength[i] = g->vEdge[i].length + vSPTDistance[g->vEdge[i].ID1] - vSPTDistance[g->vEdge[i].ID2];
    }

    vector<vector<int> > vvResult;	//Exact Path
    vector<int> vDistance;

    //Open
    vector<vector<int>> vAncestor; // Ancestors in the result path set
    vector<int> vPathParent;	//Deviated from
    vector<int> vPathParentPos;	//Deviated Pos from Parent
    vector<int> vPathDeviation; //deviation node
    vector<vector<int> > vPathLCANode; //LCA Node with path 1
    vector<int> vPathDevPrime; // The other node of the deviation edge
    vector<vector<float> > vPathFix;
    vector<float> mPathFix;
    vector<int> dEdge;
    vector<bool> bPath;

    //vector<vector<int> > vvPathCandidate;	 //nodes
    //vector<vector<int> > vvPathCandidateEdge;//edges
    vector<vector<pair<int, int> > > vvPathNonTree;
    vector<vector<pair<int, int> > > vvmArc;
    vector<vector<pair<int, int> > > vvmPathNodePos;
    vector<int> vFather;

    vector<unordered_set<int> > pResult;
    vFather.push_back(-1);

    float sim;

    //benchmark::pHeap<3, int, int, int> Arc(nodeNum);
    priority_queue<pd,vector<pd>,myComp> Arc;
    //vector<pHeap<3, int, int, int>> ArcList;
    vector<int> vPath;
    vector<int> vPathEdge;
    vector<int> Node_set;
    vPath.push_back(ID2);
    Node_set.push_back(ID2);
    //graph_data.push_back(make_pair(to_string(obj.g->adjList[ID2].size()), to_string(99999)));
    //for txt file
    //fileNodeSet.push_back(ID2);
    //for txt file
    //cout << "Find SP!" << endl;
    vector<pair<int, int> > vPathNonTree;
    int p = vSPTParent[ID2];
    int e = vSPTParentEdge[ID2];
    int oldP = ID2;
    vector<int> vTmpMatrix(7, 0);
    staMatrix.assign(5,vTmpMatrix);
    while(p != -1)
    {
        //cout << "p:" << oldP << endl;
        for(int i = 0; i < (int)g->adjListEdgeR[oldP].size(); i++)
        {
            int eID = g->adjListEdgeR[oldP][i].second;
            if(eID != e){
                Arc.push(make_pair(vEdgeReducedLength[eID], make_pair(eID,vPath.size()-1)));
            }
        }
        int loopTime = 0;
        int p1 = p;
        //cout << "p1: " << p1 << endl;
        vector<int> p1List;
        vector<int> p2List;
        p2List.push_back(p1);
        int p1ListSize = 0;
        while (loopTime < 5){
            //cout << "testloop" << endl;
            p1List = p2List;
            for(int m = p1ListSize; m < p1List.size(); m++){
                for(int i = 0; i < vSPT[p1List[m]].size(); i++){
                    vector<pair<int, int> >::iterator ivp;
                    // Change all adjList to adjListR
                    for(int j = 0; j < g->adjListR[vSPT[p1List[m]][i]].size(); j++)
                    {
                        //cout << j << endl;
                        int neighborNodeID = g->adjListR[vSPT[p1List[m]][i]][j].first;
                        int neighborLength = g->adjListR[vSPT[p1List[m]][i]][j].second;
                        if(neighborNodeID == p1List[m])
                            continue;
                        int neighborEdgeID = g->adjListEdgeR[vSPT[p1List[m]][i]][j].second;
                        int reduce = vEdgeReducedLength[neighborEdgeID];
                        if(vEdgeReducedLengthPercent[neighborEdgeID] == INF){
                            vEdgeReducedLengthPercent[neighborEdgeID] = (double)reduce / vSPTDistance[ID2];
                            //cout << "reduce: " << (double)reduce << " vSPTDistance[ID2]: " << vSPTDistance[ID2] << endl;

                            //For generate the edges
                            //staMatrix[loopTime].push_back(make_pair(g->vEdge[neighborEdgeID].ID2, neighborNodeID));
                            edge_index.push_back(make_pair(to_string(g->vEdge[neighborEdgeID].ID2), to_string(neighborNodeID)));
                            edge_attribute.push_back(make_pair(to_string(g->vEdge[neighborEdgeID].length), to_string(reduce)));
                            //graph_data.push_back(make_pair(to_string(obj.g->adjList[obj.g->vEdge[neighborEdgeID].ID2].size()), to_string(99999)));
                            //graph_data.push_back(make_pair(to_string(obj.g->adjList[neighborNodeID].size()), to_string(-1)));
                            Node_set.push_back(neighborNodeID);
                            //cout << "test Here!" << endl;

                            if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0 && vEdgeReducedLengthPercent[neighborEdgeID] < 0.001){
                                staMatrix[loopTime][0] += 1;
                            }
                            else if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0.001 && vEdgeReducedLengthPercent[neighborEdgeID] < 0.002){
                                staMatrix[loopTime][1] += 1;
                            }
                            else if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0.002 && vEdgeReducedLengthPercent[neighborEdgeID] < 0.003){
                                staMatrix[loopTime][2] += 1;
                            }
                            else if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0.003 && vEdgeReducedLengthPercent[neighborEdgeID] < 0.004){
                                staMatrix[loopTime][3] += 1;
                            }
                            else if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0.004 && vEdgeReducedLengthPercent[neighborEdgeID] < 0.005){
                                staMatrix[loopTime][4] += 1;
                            }
                            else if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0.005 && vEdgeReducedLengthPercent[neighborEdgeID] < 0.006){
                                staMatrix[loopTime][5] += 1;
                            }
                            else if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0.006 && vEdgeReducedLengthPercent[neighborEdgeID] != INF){
                                staMatrix[loopTime][6] += 1;
                            }
                        }
                    }
                    p2List.push_back(vSPT[p1List[m]][i]);
                }
            }
            loopTime += 1;
            p1ListSize = p1List.size();
        }
        vPath.push_back(p);
        //graph_data.push_back(make_pair(to_string(obj.g->adjList[p].size()), to_string(99999)));
        Node_set.push_back(p);
        oldP = p;
        vPathEdge.push_back(e);
        edge_index.push_back(make_pair(to_string(g->vEdge[e].ID1), to_string(g->vEdge[e].ID2)));
        edge_attribute.push_back(make_pair(to_string(g->vEdge[e].length), to_string(vEdgeReducedLength[e])));
        //for txt file
        /*fileNodeSet.push_back(p);
        fileEdgeSet.push_back(make_pair(e,vEdgeReducedLength[e]));*/
        //for txt file
        e = vSPTParentEdge[p];
        p = vSPTParent[p];
    }
    //cout << "Find Result!" << endl;
    reverse(vPath.begin(), vPath.end());
    reverse(vPathEdge.begin(), vPathEdge.end());

    vvPathCandidate.push_back(vPath);
    vvPathCandidateEdge.push_back(vPathEdge);

    unordered_map<int, int> nodeMapping;

    for(int i = 0; i < Node_set.size(); i++) {
        nodeMapping[Node_set[i]] = i;
        int position = find(vPath.begin(), vPath.end(), Node_set[i]) - vPath.begin();
        if(position == vPath.size()) // if node is not found in vPath
        {
            position = -1;
        }
        graph_data.push_back(make_pair(to_string(g->adjList[Node_set[i]].size()), to_string(position)));
    }

    for(auto &edge : edge_index) {
        edge.first = to_string(nodeMapping[stoi(edge.first)]);
        edge.second = to_string(nodeMapping[stoi(edge.second)]);
    }
    //cout << "return" << endl;
    return vSPTDistance[ID2];
}