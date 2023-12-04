//
// Created by Angel on 18/9/2023.
//
#include "graph.h"
#include "ClassificationSystem.h"
void Classification::SPT(int root, int ID2, vector<int>& vSPTDistance, vector<int>& vSPTParent,vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<vector<int> >& vSPT)
{
    //cout << "SPT!" << endl;
    int countSPTNodeNumber = 0;
    int countDeviationEdgeNumber = 0;
    int boundDistance = 999999999;
    benchmark::heap<2, int, int> queue(g->n);
    queue.update(root, 0);

    vector<bool> vbVisited(g->n, false);
    int topNodeID, neighborNodeID, neighborLength, neighborRoadID;
    vector<pair<int, int> >::iterator ivp;
    vSPTHeight[root] = 1;
    vSPTDistance[root] = 0;
    while(!queue.empty())
    {
        int topDistance;
        queue.extract_min(topNodeID, topDistance);
        vbVisited[topNodeID] = true;
        if(topNodeID == ID2)
        {
            //break;
            //boundDistance = topDistance;
            boundDistance = 1.2 * topDistance;
            //break;
        }
        /*if(topNodeID == ID2)
        {
            boundDistance = 1.2 * topDistance;
            //boundDistance = 50442;
            //cout << boundDistance << endl;
        }*/
        //boundDistance = 504353;
        if(topDistance > boundDistance)
            break;


        for(int i = 0; i < (int)g->adjList[topNodeID].size(); i++)
        {
            countDeviationEdgeNumber += 1;
            neighborNodeID = g->adjList[topNodeID][i].first;
            neighborLength = g->adjList[topNodeID][i].second;
            neighborRoadID = g->adjListEdge[topNodeID][i].second;
            int d = vSPTDistance[topNodeID] + neighborLength;
            if(!vbVisited[neighborNodeID])
            {
                //countDeviationEdgeNumber += 1;
                if(vSPTDistance[neighborNodeID] > d)
                {
                    vSPTDistance[neighborNodeID] = d;
                    queue.update(neighborNodeID, d);
                    vSPTParent[neighborNodeID] = topNodeID;
                    vSPTHeight[neighborNodeID] = vSPTHeight[topNodeID] + 1;
                    vSPTParentEdge[neighborNodeID] = neighborRoadID;
                }
            }
        }
    }

    //Construct SPT
    for(int i = 0; i < g->n; i++)
        if(vSPTParent[i] != -1){
            //cout << i << endl;
            //countSPTNodeNumber += 1;
            vSPT[vSPTParent[i]].push_back(i);
        }

    //cout << "SPTNodeNumber: " << countSPTNodeNumber << " SPTEdgeNumber: " << countDeviationEdgeNumber << endl;
}

