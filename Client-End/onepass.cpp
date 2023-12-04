/*
Copyright (c) 2017 Theodoros Chondrogiannis
*/

#include "kspwlo.hpp"

/*
 *
 *	onepass(RoadNetwork*, NodeID, NodeID, int, double)
 *	-----
 *	Implementation of the OnePass algorithm.
 *
 */
vector<Path> onepass(RoadNetwork *rN, NodeID source, NodeID target, unsigned int k, double theta, float &AveSim, float& minSim, float& maxSim) {
    //cout << "onepass" << endl;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    t1 = std::chrono::high_resolution_clock::now();
    int count11 = 0;
	AveSim = 0;
	minSim = 1;
	maxSim = 0;
    float sim;
    vector<Path> resPaths;

    unsigned int count = 0;
    NodeID resid;
    PriorityQueueAS2 queue;
    int newLength = 0;
    vector<double> newOverlap;
    EdgeList::iterator iterAdj;
    EdgeNew edge;
    bool check;

    unordered_map<EdgeNew, vector<int>,boost::hash<EdgeNew>> resEdges;
    unordered_map<EdgeNew, vector<int>,boost::hash<EdgeNew>>::iterator iterE;
    vector<OlLabel*> allCreatedLabels;

    pair<Path,vector<int>> resDijkstra= dijkstra_path_and_bounds(rN,source,target);
    Path resNext = resDijkstra.first;

    resPaths.push_back(resNext);
    int newLowerBound = resDijkstra.second[source];

    // Only the shortest path is requested
    if(k==1)
        return resPaths;

    for(unsigned int j = 0; j < resNext.nodes.size()-1; j++) {
        edge = make_pair(resNext.nodes[j],resNext.nodes[j+1]);
        if ((iterE = resEdges.find(edge)) == resEdges.end())
            resEdges.insert(make_pair(edge, vector<int>(1, count)));
        else
            iterE->second.push_back(count);
    }
    count++;

    newOverlap.resize(k, 0);
    queue.push(new OlLabel(source, newLength, newLowerBound, newOverlap, -1));

    while(!queue.empty()) {
        OlLabel* curLabel = static_cast<OlLabel*> (queue.top());
        queue.pop();

        if(curLabel->overlapForK < count-1) {
            check = true;

            OlLabel *tempLabel = curLabel;

            Path tempPath;
            while(tempLabel != NULL) {
                tempPath.nodes.push_back(tempLabel->node_id);
                tempLabel = static_cast<OlLabel*> (tempLabel->previous);
            }

            reverse(tempPath.nodes.begin(),tempPath.nodes.end());

            for(unsigned int j=0;j<tempPath.nodes.size()-1 && check;j++) {
                edge = make_pair(tempPath.nodes[j],tempPath.nodes[j+1]);
                if ((iterE = resEdges.find(edge)) != resEdges.end()) {
                    for(unsigned int i = 0; i < iterE->second.size(); i++) {
                        resid = iterE->second[i];
                        if (resid > curLabel->overlapForK && resid < count) {
                            curLabel->overlapList[resid] += rN->getEdgeWeight(edge.first,edge.second);
                        }
                    }
                }
            }
            curLabel->overlapForK = count-1;
        }


        if (curLabel->node_id == target) { // Found target.
         OlLabel *tempLabel = curLabel;
         Path tempPath;
         while(tempLabel != NULL) {
                tempPath.nodes.push_back(tempLabel->node_id);
             tempLabel = static_cast<OlLabel*> (tempLabel->previous);
         }
         reverse(tempPath.nodes.begin(),tempPath.nodes.end());
		 float simMax = 0;
		 float simMin = 1;
		 float simCount = 0;
         int count1 = 0;
         for(unsigned int j=0;j<tempPath.nodes.size()-1 && check;j++) {
             edge = make_pair(tempPath.nodes[j],tempPath.nodes[j+1]);
             if ((iterE = resEdges.find(edge)) != resEdges.end()) {
                 for(unsigned int i = 0; i < iterE->second.size(); i++) {
                     resid = iterE->second[i];
//                     int maxLength;
//                     if(resPaths[resid].length > curLabel->length)
//                         maxLength = resPaths[resid].length;
//                     else
//                         maxLength = curLabel->length;
                     //s1
                     sim = curLabel->overlapList[resid]/ (resPaths[resid].length + curLabel->length - curLabel->overlapList[resid]);
                     simCount += sim;
                     count1 ++;
                     if(sim > theta) {
					 //if(sim > simMax){
						 //simMax = sim;
                         check = false;
                         break;
                     }
                     if(sim > simMax) {
                         simMax = sim;
                     }
                     if(sim < simMin) {
                         simMin = sim;
                     }
                     /*else{
                         if(sim > maxSim)
                             maxSim = sim;
                         if(sim < minSim)
                             minSim = sim;
                         AveSim += sim;
                         count11++;
                     }*/
					//if(sim < simMin)
					//	simMin = sim;
                 }
             }
         }

         if (!check)
             continue;
         else
         {
             count11 += count1;
             AveSim += simCount;
             if(maxSim < simMax) {
                 maxSim = simMax;
             }
             if(simMin < minSim) {
                 minSim = simMin;
             }
             cout << sim << endl;
			 tempPath.length = curLabel->length;
             resPaths.push_back(tempPath);

             if (count == k-1)
                 break;

             for(unsigned int j = 0; j < tempPath.nodes.size()-1; j++) {
                 edge = make_pair(tempPath.nodes[j],tempPath.nodes[j+1]);
                 if ((iterE = resEdges.find(edge)) == resEdges.end())
                     resEdges.insert(make_pair(edge, vector<int>(1, count)));
                 else
                     iterE->second.push_back(count);
             }

             count++;
         }
        }


        else { // Expand Search
            // For each outgoing edge
            for(iterAdj = rN->adjListOut[curLabel->node_id].begin(); iterAdj != rN->adjListOut[curLabel->node_id].end(); iterAdj++) {
                // Avoid cycles.
                bool containsLoop = false;
                OlLabel *tempLabel = curLabel;
                while(tempLabel != NULL) {
                    if(tempLabel->node_id == iterAdj->first) {
                        containsLoop = true;
                        break;
                    }
                    tempLabel = static_cast<OlLabel*> (tempLabel->previous);
                }
                if(!containsLoop) {

                    newLength = curLabel->length + iterAdj->second;;
                    newOverlap = curLabel->overlapList;
                    newLowerBound = newLength + resDijkstra.second[iterAdj->first];
                    OlLabel* newPrevious = curLabel;
                    edge = make_pair(curLabel->node_id,iterAdj->first);
                    check = true;

                    if ((iterE = resEdges.find(edge)) != resEdges.end()) {
                        for(unsigned int j = 0; j < iterE->second.size(); j++) {
                            newOverlap[iterE->second[j]] += iterAdj->second;
                        }
                    }

                    if (check) {
                        OlLabel* label = new OlLabel(iterAdj->first, newLength, newLowerBound, newOverlap, (count-1), newPrevious);
                        queue.push(label);
                        allCreatedLabels.push_back(label);
                    }
                }
            }
        }

        //for Hybrid
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span;
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count() > 5){
            cout << "larger than 5 seconds" << endl;
            break;
        }
    }
    resEdges.clear();
	//AveSim = AveSim / (resPaths.size()*(resPaths.size()-1)/2);
    AveSim = AveSim / count11;
    cout << "AveSim: " << AveSim << " max: "<< maxSim << " min:" << minSim << endl;
	for(unsigned int i=0;i<allCreatedLabels.size();i++)
        delete allCreatedLabels[i];

    return resPaths;
}
