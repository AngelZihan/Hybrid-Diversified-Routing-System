#include "graph.h"

bool Compare2(pair<int, int> p1, pair<int, int> p2)
{
    if(p1.second > p2.second)
        return true;
    else
        return false;
}

int Graph::EdgesBlocking(int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, double t, int& countNumber, int& popPath, float& AveSim, float& minSim, float& maxSim)
{
	minSim = 1;
	maxSim = 0;
	AveSim = 0;
    //Shortest Path Tree Info
    countNumber = 0;
    popPath = 0;
    vector<int> vSPTDistance(nodeNum, INF);
    vector<int> vSPTParent(nodeNum, -1);
    vector<int> vSPTHeight(nodeNum, -1);
    vector<int> vSPTParentEdge(nodeNum, -1);
    vector<int> vSPTChildren(nodeNum, -1);
    vector<int> vTmp;
    vector<vector<int>> vSPT(nodeNum, vTmp); //Tree from root
    SPT(ID1, ID2, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPT);

    //LCA
    vector<vector<float> > vPathLCA;
    vector<vector<int> > RMQ = makeRMQ(ID1, vSPT, vSPTHeight,vSPTParent);
    vector<int> vEdgeReducedLength(vEdge.size(), INF);
    for(int i = 0; i < (int)vEdge.size(); i++)
        vEdgeReducedLength[i] = vEdge[i].length + vSPTDistance[vEdge[i].ID1] - vSPTDistance[vEdge[i].ID2];

    vector<vector<int> > vvResult;	//Exact Path
    vvResult.reserve(k);
    vector<int> vDistance;
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
    vector<vector<int> > vvPathCandidate;	 //nodes
    vector<vector<int> > vvPathCandidateEdge;//edges
    vector<unordered_map<int, int> > vmPathNodePos;	//Position of the non-fixed vertices
    //The size of mvPathNodePos indicates the fixed pos
    vector<vector<pair<int, int> > > vvPathNonTree;
    vector<multimap<int, int> > vmArc;
    vector<int> vFather;

    vector<unordered_set<int> > pResult;
    vector<unordered_set<int> > pCandidatePath; // Same as vvPathCandidateEdge just unordered_set
    unordered_set<int> pCandidateTmp;
    unordered_set<int> edgesOrderCandidate; //Candidate of the edges order heap
    vector<pair<int, int> > edgesOrderCandidate1;
    unordered_set<int> pruneOrderCandidate;
    vector<pair<int, int> > pruneOrderCandidate1;
    //vector<pair<int, int> > rollBackPath;
    unordered_set<int> unorderblockEdge;
    vector<int> blockEdge;
    unordered_set<int> unBlockEdge;

    vFather.push_back(-1);

    float sim;

    vector<int> vPath;
    vector<int> vPathEdge;
    vPath.push_back(ID2);
    vector<pair<int, int> > vPathNonTree;
    int p = vSPTParent[ID2];
    int e = vSPTParentEdge[ID2];
    multimap<int, int> mArc;
    float sim1;
    float addLength1 = 0;
    int oldP = ID2;

    while(p != -1)
    {
        vPath.push_back(p);
        for(int i = 0; i < (int)adjListEdgeR[oldP].size(); i++)
        {
            int eID = adjListEdgeR[oldP][i].second;
            if(eID != e)
                mArc.insert(make_pair(vEdgeReducedLength[eID], eID));
        }
        oldP = p;
        vPathEdge.push_back(e);
        pCandidateTmp.insert(e);
        e = vSPTParentEdge[p];
        p = vSPTParent[p];
    }

    vmArc.push_back(mArc);

    reverse(vPath.begin(), vPath.end());
    reverse(vPathEdge.begin(), vPathEdge.end());
    unordered_map<int, int> mPos;
    for(int i = 0; i < (int)vPath.size(); i++)
        mPos[vPath[i]] = i;
    vmPathNodePos.push_back(mPos);

    benchmark::heap<2, int, int> qPath(nodeNum);
    benchmark::orderHeap<2, int, int> edgesOrder(vEdge.size());
    benchmark::orderHeap<2, int, int> pruneOrder(vEdge.size());
    vvPathCandidate.push_back(vPath);
    vvPathCandidateEdge.push_back(vPathEdge);
    pCandidatePath.push_back(pCandidateTmp);
    vDistance.push_back(vSPTDistance[ID2]);
    bPath.push_back(true);
    dEdge.push_back(0);
    vPathParent.push_back(-1);
    vPathParentPos.push_back(0);
    vPathDeviation.push_back(ID2);
    vector<int> vTmpLCANode;
    vTmpLCANode.push_back(ID2);
    vPathLCANode.push_back(vTmpLCANode);
    vPathDevPrime.push_back(ID2);
    vector<float> vTmpFix;
    vTmpFix.push_back(0);
    vPathFix.push_back(vTmpFix);
    vector<int> vTmpAncestor;
    vTmpAncestor.push_back(-1);
    vAncestor.push_back(vTmpAncestor);
    vector<float> vTmpLCA;
    vTmpLCA.push_back(vSPTDistance[ID2]);
    vPathLCA.push_back(vTmpLCA);
    qPath.update(vvPathCandidate.size()-1, vSPTDistance[ID2]);

    //roll-back
    vector<multimap<int, int> > vmArcRollBack;
    vector<vector<int> > vvPathCandidateRollBack;
    vector<vector<int> > vvPathCandidateEdgeRollBack;
    vector<unordered_set<int> > pCandidatePathRollBack;
    vector<int> vDistanceRollBack;
    vector<bool> bPathRollBack;
    vector<int> dEdgeRollBack;
    vector<int> vPathParentRollBack;	//Deviated from
    vector<int> vPathParentPosRollBack;	//Deviated Pos from Parent
    vector<int> vPathDeviationRollBack; //deviation node
    vector<vector<int> > vPathLCANodeRollBack; //LCA Node with path 1
    vector<int> vPathDevPrimeRollBack; // The other node of the deviation edge
    vector<vector<float> > vPathFixRollBack;
    vector<float> mPathFixRollBack;
    vector<vector<int>> vAncestorRollBack;
    benchmark::heap<2, int, int> qPathRollBack(nodeNum);
    vector<unordered_map<int, int> > vmPathNodePosRollBack;
    vector<int> vFatherRollBack;
    vector<vector<int> > edgePath;
    vector<vector<float> > vPathLCARollBack;
    int topPathIDRollBack;
    int topPathDistanceRollBack;
    vector<int> kResultsRollBack;
    vector<vector<int> > vvResultRollBack;
    vector<vector<int> > vkPathRollBack;
    vector<int> vResultIDRollBack;


    vector<int> vResultID;
    int topPathID, topPathDistance;
    int pcount = 0;
    int oldDistance = -1;
    bool bError = false;
    int heapSize = 0, heapSizeTest = 0;
    int stopCount = 0;
    //int stopCountNextEdge = 0;
    //int stopCountTest = 0;
    bool rollBack = false;
    bool bUpdate = false;
    int stopCountHeap = 0;
    int pruneCount = 0;
    int LastCount = 0;
    bool popPathTest = false;
    bool blockEdgeTest = false;

    while((int)kResults.size() < k && !qPath.empty())
    {
        float addLength = 0;
        pcount++;
        popPath++;
        qPath.extract_min(topPathID, topPathDistance);
        if(vvPathCandidateEdge[topPathID].empty())
            continue;
        bool qPathPop = false;
        int fixLength = 0;
        //reverse(vvPathCandidateEdge[topPathID].begin(), vvPathCandidateEdge[topPathID].end());
        for (auto ie = vvPathCandidateEdge[topPathID].rbegin();
             ie != vvPathCandidateEdge[topPathID].rend() ; ie++) {
            fixLength += vEdge[*ie].length;
            //cout << "fixLength: " << fixLength << endl;
            for(int i = 0; i < blockEdge.size(); i++){
                if(*ie == blockEdge[i] && unBlockEdge.find(blockEdge[i]) == unBlockEdge.end())
                {
                    edgePath[i].push_back(topPathID);
                    pruneOrderCandidate1[i].second -= vmArc[topPathID].size();
                    //qPath.extract_min(topPathID, topPathDistance);
                    qPathPop = true;
                    if(i == blockEdge.size()-1)
                        LastCount += 1;
                    /*if(i != blockEdge.size()-1)
                        previousCount += 1;*/
                    /*heapSizeTest = qPath.size();
                    cout << "heap Size: " << heapSizeTest << " Path Number: " << vvPathCandidateEdge.size() << " rollBack: " << rollBack << endl;
                    *///cout << "popPathTest!" << endl;
                    break;
                }
            }
            if(qPathPop == true)
                break;
            if(fixLength >= vDistance[vFather[topPathID]] - vSPTDistance[vPathDeviation[topPathID]] + dEdge[topPathID])
                break;
        }
       // reverse(vvPathCandidateEdge[topPathID].begin(), vvPathCandidateEdge[topPathID].end());
        if(qPathPop == true)
        {
            stopCountHeap += 1;
            continue;
        }
        int unBlockEdgeID, pruneNumber;
        if(LastCount > 500)
        {
            if(unBlockEdge.find(blockEdge[blockEdge.size()-1]) != unBlockEdge.end())
                rollBack = false;
            else
            {
                unBlockEdge.insert(blockEdge[blockEdge.size()-1]);
                rollBack = true;

            }
            LastCount = 0;
            //break;
        }
        if(stopCountHeap > 500)
        {
            for(int i = 0; i < pruneOrderCandidate1.size(); i++)
            {
                //cout << "pruneOrderCandidate[i].first:" << pruneOrderCandidate1[i].first << " Number: " << pruneOrderCandidate1[i].second << endl;
                pruneOrder.update(pruneOrderCandidate1[i].first, pruneOrderCandidate1[i].second);
            }
            pruneOrder.extract_min(unBlockEdgeID, pruneNumber);
            unBlockEdge.insert(unBlockEdgeID);
            stopCountHeap = 0;
            //break;
        }

        /*if(popPathTest)
        {
            //cout << "stopCountHeap: " << stopCountHeap << endl;
            bool bContinue = true;
            while(bContinue){
                //cout <<  "trueFixLength: " << vDistance[vFather[topPathID]] - vSPTDistance[vPathDeviation[topPathID]] + dEdge[topPathID] << endl;
                bool qPathPop = false;
                int fixLength = 0;
				p
                for (auto ie = vvPathCandidateEdge[topPathID].begin();
                     ie != vvPathCandidateEdge[topPathID].end() ; ie++) {
                    fixLength += vEdge[*ie].length;
                    //cout << "fixLength: " << fixLength << endl;
                    for(int i = 0; i < blockEdge.size(); i++){
                        if(*ie == blockEdge[i] && unBlockEdge.find(blockEdge[i]) == unBlockEdge.end())
                        {
                            edgePath[i].push_back(topPathID);
                            pruneOrderCandidate1[i].second -= vmArc[topPathID].size();
                            //qPath.extract_min(topPathID, topPathDistance);
                            qPathPop = true;
                            if(i == blockEdge.size()-1)
                                LastCount += 1;
                            *//*if(i != blockEdge.size()-1)
                                previousCount += 1;*//*
                            *//*heapSizeTest = qPath.size();
                            cout << "heap Size: " << heapSizeTest << " Path Number: " << vvPathCandidateEdge.size() << " rollBack: " << rollBack << endl;
                            *//*//cout << "popPathTest!" << endl;
                            break;
                        }
                    }
                    if(qPathPop == true)
                        break;
                    if(fixLength >= vDistance[vFather[topPathID]] - vSPTDistance[vPathDeviation[topPathID]] + dEdge[topPathID])
                        break;
                }
                reverse(vvPathCandidateEdge[topPathID].begin(), vvPathCandidateEdge[topPathID].end());
                if(qPathPop == false)
                    bContinue = false;
                else
                    stopCountHeap += 1;

                int unBlockEdgeID, pruneNumber;
                if(LastCount > 500)
                {
                    if(unBlockEdge.find(blockEdge[blockEdge.size()-1]) != unBlockEdge.end())
                        rollBack = false;
                    else
                    {
                        unBlockEdge.insert(blockEdge[blockEdge.size()-1]);
                        rollBack = true;

                    }
                    LastCount = 0;
                    break;
                }
                if(stopCountHeap > 500)
                {
                    for(int i = 0; i < pruneOrderCandidate1.size(); i++)
                    {
                        //cout << "pruneOrderCandidate[i].first:" << pruneOrderCandidate1[i].first << " Number: " << pruneOrderCandidate1[i].second << endl;
                        pruneOrder.update(pruneOrderCandidate1[i].first, pruneOrderCandidate1[i].second);
                    }
                    pruneOrder.extract_min(unBlockEdgeID, pruneNumber);
                    unBlockEdge.insert(unBlockEdgeID);
                    *//*for(int i = 0; i < blockEdge.size(); i++)
                    {
                        if(blockEdge[i] == unBlockEdgeID){
                            for(int j = 0; j< edgePath[i].size(); j++){
                                if(!vmArc[edgePath[i][j]].empty())
                                    qPath.update(edgePath[i][j], vDistance[edgePath[i][j]]);
                            }
                        }
                    }*//*
                    *//*if(unBlockEdge.find(unBlockEdgeID) == unBlockEdge.end())
                    {
                        unBlockEdge.insert(unBlockEdgeID);
                        //cout << "repeat" << endl;
                        //rollBack = false;
                    }*//*
                   *//* else
                    {
                        unBlockEdge.insert(unBlockEdgeID);
                        rollBack = true;
                    }*//*

                  *//* if(unBlockEdge.find(blockEdge[blockEdge.size()-1]) != unBlockEdge.end())
                        rollBack = false;
                    else
                    {
                        unBlockEdge.insert(blockEdge[blockEdge.size()-1]);
                        rollBack = true;

                    }*//*
                    stopCountHeap = 0;
                    break;
                }
                //previousCount = 0;
            }
        }*/
        heapSizeTest = qPath.size();
        //cout << "blockEdgeSize: " << blockEdge.size() << " unblockEdgeSize: " << unBlockEdge.size() << endl;
        //cout << "heap Size: " << heapSizeTest << " Path Number: " << vvPathCandidateEdge.size() << " rollBack: " << rollBack << endl;

        //stopCount += 1;
        //cout << stopCount << endl;
        //cout << "stop Count: " << stopCount << ",,,,,,,,,,,,,,,,,,,,,,,,," << endl;
       /* if(stopCount >= 100 && stopCount < 106)
            rollBackPath.push_back(make_pair(topPathID, topPathDistance));*/
        //cout << "topPathID: " << topPathID << endl;
        //cout << "stop Count: " << stopCount << ",,,,,,,,,,,,,,,,,,,,,,,,," << endl;
        /*if(topPathDistance < oldDistance)
            cout<< "Error" <<endl;*/
        oldDistance = topPathDistance;

        //Loop Test
        unordered_set<int> us;
        bool bTopLoop = false;
        for(auto& v : vvPathCandidate[topPathID])
        {
            if(us.find(v) == us.end())
                us.insert(v);
            else
            {
                bTopLoop = true;
                break;
            }
        }
        if(!bTopLoop)
        {
			float simCount = 0;
            //popPath++;
            int n = 0;
            if (vvResult.size() == 0)
            {
                vvResult.push_back(vvPathCandidateEdge[topPathID]);
                kResults.push_back(topPathDistance);
                vkPath.push_back(vvPathCandidate[topPathID]);
                vResultID.push_back(topPathID);
                mPathFix.push_back(0);
                unordered_set<int> pTmp;
                for(auto ie = vvPathCandidateEdge[topPathID].begin(); ie != vvPathCandidateEdge[topPathID].end(); ie++)
                {
                    edgesOrderCandidate.insert(*ie);
                    edgesOrderCandidate1.push_back(make_pair(*ie,-1));
                    pTmp.insert(*ie);
                }
                pResult.push_back(pTmp);
            }
            else{
				float simMax = 0;
				float simMin = 1;
                for (int i = 0; i < vResultID.size(); i++) {
                    bool vFind = false;
                    /*for(int j = 0; j < vAncestor[topPathID].size(); j++)
                    {
                        if(vResultID[i] == vAncestor[topPathID][j])
                        {
                            vFind = true;
                            float simAdd = vPathFix[topPathID][j] + vPathLCA[topPathID][j] + mPathFix[i];
                            //cout << "simAdd: " << simAdd << endl;
                            //Sim 1
                            sim = simAdd / (kResults[i] + topPathDistance - simAdd);
                            if(sim <= t)
                                cout << "Ancestor Sim: " << sim << " vPathFix[topPathID][j]: " << vPathFix[topPathID][j] << " vPathLCA[topPathID][j]: " << vPathLCA[topPathID][j] << " mPathFix[i]: " << mPathFix[i] << endl;

                            //Sim 2
                            //sim = simAdd / (2*kResults[i]) + simAdd / (2*topPathDistance);

                            //Sim 3
                            //sim = sqrt((simAdd*simAdd) / ((double)kResults[i]*(double)topPathDistance));

                            //Sim 4
                            *//*int maxLength;
                            if(addLength > topPathDistance)
                                maxLength = simAdd;
                            else
                                maxLength = topPathDistance;
                            sim = simAdd / maxLength;*//*

                            //Sim 5
                            //sim = simAdd / kResults[i];
                        }
                    }*/
                    if(vFind == false)
                    {
                        for (auto m = vvResult[i].begin(); m != vvResult[i].end(); m++) {
                            for (auto ie = vvPathCandidateEdge[topPathID].begin();
                                 ie != vvPathCandidateEdge[topPathID].end(); ie++) {
                                if (*ie == *m) {
                                    addLength += vEdge[*ie].length;
                                }
                            }
                        }
                        /*if(addLength == 0) {
                            for (auto ie = vvPathCandidateEdge[topPathID].begin();
                                 ie != vvPathCandidateEdge[topPathID].end(); ie++) {
                                cout << *ie << " ";
                            }
                            cout << endl;
                            cout << vvPathCandidateEdge.size() << " " << topPathID << endl;
                            cout << "vmArc Size: " << vmArc.size() << " Candidate Size: " << vvPathCandidateEdge.size() << endl;
                            cout << "................................................" << endl;
                            for (auto m = vvResult[i].begin(); m != vvResult[i].end(); m++) {
                                cout << *m << " " ;
                            }
                            cout << endl;

                        }*/
                        //cout << "addLength: " << addLength << endl;
                        //sim = addLength / kResults[i];

                        //Sim 1
                        sim = addLength / (kResults[i] + topPathDistance - addLength);
                        //if(sim <= t)
                            //cout << "Normal Sim: " << sim << endl;
                        //Sim 2
                        //sim = addLength / (2*kResults[i]) + addLength / (2*topPathDistance);

                        //Sim 3
                        //sim = sqrt((addLength*addLength) / ((double)kResults[i]*(double)topPathDistance));

                        //Sim 4
                        //int maxLength;
						//sim = addLength / topPathDistance;
                        //if(kResults[i] > topPathDistance)
                        //	maxLength = kResults[i];
                        //else
                        //	maxLength = topPathDistance;
                        //sim = addLength / maxLength;

                        //Sim 5
                        //sim = addLength / kResults[i];
                        addLength = 0;
                    }
                    /*if (sim > t)
                        break;*/
					simCount += sim;
                    if (sim > simMax)
                        simMax = sim;
					if (sim < simMin)
						simMin = sim;
                }
                //if (sim <= t) {
				if(simMax <= t){
					AveSim += simCount;
                    if(simMax > maxSim)
						maxSim = simMax;
					if(simMin < minSim)
						minSim = simMin;
                    stopCount = 0;
                    stopCountHeap = 0;
                    popPathTest = false;
                    blockEdgeTest = false;
                    cout << "Here sim: " << sim << " topPath:" << topPathID << endl;
					//AveSim += sim;
                    /*for (int i = 0; i < vResultID.size(); i++) {
                        addLength = 0;
                        for (auto m = vvResult[i].begin(); m != vvResult[i].end(); m++) {
                            for (auto ie = vvPathCandidateEdge[topPathID].begin();
                                 ie != vvPathCandidateEdge[topPathID].end(); ie++) {
                                if (*ie == *m) {
                                    addLength += vEdge[*ie].length;
                                }
                            }
                        }
                        sim = addLength / (kResults[i] + topPathDistance - addLength);
                        cout << sim << " ";
                    }
                    cout << endl;*/
                    kResults.push_back(topPathDistance);
                    //cout << vvPathCandidateEdge[topPathID].size() << endl;
                    vvResult.push_back(vvPathCandidateEdge[topPathID]);
                    vkPath.push_back(vvPathCandidate[topPathID]);
                    vResultID.push_back(topPathID);
                    bPath[topPathID] = true;
                    mPathFix.push_back(vDistance[vFather[topPathID]] - vSPTDistance[vPathDeviation[topPathID]] + dEdge[topPathID]);
                    unordered_set<int> pTmp2;
                    for(auto ie = vvPathCandidateEdge[topPathID].begin(); ie != vvPathCandidateEdge[topPathID].end(); ie++)
                    {
                        if(edgesOrderCandidate.find(*ie) == edgesOrderCandidate.end()) {
                            edgesOrderCandidate.insert(*ie);
                            edgesOrderCandidate1.push_back(make_pair(*ie, -1));
                        }
                        pTmp2.insert(*ie);
                    }
                    pResult.push_back(pTmp2);
                }
                else
                {
                    //cout << edgesOrderCandidate1.size() << endl;
                    //cout << stopCount << endl;
                    int edge, edgeNumber;
                    //cout << vvPathCandidateEdge.size() << ".." << vResultID[vResultID.size()-1] << endl;
                    //if(vvPathCandidateEdge.size()-1-vResultID[vResultID.size()-1] == 100)
                   // if((stopCount >= 100 && !blockEdgeTest) || rollBack == true){
                   //rollBack = false;
                    if(rollBack == true || (stopCount >= 1000 && !blockEdgeTest)) {
                        popPathTest = true;
                        cout << "stopCount100........................" << topPathID << endl;
                        /*if(stopCount == 100)
                            rollBackPath.push_back(make_pair(topPathID, topPathDistance));*/
                        cout << "rollBack: " << rollBack << endl;
                        if(rollBack == false){
                            //cout << edgesOrder.size() << endl;
                            //cout << edgesOrderCandidate1.size() << endl;
                            for(int i = 0; i < edgesOrderCandidate1.size(); i++)
                            {
                                //cout << edgesOrderCandidate1[i].first << "......" << edgesOrderCandidate1[i].second << endl;
                                edgesOrder.update(edgesOrderCandidate1[i].first, edgesOrderCandidate1[i].second);
                            }
                            //cout << edgesOrder.size() << endl;
                            //Prepare to roll-back
                           /* pCandidatePathRollBack = pCandidatePath;
                            bPathRollBack = bPath;
                            dEdgeRollBack = dEdge;
                            vmArcRollBack = vmArc;
                            qPathRollBack = qPath;
                            topPathIDRollBack = topPathID;
                            topPathDistanceRollBack = topPathDistance;
                            vPathParentRollBack = vPathParent;	//Deviated from
                            vPathParentPosRollBack = vPathParentPos;
                            mPathFixRollBack = mPathFix;
                            vPathDeviationRollBack = vPathDeviation;
                            vPathLCANodeRollBack = vPathLCANode;
                            vPathDevPrimeRollBack = vPathDevPrime; // The other node of the deviation edge
                            vPathFixRollBack = vPathFix;
                            mPathFixRollBack = mPathFix;
                            vAncestorRollBack = vAncestor;
                            vmPathNodePosRollBack = vmPathNodePos;
                            vFatherRollBack = vFather;*/

                            vmArcRollBack = vmArc;
                            vvPathCandidateRollBack =  vvPathCandidate;
                            vvPathCandidateEdgeRollBack = vvPathCandidateEdge;
                            pCandidatePathRollBack = pCandidatePath;
                            vDistanceRollBack = vDistance;
                            bPathRollBack = bPath;
                            dEdgeRollBack = dEdge;
                            vPathParentRollBack = vPathParent;	//Deviated from
                            vPathParentPosRollBack = vPathParentPos;	//Deviated Pos from Parent
                            vPathDeviationRollBack = vPathDeviation; //deviation node
                            vPathLCANodeRollBack = vPathLCANode; //LCA Node with path 1
                            vPathLCARollBack = vPathLCA;
                            vPathDevPrimeRollBack = vPathDevPrime; // The other node of the deviation edge
                            vPathFixRollBack = vPathFix;
                            mPathFixRollBack = mPathFix;
                            vAncestorRollBack = vAncestor;
                            qPathRollBack = qPath;
                            topPathIDRollBack = topPathID;
                            topPathDistanceRollBack = topPathDistance;
                            vmPathNodePosRollBack = vmPathNodePos;
                            vFatherRollBack = vFather;
                            kResultsRollBack = kResults;
                            vvResultRollBack = vvResult;
                            vkPathRollBack = vkPath;
                            vResultIDRollBack = vResultID;
                            /*if(!edgesOrder.empty()){
                                edgesOrder.extract_min(edge, edgeNumber);
                                cout << edge << "edge................................................................................................................" << endl;
                                if(pruneOrderCandidate.find(edge) == pruneOrderCandidate.end())
                                {
                                    blockEdge.push_back(edge);
                                    //cout << "edge" << endl;
                                    pruneOrderCandidate.insert(edge);
                                    pruneOrderCandidate1.push_back(make_pair(edge,0));
                                }
                                //cout << edgeNumber << endl;
                            }*/
                            if(!edgesOrder.empty()){
                                bool bEdgesOrder = true;
                                //vector<pair<int,int> > edgesOrderCandidate2 = edgesOrderCandidate1;
                                sort(edgesOrderCandidate1.begin(), edgesOrderCandidate1.end(), Compare2);
                                int randomPruneEdge = int (edgesOrderCandidate1.size() * 0.6);
                                //int randomPruneEdge = rand() % int (edgesOrderCandidate1.size() * 0.4);
                                //randomPruneEdge += int (0.3 * edgesOrderCandidate1.size());
                                edge = edgesOrderCandidate1[randomPruneEdge].first;
                                edgesOrderCandidate1.erase(edgesOrderCandidate1.begin() + randomPruneEdge);
                                //edgesOrder.extract_min(edge, edgeNumber);
                                while(bEdgesOrder)
                                {
                                    if(pruneOrderCandidate.find(edge) == pruneOrderCandidate.end())
                                    {
                                        //cout << edge << "edge................................................................................................................" << endl;
                                        vector<int> v;
                                        edgePath.push_back(v);
                                        blockEdge.push_back(edge);
                                        pruneOrderCandidate.insert(edge);
                                        pruneOrderCandidate1.push_back(make_pair(edge,0));
                                        bEdgesOrder = false;
                                    }
                                    else
                                        edgesOrder.extract_min(edge, edgeNumber);
                                }
                                cout << "blockEdge Size: " << blockEdge.size() << " unblockEdge Size: " << unBlockEdge.size()<< endl;
                            }
                            //rollBack = true;
                        }
                        else{
                            //cout << "topPathID: " << topPathID << endl;
                            //cout << "vvPathCandidate: " << vvPathCandidate.size() << " vvPathCandidateRollBack: " << vvPathCandidateEdgeRollBack.size() << endl;
                            vmArc = vmArcRollBack;
                            vvPathCandidate =  vvPathCandidateRollBack;
                            vvPathCandidateEdge = vvPathCandidateEdgeRollBack;
                            //vvPathCandidate.erase(vvPathCandidate.begin()+vmArcRollBack.size(), vvPathCandidate.end());
                            //vvPathCandidateEdge.erase(vvPathCandidateEdge.begin()+vmArcRollBack.size(), vvPathCandidateEdge.end());
                            //vvPathCandidateEdge.resize(vmArcRollBack.size());
                            //vvPathCandidate.resize(vmArcRollBack.size());
                            //cout << "vvPathCandidate: " << vvPathCandidate.size() << " vvPathCandidateRollBack: " << vvPathCandidateEdgeRollBack.size() << endl;
                            pCandidatePath = pCandidatePathRollBack;
                            //pCandidatePath.erase(pCandidatePath.begin()+vmArcRollBack.size(), pCandidatePath.end());
                            //vDistance.erase(vDistance.begin()+vmArcRollBack.size(), vDistance.end());
                            //vDistance.resize(vmArcRollBack.size());
                            vDistance = vDistanceRollBack;
                            bPath = bPathRollBack;
                            dEdge = dEdgeRollBack;
                            //bPath.erase(bPath.begin()+vmArcRollBack.size(), bPath.end());
                            //dEdge.erase(dEdge.begin()+vmArcRollBack.size(), dEdge.end());
                            vPathParent = vPathParentRollBack;	//Deviated from
                            vPathParentPos = vPathParentPosRollBack;
                            //vPathParent.erase(vPathParent.begin()+vmArcRollBack.size(), vPathParent.end());
                            //vPathParentPos.erase(vPathParentPos.begin()+vmArcRollBack.size(), vPathParentPos.end());
                            //vPathDeviation.erase(vPathDeviation.begin()+vmArcRollBack.size(), vPathDeviation.end());
                            //vPathLCANode.erase(vPathLCANode.begin()+vmArcRollBack.size(), vPathLCANode.end());
                            vPathDeviation = vPathDeviationRollBack;
                            vPathLCA = vPathLCARollBack;
                            vPathLCANode  = vPathLCANodeRollBack;
                            //vPathDevPrime.erase(vPathDevPrime.begin()+vmArcRollBack.size(), vPathDevPrime.end());
                            //vPathFix.erase(vPathFix.begin()+vmArcRollBack.size(), vPathFix.end());
                            //mPathFix.erase(mPathFix.begin()+vmArcRollBack.size(), mPathFix.end());
                            vPathDevPrime  = vPathDevPrimeRollBack; // The other node of the deviation edge
                            vPathFix = vPathFixRollBack;
                            mPathFix = mPathFixRollBack;
                            vAncestor = vAncestorRollBack;
                            //vAncestor.erase(vAncestor.begin()+vmArcRollBack.size(), vAncestor.end());
                            qPath = qPathRollBack;
                            topPathID = topPathIDRollBack;
                            topPathDistance = topPathDistanceRollBack;
                            //vmPathNodePos.erase(vmPathNodePos.begin()+vmArcRollBack.size(), vmPathNodePos.end());
                            //vFather.erase(vFather.begin()+vmArcRollBack.size(), vFather.end());
                            vmPathNodePos = vmPathNodePosRollBack;
                            vFather = vFatherRollBack;
                            //qPath.update(topPathID,topPathDistance);

                            kResults = kResultsRollBack;
                            vvResult = vvResultRollBack;
                            vkPath = vkPathRollBack;
                            vResultID = vResultIDRollBack;

                            /*vmArc = vmArcRollBack;
                            vvPathCandidate =  vvPathCandidateRollBack;
                            vvPathCandidateEdge = vvPathCandidateEdgeRollBack;
                            pCandidatePath = pCandidatePathRollBack;
                            vDistance = vDistanceRollBack;
                            bPath = bPathRollBack;
                            dEdge = dEdgeRollBack;
                            vPathParent = vPathParentRollBack;	//Deviated from
                            vPathParentPos = vPathParentPosRollBack;	//Deviated Pos from Parent
                            vPathDeviation = vPathDeviationRollBack; //deviation node
                            vPathLCANode  = vPathLCANodeRollBack; //LCA Node with path 1
                            vPathDevPrime  = vPathDevPrimeRollBack; // The other node of the deviation edge
                            vPathFix = vPathFixRollBack;
                            mPathFix = mPathFixRollBack;
                            vAncestor = vAncestorRollBack;
                            qPath = qPathRollBack;
                            topPathID = topPathIDRollBack;
                            topPathDistance = topPathDistanceRollBack;
                            vmPathNodePos = vmPathNodePosRollBack;
                            vFather = vFatherRollBack;
                            qPath.update(topPathID,topPathDistance);*/
                            /*for(int i = 0; i < rollBackPath.size(); i++)
                            {
                                qPath.update(rollBackPath[i].first, rollBackPath[i].second);
                            }
                            rollBackPath.clear();*/
                        }
                        /*if(!edgesOrder.empty()){
                            edgesOrder.extract_min(edge, edgeNumber);
                            blockEdge.push_back(edge);
                        }*/
                        //heapSize = qPath.size();
                        stopCount = 1000;
                        rollBack = false;
                        bUpdate = false;
                        stopCountHeap = 0;
                        blockEdgeTest = true;
                        //stopCountTest = 100;
                    }
                    if(stopCount >= 1500){
                        cout << "stopCountNextEdge........................" << endl;
                        if(!edgesOrder.empty()){
                            bool bEdgesOrder = true;
                            sort(edgesOrderCandidate1.begin(), edgesOrderCandidate1.end(), Compare2);
                            int randomPruneEdge = int (edgesOrderCandidate1.size() * 0.6);
                            //int randomPruneEdge = rand() % int (edgesOrderCandidate1.size() * 0.2);
                            //randomPruneEdge += int (0.3 * edgesOrderCandidate1.size());
                            edge = edgesOrderCandidate1[randomPruneEdge].first;
                            edgesOrderCandidate1.erase(edgesOrderCandidate1.begin() + randomPruneEdge);
                            //edgesOrder.extract_min(edge, edgeNumber);
                            while(bEdgesOrder)
                            {
                                if(pruneOrderCandidate.find(edge) == pruneOrderCandidate.end())
                                {
                                    //cout << edge << "edge................................................................................................................" << endl;
                                    vector<int> v;
                                    edgePath.push_back(v);
                                    blockEdge.push_back(edge);
                                    pruneOrderCandidate.insert(edge);
                                    pruneOrderCandidate1.push_back(make_pair(edge,0));
                                    bEdgesOrder = false;
                                }
                                else
                                    edgesOrder.extract_min(edge, edgeNumber);
                            }
                            cout << "blockEdge Size: " << blockEdge.size() << " unblockEdge Size: " << unBlockEdge.size()<< endl;
                        }
                        stopCount = 1000;
                        stopCountHeap = 0;
                        //cout << heapSizeTest << "," << heapSize << endl;
                        /*if(heapSizeTest <= heapSize)
                            unBlockEdge.insert(blockEdge[blockEdge.size()-1]);*/
                        //cout << unBlockEdge.size() << endl;
                    }
                    //heapSizeTest = qPath.size();
                   /* cout << heapSizeTest << "," << heapSize << endl;
                    if(heapSizeTest <= heapSize)
                        unBlockEdge.insert(blockEdge[blockEdge.size()-1]);*/
                    //stopCount += 1;
                    //stopCountHeap += 1;
                }

            }
        }

        vector<int> vTwo;
        vTwo.push_back(topPathID);
        if(vFather[topPathID] != -1 &&  !vmArc[vFather[topPathID]].empty())
            vTwo.push_back(vFather[topPathID]);
        for(auto& pID : vTwo)
        {
            bool bLoop = true;
            bool bmArcTmp = false; //if all mArcTmp has the block edge
            while(bLoop)
            {
                //No More Candidate from current class
                if(vmArc[pID].empty())
                {
                    vvPathCandidate[pID].clear();
                    vvPathCandidateEdge[pID].clear();
                    vmPathNodePos[pID].clear();
                    if(bmArcTmp)
                        stopCountHeap += 1;
                    break;
                }
                int mineID;
                int eReducedLen;
                auto it = vmArc[pID].begin();
                eReducedLen = (*it).first;
                mineID = (*it).second;
                //countNumber += 1;
                vmArc[pID].erase(it);

                //eNodeID1 is also the first point from the deviation edge
                int eNodeID1 = vEdge[mineID].ID1;
                int eNodeID2 = vEdge[mineID].ID2;

                bool bFixedLoop = false;
                unordered_set<int> sE;
                for(int i = vmPathNodePos[pID][eNodeID2]; i < (int)vvPathCandidate[pID].size(); i++)
                {
                    if(sE.find(vvPathCandidate[pID][i]) == sE.end())
                        sE.insert(vvPathCandidate[pID][i]);
                    else
                    {
                        bFixedLoop = true;
                        break;
                    }
                }

                if(bFixedLoop)
                    continue;

                for(int i = 0; i < blockEdge.size(); i++)
                {
                    if(unBlockEdge.find(blockEdge[i]) == unBlockEdge.end())
                    {
                        if(mineID == blockEdge[i])
                            break;
                        int _p, _q, _m;
                        _p = rEulerSeq[vEdge[mineID].ID1];
                        _q = rEulerSeq[vEdge[blockEdge[i]].ID1];
                        _m = rEulerSeq[vEdge[blockEdge[i]].ID2];
                        int LCA1 = LCAQuery(_p, _q, vSPT,  vSPTHeight,vSPTParent);
                        int LCA2 = LCAQuery(_p, _m, vSPT,  vSPTHeight,vSPTParent);
                        if(LCA1 ==vEdge[blockEdge[i]].ID1 && LCA2 == vEdge[blockEdge[i]].ID2)
                        {
                            if(i == blockEdge.size()-1)
                                LastCount += 1;
                            pruneOrderCandidate1[i].second -= 1;
                            bmArcTmp = true;
                            break;
                        }
                    }
                }
                if(bmArcTmp)
                    continue;


                int distTmp = vDistance[pID] - vSPTDistance[eNodeID2] + vEdge[mineID].length;
                bLoop = false;
                vPathDevPrime.push_back(eNodeID1);
                vPathDeviation.push_back(eNodeID2);
                multimap<int, int> mArcTmp;
                vector<pair<int, int>> mArcTmpTest;
                vPath.clear();
                vPathEdge.clear();
                pCandidateTmp.clear();
                int p = eNodeID1;
                int e = vSPTParentEdge[p];
                while(p != -1)
                {
                    sE.insert(p);
                    vPath.push_back(p);
                    for(int i = 0; i < (int)adjListEdgeR[p].size(); i++)
                    {
                        int reID = adjListEdgeR[p][i].second;
                        int eID1 = vEdge[reID].ID1;
                        int dTmp = distTmp + vEdge[reID].length;

                        if(sE.find(eID1) != sE.end())
                            continue;

                        if(reID != e && reID != mineID){
                            mArcTmp.insert(make_pair(vEdgeReducedLength[reID], reID));
                        }
                    }
                    vPathEdge.push_back(e);
                    pCandidateTmp.insert(e);
                    distTmp += vEdge[e].length;

                    p = vSPTParent[p];
                    if(vSPTParent[p] == -1)
                    {
                        vPath.push_back(p);
                        break;
                    }
                    e = vSPTParentEdge[p];
                }


                dEdge.push_back(vEdge[mineID].length);
                int dist = vDistance[pID] - vSPTDistance[eNodeID2] + vSPTDistance[eNodeID1] + vEdge[mineID].length;
                vDistance.push_back(dist);
                if (bPath[pID])
                {
                    if(pID == 0)
                    {
                        vector<int> tmpAnc;
                        tmpAnc.push_back(pID);
                        vAncestor.push_back(tmpAnc);
                    }
                    else{
                        vAncestor.push_back(vAncestor[pID]);
                        vAncestor[vDistance.size()-1].push_back(pID);
                    }
                }
                else{
                    vAncestor.push_back(vAncestor[pID]);
                }

                for(int i = 0; i < vAncestor[vDistance.size()-1].size(); i++)
                {
                    //LCA
                    int _p, _q;
                    _p = rEulerSeq[vPathDevPrime[vAncestor[vDistance.size()-1][i]]];
                    _q = rEulerSeq[eNodeID1];
                    int LCA = LCAQuery(_p, _q, vSPT,  vSPTHeight,vSPTParent);
                    int dLCA = vSPTDistance[LCA];
                    if(i == 0)
                    {
                        vector<float> mTmpLCA;
                        vector<int> mTmpLCANode;
                        mTmpLCA.push_back(dLCA);
                        vPathLCA.push_back(mTmpLCA);
                        mTmpLCANode.push_back(LCA);
                        vPathLCANode.push_back(mTmpLCANode);
                    }
                    else{
                        vPathLCA[vDistance.size()-1].push_back(dLCA);
                        vPathLCANode[vDistance.size()-1].push_back(LCA);
                    }
                    vector<float> vTmpdFix;
                    if(i == vPathLCANode[pID].size())
                    {
                        vPathLCANode[pID][i] = vPathLCANode[pID][0];
                    }
                    if(vSPTHeight[vPathLCANode[pID][i]] > vSPTHeight[eNodeID2])
                    {
                        int dFix = vSPTDistance[vPathDevPrime[pID]] - vSPTDistance[eNodeID2] + vPathFix[pID][i];
                        if(i == 0)
                        {
                            int dFix = vSPTDistance[vPathLCANode[pID][i]] - vSPTDistance[eNodeID2] + vPathFix[pID][0];
                            vTmpdFix.push_back(dFix);
                            vPathFix.push_back(vTmpdFix);
                        }
                        else{
                            if(vAncestor[vDistance.size()-1][i] == pID && bPath[pID])
                                vPathFix[vDistance.size()-1].push_back(dFix);
                            else{
                                int dFix = vSPTDistance[vPathLCANode[pID][i]] -vSPTDistance[eNodeID2] + vPathFix[pID][i];
                                vPathFix[vDistance.size()-1].push_back(dFix);
                            }
                        }
                    }
                    else{
                        if(i == 0)
                        {
                            vTmpdFix.push_back(vPathFix[pID][0]);
                            vPathFix.push_back(vTmpdFix);
                        }
                        else{
                            if(vAncestor[vDistance.size()-1][i] == pID && bPath[pID])
                                vPathFix[vDistance.size()-1].push_back(vPathFix[pID][i] + vSPTDistance[vPathDevPrime[pID]] - vSPTDistance[eNodeID2]);
                            else
                                vPathFix[vDistance.size()-1].push_back(vPathFix[pID][i]);
                        }
                    }
                }
                vFather.push_back(pID);
                bPath.push_back(false);
                //qPath.update(vDistance.size()-1, dist);

                reverse(vPath.begin(), vPath.end());
                reverse(vPathEdge.begin(), vPathEdge.end());
                unordered_map<int, int> mE;
                int i;
                //Pos stop at eNodeID1 as it is boundary of fixed
                for(i = 0; i < (int)vPath.size(); i++)
                    mE[vPath[i]] = i;
                vmPathNodePos.push_back(mE);
                vPath.push_back(eNodeID2);
                vPathEdge.push_back(mineID);
                pCandidateTmp.insert(mineID);

                for(int j = vmPathNodePos[pID][eNodeID2]; j+1 < (int)vvPathCandidate[pID].size(); j++)
                {
                    int nodeID = vvPathCandidate[pID][j+1];
                    vPath.push_back(nodeID);
                    int edgeID = vvPathCandidateEdge[pID][j];
                    vPathEdge.push_back(edgeID);
                    pCandidateTmp.insert(edgeID);
                }
                vmArc.push_back(mArcTmp);
                vvPathCandidate.push_back(vPath);
                vvPathCandidateEdge.push_back(vPathEdge);
                /*cout << vDistance.size()-1 << " vPathEdgeSize:"<< vPathEdge.size() << endl;
                cout << " vPathEdge: ";
                for(int i = 0; i < vPathEdge.size(); i++)
                {
                    cout << vPathEdge[i] << " ";
                }
                cout << endl;
                cout << "vvPathCandidateSize: " << vvPathCandidateEdge.size() << endl;
                //if(vvPathCandidateEdge[vvPathCandidateEdge.size()-1])
                if(vPathEdge.empty())
                    cout << "empty!" << " Distance: " << dist << endl;*/
                pCandidatePath.push_back(pCandidateTmp);
                stopCount +=1;
                qPath.update(vDistance.size() - 1, dist);
                countNumber += 1;
                //cout << "update!............................" << endl;
                stopCountHeap = 0;
                LastCount = 0;


                for(int i = 0; i < edgesOrderCandidate1.size(); i++){
                    if(pCandidateTmp.find(edgesOrderCandidate1[i].first) != pCandidateTmp.end() )
                        edgesOrderCandidate1[i].second -= 1;
                }

                /*bool bBlockEdge = false;
                //cout << "bE: " << blockEdge.size() << endl;
                if(blockEdge.size() > 0)
                {
                    for(int j = 0; j < blockEdge.size(); j++)
                    {
                        if(pCandidateTmp.find(blockEdge[j]) != pCandidateTmp.end() && unBlockEdge.find(blockEdge[j]) == unBlockEdge.end())
                        {
                            //cout << "true" << endl;
                            bBlockEdge = true;
                            break;
                        }
                    }
                }*/
                //cout << bBlockEdge << endl;
                //if(!bmArcTmp && stopCountHeap != 100) {
                //cout << "bmArcTmp:" << bmArcTmp << endl;
                /*if( stopCount < 1000) {
                    //cout << "update........................................" << endl;
                    qPath.update(vDistance.size() - 1, dist);
                    stopCountHeap = 0;
                    //bUpdate = true;
                }*/

                /*heapSizeTest = qPath.size();
                cout << "heap Size: " << heapSizeTest << " Path Number: " << vvPathCandidateEdge.size() << endl;*/
            }
        }
    }
		//AveSim = AveSim /(kResults.size()-1);
		AveSim = AveSim / (kResults.size()*(kResults.size()-1)/2);
		cout << AveSim << endl;
    cout << "EdgesBlocking countNumber: "<< countNumber << " Pop Path: " << popPath << endl;
    return -1;
}
