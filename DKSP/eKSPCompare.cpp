#include "graph.h"

int Graph::eKSPCompare(int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, double t, int& countNumber, int& popPath, float& percentage, float& SimTime, float& AveSim, float& minSim, float& maxSim)
{
	AveSim = 0;
	minSim = 1;
	maxSim = 0;
//	float simCount = 0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::high_resolution_clock::time_point t3;
    std::chrono::high_resolution_clock::time_point t4;
    t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span;
    std::chrono::duration<double> time_span1;
    //float percentage = 0;
    //Shortest Path Tree Info
    float countAncestor = 0;
    float countNonAncestor = 0;
    countNumber = 0;
    popPath = 0;
    bool bCountNumber = true;
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

    /*for(int i = 0; i < rEulerSeq.size(); i++){
        cout << rEulerSeq[i] << endl;
    }*/
    double time = 0;
    SimTime = 0;
    double edgeTime = 0;
	double sETime = 0;
    double pathTime = 0;
    double loopTime = 0;

    vector<int> vEdgeReducedLength(vEdge.size(), INF);
    for(int i = 0; i < (int)vEdge.size(); i++)
        vEdgeReducedLength[i] = vEdge[i].length + vSPTDistance[vEdge[i].ID1] - vSPTDistance[vEdge[i].ID2];

    vector<vector<int> > vvResult;	//Exact Path
    vvResult.reserve(k);
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

    vector<vector<int> > vvPathCandidate;	 //nodes
    vector<vector<int> > vvPathCandidateEdge;//edges
	vector<set<int>> vvsE;
    //vector<unordered_map<int, int> > vmPathNodePos;	//Position of the non-fixed vertices
    //The size of mvPathNodePos indicates the fixed pos
    vector<vector<pair<int, int> > > vvPathNonTree;
    //vector<multimap<int, int> > vmArc;
    vector<vector<pair<int, int> > > vvmArc;
    vector<vector<pair<int, int> > > vvmPathNodePos;
    vector<int> vFather;

    vector<unordered_set<int> > pResult;
    vFather.push_back(-1);

    float sim;

    benchmark::pHeap<3, int, int, int> Arc(2*nodeNum);
    //vector<pHeap<3, int, int, int>> ArcList;
    vector<int> vPath;
    vector<int> vPathEdge;
    vPath.push_back(ID2);
    vector<pair<int, int> > vPathNonTree;
    int p = vSPTParent[ID2];
    int e = vSPTParentEdge[ID2];
    //multimap<int, int> mArc;
    float sim1;
    float addLength1 = 0;
    int oldP = ID2;
    while(p != -1)
    {
        //cout << "p:" << oldP << endl;
        for(int i = 0; i < (int)adjListEdgeR[oldP].size(); i++)
        {
            int eID = adjListEdgeR[oldP][i].second;
            if(eID != e){
                //mArc.insert(make_pair(vEdgeReducedLength[eID], eID));
                Arc.update(eID,vEdgeReducedLength[eID],vPath.size()-1);
                //cout << vEdge[eID].ID2 << " "<< vPath.size()-1 << endl;
            }
            //	else
            //		cout << "Skip " << e << endl;
        }
        vPath.push_back(p);
        oldP = p;
        vPathEdge.push_back(e);
        //cout << p << " " << vEdge[e].ID1 << endl;
        e = vSPTParentEdge[p];
        p = vSPTParent[p];
    }
    int mmArcEdgeID, mmArcReducedLength, mmArcPosition;
    vector<pair<int, int> > mmArc;
    vector<pair<int, int> > mmPos;
	/*pair<int,int> mmArcPosTmp;
	mmArc.assign(Arc.size(),mmArcPosTmp);
	mmPos.assign(Arc.size(),mmArcPosTmp);*/
    while(!Arc.empty())
    {
        Arc.extract_min(mmArcEdgeID, mmArcReducedLength, mmArcPosition);
		//mmArc[Arc.size()] = make_pair(mmArcEdgeID,mmArcReducedLength);
		//mmPos[Arc.size()] = make_pair(mmArcEdgeID,mmArcPosition);
        mmArc.push_back(make_pair(mmArcEdgeID,mmArcReducedLength));
        mmPos.push_back(make_pair(mmArcEdgeID,mmArcPosition));
        //cout << mmArcEdgeID << "  " << mmArcReducedLength << " " << mmArcPosition << endl;
    }
    //ArcList.push_back(Arc);
    //vmArc.push_back(mArc);
    vvmArc.push_back(mmArc);
    vvmPathNodePos.push_back(mmPos);

    reverse(vPath.begin(), vPath.end());
    reverse(vPathEdge.begin(), vPathEdge.end());
//	cout << "finish Arc" << endl;
    //cout << vPath[vPath.size()-1-173] << endl;
   /* unordered_map<int, int> mPos;
    for(int i = 0; i < (int)vPath.size(); i++)
        mPos[vPath[i]] = i;
    vmPathNodePos.push_back(mPos);*/

    benchmark::heap<2, int, int> qPath(nodeNum);
    vvPathCandidate.push_back(vPath);
    vvPathCandidateEdge.push_back(vPathEdge);
    vDistance.push_back(vSPTDistance[ID2]);
	//set<int> tempsE;
	//vvsE.push_back(tempsE);

    //Open
    bPath.push_back(true);
    dEdge.push_back(0);
    vPathParent.push_back(-1);
    vPathParentPos.push_back(0);
    vPathDeviation.push_back(ID2);


    vector<int> vTmpLCANode;
    vTmpLCANode.push_back(ID2);

    //Open
    vPathLCANode.push_back(vTmpLCANode);
    vPathDevPrime.push_back(ID2);


    vector<float> vTmpFix;
    vTmpFix.push_back(0);

    //Open
    vPathFix.push_back(vTmpFix);


    vector<int> vTmpAncestor;
    vTmpAncestor.push_back(-1);

    //Open
    vAncestor.push_back(vTmpAncestor);

    vector<float> vTmpLCA;
    vTmpLCA.push_back(vSPTDistance[ID2]);

    //Open
    vPathLCA.push_back(vTmpLCA);

    qPath.update(vvPathCandidate.size()-1, vSPTDistance[ID2]);

    vector<int> vResultID;
    int topPathID, topPathDistance;
    int pcount = 0;
    int oldDistance = -1;
    bool bError = false;
    //t2 = std::chrono::high_resolution_clock::now();
    //time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    //cout << "before Time:" << time_span.count() << endl;
    //t3 = std::chrono::high_resolution_clock::now();
    while((int)kResults.size() < k && !qPath.empty())
    {
        //t1 = std::chrono::high_resolution_clock::now();
        float addLength = 0;
        pcount++;
        popPath++;
        qPath.extract_min(topPathID, topPathDistance);
        //cout << topPathID << "\t" << topPathDistance << endl;
        //cout << topPathID << endl;
        if(topPathDistance < oldDistance)
            cout<< "Error" <<endl;
        oldDistance = topPathDistance;
        //Loop Test

        //Change to vector
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
        /*t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        loopTime += time_span.count();*/

        t1 = std::chrono::high_resolution_clock::now();
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

                //Open
                mPathFix.push_back(0);

                unordered_set<int> pTmp;
                for(auto ie = vvPathCandidateEdge[topPathID].begin(); ie != vvPathCandidateEdge[topPathID].end(); ie++)
                {
                    pTmp.insert(*ie);
                }
                pResult.push_back(pTmp);
            }
            else{
				float simMax = 0;
				float simMin = 1;
                for (int i = 0; i < vResultID.size(); i++) {
                    bool vFind = false;

                    //Open
                    for(int j = 0; j < vAncestor[topPathID].size(); j++)
                    {
                        if(vResultID[i] == vAncestor[topPathID][j])
                        {
                            countAncestor += 1;
                            vFind = true;
                            float simAdd = vPathFix[topPathID][j] + vPathLCA[topPathID][j] + mPathFix[i];
                            //Sim 1
							sim = simAdd / (kResults[i] + topPathDistance - simAdd);
                            //cout << "vPathFix[topPathID][j]: "<< vPathFix[topPathID][j] << " vPathLCA[topPathID][j]: " << vPathLCA[topPathID][j] << " mPathFix[i]: " << mPathFix[i] << endl;
                            //Sim 2
                            //sim = simAdd / (2*kResults[i]) + simAdd / (2*topPathDistance);

                            //Sim 3
							//sim = sqrt((simAdd*simAdd) / ((double)kResults[i]*(double)topPathDistance));

                            //Sim 4
							//sim = simAdd / topPathDistance;
                            //int maxLength;
                            //if(addLength > topPathDistance)
                            //    maxLength = simAdd;
                            //else
                            //    maxLength = topPathDistance;
                            //sim = simAdd / maxLength;


                            //Sim 5
							//sim = simAdd / kResults[i];
                            break;
                        }
                    }


                    if(vFind == false)
                    {
                        countNonAncestor += 1;
                        for(auto ie = vvPathCandidateEdge[topPathID].begin(); ie != vvPathCandidateEdge[topPathID].end(); ie++){
                            if(pResult[i].find(*ie) != pResult[i].end())
                                addLength += vEdge[*ie].length;
                        }
                        //Sim 1
                        sim = addLength / (kResults[i] + topPathDistance - addLength);
                        //cout << "addLength: "<< addLength << " kResults[i] + topPathDistance - addLength: " << (kResults[i] + topPathDistance - addLength) << endl;
                        //Sim 2
                        //sim = addLength / (2*kResults[i]) + addLength / (2*topPathDistance);

                        //Sim 3
                        //sim = sqrt((addLength*addLength) / ((double)kResults[i]*(double)topPathDistance));

                        //Sim 4
						//sim = addLength/ topPathDistance;
                        //int maxLength;
                        //if(addLength > topPathDistance)
                        //    maxLength = addLength;
                        //else
                        //    maxLength = topPathDistance;
                        //sim = addLength / maxLength;


                        //Sim 5
                        //sim = addLength / kResults[i];
                        addLength = 0;
                    }
					simCount += sim;
					//if(sim > t)
					//	break;
                    if (sim > simMax)
                        simMax = sim;
					if(sim < simMin)
						simMin = sim;
						//break;
					//else
					//	simCount += sim;
					
                }
				//simMax = 0;
                //if (sim <= t) {
				if(simMax <= t){
                    if(simMax > maxSim)
						maxSim = simMax;
					if(simMin < minSim)
						minSim = simMin;
					//cout << "max: " << maxSim << endl;
                    //cout << "sim: " << sim << " topPath:" << topPathID << endl;
					//simCount += sim;
					AveSim += simCount;
                    kResults.push_back(topPathDistance);
                    vvResult.push_back(vvPathCandidateEdge[topPathID]);
                    vkPath.push_back(vvPathCandidate[topPathID]);
                    vResultID.push_back(topPathID);

                    //Open
                    bPath[topPathID] = true;
                    mPathFix.push_back(vDistance[vFather[topPathID]] - vSPTDistance[vPathDeviation[topPathID]] + dEdge[topPathID]);

                    unordered_set<int> pTmp2;
                    for(auto ie = vvPathCandidateEdge[topPathID].begin(); ie != vvPathCandidateEdge[topPathID].end(); ie++)
                    {
                        pTmp2.insert(*ie);
                    }
                    pResult.push_back(pTmp2);
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        SimTime += time_span.count();

       // t1 = std::chrono::high_resolution_clock::now();
        vector<int> vTwo;
        vTwo.push_back(topPathID);
        //if(vFather[topPathID] != -1 &&  !vmArc[vFather[topPathID]].empty())
        if(vFather[topPathID] != -1 &&  !vvmArc[vFather[topPathID]].empty())
        //if(vFather[topPathID] != -1 &&  !ArcList[vFather[topPathID]].empty())
            vTwo.push_back(vFather[topPathID]);
        for(auto& pID : vTwo)
        {
            //cout << "path ID: " << pID << endl;
            bool bLoop = true;
            while(bLoop)
            {
                //t1 = std::chrono::high_resolution_clock::now();
                //No More Candidate from current class
                //if(vmArc[pID].empty())
                if(vvmArc[pID].empty())
                //if(ArcList[pID].empty())
                {
                    vvPathCandidate[pID].clear();
                    vvPathCandidateEdge[pID].clear();
                    //vmPathNodePos[pID].clear();
                    break;
                }
                int mineID;
                int eReducedLen;
                //auto it = vmArc[pID].begin();
                auto it = vvmArc[pID].begin();
                //auto it = vvmArc[pID].end()-1;
                /*int mmArcEdgeID, mmArcReducedLength, mmArcPosition;
                ArcList[pID].extract_min(mmArcEdgeID,mmArcReducedLength,mmArcPosition);*/
                //eReducedLen = (*it).first;
                //mineID = (*it).second;
                eReducedLen = (*it).second;
                mineID = (*it).first;
				
                /*eReducedLen = mmArcReducedLength;
                mineID = mmArcEdgeID;*/
                countNumber += 1;
                //vmArc[pID].erase(it);
                //vvmArc[pID].erase(it);
                vvmArc[pID].erase(it);

                //eNodeID1 is also the first point from the deviation edge
                int eNodeID1 = vEdge[mineID].ID1;
                int eNodeID2 = vEdge[mineID].ID2;

                bool bFixedLoop = false;

                //Change to vector
                auto itp = vvmPathNodePos[pID].begin();
                //auto itp = vvmPathNodePos[pID].end()-1;
				//cout << "itp firts: " << (*itp).first << " itp second: " << (*itp).second << endl;
                int ReID2Pos = (*itp).second;
                //int ReID2Pos = mmArcPosition;
                int eID2Pos = vvPathCandidate[pID].size()-ReID2Pos-1;
                /*if(vmPathNodePos[pID][eNodeID2] != eID2Pos)
                {
                    cout << "false" << endl;
                    cout <<vmPathNodePos[pID][eNodeID2] << "    " << eID2Pos << endl;
                    //eID2Pos += 1;
                    //eID2Pos = vmPathNodePos[pID][eNodeID2];
                }*/
                vvmPathNodePos[pID].erase(itp);
				unordered_set<int> sE;
                //set<int> sE = vvsE[pID];
                //for(int i = vmPathNodePos[pID][eNodeID2]; i < (int)vvPathCandidate[pID].size(); i++)
                //t3 = std::chrono::high_resolution_clock::now();
				//int pID1Pos = vvPathCandidate[pID].size()-sE.size();
                //for(int i = eID2Pos; i <pID1Pos; i++)
				for(int i = eID2Pos; i < (int) vvPathCandidate[pID].size(); i++)
                {
                    if(sE.find(vvPathCandidate[pID][i]) == sE.end())
                        sE.insert(vvPathCandidate[pID][i]);
                    else
                    {
                        bFixedLoop = true;
                        break;
                    }
                }
                /*t4 = std::chrono::high_resolution_clock::now();
                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3);
				//cout << time_span.count << endl;
                sETime += time_span.count();*/

                if(bFixedLoop) {
                    continue;
                }
				//vvsE.push_back(sE);
                int distTmp = vDistance[pID] - vSPTDistance[eNodeID2] + vEdge[mineID].length;
                bLoop = false;

                //Open
                vPathDevPrime.push_back(eNodeID1);
                vPathDeviation.push_back(eNodeID2);

                //multimap<int, int> mArcTmp;
                vector<pair<int, int> > mmArcTmp;
                vPath.clear();
                vPathEdge.clear();
                int p = eNodeID1;
                int e = vSPTParentEdge[p];
                Arc.clear();
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

                        if(reID != e && reID != mineID)
                            Arc.update(reID,vEdgeReducedLength[reID],vPath.size()+ReID2Pos);
                            //mArcTmp.insert(make_pair(vEdgeReducedLength[reID], reID));
                    }
                    vPathEdge.push_back(e);
                    distTmp += vEdge[e].length;

                    p = vSPTParent[p];
                    if(vSPTParent[p] == -1)
                    {
                        vPath.push_back(p);
                        break;
                    }
                    e = vSPTParentEdge[p];
                }
                //ArcList.push_back(Arc);
				//pair<int, int> mmETmp;
                vector<pair<int, int> > mmE;
				//mmE.assign(Arc.size(),mmETmp);
				//mmArcTmp.assign(Arc.size(),mmETmp);
				//int AcrSize = Arc.size();
				//int n = 0;
                while(!Arc.empty())
                {
                    Arc.extract_min(mmArcEdgeID, mmArcReducedLength, mmArcPosition);
					//mmArcTmp[Arc.size()] = make_pair(mmArcEdgeID,mmArcReducedLength);
					//mmE[Arc.size()] = make_pair(mmArcEdgeID,mmArcPosition);
					mmArcTmp.push_back(make_pair(mmArcEdgeID,mmArcReducedLength));
                    mmE.push_back(make_pair(mmArcEdgeID,mmArcPosition));
                }
                //cout << vvmArc.size() << "    " << vvmPathNodePos.size()<< endl;
				//Open
                dEdge.push_back(vEdge[mineID].length);

                int dist = vDistance[pID] - vSPTDistance[eNodeID2] + vSPTDistance[eNodeID1] + vEdge[mineID].length;
                vDistance.push_back(dist);
                /*t2 = std::chrono::high_resolution_clock::now();
                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
                edgeTime += time_span.count();*/


                //t1 = std::chrono::high_resolution_clock::now();

                //Open
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
                    //cout << "_p:" << _p << " _q:" << _q << endl;
                    int LCA = LCAQuery(_p, _q, vSPT,  vSPTHeight,vSPTParent);
                    int dLCA = vSPTDistance[LCA];
                    //cout << "dLCA: " << dLCA << endl;
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
                /*t2 = std::chrono::high_resolution_clock::now();
                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
                time += time_span.count();

                t1 = std::chrono::high_resolution_clock::now();*/
                vFather.push_back(pID);

                //Open
                bPath.push_back(false);
                qPath.update(vDistance.size()-1, dist);

                //rbegin
                reverse(vPath.begin(), vPath.end());
                reverse(vPathEdge.begin(), vPathEdge.end());
                //Dense hash map?
               /* unordered_map<int, int> mE;
                //Pos stop at eNodeID1 as it is boundary of fixed
                for(int i = 0; i < (int)vPath.size(); i++)
                    mE[vPath[i]] = i;
                vmPathNodePos.push_back(mE);*/
                vPath.push_back(eNodeID2);
                vPathEdge.push_back(mineID);
                vPath.insert(vPath.end(), vvPathCandidate[pID].begin()+eID2Pos+1, vvPathCandidate[pID].end());
                vPathEdge.insert(vPathEdge.end(), vvPathCandidateEdge[pID].begin()+eID2Pos, vvPathCandidateEdge[pID].end());
                //for(int j = vmPathNodePos[pID][eNodeID2]; j+1 < (int)vvPathCandidate[pID].size(); j++)
                /*for(int j = eID2Pos; j < (int)vvPathCandidate[pID].size()-1; j++)
                {
                    int nodeID = vvPathCandidate[pID][j+1];
                    vPath.push_back(nodeID);
                    int edgeID = vvPathCandidateEdge[pID][j];
                    vPathEdge.push_back(edgeID);
                }*/
                //vmArc.push_back(mArcTmp);
                vvmArc.push_back(mmArcTmp);
                vvmPathNodePos.push_back(mmE);
                vvPathCandidate.push_back(vPath);
                vvPathCandidateEdge.push_back(vPathEdge);

                /*t2 = std::chrono::high_resolution_clock::now();
                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
                pathTime += time_span.count();*/
            }
        }

        t4 = std::chrono::high_resolution_clock::now();
        time_span1 = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3);
        if(time_span1.count() > 5){
            //cout << "larger than 5 seconds" << endl;
            break;
        }
        /*if(countNumber >= 33836288){
            bCountNumber = false;
            break;
        }*/
    }

    if(!bCountNumber){
        cout << "Skip!" << endl;
        popPath = -1;
    }
    else{
		AveSim = AveSim / (kResults.size()*(kResults.size()-1)/2);
		cout << "AveSim: " << AveSim << endl;

		cout << "max: " << maxSim << endl;
        percentage = countAncestor/(countNonAncestor + countAncestor);
        cout << "Sim Time:" << SimTime << endl;
        //cout << "loop Time: " << loopTime << endl;
        //cout << "edgeTime: " << edgeTime << endl;
        //cout << "LCA Time: " << time << endl;
		//cout << "sETime: " << sETime << endl;
        //cout << "pathTime: " << pathTime << endl;
        cout << "eKSPCompare countNumber: "<< countNumber << " Pop Path: " << popPath << " Percentage: " << percentage << endl;
    }
    //t2 = std::chrono::high_resolution_clock::now();
    //time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    //cout << "while Time:" << time_span.count() << endl;


    /*for(int i = 0; i < vvPathCandidateEdge.size(); i++){
        cout << "i: " << i << endl;
        for(int j = 0; j < vvPathCandidateEdge[i].size(); j++){
            cout << vvPathCandidate[i][j] << ",";
        }
        cout << endl;
    }

    for(int i = 0; i < vvPathCandidateEdge.size(); i++){
        cout << "i: " << i << endl;
        for(int j = 0; j < vvPathCandidateEdge[i].size(); j++){
            cout << vEdge[vvPathCandidate[i][j]].length << ",";
        }
        cout << endl;
    }
    cout << endl;*/
    return -1;
}
