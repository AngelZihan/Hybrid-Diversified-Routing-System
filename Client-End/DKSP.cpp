#include "graph.h"
#include "ClassificationSystem.h"

int Classification::eKSPCompare(Graph* _g, int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, double t, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<int>& vSPTChildren, vector<vector<int> >& vSPT, vector<vector<int> >& vvPathCandidate, vector<vector<int> >& vvPathCandidateEdge, float& maxSim)
{
    maxSim = 0;
    cout << "eKSPCompare" << endl;
    //Classification obj2;
    g = _g;
    int countNumber = 0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span;
    float countAncestor = 0;
    float countNonAncestor = 0;
    bool bCountNumber = true;

    //Add

    /*vSPTDistance.resize(g->n, INF);
    vSPTParent.resize(g->n, -1);
    vSPTHeight.resize(g->n, -1);
    vSPTParentEdge.resize(g->n, -1);
    vSPTChildren.resize(g->n, -1);
    vector<int> vTmp;
    vSPT.resize(g->n, vTmp);
    SPT(ID1, ID2, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPT);*/


    //LCA
    vector<vector<float> > vPathLCA;
    vector<vector<int> > RMQ = makeRMQ(g, ID1, vSPT, vSPTHeight,vSPTParent);

    /*for(int i = 0; i < rEulerSeq.size(); i++){
        cout << rEulerSeq[i] << endl;
    }*/
    vector<int> vEdgeReducedLength(g->vEdge.size(), INF);
    for(int i = 0; i < (int)g->vEdge.size(); i++)
        vEdgeReducedLength[i] = g->vEdge[i].length + vSPTDistance[g->vEdge[i].ID1] - vSPTDistance[g->vEdge[i].ID2];
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


    //Add
    /*vvPathCandidate.clear();	 //nodes
    vvPathCandidateEdge.clear();*///edges

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

    benchmark::pHeap<3, int, int, int> Arc(2*g->n);
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
        for(int i = 0; i < (int)g->adjListEdgeR[oldP].size(); i++)
        {
            int eID = g->adjListEdgeR[oldP][i].second;
            //cout << "eID: " << eID << endl;
            if(eID != e){
                Arc.update(eID,vEdgeReducedLength[eID],vPath.size()-1);
            }
        }
        vPath.push_back(p);
        oldP = p;
        vPathEdge.push_back(e);
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
        mmArc.push_back(make_pair(mmArcEdgeID,mmArcReducedLength));
        mmPos.push_back(make_pair(mmArcEdgeID,mmArcPosition));
    }
    vvmArc.push_back(mmArc);
    vvmPathNodePos.push_back(mmPos);

    /*reverse(vPath.begin(), vPath.end());
    reverse(vPathEdge.begin(), vPathEdge.end());*/

    benchmark::heap<2, int, int> qPath(g->n);

    //Add
    /*vvPathCandidate.push_back(vPath);
    vvPathCandidateEdge.push_back(vPathEdge);*/


    vDistance.push_back(vSPTDistance[ID2]);

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

    while((int)kResults.size() < k && !qPath.empty())
    {
        float addLength = 0;
        pcount++;
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

        //t1 = std::chrono::high_resolution_clock::now();
        if(!bTopLoop)
        {
            float simCount = 0;
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

                            break;
                        }
                    }


                    if(vFind == false)
                    {
                        countNonAncestor += 1;
                        for(auto ie = vvPathCandidateEdge[topPathID].begin(); ie != vvPathCandidateEdge[topPathID].end(); ie++){
                            if(pResult[i].find(*ie) != pResult[i].end())
                                addLength += g->vEdge[*ie].length;
                        }
                        //Sim 1
                        sim = addLength / (kResults[i] + topPathDistance - addLength);
                        //cout << "addLength: "<< addLength << " kResults[i] + topPathDistance - addLength: " << (kResults[i] + topPathDistance - addLength) << endl;
                        addLength = 0;
                    }
                    simCount += sim;
                    //if(sim > t)
                    //	break;
                    if (sim > simMax)
                        simMax = sim;
                    if(sim < simMin)
                        simMin = sim;

                }

                if(simMax <= t){
                    if(simMax > maxSim)
                        maxSim = simMax;
                    //AveSim += simCount;
                    //cout << "sim: " << sim << " topPath:" << topPathID << endl;
                    kResults.push_back(topPathDistance);
                    vvResult.push_back(vvPathCandidateEdge[topPathID]);
                    vkPath.push_back(vvPathCandidate[topPathID]);
                    vResultID.push_back(topPathID);
                    //cout << "topPathDistance: " << topPathDistance << endl;

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

        // t1 = std::chrono::high_resolution_clock::now();
        vector<int> vTwo;
        vTwo.push_back(topPathID);
        if(vFather[topPathID] != -1 &&  !vvmArc[vFather[topPathID]].empty())
            vTwo.push_back(vFather[topPathID]);
        for(auto& pID : vTwo)
        {
            //cout << "path ID: " << pID << endl;
            bool bLoop = true;
            while(bLoop)
            {
                if(vvmArc[pID].empty())
                {
                    vvPathCandidate[pID].clear();
                    vvPathCandidateEdge[pID].clear();
                    break;
                }
                int mineID;
                int eReducedLen;
                auto it = vvmArc[pID].begin();
                eReducedLen = (*it).second;
                mineID = (*it).first;

                countNumber += 1;
                vvmArc[pID].erase(it);

                //eNodeID1 is also the first point from the deviation edge
                int eNodeID1 = g->vEdge[mineID].ID1;
                int eNodeID2 = g->vEdge[mineID].ID2;

                bool bFixedLoop = false;

                //Change to vector
                auto itp = vvmPathNodePos[pID].begin();
                int ReID2Pos = (*itp).second;
                //int ReID2Pos = mmArcPosition;
                int eID2Pos = vvPathCandidate[pID].size()-ReID2Pos-1;
                vvmPathNodePos[pID].erase(itp);
                unordered_set<int> sE;
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

                if(bFixedLoop) {
                    continue;
                }
                //vvsE.push_back(sE);
                int distTmp = vDistance[pID] - vSPTDistance[eNodeID2] + g->vEdge[mineID].length;
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
                    for(int i = 0; i < (int)g->adjListEdgeR[p].size(); i++)
                    {
                        int reID = g->adjListEdgeR[p][i].second;
                        int eID1 = g->vEdge[reID].ID1;
                        int dTmp = distTmp + g->vEdge[reID].length;

                        if(sE.find(eID1) != sE.end())
                            continue;

                        if(reID != e && reID != mineID)
                            Arc.update(reID,vEdgeReducedLength[reID],vPath.size()+ReID2Pos);
                        //mArcTmp.insert(make_pair(vEdgeReducedLength[reID], reID));
                    }
                    vPathEdge.push_back(e);
                    distTmp += g->vEdge[e].length;

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
                    mmArcTmp.push_back(make_pair(mmArcEdgeID,mmArcReducedLength));
                    mmE.push_back(make_pair(mmArcEdgeID,mmArcPosition));
                }
                dEdge.push_back(g->vEdge[mineID].length);

                int dist = vDistance[pID] - vSPTDistance[eNodeID2] + vSPTDistance[eNodeID1] + g->vEdge[mineID].length;
                vDistance.push_back(dist);
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
                vFather.push_back(pID);

                //Open
                bPath.push_back(false);
                qPath.update(vDistance.size()-1, dist);

                //rbegin
                reverse(vPath.begin(), vPath.end());
                reverse(vPathEdge.begin(), vPathEdge.end());
                vPath.push_back(eNodeID2);
                vPathEdge.push_back(mineID);
                vPath.insert(vPath.end(), vvPathCandidate[pID].begin()+eID2Pos+1, vvPathCandidate[pID].end());
                vPathEdge.insert(vPathEdge.end(), vvPathCandidateEdge[pID].begin()+eID2Pos, vvPathCandidateEdge[pID].end());
                vvmArc.push_back(mmArcTmp);
                vvmPathNodePos.push_back(mmE);
                vvPathCandidate.push_back(vPath);
                vvPathCandidateEdge.push_back(vPathEdge);
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count() > 5){
        //if(countNumber >= 33836288){
            bCountNumber = false;
            break;
        }
    }

    if(!bCountNumber){
        cout << "Skip!" << endl;
    }
    else{

        cout << "max: " << maxSim << endl;
        cout << "eKSPCompare countNumber: "<< countNumber << endl;
    }

    /*for(int i = 0; i < vvPathCandidateEdge.size(); i++){
        cout << "i: " << i << endl;
        for(int j = 0; j < vvPathCandidateEdge[i].size(); j++){
            cout << vvPathCandidate[i][j] << ",";
        }
        cout << endl;
    }
    cout << endl;

    for(int i = 0; i < vvPathCandidateEdge.size(); i++){
        cout << "i: " << i << endl;
        for(int j = 0; j < vvPathCandidateEdge[i].size(); j++){
            cout << g->vEdge[vvPathCandidate[i][j]].length << ",";
        }
        cout << endl;
    }
    cout << endl;*/
    return -1;
}



vector<vector<int>> Classification::makeRMQ(Graph* _g, int p, vector<vector<int> >& vSPT, vector<int>& vSPTHeight, vector<int>& vSPTParent){
    g = _g;
    EulerSeq.clear();
    toRMQ.assign(g->n,0);
    RMQIndex.clear();
    makeRMQDFS(g, p, vSPT,vSPTParent);
    RMQIndex.push_back(EulerSeq);

    int m = EulerSeq.size();
    for (int i = 2, k = 1; i < m; i = i * 2, k++){
        vector<int> tmp;
        //tmp.clear();
        tmp.assign(m,0);
        for (int j = 0; j < m - i; j++){
            int x = RMQIndex[k - 1][j], y = RMQIndex[k - 1][j + i / 2];
            if (vSPTHeight[x] < vSPTHeight[y])
                tmp[j] = x;
            else tmp[j] = y;
        }
        RMQIndex.push_back(tmp);
    }
    //cout << "Table Time" << time_span.count() << endl;
    return RMQIndex;
}

vector<int> Classification::makeRMQDFS(Graph* _g, int p, vector<vector<int> >& vSPT, vector<int>& vSPTParent){
    g = _g;
    stack<int> sDFS;
    sDFS.push(p);
    vector<bool> vbVisited(g->n, false);
    while(!sDFS.empty())
    {
        int u = sDFS.top();
        if(vbVisited[u] == false)
        {
            int u = sDFS.top();
            EulerSeq.push_back(u);
            for(auto v = vSPT[u].end()-1; v != vSPT[u].begin()-1; v--)
                sDFS.push(*v);
            vbVisited[u] = true;
        }
        else
        {
            if(vSPTParent[u] != -1)
                EulerSeq.push_back(vSPTParent[u]);
            sDFS.pop();
        }
    }
    rEulerSeq.assign(g->n,-1);
    for(int i = 0; i < EulerSeq.size(); i++)
        rEulerSeq[EulerSeq[i]] = i;
    return EulerSeq;
}


int Classification::LCAQuery(int _p, int _q, vector<vector<int> >& vSPT, vector<int>& vSPTHeight, vector<int>& vSPTParent)
{
    //int p = toRMQ[_p], q = toRMQ[_q];
    int p = _p, q = _q;
    //cout << p << "..." << q << endl;
    //cout << EulerSeq[p] << "," << vSPTHeight[114572] << endl;
    if (p > q){
        int x = p;
        p = q;
        q = x;
    }

    int len = q - p + 1;
    int i = 1, k = 0;
    while (i * 2 < len){
        i *= 2;
        k++;
    }
    q = q - i + 1;
    if (vSPTHeight[RMQIndex[k][p]] < vSPTHeight[RMQIndex[k][q]]){
        /*cout << "1: " << vSPTHeight[RMQIndex[k][p]] << " 2: " << vSPTHeight[RMQIndex[k][q]] << endl;
        cout << "k: " << k << " p: " << p << " q: " << q << endl;*/
        return RMQIndex[k][p];
    }
    else return RMQIndex[k][q];
}
