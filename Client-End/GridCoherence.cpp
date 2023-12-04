#include "GridCoherence.h"
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

typedef pair<int, int> pd2;
struct myComp2 {
    constexpr bool operator()(
            pair<int, int> const& a,
            pair<int, int> const& b)
    const noexcept
    {
        return a.second < b.second;
    }
};

void GridCoherence::initCoherence()
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

void GridCoherence::createCoherence(int mm, int nn)
{
	//First create the lowest level
	//Collect nodes in each grid

    //Change to global
	//vector<vector<vector<int> > > vvGridNode;
    //vector<vector<int> > vvGridCentreNode;
    //vector<vector<vector<int> > > vvGridDistance;

	vector<int> vTmp;
	vector<vector<int> > vvTmp;
	vvTmp.assign(vBase[level], vTmp);
	g->vvGridNode.assign(vBase[level], vvTmp);

    vector<int> vvTmp4;
    vvTmp4.assign(vBase[level],INF);
    g->vvGridCentreNode.assign(vBase[level],vvTmp4);

    //vector<vector<int> > vvGridLeftTop;
    vector<int> vvTmp5;
    vvTmp5.assign(vBase[level],INF);
    g->vvGridLeftTop.assign(vBase[level],vvTmp5);

    vector<vector<int> > vvGridLeftBottom;
    vector<int> vvTmp6;
    vvTmp6.assign(vBase[level],INF);
    vvGridLeftBottom.assign(vBase[level],vvTmp6);

    vector<vector<int> > vvGridRightTop;
    vector<int> vvTmp7;
    vvTmp7.assign(vBase[level],INF);
    vvGridRightTop.assign(vBase[level],vvTmp7);

    vector<vector<int> > vvGridRightBottom;
    vector<int> vvTmp8;
    vvTmp8.assign(vBase[level],INF);
    vvGridRightBottom.assign(vBase[level],vvTmp8);

    g->vvNodeGrid.assign(g->n,-1);
    //cout << "vvNodeGrid size: " << g->vvNodeGrid.size() << endl;


    //cout << " Center Node: " << vvGridCentreNode[0][0] << endl;
    vector<double> vTmp2;
    vector<vector<double> > vvTmp2;
    vvTmp2.assign(vBase[level]*vBase[level],vTmp2);
    vvGridDistance.assign(vBase[level]*vBase[level], vvTmp2);
    vector<pair<double,int> > vTmp3;
    vector<vector<pair<double,int> > > vvTmp3;
    vvTmp3.assign(vBase[level]*vBase[level],vTmp3);
    KSPNumber.assign(vBase[level]*vBase[level], vvTmp3);
	double xBase = vpXYBase[level].first;
	double yBase = vpXYBase[level].second;
    //cout << vBase[level] << endl;
	int x, y;
    //cout  << "vBase[level]" << vBase[level] << endl;
    //Put all nodes to different grids
	for(int i = 0; i < g->n; i++)
	{
		x = int((g->Locat[i].second - g->minX) / xBase);
		y = int((g->Locat[i].first - g->minY) / yBase);
        g->vvGridNode[x][y].push_back(i);
        g->vvNodeGrid[i] = vBase[level]*x + y;
        //cout << "x: " << x << " y: "<< y << endl;
        if(g->vvGridCentreNode[x][y] == INF){
            g->vvGridCentreNode[x][y] = i;
            g->vvGridLeftTop[x][y] = i;
            vvGridLeftBottom[x][y] = i;
            vvGridRightTop[x][y] = i;
            vvGridRightBottom[x][y] = i;
        }
        if(abs(g ->Locat[i].second - (x*xBase+minX+xBase/2)) < abs(g ->Locat[g->vvGridCentreNode[x][y]].second - (x*xBase+minX+xBase/2))  && abs(g ->Locat[i].first - (y*yBase+minY+yBase/2)) < abs(g ->Locat[g->vvGridCentreNode[x][y]].first - (y*yBase+minY+yBase/2)))
            g->vvGridCentreNode[x][y] = i;
        if(abs(g ->Locat[i].second - (x*xBase+minX)) < abs(g ->Locat[vvGridLeftBottom[x][y]].second - (x*xBase+minX)) && abs(g->Locat[i].first - (y*yBase+minY+yBase/2)) < abs(g->Locat[vvGridLeftBottom[x][y]].first - (y*yBase+minY)))
            vvGridLeftBottom[x][y] = i;
        if(abs(g ->Locat[i].second - (x*xBase+minX)) < abs(g ->Locat[vvGridLeftBottom[x][y]].second - (x*xBase+minX)) && abs(g->Locat[i].first - (y*yBase+minY+yBase/2)) > abs(g->Locat[vvGridLeftBottom[x][y]].first - (y*yBase+minY)))
            g->vvGridLeftTop[x][y] = i;
        if(abs(g ->Locat[i].second - (x*xBase+minX)) > abs(g ->Locat[vvGridLeftBottom[x][y]].second - (x*xBase+minX)) && abs(g->Locat[i].first - (y*yBase+minY+yBase/2)) > abs(g->Locat[vvGridLeftBottom[x][y]].first - (y*yBase+minY)))
            vvGridRightTop[x][y] = i;
        if(abs(g ->Locat[i].second - (x*xBase+minX)) > abs(g ->Locat[vvGridLeftBottom[x][y]].second - (x*xBase+minX)) && abs(g->Locat[i].first - (y*yBase+minY+yBase/2)) < abs(g->Locat[vvGridLeftBottom[x][y]].first - (y*yBase+minY)))
            vvGridRightBottom[x][y] = i;
        //vvGridCentreNode[x][y] = i;
        //cout << "x: " << x << " y: " << y << endl;
	}


    /*for(int i = 0; i < g->vvGridLeftTop.size(); i++){
        for(int j = 0; j < g->vvGridLeftTop[i].size(); j++){
            int level1 = i * vBase[level] + j;
            //cout << "x: " << i << " j: " << j << " LeftTop Node: " << g->vvGridLeftTop[i][j] << endl;
            cout << "x: " << i << " j: " << j << " level1: " << level1 << " Center Node: " << g->vvGridCentreNode[i][j] << endl;
        }
    }*/
    //cout center node for each grid
    /*for(int i = 0; i < vvGridCentreNode.size(); i++){
        for(int j = 0; j < vvGridCentreNode[i].size(); j++){
            cout << "x: " << i << " j: " << j << " Center Node: " << vvGridCentreNode[i][j] << endl;
        }
    }*/

    //Generate 100 query for each grid pair
    /*vector<int> ID1List;
    vector<int> ID2List;
    for(int i = 0; i < g -> vvGridNode.size(); i++){
        for(int j = 0; j < g -> vvGridNode[i].size(); j++){
            for (int l = 0; l < g -> vvGridNode.size(); ++l) {
                for (int n = 0; n < g -> vvGridNode[l].size(); ++n) {
                    if(g->vvGridNode[i][j].size() == 0 || g -> vvGridNode[l][n].size() == 0)
                        continue;
                    int gridCenterNode1 = g->vvGridCentreNode[i][j];
                    int gridCenterNode2 = g->vvGridCentreNode[l][n];
                    int dij = Dijkstra(gridCenterNode1,gridCenterNode2);
                    int querySize;
                    querySize = 3;
                    //if( dij >= 190000 && dij <= 220000)
                    //    querySize = 100;
                    //else
                    //    querySize = 3;
                    if (g->vvGridNode[i][j].size() < querySize)
                        querySize = g->vvGridNode[i][j].size();
                    if(g->vvGridNode[l][n].size() < querySize)
                        querySize = g->vvGridNode[l][n].size();
                    for(int k = 0; k < querySize; k++){
                        int firstGridNode;
                        int secondGridNode;
                        bool bExist = false;
                        while(!bExist){
                            int random = rand() % g -> vvGridNode[i][j].size();
                            firstGridNode = g->vvGridNode[i][j][random];
                            int random2 = rand() % g -> vvGridNode[l][n].size();
                            secondGridNode = g -> vvGridNode[l][n][random2];
                            //cout << "ID1List: " << ID1List.size() << endl;
                            if(ID1List.size() == 0)
                                bExist = true;
                            for(int h = 0; h < ID1List.size(); h++){
                                //cout << "ID1List: " << ID1List.size() << " h: " << h << endl;
                                if((ID1List[h] == firstGridNode && ID2List[h] == secondGridNode) || (ID1List[h] == secondGridNode && ID2List[h] == firstGridNode))
                                {
                                    //cout << "exist!" << endl;
                                    break;
                                }
                                if(h == ID1List.size()-1)
                                {
                                    bExist = true;
                                }
                            }
                        }
                        ID1List.push_back(firstGridNode);
                        ID2List.push_back(secondGridNode);
                        cout << firstGridNode << " " << secondGridNode << endl;
                        //cout << "firstNode: " << firstGridNode << " secondNode: " << secondGridNode << " ID1 size: " << ID1List.size() << " ID2 size: " << ID2List.size() << endl;
                    }
                }
            }
        }
    }
    cout << "ID1List size: " << ID1List.size() << " ID2List size: " << ID2List.size() << endl;*/

	double a2 = PI/2;
	double a3 = PI;
	double a4 = PI*3/2;
	double a5 = 2*PI;

	//Roads are inferred by node's neighbors
	for(int i = 0; i < vBase[level]; i++)
	{
		for(int j = 0; j < vBase[level]; j++)
		{
			if(g->vvGridNode[i][j].size() == 0)
			{
				vvvCoherence[level][i][j] = make_pair(0,0);
				continue;
			}

			//road length sum
			int roadNum = 0;
			double angleSum = 0;
			//For each road, compute direction
			for(int k = 0; k < (int)g->vvGridNode[i][j].size(); k++)
			{
				int nodeID = g->vvGridNode[i][j][k];
				for(int r = 0; r < (int)g->Edges[nodeID].size(); r++)
				{
//					roadNum += g->Edges[nodeID][r].second;
					roadNum++;
					int neighbor = g->Edges[nodeID][r].first;
					double angle = g->Angle(nodeID, neighbor);
					double ad = angle;
					double ad2 = abs(angle-a2);
					double ad3 = abs(angle-a3);
					double ad4 = abs(angle-a4);
					double ad5 = abs(angle-a5);

					if(ad2 < ad)
						ad = ad2;
					if(ad3 < ad)
						ad = ad3;
					if(ad4 < ad)
						ad = ad4;
					if(ad5 < ad)
						ad = ad5;

					if(ad < 0.001)
						ad = 0;
					//angleSum += ad * g->Edges[nodeID][r].second;
					angleSum += ad;

//					cout << ad << "\t" << angle << endl;
				}
			}
			vvvCoherence[level][i][j] = make_pair(angleSum / roadNum, roadNum) ;
		}
	}
	
	//Summarizing upward
	for(int i = level - 1; i >=0; i--)
	{
		for(int j =0; j < vBase[i]; j++)
		{
			for(int k = 0; k < vBase[i]; k++)
			{
				pair<double, int> p1,p2,p3,p4;
				p1 = vvvCoherence[i+1][2*j][2*k];
				p2 = vvvCoherence[i+1][2*j+1][2*k];
				p3 = vvvCoherence[i+1][2*j][2*k+1];
				p4 = vvvCoherence[i+1][2*j+1][2*k+1];
				
				int roadNum = p1.second + p2.second + p3.second + p4.second;
				if(roadNum == 0)
					vvvCoherence[i][j][k] = make_pair(0, 0);
				else
				{
					double angleSum = p1.first*p1.second + p2.first*p2.second + p3.first*p3.second + p4.first*p4.second;
					vvvCoherence[i][j][k] = make_pair(angleSum / roadNum, roadNum);
				}
//				cout << roadNum << " ";
			}
//			cout << endl;
		}
//		cout << endl;
	}


	//Output Coherence Girds
/*	for(int i = 0; i <= level; i++)
	{
		cout << "Level:" << i << endl;
		for(int j =0; j < vBase[i]; j++)
		{
			for(int k = 0; k < vBase[i]; k++)
			{
				cout << j <<"\t" << k << "\t" << vvvCoherence[i][j][k].first <<"\t" << vvvCoherence[i][j][k].second << endl;
			}
		}
		cout <<endl;
	}
*/


//Rewrite CoverNumber for pair in the same grid
    /*vector<int> sameLevel;
    for(int i = 0; i < g->vvGridNode.size(); i++){
        for(int j = 0; j < g-> vvGridNode[i].size(); j++){
            sameLevel.push_back(g-> vvGridNode[i][j].size());
            //cout << "i: " << i << " j: " << j << " size: " << g-> vvGridNode[i][j].size() << endl;
        }
    }
    //cout << "sameLevel: " << sameLevel.size() << endl;
    string NodeNumberFile = "./COL_CoverNumber";
    ifstream in1(NodeNumberFile);
    string line1;
    if(in1){
        getline(in1, line1);
        while (!in1.eof()) {
            vector<string> vs1;
            boost::split(vs1,line1,boost::is_any_of(" "),boost::token_compress_on);
            int i = stoi(vs1[1]);
            int j = stoi(vs1[3]);
            if(i == j){
                cout << "i: " << i << " j: " << j << " number: " << sameLevel[i] << endl;
            }
            else
                cout << line1 << endl;
            getline(in1, line1);
        }
    }*/


    //Rewrite KSPNumber for pair in the same pair
    //open for edgeNumber
    /*vector<vector<double > > edgeNodeNumber;
    vector<double> vvvTmp;
    vvvTmp.assign(16*16,-1);
    edgeNodeNumber.assign(16*16, vvvTmp);
    //open for edgeNumber

    //cout << "vvGridLeftTop size: " << g->vvGridLeftTop.size() << endl;
    for(int i = 0; i < g->vvGridLeftTop.size(); i++){
        //cout << "vvGridLeftTop[i] size: " << g->vvGridLeftTop[i].size() << endl;
        for(int j = 0; j < g->vvGridLeftTop[i].size(); j++) {
            //i = 2;
            //j = 4;
            int xLevel = i*vBase[level] + j;
            int yLevel = xLevel;
            //cout << g->vvGridLeftTop[i][j] << endl;
            //cout << "Node Size: " << g->vvGridNode[i][j].size() << "  g->vvGridNode[i][j][0]: " <<  g->vvGridNode[0][0][0]<< endl;
            if (g->vvGridLeftTop[i][j] != INF && g->vvGridLeftTop[i][j] != vvGridRightBottom[i][j]) {
                int gridLeftTop = g->vvGridLeftTop[i][j];
                int gridRightBottom= vvGridRightBottom[i][j];
                vector<int> kResults;
                vector<vector<int> >vkPath;
                vector<vector<int> > vvResult;
                //cout << "i: " << i << " j: " << j << " gridLeftTop: " <<  gridLeftTop << " gridRightBottom: " << gridRightBottom << endl;
                GridCoherence::eKSP(gridLeftTop, gridRightBottom, 100000, kResults, vkPath, xLevel, yLevel, KSPNumber, vvResult);

                //open for edgeNumber
                double edgeNumberResult;
                int SP = kResults[0];
                GridCoherence::edgeNumber(vvResult, edgeNumberResult, SP);
                //cout << "xLevel: " << xLevel << " yLevel: " << yLevel << " gridCenterNode1: " << gridLeftTop << " gridCenterNode2: " << gridRightBottom << " edgeNumberResult: " << edgeNumberResult << endl;
                edgeNodeNumber[xLevel][yLevel] = edgeNumberResult;
                //open for edgeNumber

                //sameLevel.push_back(g-> vvGridNode[i][j].size());
                //cout << "i: " << i << " j: " << j << " size: " << g-> vvGridNode[i][j].size() << endl;
            }
            else
                KSPNumber[xLevel][yLevel].push_back(make_pair(i, 100000));
            //{
            //    for(double i = 0.01; i < 0.21; i+=0.01) {
            //        KSPNumber[xLevel][yLevel].push_back(make_pair(i, 100000));
            //    }
            //}
            //break;
        }
        //break;
       //if(bPairEnd == true)
       //     break;
    }

    vector<vector<pair<double, int> > > leftTopRightBottom;
    vector<pair<double,int> > vvTmp9;
    leftTopRightBottom.assign(vBase[level]*vBase[level], vvTmp9);
    for (int i = 0; i < KSPNumber.size(); i++) {
        for (int j = 0; j < KSPNumber[i].size(); ++j) {
            if(i == j){
                for(int m = 0; m < KSPNumber[i][j].size() ; m++)
                //for(int m = 0; m < 21 ; m++)
                {
                    leftTopRightBottom[i].push_back(make_pair(KSPNumber[i][j][m].first, KSPNumber[i][j][m].second));
                    //cout << "xLevel: " << i << " yLevel: " << j << " percentage: " << KSPNumber[i][j][m].first << " Number: " << KSPNumber[i][j][m].second << endl;
                }
            }
        }
    }

    KSPNumber.clear();
    KSPNumber.assign(vBase[level]*vBase[level], vvTmp3);
    for(int i = 0; i < vvGridRightTop.size(); i++){
        for(int j = 0; j < vvGridRightTop[i].size(); j++) {
            //i = 2;
            //j = 4;
            int xLevel = i*vBase[level] + j;
            int yLevel = xLevel;
            //cout << "Node Size: " << g->vvGridNode[i][j].size() << "  g->vvGridNode[i][j][0]: " <<  g->vvGridNode[0][0][0]<< endl;
            if (vvGridLeftBottom[i][j] != INF && vvGridLeftBottom[i][j] != vvGridRightTop[i][j]) {
                int gridLeftBottom = vvGridLeftBottom[i][j];
                int gridRightTop = vvGridRightTop[i][j];
                vector<int> kResults;
                vector<vector<int> >vkPath;
                vector<vector<int> > vvResult;
                //cout << "i: " << i << " j: " << j << " gridLeftBottom: " <<  gridLeftBottom << " gridRightTop: " << gridRightTop << endl;
                GridCoherence::eKSP(gridLeftBottom, gridRightTop, 100000, kResults, vkPath, xLevel, yLevel, KSPNumber, vvResult);

                //open for edgeNumber
                double edgeNumberResult;
                int SP = kResults[0];
                GridCoherence::edgeNumber(vvResult, edgeNumberResult, SP);
                //cout << "xLevel: " << xLevel << " yLevel: " << yLevel << " gridCenterNode1: " << gridLeftBottom << " gridCenterNode2: " << gridRightTop << " edgeNumberResult: " << edgeNumberResult << endl;
                edgeNodeNumber[xLevel][yLevel] = (edgeNodeNumber[xLevel][yLevel] + edgeNumberResult)/2;
                //open for edgeNumber

                //sameLevel.push_back(g-> vvGridNode[i][j].size());
                //cout << "i: " << i << " j: " << j << " size: " << g-> vvGridNode[i][j].size() << endl;
            }
            else
            {
                KSPNumber[xLevel][yLevel].push_back(make_pair(i, 100000));
            }
            //else
            //{
            //    for(double i = 0.01; i < 0.21; i+=0.01) {
            //        KSPNumber[xLevel][yLevel].push_back(make_pair(i, 100000));
            //    }
            //}
            //break;
        }
        //break;
    }

    //cout << "stop!: " << endl;
    //Close when rewrite edgeNumber
    vector<vector<pair<double, int> > > leftBottomRightTop;
    vector<pair<double,int> > vvTmp20;
    leftBottomRightTop.assign(vBase[level]*vBase[level], vvTmp20);
    for (int i = 0; i < KSPNumber.size(); i++) {
        for (int j = 0; j < KSPNumber[i].size(); ++j) {
            if(i == j){
                for(int m = 0; m < KSPNumber[i][j].size() ; m++)
                {
                    leftBottomRightTop[i].push_back(make_pair(KSPNumber[i][j][m].first, KSPNumber[i][j][m].second));
                    //cout << "xLevel: " << i << " yLevel: " << j << " percentage: " << KSPNumber[i][j][m].first << " Number: " << KSPNumber[i][j][m].second << endl;
                }
            }
        }
    }
    vector<vector<pair<double, int> > > sameLevelKSP;
    vector<pair<double,int> > vvTmp21;
    sameLevelKSP.assign(vBase[level]*vBase[level], vvTmp21);
    int minSize;
    for(int i = 0; i < leftBottomRightTop.size(); i++){
        //cout << "leftTop: " << leftTopRightBottom[i].size() << "leftBottom" << leftBottomRightTop[i].size() << endl;
        if(leftTopRightBottom[i].size() < leftBottomRightTop[i].size())
            minSize = leftTopRightBottom[i].size();
        else
            minSize = leftBottomRightTop[i].size();
        //cout << "minSize: " << minSize << endl;
        if(leftBottomRightTop[i].size () > 0 && leftTopRightBottom.size() > 0){
            for(int j = 0; j < minSize; j++){
            //for (int j = 0; j < leftBottomRightTop[i].size(); ++j) {
                int newNumber = (leftBottomRightTop[i][j].second + leftTopRightBottom[i][j].second)/2;
                sameLevelKSP[i].push_back(make_pair(leftBottomRightTop[i][j].first,newNumber));
            }
        }
    }

    string NodeNumberFile = "./COL_KSPNumber";
    ifstream in11(NodeNumberFile);
    string line11;
    if(in11){
        getline(in11, line11);
        while (!in11.eof()) {
            bool bGetline = true;
            vector<string> vs1;
            boost::split(vs1,line11,boost::is_any_of(" "),boost::token_compress_on);
            int i = stoi(vs1[1]);
            int j = stoi(vs1[3]);
            double percent = stod(vs1[5]);
            if(i == j && sameLevelKSP[i].size() > 0){
                for(int m = 0; m < sameLevelKSP[i].size(); m++){
                //for(int m = 0; m < 20; m++){
                    cout << "xLevel: " << i << " yLevel: " << j << " percentage: " << sameLevelKSP[i][m].first << " Number: " << sameLevelKSP[i][m].second << endl;
                }
                //for(int i = 0; i < 39; i++){
                //    getline(in11, line11);
                //}
                while (i == j){
                    getline(in11, line11);
                    bGetline = false;
                    if(line11[0] == 'x'){
                        vector<string> vs2;
                        boost::split(vs2,line11,boost::is_any_of(" "),boost::token_compress_on);
                        //cout << vs2[0] << endl;
                        i = stoi(vs2[1]);
                        j = stoi(vs2[3]);
                        if(i == j)
                            continue;
                    }
                    else
                        break;
                }

                //cout << "i: " << i << " j: " << j << " number: " << sameLevel[i] << endl;
                //cout << "xLevel: " << i << " yLevel: " << j << " percentage: " << sameLevelKSP[i][j].first << " Number: " << sameLevelKSP[i][j].second << endl;
            }
            //if(i == j)
            //    cout << "xLevel: " << i << " yLevel: " << j << " percentage: " << percent << " Number: " << "100000" << endl;
            else
                cout << line11 << endl;
            if(bGetline == true)
                getline(in11, line11);
        }
    }
    //Close when rewrite edgeNumber


    //Open when generate CEONumber
    string NodeNumberFile = "./MAN_CEONumber";
    ifstream in11(NodeNumberFile);
    string line11;
    if(in11){
        getline(in11, line11);
        while (!in11.eof()) {
            bool bGetline = true;
            vector<string> vs1;
            boost::split(vs1,line11,boost::is_any_of(" "),boost::token_compress_on);
            int i = stoi(vs1[1]);
            int j = stoi(vs1[3]);
            //int edgeNumber = stoi(vs1[5]);
            if(i == j){
                cout << "xLevel: " << i << " yLevel: " << j << " edgeNumber: " <<  edgeNodeNumber[i][j] <<endl;
            }
            else
                cout << line11 << endl;
            if(bGetline == true)
                getline(in11, line11);
        }
    }*/
    //Open when generate CEONumber

    //Read query file Generate Training data
    //Remember change xLevel an yLevel!!!!
    //ofstream file("NY_Query");


/*    vector<int> ID1List;
    vector<int> ID2List;
    //string queryFilename = "/Users/angel/CLionProjects/ZIGZAG/USA-NY-Q3.txt";
    //string queryFilename = "./USA-query/USA-NY-Q5.txt";
    //string queryFilename = "./GridNodePair";
    //string queryFilename = "./MAN_NodePair";
    //string queryFilename = "./NY_NodePair";
    string queryFilename = "./TestSet/Manhattan/Manhattan-Q2";
    //string queryFilename = "./TestSet/COL/USA-COL-Q4.txt";
    //string queryFilename = "./COL_NodePair";
    //string queryFilename = "./test";
    //string queryFilename = "/Users/angel/CLionProjects/ZIGZAG/GridNodePair";
    ifstream inGraph(queryFilename);
    if(!inGraph)
        cout << "Cannot open Map " << queryFilename << endl;
    int pID1, pID2;
    string line4;
    getline(inGraph,line4);
    //cout << line4 << endl;
    while(!inGraph.eof())
    {
        vector<string> vs4;
        boost::split(vs4,line4,boost::is_any_of(" "),boost::token_compress_on);
        //cout << vs4.size() << endl;
        pID1 = stoi(vs4[0]) - 1;
        pID2 = stoi(vs4[1]) - 1;
        //pID1 = stoi(vs4[0]);
        //pID2 = stoi(vs4[1]);
        //cout << "pID1: " << pID1 << " pID2: " << pID2 << endl;
        //cout << pID1 << endl;
        ID1List.push_back(pID1);
        ID2List.push_back(pID2);
        getline(inGraph, line4);
    }
    inGraph.close();
    cout << "ID1 size: " << ID1List.size() << endl;

    // Generate First path Covered Grid Level
    *//*for(int i = 0; i < ID1List.size(); i++){
        int ID1 = ID1List[i];
        int ID2 = ID2List[i];
        cout << "ID1: " << ID1 << " ID2: " << ID2;
        int xLevel;
        int yLevel;
        vector<int> passLevel;
        vector<int> kResults;
        vector<vector<int> >vkPath;
        vector<int> vTmpMatrix(7, 0);
        vector<vector<int>> staMatrix;
        staMatrix.assign(5,vTmpMatrix);
        xLevel = g->vvNodeGrid[ID1];
        yLevel = g->vvNodeGrid[ID2];

        cout << " ID1 Level: " << xLevel << " ID2 Level: " <<  yLevel;
        kResults.clear();
        vkPath.clear();
        int k = 1;
        int t = 0.99;
        std::chrono::high_resolution_clock::time_point t1;
        std::chrono::high_resolution_clock::time_point t2;
        std::chrono::duration<double> time_span;
        t1 = std::chrono::high_resolution_clock::now();
        eKSPCompare(ID1, ID2, k, kResults, vkPath, t, staMatrix);
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

        for(int o = 0; o < vkPath[0].size(); o++){
            bool bAdd = true;
            for (int oo = 0; oo < passLevel.size(); oo++){
                if(passLevel[oo] == g->vvNodeGrid[vkPath[0][o]]){
                    bAdd = false;
                    break;
                }
            }
            if(bAdd == true){
                passLevel.push_back(g->vvNodeGrid[vkPath[0][o]]);
            }
        }

        cout << " passLevel: ";
        for(int jj = 0; jj < passLevel.size(); jj++){
            cout << passLevel[jj] << " ";
        }
        cout << endl;
    }*//*
    // Generate First path Covered Grid Level


    //Open when generate result
    for (int i = 0; i < ID1List.size(); ++i) {
    //for (int i = mm; i < nn; i++){
        if(i == 0){
            cout << "ID1" << "," << "ID2" << "," << "k" << "," << "t" << "," << "xLevel" << "," << "yLevel" << "," << "coverNumber"  << "," << "CEO Number" << "," << "SP" <<  "," << "time" << "," << "result";
            for(double i = 0.0002; i <= 0.04; i+=0.0002){
                //cout << percentage[i] << "," << percentageNumber[i] << ",";
                cout << "," << i ;
            }
            cout << "," << "1_0.001" << "," << "1_0.002" << "," << "1_0.003" << "," << "1_0.004" << "," << "1_0.005" "," << "1_0.006" << "," << "1_INF"
                    << "," << "2_0.001" << "," << "2_0.002" << "," << "2_0.003" << "," << "2_0.004" << "," << "2_0.005" "," << "2_0.006" << "," << "2_INF"
                    << "," << "3_0.001" << "," << "3_0.002" << "," << "3_0.003" << "," << "3_0.004" << "," << "3_0.005" "," << "3_0.006" << "," << "3_INF"
                    << "," << "4_0.001" << "," << "4_0.002" << "," << "4_0.003" << "," << "4_0.004" << "," << "4_0.005" "," << "4_0.006" << "," << "4_INF"
                    << "," << "5_0.001" << "," << "5_0.002" << "," << "5_0.003" << "," << "5_0.004" << "," << "5_0.005" "," << "5_0.006" << "," << "5_INF";
            cout << endl;
        }
    //for (int i = 0; i < ID1List.size(); ++i) {
        //find the xLevel and yLevel for the pair
        int ID1 = ID1List[i];
        int ID2 = ID2List[i];
        //cout << "ID1: " << ID1 << " ID2: " << ID2 << endl;
        vector<int> kSet;
        kSet.push_back(3);
        kSet.push_back(5);
        kSet.push_back(10);
        kSet.push_back(10);

        vector<double> tSet;
        tSet.push_back(0.1);
        tSet.push_back(0.5);
        tSet.push_back(0.8);
        tSet.push_back(0.9);

        for(int ii = 0 ; ii < kSet.size(); ii++){
            for(int tt = 0 ; tt < kSet.size(); tt++){
            //for(double tt = 0.5; tt < 1; tt+=0.2){
                int k = kSet[ii];
                //double t = tt;
                double t = tSet[tt];
                //cout << "k: " << k << " t: " << t << endl;
                //int k = 10;
                //double t = 0.1;
                int xLevel;
                int yLevel;
                vector<int> kResults;
                vector<vector<int> >vkPath;
                vector<int> vTmpMatrix(7, 0);
                vector<vector<int>> staMatrix;
                staMatrix.assign(5,vTmpMatrix);
                //cout << "xLevel: " << xLevel << " yLevel: " << yLevel << endl;
                for(int h = 0; h < g->vvGridNode.size(); h++){
                    for(int l = 0; l < g->vvGridNode[h].size(); l++) {
                        if(g->vvGridNode[h][l].size() != 0){
                            for (int b = 0; b < g->vvGridNode[h][l].size(); b++) {
                                int gridNode = g->vvGridNode[h][l][b];
                                if(gridNode == ID1){
                                    //xLevel = j*vBase[level] + i;
                                    xLevel = h*vBase[level] + l;
                                    //cout << gridNode << " i: " << i << " j: " << j << endl;
                                }
                                if(gridNode == ID2){
                                    //yLevel = j*vBase[level] + i;
                                    yLevel = h*vBase[level] + l;
                                    //cout << gridNode << " i: " << i << " j: " << j << endl;
                                }
                            }
                        }
                    }
                }
                //cout << "xLevel: " << xLevel << " yLevel: " << yLevel << endl;
                //cout << "xlevel: " << xLevel << " ylevel: " << yLevel << endl;
                //find the KSP number of two grids
                //rewrite percentageNumber
                vector<double> percentage;
                vector<int> percentageNumber;
                bool bFind = false;
                //string KSPFile = "/Users/angel/CLionProjects/ZIGZAG/NY_KSPNumber";
                string KSPFile = "./MAN_KSPNumber2";
                ifstream in(KSPFile);
                string line;
                if(in) {
                    while (!in.eof()) {
                        //bool bGetline2 = true;
                        getline(in, line);
                        //cout << " Here!" << endl;
                        vector<string> vs;
                        boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
                        int x = stoi(vs[1]);
                        int y = stoi(vs[3]);
                        double l = stod(vs[5]);
                        int b = stoi(vs[7]);
                        //cout << "x: " << x << " y: " << y << endl;
                        bool bTest = false;
                        while(x == xLevel && y == yLevel) {
                            //cout << "x: " << x << " y: " << y << endl;
                            //cout << "percentage: " << l << " percentageNumber: " << b << endl;
                            //percentage.push_back(stod(vs[5]));
                            //percentageNumber.push_back(stoi(vs[7]));
                            percentage.push_back(l);
                            percentageNumber.push_back(b);
                            getline(in, line);
                            vector<string> vs;
                            boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
                            //cout << vs2[0] << endl;
                            x = stoi(vs[1]);
                            y = stoi(vs[3]);
                            l = stod(vs[5]);
                            b = stoi(vs[7]);
                            //cout << l << endl;
                            //cout <<"end:" << stod(vs[5]) << endl;
                            bFind = true;
                            //for (int i = 0; i < 20; i++) {
                            //cout << line << endl;
                            //cout << "x: " << x << " y: " << y << endl;
                            //    getline(in, line);
                            //    vector<string> vvs;
                            //    boost::split(vvs,line,boost::is_any_of(" "),boost::token_compress_on);
                            //    percentageNumber.push_back(stoi(vs2[7]));
                            //}
                            //break;
                        }
                        //cout << "count: " << count << endl;
                        if(bFind == true)
                            break;
                        //cout << stoi(vs[4]) << " " << stoi(vs[5]) << " " << stoi(vs[6]) << " " << stoi(vs[7]) << endl;
                    }
                }
                //cout << "KSP Number read finish" << endl;

                //find covered node number
                //string NodeNumberFile = "/Users/angel/CLionProjects/ZIGZAG/NodeNumber";
                int coverNumber;
                string NodeNumberFile = "./MAN_CoverNumber2";
                ifstream in1(NodeNumberFile);
                string line1;
                if(in1){
                    while (!in1.eof()) {
                        getline(in1, line1);
                        vector<string> vs1;
                        boost::split(vs1,line1,boost::is_any_of(" "),boost::token_compress_on);
                        int i = stoi(vs1[1]);
                        int j = stoi(vs1[3]);
                        //cout << x << " y: " << y << endl;
                        if(i == xLevel && j == yLevel) {
                            coverNumber = stoi(vs1[5]);
                            //cout << line1 << endl;
                            break;
                        }
                    }
                }
                //cout << "ID1: " << ID1 << " ID2: " << ID2 << " k: " << k << " t: " << t << endl;
                //cout << "Cover Number read finish" << endl;

                //find CEO node number
                int CEOEdgeNumber;
                string CEOEdgeNumberFile = "./MAN_CEONumber2";
                ifstream in2(CEOEdgeNumberFile);
                string line2;
                if(in2){
                    while (!in2.eof()) {
                        getline(in2, line2);
                        vector<string> vs2;
                        boost::split(vs2,line2,boost::is_any_of(" "),boost::token_compress_on);
                        int i = stoi(vs2[1]);
                        int j = stoi(vs2[3]);
                        //cout << x << " y: " << y << endl;
                        if(i == xLevel && j == yLevel) {
                            CEOEdgeNumber = stoi(vs2[5]);
                            //cout << line1 << endl;
                            break;
                        }
                    }
                }
                //cout << "CEO Number read finish" << endl;

                //run ksp and find its complex or simple
//                if(k == 10 && t > 0.7) {
//                    kResults.clear();
//                    vkPath.clear();
//                    std::chrono::high_resolution_clock::time_point t1;
//                    std::chrono::high_resolution_clock::time_point t2;
//                    std::chrono::duration<double> time_span;
//                    t1 = std::chrono::high_resolution_clock::now();
//                    eKSPCompare(ID1, ID2, k, kResults, vkPath, t);
//                    t2 = std::chrono::high_resolution_clock::now();
//                    time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//                    //cout << "eKSPCompare Time:" << time_span.count() << endl;
//                    string result;
//                    if(time_span.count() < 5){
//                        result = "simple!!!";
//                        //cout << "simple!!!" << endl;
//                    }
//                    else{
//                        result = "complex!!!";
//                        //cout << "complex!!!" << endl;
//                    }
//                    cout << ID1 << "," << ID2 << "," << k << "," << t << "," << xLevel << "," << yLevel << ",";
//                    cout << coverNumber << "," << CEOEdgeNumber << "," << kResults[0] << "," << time_span.count() << ","
//                         << result << endl;
//                    //cout << "0-0.001: " << count << endl;
//                }
                kResults.clear();
                vkPath.clear();
                //dataMatrix.clear();
                std::chrono::high_resolution_clock::time_point t1;
                std::chrono::high_resolution_clock::time_point t2;
                std::chrono::duration<double> time_span;
                t1 = std::chrono::high_resolution_clock::now();
                eKSPCompare(ID1, ID2, k, kResults, vkPath, t, staMatrix);
                t2 = std::chrono::high_resolution_clock::now();
                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

                //cout << "eKSPCompare Time:" << time_span.count() << endl;
                string result;
                if(time_span.count() < 5){
                    result = "simple!!!";
                    //cout << "simple!!!" << endl;
                }
                else{
                    result = "complex!!!";
                    //cout << "complex!!!" << endl;
                }
                //for(auto& d : kResults)
                //    cout << d << "\t";
                //cout << endl;
                //cout << endl;

                //cout << endl;
                cout << ID1 << "," << ID2 << "," << k << "," << t << "," << xLevel << "," << yLevel << ",";
                cout << coverNumber << "," << CEOEdgeNumber << "," << kResults[0] << "," << time_span.count() << "," << result;
                for(int i = 0; i < percentageNumber.size(); i++){
                    //cout << percentage[i] << "," << percentageNumber[i] << ",";
                    cout << "," <<percentageNumber[i] ;
                }
                if(percentageNumber.size() < 200){
                    for(int j = percentageNumber.size(); j< 200; j++){
                        cout << "," << 100000;
                    }
                }
                //staMatrix
                for(int i = 0; i < staMatrix.size(); i++)
                {
                    for(int j = 0; j < staMatrix[i].size(); j++){
                        cout << "," << staMatrix[i][j];
                        //cout << "loopTime: " << i << " range: " << double(j)/0.001 << "-" << double(j)/0.001+0.001 << " value: " << staMatrix[i][j];
                    }
                    //cout << endl;
                }
                cout << endl;
            }
        }
//        cout << ID1 << "," << ID2 << "," << k << "," << t << "," << xLevel << "," << yLevel << "," << percentageNumber[0] << "," << percentageNumber[1] << "," << percentageNumber[2] << "," << percentageNumber[3] << "," << percentageNumber[4] << "," << percentageNumber[5] << "," << percentageNumber[6] << "," << percentageNumber[7] << "," << percentageNumber[8]
//               << "," << percentageNumber[9] << "," << percentageNumber[10] << "," << percentageNumber[11] << "," << percentageNumber[12] << "," << percentageNumber[13] << "," << percentageNumber[14] << "," << percentageNumber[15] << "," << percentageNumber[16] << "," << percentageNumber[17] << "," << percentageNumber[18] << "," << percentageNumber[19] << "," << coverNumber
//               << "," << kResults[0] << "," << time_span.count() << "," << result << endl;
//        if(file)
//        {
//            file << ID1 << "," << ID2 << "," << k << "," << t << "," << xLevel << "," << yLevel << "," << percentageNumber[0] << "," << percentageNumber[1] << "," << percentageNumber[2] << "," << percentageNumber[3] << "," << percentageNumber[4] << "," << percentageNumber[5] << "," << percentageNumber[6] << "," << percentageNumber[7] << "," << percentageNumber[8]
//            << "," << percentageNumber[9] << "," << percentageNumber[10] << "," << percentageNumber[11] << "," << percentageNumber[12] << "," << percentageNumber[13] << "," << percentageNumber[14] << "," << percentageNumber[15] << "," << percentageNumber[16] << "," << percentageNumber[17] << "," << percentageNumber[18] << "," << percentageNumber[19] << "," << coverNumber
//            << "," << kResults[0] << "," << time_span.count() << "," << result << endl;
//        }
    }*/
    //file.close();


//generate 100000 KSP path for each grid pair
//boost::thread_group threads;
    /*gridDistance GD;
    GD.distance = vvGridDistance;
    int kNumber = 100000;
//Statistics
    bool bTest = false;
    int count = 0;
    //cout << "vvGridNode size: "<< vvGridNode.size() << endl;
    //for(int i = 0; i < g->vvGridNode.size(); i++){
    for (int i = mm; i < mm+1; i++){
        //for (int j = 0; j < g->vvGridNode[i].size(); ++j) {
        for (int j = nn; j < nn+1; ++j) {
            for (int k = 0; k < g->vvGridNode.size(); ++k) {
                for (int l = 0; l < g->vvGridNode[k].size(); ++l) {
            //for (int k = 15; k < 16; ++k) {
                //for (int l = 14; l < 15; ++l) {
                    cout <<  "i: " << i << " j: " << j << " k: " << k << " l: " << l << endl;
                    //int xLevel = (j+1)*vBase[level] - (i+1);
                    //int yLevel = (l+1)*vBase[level] - (k+1);
                    //int xLevel = j*vBase[level] + i;
                    //int yLevel = l*vBase[level] + k;
                    int xLevel = i*vBase[level] + j;
                    int yLevel = k*vBase[level] + l;
                    //cout << "xLevel: " << xLevel << " yLevel: " << yLevel << endl;
                    //if(g->vvGridNode[i][j].size() > 0 && g->vvGridNode[k][l].size() > 0){
                    if(g->vvGridCentreNode[i][j] != INF && g->vvGridCentreNode[k][l] != INF){
                        int gridCenterNode1 = g->vvGridCentreNode[i][j];
                        int gridCenterNode2 = g->vvGridCentreNode[k][l];
                        //int gridCenterNode1 = 17409;
                        //int gridCenterNode2 = 6358;
                        //int gridCenterNode1 = 11478;
                        //int gridCenterNode2 = 29358;
                        //int gridCenterNode1 = 17409;
                        //int gridCenterNode2 = 6358;
                        //int gridCenterNode1 = 4181;
                        //int gridCenterNode2 = 25390;
                        //open
                        //cout << "xLevel: " << xLevel << " yLevel: " << yLevel << " gridCenterNode1: " << gridCenterNode1 << " gridCenterNode2: " << gridCenterNode2 << endl;
                        //open
                        //cout << "level: " << vBase[level] << "ij: " << (j+1)*vBase[level] - (i+1)  << " lk: " << (l+1)*vBase[level] - (k+1) << endl;
                        //threads.add_thread(new boost::thread(&GridCoherence::Dijkstra,boost::ref(gridCenterNode1), boost::ref(gridCenterNode2), boost::ref(xLevel), boost::ref(yLevel), boost::ref(vvGridDistance)));
                        //int dij = Dijkstra(gridCenterNode1,gridCenterNode2, xLevel, yLevel, GD);
                        vector<int> kResults;
                        vector<vector<int> >vkPath;
                        vector<vector<int> > vvResult;
                        std::chrono::duration<double> time_span;
                        std::chrono::high_resolution_clock::time_point t1;
                        t1 = std::chrono::high_resolution_clock::now();
                        GridCoherence::eKSP(gridCenterNode1, gridCenterNode2, kNumber, kResults, vkPath, xLevel, yLevel, KSPNumber, vvResult);
                        std::chrono::high_resolution_clock::time_point t2;
                        t2 = std::chrono::high_resolution_clock::now();
                        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
                        cout << "KSP Time: " << time_span.count() << endl;
                        cout << "SP: " << kResults[0] << " Last path: " << kResults[kResults.size()-1] << " xLevel: " << xLevel << " yLevel: " << yLevel << " gridCenterNode1: " << gridCenterNode1 << " gridCenterNode2: " << gridCenterNode2<< endl;
                        for(auto& d : kResults)
                            cout << double (d-kResults[0])/double(kResults[0]) << "\t";
                        cout << endl;
                        cout << endl;
                        double edgeNumberResult;
                        int SP = kResults[0];
                        GridCoherence::edgeNumber(vvResult, edgeNumberResult, SP);
                        cout << "xLevel: " << xLevel << " yLevel: " << yLevel << " gridCenterNode1: " << gridCenterNode1 << " gridCenterNode2: " << gridCenterNode2 << " edgeNumberResult: " << edgeNumberResult << endl;
                    } else{
                        //vvGridDistance[xLevel][yLevel].push_back(0);
                        //vvGridDistance[xLevel][yLevel].push_back(-1);
                        KSPNumber[xLevel][yLevel].push_back(make_pair(-1,0));
                    }
                }
            }
        }
    }

    //Motivation graph
    vector<int> kSet;
    kSet.push_back(3);
    kSet.push_back(5);
    kSet.push_back(10);

    vector<int> kResults;
    vector<vector<int> >vkPath;
    vector<vector<int> > vvResult;
    //Motivation node
    for(int i = 0; i < 100; i++){
        int ID1 = rand() % g->n;
        int ID2 = rand() % g->n;
        cout << "ID1: " <<ID1 << " ID2: " << ID2 << endl;
        int k = 10;
        double t = 0.9;
        kResults.clear();
        vkPath.clear();
        vvResult.clear();
        std::chrono::high_resolution_clock::time_point t1;
        std::chrono::high_resolution_clock::time_point t2;
        std::chrono::duration<double> time_span;
        t1 = std::chrono::high_resolution_clock::now();
        eKSPCompare(ID1, ID2, k, kResults, vkPath, t);
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast < std::chrono::duration < double > > (t2 - t1);
        cout << "ID1: " << ID1 << " ID2: " << ID2 << " k: " << k << " t: " << t << " SP: " << kResults[0] << " eKSPCompare Time: " << time_span.count() << endl;
    }*/

    //cout << "Coherence Finish" << endl;


    //For visualize the 5 hop alternative edges
    /*int ID1 = 130844;
    int ID2 = 86632;*/
    int ID1 = 127949;
    int ID2 = 86549;
    int k = 10;
    double t = 0.9;
    cout << "ID1: " << ID1 << " ID2: " << ID2 << " k: " << k << " t: " << t << endl;
    vector<pair<int, int> > tmpMatrix;
    vector<vector<pair<int, int> > > staMatrix;
    staMatrix.assign(5, tmpMatrix);
    vector<int> kResults;
    vector<vector<int> > vkPath;
    eKSPCompare(ID1, ID2, k, kResults, vkPath, t, staMatrix);
    cout << "kResults: " << kResults[0] << endl;
    /*for (int si = 0; si < staMatrix.size(); ++si) {
        for (int sj = 0; sj < staMatrix[si].size(); ++sj) {
            cout << " Node1: " << staMatrix[si][sj].first << " Node2: " << staMatrix[si][sj].second << endl;
        }
    }
    for (int i = 0; i < vkPath[0].size(); ++i) {
        cout << vkPath[0][i] << " ";
    }
    cout << endl;*/

    string pathFileName = "127949_86549_s910";
    //string pathFileName = "130844_86632_c910";
    ofstream outfile;
    outfile.open(pathFileName);
   /* for(int i = 0; i < vkPath[0].size(); i++){
        outfile << vkPath[0][i] << " ";
    }
    outfile << endl;*/
    for (int i = 0; i < vkPath.size(); ++i) {
        for(int j = 0; j < vkPath[i].size(); j++){
            outfile << vkPath[i][j] << " ";
        }
        outfile << endl;
    }
    /*for (int si = 0; si < staMatrix.size(); ++si) {
        for (int sj = 0; sj < staMatrix[si].size(); ++sj) {
            outfile <<staMatrix[si][sj].second << " " << staMatrix[si][sj].first << endl;
        }
    }*/


}

double GridCoherence::edgeNumber(vector<vector<int> > vvResult, double& edgeNumberResult, int SP){
    priority_queue<pd2,vector<pd2>,myComp2> edgeOrder;
    unordered_set<int> edgesOrderCandidate;
    vector<pair<int, int> > edgesOrderCandidate1;

    for(int i = 0; i< vvResult.size(); i++){
        for(auto ie = vvResult[i].begin(); ie != vvResult[i].end(); ie++){
            if(edgesOrderCandidate.find(*ie) == edgesOrderCandidate.end()) {
                edgesOrderCandidate.insert(*ie);
                edgesOrderCandidate1.push_back(make_pair(*ie, 1));
            }
            else{
                bool bTest = false;
                for(int j = 0; j < edgesOrderCandidate1.size(); j++){
                    if(edgesOrderCandidate1[j].first == *ie) {
                        bTest = true;
                        edgesOrderCandidate1[j].second += 1;
                        break;
                    }
                }
                //cout << bTest << endl;
            }
        }
    }

    for(int m = 0; m < edgesOrderCandidate1.size(); m++){
        edgeOrder.push(make_pair(edgesOrderCandidate1[m].first,edgesOrderCandidate1[m].second));
    }

    int edgeTotalLength = 0;
    int edgeCountNumber = 0;
    while(!edgeOrder.empty())
    {
        int edgeIDN = edgeOrder.top().first;
        int edgeNumber = edgeOrder.top().second;
        edgeCountNumber += edgeNumber;
        edgeOrder.pop();
        edgeTotalLength += g->vEdge[edgeIDN].length;
        //cout << "edge: " << edgeIDN << " edge number: " << edgeNumber << endl;
        if(edgeTotalLength >= SP)
            break;
    }
    edgeNumberResult = double(edgeCountNumber) / vvResult.size();
    //cout << "edgeNumberResult: " << edgeNumberResult << endl;
    /*for(int i = 0; i < vvResult.size(); i++){
        for(int j = 0; j < vvResult[i].size(); j++){
            cout << vvResult[i][j] << "\t";
        }
        cout << endl;
        cout << endl;
    }*/
    return edgeNumberResult;
}
void GridCoherence::coveredGrids(double x1, double y1, double x2, double y2, double constDist, set<pair<int, int> >& sGrids)  
{
	//<level, <x,y>>
//	vector<pair<int, int> > vGrids;
	queue<pair<int, pair<int, int > > > gridQ;

	//<lon, lat>, in/not ellipse 
	//For fast check
	map<pair<double, double>, bool> mpB;
	map<pair<double, double>, bool>::iterator impB; 
	int pLevel = -2;
	for(int i = 0; i < level; i++)
	{
		double xBase = vpXYBase[i].first;
		double yBase = vpXYBase[i].second;
		for(int j = 0; j < vBase[i]; j++)
		{
			for(int k = 0; k < vBase[i]; k++)
			{
				if(vvvCoherence[i][j][k].second == 0)
					continue;
				
				pair<double, double> SW = make_pair(minX + j*xBase, minY + k*yBase);
				pair<double, double> SE = make_pair(minX + (j+1)*xBase, minY + k*yBase);
				pair<double, double> NW = make_pair(minX + j*xBase, minY + (k+1)*yBase);
				pair<double, double> NE = make_pair(minX + (j+1)*xBase, minY + (k+1)*yBase);
				vector<pair<double, double> > vpP;
				vector<pair<double, double> >::iterator ivpP;
				vpP.push_back(SW);
				vpP.push_back(SE);
				vpP.push_back(NW);
				vpP.push_back(NE);
				
				int pCount = 0;
				for(ivpP = vpP.begin(); ivpP != vpP.end(); ivpP++)
				{
					impB = mpB.find(*ivpP);
					if(impB != mpB.end())
					{
						if((*impB).second)
							pCount++;
					}
					else
					{
						int dp = Eucli((*ivpP).first, (*ivpP).second, x1, y1) + Eucli((*ivpP).first, (*ivpP).second, x2, y2);
						if(dp <= constDist)
						{
							mpB[*ivpP] = true;
							pCount++;
						}
						else
							mpB[*ivpP] = false;
					}
				}
		
				//fully covered, add all grids
				if(pCount == 4)
				{
					addAllChildren(i, j, k, sGrids);
				}
				else if(pCount > 0) //Partially covered, push to queue 
				{
	//				if(gridQ.empty())
	//					pLevel = i;
					gridQ.push(make_pair(i, make_pair(j,k))); 
				}
			}
		}

//		if(i == pLevel + 1)
//			break;
		if(!gridQ.empty())
			break;
	}
    //cout << "gridSize: " << gridQ.size() << endl;
//	cout << "Queue size:" << gridQ.size() << endl;
	pair<int, pair<int, int > > currentGrid; 
	while(!gridQ.empty())
	{
        //cout << "gridSize: " << gridQ.size() << endl;
		currentGrid = gridQ.front();
		gridQ.pop();
		int currentLevel = currentGrid.first;
		if(currentLevel + 1 < level)
		{
			int gridX = currentGrid.second.first;
			int gridY = currentGrid.second.second;
			double xBase = vpXYBase[currentLevel+1].first;
			double yBase = vpXYBase[currentLevel+1].second;
			for(int j = 0; j < 2; j++)
			{
				for(int k = 0; k < 2; k++)
				{
					if(vvvCoherence[currentLevel+1][gridX*2 + j][gridY*2 + k].second == 0)
						continue;
					pair<double, double> SW = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> SE = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> NW = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + (k+1))*yBase);
					pair<double, double> NE = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + (k+1))*yBase);
			
					vector<pair<double, double> > vpP;
					vector<pair<double, double> >::iterator ivpP;
					vpP.push_back(SW);
					vpP.push_back(SE);
					vpP.push_back(NW);
					vpP.push_back(NE);
		
					int pCount = 0;
					for(ivpP = vpP.begin(); ivpP != vpP.end(); ivpP++)
					{
						impB = mpB.find(*ivpP);
						if(impB != mpB.end())
						{
							if((*impB).second)
								pCount++;
						}
						else
						{
							int dp = Eucli((*ivpP).first, (*ivpP).second, x1, y1) + Eucli((*ivpP).first, (*ivpP).second, x2, y2);
							if(dp <= constDist)
							{
								mpB[*ivpP] = true;
								pCount++;
							}
							else
								mpB[*ivpP] = false;
						}
					
						//fully covered, add all grids
						if(pCount == 4) 
						{
							addAllChildren(currentLevel+1, 2*gridX+j, 2*gridY+k, sGrids); 
						}
						else if(pCount > 0) //Partially covered, push to queue
							gridQ.push(make_pair(currentLevel+1, make_pair(2*gridX+j, 2*gridY+k)));
					}
				}	
			}
		}
		else
		{
			int gridX = currentGrid.second.first;
			int gridY = currentGrid.second.second;
//			addAllChildren(currentLevel+1, 2*gridX, 2*gridY, sGrids); 
			double xBase = vpXYBase[currentLevel+1].first;
			double yBase = vpXYBase[currentLevel+1].second;
			for(int j = 0; j < 2; j++)
			{
				for(int k = 0; k < 2; k++)
				{
					if(vvvCoherence[level][gridX*2 +j][gridY*2 +k].second == 0)
						continue;

					pair<double, double> SW = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> SE = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> NW = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + (k+1))*yBase);
					pair<double, double> NE = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + (k+1))*yBase);
			
					vector<pair<double, double> > vpP;
					vector<pair<double, double> >::iterator ivpP;
					vpP.push_back(SW);
					vpP.push_back(SE);
					vpP.push_back(NW);
					vpP.push_back(NE);
		
					int pCount = 0;
					for(ivpP = vpP.begin(); ivpP != vpP.end(); ivpP++)
					{
						impB = mpB.find(*ivpP);
						if(impB != mpB.end())
						{
							if((*impB).second)
								pCount++;
						}
						else
						{
							int dp = Eucli((*ivpP).first, (*ivpP).second, x1, y1) + Eucli((*ivpP).first, (*ivpP).second, x2, y2);
							if(dp <= constDist)
							{
								mpB[*ivpP] = true;
								pCount++;
							}
							else
								mpB[*ivpP] = false;
						}
					
						//fully covered, add all grids
						if(pCount > 0) 
						{
							sGrids.insert(make_pair(2*gridX+j, 2*gridY+k));
		//					addAllChildren(currentLevel+1, 2*gridX+j, 2*gridY+k, sGrids); 
						}
					}
				}	
			}
		}
	}
}

void GridCoherence::addAllChildren(int currentLevel, int x, int y, set<pair<int, int> >& sGrids)  
{
	int ldiff = level - currentLevel;
	int b1 = pow(2, ldiff);
	int b2 = pow(2, ldiff-1);
	for(int i = b1 * x; i < b1 * x + b1; i++)
	{
		for(int j = b1 * y; j < b1 * y + b1; j++)
		{
			if(vvvCoherence[level][i][j].second != 0)
				sGrids.insert(make_pair(i,j));
		}
	}
}

int GridCoherence::Dijkstra(int ID1, int ID2)
//int GridCoherence::Dijkstra(int ID1, int ID2, int i, int j, gridDistance& GridDistance)
{
    benchmark::heap<2, int, int> queue(g->adjList.size());
    queue.update(ID1, 0);

    vector<int> vDistance(g->n, INF);
    vector<bool> vbVisited(g->n, false);
    int topNodeID, neighborNodeID, neighborLength;
    vector<pair<int, int> >::iterator ivp;

    vDistance[ID1] = 0;

    while(!queue.empty())
    {
        int topDistance;
        queue.extract_min(topNodeID, topDistance);
        vbVisited[topNodeID] = true;
        if(topNodeID == ID2)
            break;

        for(ivp = g->adjList[topNodeID].begin(); ivp != g->adjList[topNodeID].end(); ivp++)
        {
            neighborNodeID = (*ivp).first;
            neighborLength = (*ivp).second;
            int d = vDistance[topNodeID] + neighborLength;
            if(!vbVisited[neighborNodeID])
            {
                if(vDistance[neighborNodeID] > d)
                {
                    vDistance[neighborNodeID] = d;
                    queue.update(neighborNodeID, d);
                }
            }
        }
    }
    //GridDistance.distance[i][j].push_back(vDistance[ID2]);
    //GridDistance[i][j].push_back(vDistance[ID2]);
    //cout << "gridDistance: " << GridDistance.distance[i][j][0] << endl;
    //distance=vDistance[ID2];
    return vDistance[ID2];
}


void GridCoherence::SPT(int root, int ID2, vector<int>& vSPTDistance, vector<int>& vSPTParent,vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<vector<int> >& vSPT)
{
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
        /*if(topNodeID == ID2)
        {
            //break;
            boundDistance = topDistance;
            //boundDistance = 1.1 * topDistance;
            break;
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
            countSPTNodeNumber += 1;
            vSPT[vSPTParent[i]].push_back(i);
        }

    //cout << "SPTNodeNumber: " << countSPTNodeNumber << " SPTEdgeNumber: " << countDeviationEdgeNumber << endl;
}

vector<int> GridCoherence::makeRMQDFS(int p, vector<vector<int> >& vSPT, vector<int>& vSPTParent){
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
    rEulerSeq.assign(EulerSeq.size(),-1);
    for(int i = 0; i < EulerSeq.size(); i++)
        rEulerSeq[EulerSeq[i]] = i;
    return EulerSeq;
}

vector<vector<int>> GridCoherence::makeRMQ(int p, vector<vector<int> >& vSPT, vector<int>& vSPTHeight, vector<int>& vSPTParent){
    EulerSeq.clear();
    toRMQ.assign(g->n,0);
    RMQIndex.clear();
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    t1 = std::chrono::high_resolution_clock::now();
    makeRMQDFS(p, vSPT,vSPTParent);
    t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    //cout << "Euler Time" << time_span.count() << endl;
    RMQIndex.push_back(EulerSeq);

    int m = EulerSeq.size();
    t1 = std::chrono::high_resolution_clock::now();
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
    t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    //cout << "Table Time" << time_span.count() << endl;
    return RMQIndex;
}

//int GridCoherence::eKSPCompare(int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, double t, vector<vector<int> >& staMatrix)
int GridCoherence::eKSPCompare(int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, double t, vector<vector<pair<int,int> > >& staMatrix)
{
    //cout << "ID1: " << ID1 << " ID2: " << ID2 << endl;
    int generatePath = 0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    t1 = std::chrono::high_resolution_clock::now();
    int percentage = 0;
    //Shortest Path Tree Info
    float countAncestor = 0;
    float countNonAncestor = 0;
    int countNumber = 0;
    bool bCountNumber = true;
    vector<int> vSPTDistance(g->n, INF);
    vector<int> vSPTParent(g->n, -1);
    vector<int> vSPTHeight(g->n, -1);
    vector<int> vSPTParentEdge(g->n, -1);
    vector<int> vSPTChildren(g->n, -1);
    vector<int> vTmp;
    vector<vector<int>> vSPT(g->n, vTmp); //Tree from root

    SPT(ID1, ID2, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPT);

    vector<vector<float> > vPathLCA;
    vector<vector<int> > RMQ = makeRMQ(ID1, vSPT, vSPTHeight,vSPTParent);
    double time = 0;
    double SimTime = 0;
    double edgeTime = 0;
    double pathTime = 0;
    double loopTime = 0;

    vector<int> vEdgeReducedLength(g->vEdge.size(), INF);
    vector<double> vEdgeReducedLengthPercent(g->vEdge.size(), INF);
    for(int i = 0; i < (int)g->vEdge.size(); i++) {
        vEdgeReducedLength[i] = g->vEdge[i].length + vSPTDistance[g->vEdge[i].ID1] - vSPTDistance[g->vEdge[i].ID2];
        //cout << vEdgeReducedLength[i] << endl;
    }

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
    vPath.push_back(ID2);
    vector<pair<int, int> > vPathNonTree;
    int p = vSPTParent[ID2];
    int e = vSPTParentEdge[ID2];
    //multimap<int, int> mArc;
    float sim1;
    float addLength1 = 0;
    int oldP = ID2;
    /*vector<int> vTmpMatrix(7, 0);
    vector<vector<int>> staMatrix;
    staMatrix.assign(5,vTmpMatrix);*/
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
            p1List = p2List;
            //cout << "loopTime: " << loopTime << endl;
            for(int m = p1ListSize; m < p1List.size(); m++){
                //cout << "p1ListSize: " << p1ListSize << " p1List size: " << p1List.size() << endl;
                //cout << "neighbour size: " << vSPT[p1List[m]].size() << endl;
                for(int i = 0; i < vSPT[p1List[m]].size(); i++){
                    vector<pair<int, int> >::iterator ivp;
                    // Change all adjList to adjListR
                    for(int j = 0; j < g->adjListR[vSPT[p1List[m]][i]].size(); j++)
                    {
                        int neighborNodeID = g->adjListR[vSPT[p1List[m]][i]][j].first;
                        //cout << "neighborNodeID: " << neighborNodeID << endl;
                        int neighborLength = g->adjListR[vSPT[p1List[m]][i]][j].second;
                        if(neighborNodeID == p1List[m])
                            continue;
                        int neighborEdgeID = g->adjListEdgeR[vSPT[p1List[m]][i]][j].second;
                        int reduce = vEdgeReducedLength[neighborEdgeID];
                        //cout << "eID:" << neighborEdgeID << "\tID1:" << g->vEdge[neighborEdgeID].ID1 << "\tID2:" << g->vEdge[neighborEdgeID].ID2 << "\tlength:" << g->vEdge[neighborEdgeID].length << endl;
                        //cout << "R:" << g->vEdge[neighborEdgeID].length+vSPTDistance[g->vEdge[neighborEdgeID].ID1]-vSPTDistance[g->vEdge[neighborEdgeID].ID2]<<endl;
                        //cout << vSPTDistance[g->vEdge[neighborEdgeID].ID1] << "\t" << vSPTDistance[g->vEdge[neighborEdgeID].ID2] << endl;
                        //cout << "Reduce:" << reduce << "\t" << vSPTDistance[ID2] << endl;
                        if(vEdgeReducedLengthPercent[neighborEdgeID] == INF){
                            vEdgeReducedLengthPercent[neighborEdgeID] = (double)reduce / vSPTDistance[ID2];
                            //cout << "reduce: " << (double)reduce << " vSPTDistance[ID2]: " << vSPTDistance[ID2] << endl;

                            //For generate the edges
                            //staMatrix[loopTime].push_back(make_pair(neighborEdgeID, vSPT[p1List[m]][i]));
                            staMatrix[loopTime].push_back(make_pair(neighborNodeID, g->vEdge[neighborEdgeID].ID2));
                            //For count the path numbers
			                /*if(vEdgeReducedLengthPercent[neighborEdgeID] >= 0 && vEdgeReducedLengthPercent[neighborEdgeID] < 0.001){
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
                            }*/
                        }
                        //vEdgeReducedLengthPercent[neighborEdgeID] = (double)reduce / vSPTDistance[ID2];
                        //cout << vEdgeReducedLengthPercent[neighborEdgeID] << endl;
                    }
                 /*   for(ivp = g->adjListR[vSPT[p1List[m]][i]].begin(); ivp != g->adjListR[vSPT[p1List[m]][i]].end(); ivp++)
                    {
                        int neighborNodeID = (*ivp).first;
                        //cout << "neighborNodeID: " << neighborNodeID << endl;
                        int neighborLength = (*ivp).second;
                        if(vEdgeReducedLengthPercent[vSPTParentEdge[neighborNodeID]] == INF && neighborNodeID != p1List[m])
                        {
                            vEdgeReducedLengthPercent[vSPTParentEdge[neighborNodeID]]= vEdgeReducedLength[vSPTParentEdge[neighborNodeID]]/vSPTDistance[ID2];
                            cout << "Reduce Length: " << vEdgeReducedLengthPercent[vSPTParentEdge[neighborNodeID]] << endl;
                            //cout << g->vEdge[vSPTParentEdge[neighborNodeID]].length << " " << vSPTDistance[g->vEdge[vSPTParentEdge[neighborNodeID]].ID1] << " " << vSPTDistance[g->vEdge[vSPTParentEdge[neighborNodeID]].ID2] << endl;
                        }
                    }*/
                    p2List.push_back(vSPT[p1List[m]][i]);
                    //cout  << "neighbour size: " << vSPT[p1List[m]].size() << " node: " << p1List[m] << endl;
                    /*if(vEdgeReducedLengthPercent[vSPTParentEdge[vSPT[p1List[m]][i]]] == INF){
                        //cout << "SPT" << endl;
                        p2List.push_back(vSPT[p1List[m]][i]);
                        //cout << vSPT[p1List[m]][i] << endl;
                        //vEdgeReducedLengthPercent[vSPTParentEdge[vSPT[p1List[m]][i]]] = g->vEdge[vSPTParentEdge[vSPT[p1List[m]][i]]].length + vSPTDistance[g->vEdge[vSPTParentEdge[vSPT[p1List[m]][i]]].ID1] - vSPTDistance[g->vEdge[vSPTParentEdge[vSPT[p1List[m]][i]]].ID2];
                        //vEdgeReducedLengthPercent[vSPTParentEdge[vSPT[p1List[m]][i]]] = vEdgeReducedLength[vSPTParentEdge[vSPT[p1List[m]][i]]]/vSPTDistance[ID2];
                        //cout << "ReduceLength: " << vEdgeReducedLength[vSPTParentEdge[vSPT[p1List[m]][i]]] << " Percent: " << vEdgeReducedLengthPercent[vSPTParentEdge[vSPT[p1List[m]][i]]] << endl;
                        //cout << vSPTParentEdge[vSPT[p1List[m]][i]] << endl;
                        //cout << g->vEdge[vSPTParentEdge[vSPT[p1List[m]][i]]].length << " " << vSPTDistance[g->vEdge[vSPTParentEdge[vSPT[p1List[m]][i]]].ID1] << " " << vSPTDistance[g->vEdge[vSPTParentEdge[vSPT[p1List[m]][i]]].ID2] << endl;
                        //cout << vEdgeReducedLengthPercent[vSPTParentEdge[vSPT[p1List[m]][i]]] << endl;
                    }*/
                }
            }
            loopTime += 1;
            //cout << "Loop Time: " << loopTime << endl;
            p1ListSize = p1List.size();
        }
        vPath.push_back(p);
        oldP = p;
        vPathEdge.push_back(e);
        e = vSPTParentEdge[p];
        p = vSPTParent[p];
    }
    /*int count = 0;
    for(int j = 0; j < vEdgeReducedLengthPercent.size(); j++){
        //if(vEdgeReducedLength[j] >= 0 && vEdgeReducedLength[j] < 500){
        if(vEdgeReducedLengthPercent[j] >= 0 && vEdgeReducedLengthPercent[j] < 0.001){
            count += 1;
        }
    }
    cout << "0.001: " << count << endl;*/



    /*for(int i = 0; i < staMatrix.size(); i++)
    {
        for(int j = 0; j < staMatrix[i].size(); j++){
            cout << staMatrix[i][j] << " ";
            //cout << "loopTime: " << i << " range: " << double(j)/0.001 << "-" << double(j)/0.001+0.001 << " value: " << staMatrix[i][j];
        }
        //cout << endl;
    }
    cout << endl;*/

    /*int count = 0;
    int count1 = 0;
    int count2 = 0;
    int count3 = 0;
    int count4 = 0;
    int count5 = 0;
    int count6 = 0;
    for(int j = 0; j < vEdgeReducedLengthPercent.size(); j++){
        //if(vEdgeReducedLength[j] >= 0 && vEdgeReducedLength[j] < 500){
        if(vEdgeReducedLengthPercent[j] >= 0 && vEdgeReducedLengthPercent[j] < 0.001){
            count += 1;
        }
            //else if(vEdgeReducedLength[j] >= 500 && vEdgeReducedLength[j] < 1000){
        else if(vEdgeReducedLengthPercent[j] >= 0.001 && vEdgeReducedLengthPercent[j] < 0.002){
            count1 += 1;
        }
            //else if(vEdgeReducedLength[j] >= 1000 && vEdgeReducedLength[j] < 1500){
        else if(vEdgeReducedLengthPercent[j] >= 0.002 && vEdgeReducedLengthPercent[j] < 0.003){
            count2 += 1;
        }
            //else if(vEdgeReducedLength[j] >= 1500 && vEdgeReducedLength[j] < 2000){
        else if(vEdgeReducedLengthPercent[j] >= 0.003 && vEdgeReducedLengthPercent[j] < 0.004){
            count3 += 1;
        }
            //else if(vEdgeReducedLength[j] >= 2000 && vEdgeReducedLength[j] < 2500){
        else if(vEdgeReducedLengthPercent[j] >= 0.004 && vEdgeReducedLengthPercent[j] < 0.005){
            count4 += 1;
        }
            //else if(vEdgeReducedLength[j] >= 2500 && vEdgeReducedLength[j] < 3000){
        else if(vEdgeReducedLengthPercent[j] >= 0.005 && vEdgeReducedLengthPercent[j] < 0.006){
            count5 += 1;
        }
            //else if(vEdgeReducedLength[j] >= 3000 && vEdgeReducedLength[j] != INF){
        else if(vEdgeReducedLengthPercent[j] >= 0.006 && vEdgeReducedLengthPercent[j] != INF){
            count6 += 1;
        }
    }*/
     //cout << "0-500:" << count << " 500-1000:" << count1 << " 1000-1500:" << count2 << " 1500-2000:" << count3 << " 2000-2500:" << count4 << " 2500-3000:" << count5 << " >3000:" << count6 << endl;
     //cout << "0-0.001:" << count << " 0.001-0.002:" << count1 << " 0.002-0.003:" << count2 << " 0.003-0.004:" << count3 << " 0.004-0.005:" << count4 << " 0.005-0.006:" << count5 << " >0.006:" << count6 << endl;

    int mmArcEdgeID, mmArcReducedLength, mmArcPosition;
    vector<pair<int, int> > mmArc;
    vector<pair<int, int> > mmPos;
    while(!Arc.empty())
    {
        mmArcReducedLength = Arc.top().first;
        mmArcEdgeID = Arc.top().second.first;
        mmArcPosition = Arc.top().second.second;
        Arc.pop();
        mmArc.push_back(make_pair(mmArcEdgeID,mmArcReducedLength));
        mmPos.push_back(make_pair(mmArcEdgeID,mmArcPosition));
    }
    vvmArc.push_back(mmArc);
    vvmPathNodePos.push_back(mmPos);

    reverse(vPath.begin(), vPath.end());
    reverse(vPathEdge.begin(), vPathEdge.end());

    benchmark::heap<2, int, int> qPath(g->n);
    vvPathCandidate.push_back(vPath);
    vvPathCandidateEdge.push_back(vPathEdge);
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
        if(topPathDistance < oldDistance)
            cout<< "Error" <<endl;
        oldDistance = topPathDistance;
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
            int n = 0;
            if (vvResult.size() == 0)
            {
                vvResult.push_back(vvPathCandidateEdge[topPathID]);
                kResults.push_back(topPathDistance);
                //cout << generatePath << endl;
                generatePath = 0;
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
                        addLength = 0;
                    }

                    if (sim > t)
                        break;
                }

                if (sim <= t) {
                    //cout << generatePath << endl;
                    generatePath = 0;
                    //cout << "sim: " << sim << " topPath:" << topPathID << endl;
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


        vector<int> vTwo;
        vTwo.push_back(topPathID);
        if(vFather[topPathID] != -1 &&  !vvmArc[vFather[topPathID]].empty())
            vTwo.push_back(vFather[topPathID]);
        for(auto& pID : vTwo)
        {
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
                int eID2Pos = vvPathCandidate[pID].size()-ReID2Pos-1;

                vvmPathNodePos[pID].erase(itp);
                unordered_set<int> sE;
                for(int i = eID2Pos; i < (int)vvPathCandidate[pID].size(); i++)
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
                int distTmp = vDistance[pID] - vSPTDistance[eNodeID2] + g->vEdge[mineID].length;
                bLoop = false;

                //Open
                vPathDevPrime.push_back(eNodeID1);
                vPathDeviation.push_back(eNodeID2);

                vector<pair<int, int> > mmArcTmp;
                vPath.clear();
                vPathEdge.clear();
                int p = eNodeID1;
                int e = vSPTParentEdge[p];
                //Arc.clear();
                priority_queue<pd,vector<pd>,myComp> Arc1;
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
                            Arc1.push(make_pair(vEdgeReducedLength[reID], make_pair(reID,vPath.size()+ReID2Pos)));
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
                vector<pair<int, int> > mmE;
                while(!Arc1.empty())
                {
                    //Arc.extract_min(mmArcEdgeID, mmArcReducedLength, mmArcPosition);
                    mmArcReducedLength = Arc1.top().first;
                    mmArcEdgeID = Arc1.top().second.first;
                    mmArcPosition = Arc1.top().second.second;
                    Arc1.pop();
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
                generatePath += 1;
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span;
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count() > 5){
            //cout << "larger than 5 seconds" << endl;
            break;
        }
        /*if(countNumber >= 900000){
            bCountNumber = false;
            cout << "larger than 5 seconds" << endl;
            break;
        }*/
    }
    return -1;
}

int GridCoherence::eKSP(int ID1, int ID2, int k, vector<int>& kResults, vector<vector<int> >& vkPath, int xLevel, int yLevel,vector<vector<vector<pair<double,int> > > >& KSPNumber,  vector<vector<int> >& vvResult)
{
    bool bStop = false;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    t1 = std::chrono::high_resolution_clock::now();
    /*bool bTest1 = false;
    bool bTest2 = false;
    bool bTest3 = false;
    bool bTest4 = false;*/
    vector<int> vSPTDistance(g->n, INF);
    vector<int> vSPTParent(g->n, -1);
    vector<int> vSPTHeight(g->n, -1);
    vector<int> vSPTParentEdge(g->n, -1);
    vector<int> vSPTChildren(g->n, -1);
    vector<int> vTmp;
    vector<vector<int>> vSPT(g->n, vTmp); //Tree from root

    SPT(ID1, ID2, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPT);

    vector<int> vEdgeReducedLength(g->vEdge.size(), INF);
    for(int i = 0; i < (int)g->vEdge.size(); i++)
        vEdgeReducedLength[i] = g->vEdge[i].length + vSPTDistance[g->vEdge[i].ID1] - vSPTDistance[g->vEdge[i].ID2];

    //vector<vector<int> > vvResult;	//Exact Path
    vvResult.reserve(k);
    vector<int> vDistance;

    vector<vector<int> > vvPathCandidate;	 //nodes
    vector<vector<int> > vvPathCandidateEdge;//edges
    vector<vector<pair<int, int> > > vvPathNonTree;
    //vector<multimap<int, int> > vmArc;
    vector<vector<pair<int, int> > > vvmArc;
    vector<vector<pair<int, int> > > vvmPathNodePos;
    vector<int> vFather;

    vector<unordered_set<int> > pResult;
    vFather.push_back(-1);

    //benchmark::pHeap<3, int, int, int> Arc(g->n);
    priority_queue<pd,vector<pd>,myComp> Arc;
    vector<int> vPath;
    vector<int> vPathEdge;
    vPath.push_back(ID2);
    vector<pair<int, int> > vPathNonTree;
    int p = vSPTParent[ID2];
    int e = vSPTParentEdge[ID2];
    //multimap<int, int> mArc;
    float addLength1 = 0;
    int oldP = ID2;

    while(p != -1)
    {
        //cout << "p:" << oldP << endl;
        for(int i = 0; i < (int)g->adjListEdgeR[oldP].size(); i++)
        {
            int eID = g->adjListEdgeR[oldP][i].second;
            if(eID != e){
                //mArc.insert(make_pair(vEdgeReducedLength[eID], eID));
                //cout << "eID: " << eID << " ReduceLength: " << vEdgeReducedLength[eID] << " position: " << vPath.size()-1 << endl;
                //Arc.update(eID,vEdgeReducedLength[eID],vPath.size()-1);
                Arc.push(make_pair(vEdgeReducedLength[eID], make_pair(eID,vPath.size()-1)));
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
    while(!Arc.empty())
    {
        //Arc.extract_min(mmArcEdgeID, mmArcReducedLength, mmArcPosition);

        mmArcReducedLength = Arc.top().first;
        mmArcEdgeID = Arc.top().second.first;
        mmArcPosition = Arc.top().second.second;
        Arc.pop();
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


    benchmark::heap<2, int, int> qPath(g->n);
    vvPathCandidate.push_back(vPath);
    vvPathCandidateEdge.push_back(vPathEdge);
    vDistance.push_back(vSPTDistance[ID2]);


    qPath.update(vvPathCandidate.size()-1, vSPTDistance[ID2]);

    vector<int> vResultID;
    int topPathID, topPathDistance;
    int pcount = 0;
    int oldDistance = -1;
    bool bError = false;
    int SPDistance = vSPTDistance[ID2];

    while((int)kResults.size() < k && !qPath.empty())
    //while(!qPath.empty())
    {
        //t1 = std::chrono::high_resolution_clock::now();
        float addLength = 0;
        pcount++;
        qPath.extract_min(topPathID, topPathDistance);
        //cout << topPathID << endl;
        /*if(topPathDistance > 1.001* SPDistance && bTest1 == false)
        {
            //cout << "Added!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << " xLevel: " << xLevel << " yLevel: " << yLevel << " size: " << KSPNumber[xLevel][yLevel].size() << endl;
            KSPNumber[xLevel][yLevel].push_back(make_pair(1.001,kResults.size()));
            //cout << "Added!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << " xLevel: " << xLevel << " yLevel: " << yLevel << " size: " << KSPNumber[xLevel][yLevel].size() << endl;
            //cout << KSPNumber[xLevel][yLevel].size() << endl;
              //cout << KSPNumber[xLevel][yLevel][0].first << " KSPNumber: " << KSPNumber[xLevel][yLevel][0].second << endl;
          bTest1 = true;
        }
        if(topPathDistance > 1.002* SPDistance && bTest2 == false)
        {
           KSPNumber[xLevel][yLevel].push_back(make_pair(1.002,kResults.size()));
            bTest2 = true;
        }
        if(topPathDistance > 1.003* SPDistance && bTest3 == false)
        {
            KSPNumber[xLevel][yLevel].push_back(make_pair(1.003,kResults.size()));
            bTest3 = true;
        }
        if(topPathDistance > 1.1* SPDistance && bTest4 == false)
        {
            KSPNumber[xLevel][yLevel].push_back(make_pair(1.1,kResults.size()));
            bTest4 = true;
            break;
        }*/
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

        if(!bTopLoop)
        {
            //popPath++;
            int n = 0;
            vvResult.push_back(vvPathCandidateEdge[topPathID]);
            kResults.push_back(topPathDistance);
            vkPath.push_back(vvPathCandidate[topPathID]);
            vResultID.push_back(topPathID);
            unordered_set<int> pTmp;
            for(auto ie = vvPathCandidateEdge[topPathID].begin(); ie != vvPathCandidateEdge[topPathID].end(); ie++)
            {
                pTmp.insert(*ie);
            }
            pResult.push_back(pTmp);
        }

        //t1 = std::chrono::high_resolution_clock::now();
        vector<int> vTwo;
        vTwo.push_back(topPathID);
        //if(vFather[topPathID] != -1 &&  !vmArc[vFather[topPathID]].empty())
        if(vFather[topPathID] != -1 &&  !vvmArc[vFather[topPathID]].empty())
            //if(vFather[topPathID] != -1 &&  !ArcList[vFather[topPathID]].empty())
            vTwo.push_back(vFather[topPathID]);
        for(auto& pID : vTwo)
        {
            bool bLoop = true;
            while(bLoop)
            {
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
                eReducedLen = (*it).second;
                mineID = (*it).first;
                vvmArc[pID].erase(it);

                //eNodeID1 is also the first point from the deviation edge
                int eNodeID1 = g->vEdge[mineID].ID1;
                int eNodeID2 = g->vEdge[mineID].ID2;

                bool bFixedLoop = false;

                //Change to vector
                auto itp = vvmPathNodePos[pID].begin();
                int ReID2Pos = (*itp).second;
                int eID2Pos = vvPathCandidate[pID].size()-ReID2Pos-1;
                vvmPathNodePos[pID].erase(itp);
                unordered_set<int> sE;
                for(int i = eID2Pos; i < (int)vvPathCandidate[pID].size(); i++)
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
                int distTmp = vDistance[pID] - vSPTDistance[eNodeID2] + g->vEdge[mineID].length;
                bLoop = false;

                //multimap<int, int> mArcTmp;
                vector<pair<int, int> > mmArcTmp;
                vPath.clear();
                vPathEdge.clear();
                int p = eNodeID1;
                int e = vSPTParentEdge[p];
                //Arc.clear();
                priority_queue<pd,vector<pd>,myComp> Arc1;
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
                            //Arc.update(reID,vEdgeReducedLength[reID],vPath.size()+ReID2Pos);
                            Arc1.push(make_pair(vEdgeReducedLength[reID], make_pair(reID,vPath.size()+ReID2Pos)));
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
                vector<pair<int, int> > mmE;
                while(!Arc1.empty())
                {
                    //Arc.extract_min(mmArcEdgeID, mmArcReducedLength, mmArcPosition);
                    mmArcReducedLength = Arc1.top().first;
                    mmArcEdgeID = Arc1.top().second.first;
                    mmArcPosition = Arc1.top().second.second;
                    Arc1.pop();
                    mmArcTmp.push_back(make_pair(mmArcEdgeID,mmArcReducedLength));
                    mmE.push_back(make_pair(mmArcEdgeID,mmArcPosition));
                }
                //cout << vvmArc.size() << "    " << vvmPathNodePos.size()<< endl;

                int dist = vDistance[pID] - vSPTDistance[eNodeID2] + vSPTDistance[eNodeID1] + g->vEdge[mineID].length;
                vDistance.push_back(dist);

                vFather.push_back(pID);

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
                vvmArc.push_back(mmArcTmp);
                vvmPathNodePos.push_back(mmE);
                vvPathCandidate.push_back(vPath);
                vvPathCandidateEdge.push_back(vPathEdge);
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span;
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count() > 5){
            bStop = true;
            //cout << "larger than 50 seconds" << endl;
            break;
        }
    }
    if(bStop == true){
        /*for(double i = 0.01; i < 0.21; i+=0.01) {
            KSPNumber[xLevel][yLevel].push_back(make_pair(i, 100000));
        }*/
        bool bEnd = false;
        for(double i = 0.0002; i <= 0.04; i+=0.0002){
        //for(double i = 0.01; i < 0.21; i+=0.01){
            int kNumber = 0;
            bool vInsert = false;
            for(int j = 0; j < kResults.size(); j++){
                if((double(kResults[j]-kResults[0])/double(kResults[0])) < i){
                    kNumber++;
                }
                else{
                    vInsert = true;
                    //GridKSPNumber[xLevel][yLevel].push_back(make_pair(i,kNumber));
                    if(kNumber == kResults.size()){
                        bEnd = true;
                        KSPNumber[xLevel][yLevel].push_back(make_pair(i,100000));
                    }
                    else
                        KSPNumber[xLevel][yLevel].push_back(make_pair(i,kNumber));
                    //GridKSPNumber[xLevel][yLevel][(i/0.05)-1] = make_pair(i,kNumber);
                    break;
                }
                if(j == kResults.size()-1){
                    vInsert = true;
                    //GridKSPNumber[xLevel][yLevel].push_back(make_pair(i,kNumber));
                    if(kNumber == kResults.size())
                    {
                        bEnd = true;
                        KSPNumber[xLevel][yLevel].push_back(make_pair(i,100000));
                    }
                    else
                        KSPNumber[xLevel][yLevel].push_back(make_pair(i,kNumber));
                }
            }
            if(vInsert == false)
            {
                bEnd = true;
                KSPNumber[xLevel][yLevel].push_back(make_pair(i,100000));
            }
            if(bEnd == true)
                break;
        }
    }
    else{
        bool bEnd = false;
        //cout << "kSPNumber: " << kResults.size() << endl;
        for(double i = 0.0002; i <= 0.04; i+=0.0002){
            int kNumber = 0;
            bool vInsert = false;
            for(int j = 0; j < kResults.size(); j++){
                if((double(kResults[j]-kResults[0])/double(kResults[0])) < i){
                    kNumber++;
                }
                else{
                    vInsert = true;
                    if(kNumber == 100000)
                        bEnd = true;
                    KSPNumber[xLevel][yLevel].push_back(make_pair(i,kNumber));
                    break;
                }

                if(j == kResults.size() -1){
                    vInsert = true;
                    if(kNumber == 100000)
                        bEnd = true;
                    KSPNumber[xLevel][yLevel].push_back(make_pair(i,kNumber));
                }
            }
            if(vInsert == false)
            {
                bEnd = true;
                KSPNumber[xLevel][yLevel].push_back(make_pair(i,-1));
            }
            if(bEnd == true)
                break;
        }
    }
    return -1;
}

int GridCoherence::Eucli(double x1, double y1, double x2, double y2)
{
	/*int lat=(int)(abs(y1-y2)*111319);
	int lon=(int)(abs(x1-x2)*83907);*/
    //COL 1 lon 86km. 1 lat 111km
    int lat=(int)(abs(y1-y2)*1110000);
    int lon=(int)(abs(x1-x2)*860000);
    //coveredGridsint lon=(int)(abs(x1-x2)*310000);
	int min,max;
	min=(lat>lon)?lon:lat;
	max=(lat>lon)?lat:lon;
	int approx=max*1007+min*441;
	
	if(max<(min<<4))
		approx-=max*40;
	return (approx+512)>>10;
	}

void GridCoherence::loadCoherence()
{
}

void GridCoherence::saveCoherence()
{
}

void GridCoherence::coveredGridsRadius(double x, double y, double r, set<pair<int, int> >& sGrids)  
{
	//<level, <x,y>>
//	vector<pair<int, int> > vGrids;
	queue<pair<int, pair<int, int > > > gridQ;
	
	//<lon, lat>, in/not ellipse 
	//For fast check
	map<pair<double, double>, bool> mpB;
	map<pair<double, double>, bool>::iterator impB; 
	int pLevel = -2;
	for(int i = 0; i < level; i++)
	{
		double xBase = vpXYBase[i].first;
		double yBase = vpXYBase[i].second;
		for(int j = 0; j < vBase[i]; j++)
		{
			for(int k = 0; k < vBase[i]; k++)
			{
//				if(vvvCoherence[i][j][k].second == 0)
//					continue;
				
				pair<double, double> SW = make_pair(minX + j*xBase, minY + k*yBase);
				pair<double, double> SE = make_pair(minX + (j+1)*xBase, minY + k*yBase);
				pair<double, double> NW = make_pair(minX + j*xBase, minY + (k+1)*yBase);
				pair<double, double> NE = make_pair(minX + (j+1)*xBase, minY + (k+1)*yBase);
				vector<pair<double, double> > vpP;
				vector<pair<double, double> >::iterator ivpP;
				vpP.push_back(SW);
				vpP.push_back(SE);
				vpP.push_back(NW);
				vpP.push_back(NE);
				
				int pCount = 0;
				for(ivpP = vpP.begin(); ivpP != vpP.end(); ivpP++)
				{
					impB = mpB.find(*ivpP);
					if(impB != mpB.end())
					{
						if((*impB).second)
							pCount++;
					}
					else
					{
						int dp = Eucli((*ivpP).first, (*ivpP).second, x, y);
						if(dp <= r)
						{
							mpB[*ivpP] = true;
							pCount++;
							cout <<"Upper Grid:" << i <<endl;
							cout << dp <<"\t" <<r << endl;
						}
						else
							mpB[*ivpP] = false;
					}
				}
		
				//fully covered, add all grids
				if(pCount == 4)
				{
					addAllChildren(i, j, k, sGrids);
				}
				else if(pCount > 0) //Partially covered, push to queue 
				{
	//				if(gridQ.empty())
	//					pLevel = i;
					gridQ.push(make_pair(i, make_pair(j,k))); 
				}
			}
		}

//		if(i == pLevel + 1)
//			break;
		if(!gridQ.empty())
			break;
	}
//	cout << "Queue size:" << gridQ.size() << endl;
	pair<int, pair<int, int > > currentGrid; 
	while(!gridQ.empty())
	{
		currentGrid = gridQ.front();
		gridQ.pop();
		int currentLevel = currentGrid.first;
		if(currentLevel + 1 < level)
		{
			int gridX = currentGrid.second.first;
			int gridY = currentGrid.second.second;
			double xBase = vpXYBase[currentLevel+1].first;
			double yBase = vpXYBase[currentLevel+1].second;
			for(int j = 0; j < 2; j++)
			{
				for(int k = 0; k < 2; k++)
				{
//					if(vvvCoherence[currentLevel+1][gridX*2 + j][gridY*2 + k].second == 0)
//						continue;
					pair<double, double> SW = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> SE = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> NW = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + (k+1))*yBase);
					pair<double, double> NE = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + (k+1))*yBase);
			
					vector<pair<double, double> > vpP;
					vector<pair<double, double> >::iterator ivpP;
					vpP.push_back(SW);
					vpP.push_back(SE);
					vpP.push_back(NW);
					vpP.push_back(NE);
		
					int pCount = 0;
					for(ivpP = vpP.begin(); ivpP != vpP.end(); ivpP++)
					{
						impB = mpB.find(*ivpP);
						if(impB != mpB.end())
						{
							if((*impB).second)
								pCount++;
						}
						else
						{
							int dp = Eucli((*ivpP).first, (*ivpP).second, x, y);
							if(dp <= r)
							{
								mpB[*ivpP] = true;
								pCount++;
								cout <<"High Grid:" << currentLevel <<endl;
								cout << dp <<"\t" <<r << endl;
							}
							else
								mpB[*ivpP] = false;
						}
					
						//fully covered, add all grids
						if(pCount == 4) 
						{
							addAllChildren(currentLevel+1, 2*gridX+j, 2*gridY+k, sGrids); 
						}
						else if(pCount > 0) //Partially covered, push to queue
							gridQ.push(make_pair(currentLevel+1, make_pair(2*gridX+j, 2*gridY+k)));
					}
				}	
			}
		}
		else
		{
			int gridX = currentGrid.second.first;
			int gridY = currentGrid.second.second;
//			addAllChildren(currentLevel+1, 2*gridX, 2*gridY, sGrids); 
			double xBase = vpXYBase[currentLevel+1].first;
			double yBase = vpXYBase[currentLevel+1].second;
			for(int j = 0; j < 2; j++)
			{
				for(int k = 0; k < 2; k++)
				{
//					if(vvvCoherence[level][gridX*2 +j][gridY*2 +k].second == 0)
//						continue;

					pair<double, double> SW = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> SE = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + k)*yBase);
					pair<double, double> NE = make_pair(minX + (gridX*2 + (j+1))*xBase, minY + (gridY*2 + (k+1))*yBase);
					pair<double, double> NW = make_pair(minX + (gridX*2 + j)*xBase, minY + (gridY*2 + (k+1))*yBase);
			
					vector<pair<double, double> > vpP;
					vector<pair<double, double> >::iterator ivpP;
					vpP.push_back(SW);
					vpP.push_back(SE);
					vpP.push_back(NW);
					vpP.push_back(NE);
		
					int pCount = 0;
					for(ivpP = vpP.begin(); ivpP != vpP.end(); ivpP++)
					{
						impB = mpB.find(*ivpP);
						if(impB != mpB.end())
						{
							if((*impB).second)
								pCount++;
						}
						else
						{
							int dp = Eucli((*ivpP).first, (*ivpP).second, x, y);
							if(dp <= r)
							{
								cout <<"Lowest Grid:" <<endl;
								cout << dp <<"\t" <<r << endl;
								mpB[*ivpP] = true;
								pCount++;
							}
							else
								mpB[*ivpP] = false;
						}
					
						//fully covered, add all grids
						if(pCount > 0) 
						{
							sGrids.insert(make_pair(2*gridX+j, 2*gridY+k));
		//					addAllChildren(currentLevel+1, 2*gridX+j, 2*gridY+k, sGrids); 
						}
					}
				}	
			}
		}
				
	}
}
	
