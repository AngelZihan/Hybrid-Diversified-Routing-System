#include "graph.h"

int Graph::readBeijingMapDirected(string filename)
{
	ifstream inGraph(filename.c_str());
	if(!inGraph)
	{
		cout << "Cannot open Beijing Map " << filename << endl;
		return -1;
	}

	int nodeNum, i;
	inGraph >> nodeNum >> minX >> maxX >> minY >> maxY;
	cout << "Beijing Node number: " << nodeNum << endl;
	this->nodeNum = nodeNum;

	vector<Edge> vEdges; 
	vCoor.reserve(nodeNum);
	

	double x, y;
	int nodeID, type, j, k;
	int ID2, length;
	for(i = 0; i < nodeNum; i++)
	{
		inGraph >> nodeID >> type >> x >> y >> j >> j; 
		vCoor.push_back(make_pair(x, y));

		for(k = 0; k < j; k++)
		{
			inGraph >> ID2 >> length; 
			struct Edge e, eR;
			e.ID1 = nodeID;
			e.ID2 = ID2;
			e.length = length;
			vEdges.push_back(e);

			eR.ID1 = ID2;
			eR.ID2 = nodeID;
			eR.length = length;
			vEdgeR.push_back(eR);
		}

		inGraph >> j;
		for(k = 0; k < j; k++)
			inGraph >> ID2 >> length;

		inGraph >> j;
		for(k = 0; k < j; k++)
			inGraph >> ID2 >> length;
	}

	cout << "Finish Reading " << filename << endl;
	inGraph.close();

	adjList.resize(nodeNum);
	adjListR.resize(nodeNum);
	adjListEdge.resize(nodeNum);
	adjListEdgeR.resize(nodeNum);
	int edgeCount = 0;
	for(auto &edge: vEdges)
	{
		int ID1 = edge.ID1;
		int ID2 = edge.ID2;
		int length = edge.length;

		bool b = false;
		for(auto ivp = adjList[ID1].begin(); ivp != adjList[ID1].end(); ivp++)
		{
			if((*ivp).first == ID2) 
			{
				b = true;
				break;
			}
		}

		if(!b) 
		{
			adjList[ID1].push_back(make_pair(ID2, length)); 
			adjListR[ID2].push_back(make_pair(ID1, length));

			edge.edgeID = edgeCount; 
			cout << ID2 << "\t" << ID1 << "\t" << edgeCount << endl;
			adjListEdge[ID1].push_back(make_pair(ID2, edgeCount)); 
			adjListEdgeR[ID2].push_back(make_pair(ID1, edgeCount));
			vEdge.push_back(edge);
			edgeCount++;
		}
	}

	cout << "Beijing Road number: " << edgeCount << endl;

	return nodeNum;
}
//Edited
int Graph::readUSMap(string filename)
{
    //cout << "us map" << endl;
    ifstream inGraph(filename);
    if(!inGraph)
        cout << "Cannot open Map " << filename << endl;
    cout << "Reading " << filename << endl;

    string line;
    do
    {
        getline(inGraph,line);
        if(line[0]=='p')
        {
            vector<string> vs;
            boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
            nodeNum = stoi(vs[2]); //edgenum=stoi(vs[3]);
            cout << "Nodenum " << nodeNum<<endl;
            edgeNum = 0;
        }
    }while(line[0]=='c'|| line[0]=='p');

    int ID1, ID2, length;
    adjList.resize(nodeNum);
    adjListR.resize(nodeNum);
    adjListEdge.resize(nodeNum);
    adjListEdgeR.resize(nodeNum);
    int edgeCount = 0;
    string a;
    while(!inGraph.eof())
    {
		vector<string> vs;
		boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
		ID1 = stoi(vs[1]) - 1;
		ID2 = stoi(vs[2]) - 1;
		length = stoi(vs[3]);
        /*inGraph >> a >> ID1 >> ID2 >> length;
        ID1 -= 1;
        ID2 -= 1;*/
        //cout << ID1 << "\t" << ID2 << "\t" << length << endl;

        struct Edge e;
        e.ID1 = ID1;
        e.ID2 = ID2;
        e.length = length;
        e.edgeID = edgeCount;

        bool bExisit = false;
        for(int i = 0; i < (int)adjList[ID1].size(); i++)
        {
            if(adjList[ID1][i].first == ID2)
            {
                bExisit = true;
                break;
            }
        }

        //cout << ID1 << "\t" << ID2 << "\t" << length << endl;
        if(!bExisit)
        {
            vEdge.push_back(e);
            adjList[ID1].push_back(make_pair(ID2, length));
            adjListR[ID2].push_back(make_pair(ID1, length));
            adjListEdge[ID1].push_back(make_pair(ID2, edgeCount));
            adjListEdgeR[ID2].push_back(make_pair(ID1, edgeCount));
            edgeCount++;
        }
        getline(inGraph,line);
    }

    vbISO.assign(nodeNum, false);
    inGraph.close();

    /*string coorFile = filename.substr(0, filename.size()-2)+"co";
    ifstream inCoor(coorFile.c_str());
    if(!inCoor)
        cout << "Cannot open Coordinate " << coorFile << endl;
    cout << "Reading " << coorFile << endl;

    do
    {
        getline(inCoor,line);
    }while(line[0]=='c'|| line[0]=='p');

    double lon, lat;
    vCoor.resize(nodeNum);
    while(!inCoor.eof())
    {
        vector<string> vs;
        boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
        lon = stod(vs[2].substr(0,3) + "." + vs[2].substr(3));
        lat = stod(vs[3].substr(0,2) + "." + vs[3].substr(2));
        vCoor.push_back(make_pair(lon, lat));
        getline(inCoor,line);
    }*/

//	mCoor["NY"] = make_pair(84000, 69000);

    return nodeNum;
}

int Graph:: readTestCoor(string filename)
{
    string line;
    ifstream inCoor(filename);
    vCoor.resize(nodeNum);
    if(!inCoor)
        cout << "Cannot open Coordinate " << filename << endl;
    cout << "Reading " << filename << endl;

    do
    {
        getline(inCoor,line);
    }while(line[0]=='c'|| line[0]=='p');

    double lon, lat;
    while(!inCoor.eof())
    {
        vector<string> vs;
        boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
        lon = stod(vs[2].substr(0,3) + "." + vs[2].substr(3));
        lat = stod(vs[3].substr(0,2) + "." + vs[3].substr(2));
        vCoor.push_back(make_pair(lon, lat));
        getline(inCoor,line);
    }

//	mCoor["NY"] = make_pair(84000, 69000);
    return nodeNum;
}

int Graph:: readTestMap(string filename)
{
    ifstream inGraph(filename);
    if(!inGraph)
        cout << "Cannot open Map " << filename << endl;
    cout << "Reading " << filename << endl;

    string line;
    do
    {
        getline(inGraph,line);
        if(line[0]=='p')
        {
            vector<string> vs;
            boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
            nodeNum = stoi(vs[2]); //edgenum=stoi(vs[3]);
            cout << "Nodenum " << nodeNum<<endl;
            edgeNum = 0;
        }
    }while(line[0]=='c'|| line[0]=='p');

    //inGraph >> nodeNum;
    int ID1, ID2, length;
    adjList.resize(nodeNum);
    adjListR.resize(nodeNum);
    adjListEdge.resize(nodeNum);
    adjListEdgeR.resize(nodeNum);
    int edgeCount = 0;
    string a;
    while(!inGraph.eof())
    {
        inGraph >> a >> ID1 >> ID2 >> length;
        ID1 -= 1;
        ID2 -= 1;

        struct Edge e;
        e.ID1 = ID1;
        e.ID2 = ID2;
        e.length = length;
        e.edgeID = edgeCount;
        //vEdge.push_back(e);
        bool bExisit = false;

        for(int i = 0; i < (int)adjList[ID1].size(); i++)
        {
            if(adjList[ID1][i].first == ID2)
            {
                bExisit = true;
                break;
            }
        }
        if(!bExisit)
        {
            vEdge.push_back(e);
            adjList[ID1].push_back(make_pair(ID2, length));
            adjListR[ID2].push_back(make_pair(ID1, length));
            adjListEdge[ID1].push_back(make_pair(ID2, edgeCount));
            adjListEdgeR[ID2].push_back(make_pair(ID1, edgeCount));
            edgeCount++;
        }
        //getline(inGraph,line);
    }
    vbISO.assign(nodeNum, false);
    inGraph.close();

    return nodeNum;
}

int Graph::readUSMapCost(string filename)
{ 
	ifstream inGraph(filename);
	if(!inGraph)
		cout << "Cannot open Map " << filename << endl; 
	cout << "Reading " << filename << endl;

	string line; 
	inGraph >> nodeNum;
	int ID1, ID2, length, cost;
	adjList.resize(nodeNum);
	adjListR.resize(nodeNum);
	adjListEdge.resize(nodeNum);
	adjListEdgeR.resize(nodeNum); 
	adjListCost.resize(nodeNum);
	adjListCostR.resize(nodeNum);
	int edgeCount = 0;
	while(!inGraph.eof())
	{
		inGraph >> ID1 >> ID2 >> length >> cost;
        //cout << ID1 << "," << ID2 << "," << length << "," << cost << endl;
		ID1 -= 1; 
		ID2 -= 1;

		struct Edge e; 
		e.ID1 = ID1;
		e.ID2 = ID2;
		e.length = length;
		e.edgeID = edgeCount;  
		e.cost = cost;

		bool bExisit = false;
		for(int i = 0; i < (int)adjListCost[ID1].size(); i++) 
		{
			if(adjListCost[ID1][i].first == ID2)
			{
				bExisit = true;
				if(cost < adjListCost[ID1][i].second)
				{
//					cout << "Cost update:" << adjListCost[ID1][i].second << "\t" << cost << endl; 
					adjListCost[ID1][i].second = cost; 
					for(int j = 0; j < (int)adjListCostR[ID2].size(); j++)
					{
						if(adjListCostR[ID2][j].first == ID1)
						{
							adjListCostR[ID2][j].second = cost;    
							break;
						}
					}

					int eID = adjListEdge[ID1][i].second;
					vEdge[eID].cost = cost;
				}
				break;
			}
		}

		if(!bExisit)
		{
			vEdge.push_back(e);
			adjList[ID1].push_back(make_pair(ID2, length));
			adjListR[ID2].push_back(make_pair(ID1, length));
			adjListEdge[ID1].push_back(make_pair(ID2, edgeCount)); 
			adjListEdgeR[ID2].push_back(make_pair(ID1, edgeCount));
			adjListCost[ID1].push_back(make_pair(ID2, cost));
			adjListCostR[ID2].push_back(make_pair(ID1, cost)); 
			edgeCount++;
		}
	//	getline(inGraph,line);
	}

/*	for(int i = 0; i < (int)adjListCost.size(); i++)
	{
		unordered_map<int, int> us;
		for(auto& c: adjListCost[i])
		{
			if(us.find(c.first) != us.end())
			{
				cout << "Repeated! " << i << "\t" << c.first << "\t" << us[c.first] << "\t" << c.second << endl;
			}
			else
				us[c.first] = c.second;
		}
	}
*/
	vbISO.assign(nodeNum, false); 
	inGraph.close();
/*	
	vector<int> vCorrect;
	ifstream ic("./c"); 
	int icc;
	while(ic >> icc)
		vCorrect.push_back(icc); 
	int c= 0;
	int d = 0;
	int oldv = -1; 
	for(auto&v : vCorrect)
	{
		cout << v << "\t";
		bool b = false;
		if(oldv > -1)
		{
			for(int j= 0 ; j < (int)adjList[oldv].size(); j++)
			{
				if(adjList[oldv][j].first == v)
				{
					d += adjList[oldv][j].second;
					c += adjListCost[oldv][j].second;
					b = true;
					break;
				}
			}
			if(!b)
			{
				cout << endl << "Disconnected!" << endl;
			}
		}
		oldv = v;
		cout << endl << d << "\t" << c << "\t" << endl;
	}
	*/
	return nodeNum;
}

int Graph::readExampleMap(string filename)
{
	ifstream inGraph(filename);
	if(!inGraph)
		cout << "Cannot open Map " << filename << endl; 
	cout << "Reading " << filename << endl;

	string line;
	do
	{
		getline(inGraph,line);
		if(line[0]=='p')
		{ 
			vector<string> vs;
			boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
			nodeNum = stoi(vs[2]); //edgenum=stoi(vs[3]);
			cout << "Nodenum " << nodeNum<<endl;
			edgeNum = 0;
		}
	}while(line[0]=='c'|| line[0]=='p');

	int ID1, ID2, length;
	adjList.resize(nodeNum);
	adjListR.resize(nodeNum);
	adjListEdge.resize(nodeNum);
	adjListEdgeR.resize(nodeNum); 
	int edgeCount = 0;
	while(!inGraph.eof())
	{
		vector<string> vs;
		boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
		ID1 = stoi(vs[1]); 
		ID2 = stoi(vs[2]); 
		length = stoi(vs[3]);
		ID1 -= ID1;
		ID2 -= ID2;
		
		struct Edge e; 
		e.ID1 = ID1;
		e.ID2 = ID2;
		e.length = length;
		e.edgeID = edgeCount; 
		vEdge.push_back(e);

		cout << ID1 << "\t" << ID2 << "\t" << length << endl;
		adjList[ID1].push_back(make_pair(ID2, length));
		adjListR[ID2].push_back(make_pair(ID1, length));
		adjListEdge[ID1].push_back(make_pair(ID2, edgeCount)); 
		adjListEdgeR[ID2].push_back(make_pair(ID1, edgeCount));
		edgeCount++;
		getline(inGraph,line);
	}

	vbISO.assign(nodeNum, false); 
	inGraph.close();

//	mCoor["NY"] = make_pair(84000, 69000);

	return nodeNum;
}
	
int Graph::readUSCost(string filename)
{
	ifstream inGraph(filename);
	if(!inGraph)
		cout << "Cannot open Map " << filename << endl; 
	cout << "Reading " << filename << endl;

	string line;
	do
	{
		getline(inGraph,line);
	}while(line[0]=='c'||line[0]=='p');

	int ID1, ID2, cost; 
	adjListCost.resize(nodeNum);
	adjListCostR.resize(nodeNum);
	int edgeCount = 0;
	while(!inGraph.eof())
	{
		vector<string> vs;
		boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
		ID1 = stoi(vs[1]) - 1;  
		ID2 = stoi(vs[2]) - 1; 
		cost = stoi(vs[3]);
		
		adjListCost[ID1].push_back(make_pair(ID2, cost));
		adjListCostR[ID2].push_back(make_pair(ID1, cost)); 
		vEdge[edgeCount].cost = cost;
		edgeCount++;
		getline(inGraph,line);
	}
	
	cout << vEdge.size() << endl;
	return nodeNum;
}

int Graph::DijkstraPath(int ID1, int ID2, vector<int>& vPath, vector<int>& vPathEdge)
{
	benchmark::heap<2, int, int> queue(adjList.size());
	queue.update(ID1, 0);
    vector<pair<int, int> >::iterator ivp;

	vector<int> vDistance(adjList.size(), INF);
	vector<int> vPrevious(adjList.size(), -1);
	vector<int> vPreviousEdge(adjList.size(), -1);
	vector<bool> vbVisited(adjList.size(), false);
	int topNodeID, neighborNodeID, neighborLength, neighborRoadID;

	vDistance[ID1] = 0;
	while(!queue.empty())
	{
		int topDistance;
		queue.extract_min(topNodeID, topDistance);
		//cout << topNodeID << "\t" << vDistance[topNodeID] << endl;
		vbVisited[topNodeID] = true;
		if(topNodeID == ID2)
			break;
        for(int i = 0; i < (int)adjList[topNodeID].size(); i++)
        {
            neighborNodeID = adjList[topNodeID][i].first;
            neighborLength = adjList[topNodeID][i].second;
            neighborRoadID = adjListEdge[topNodeID][i].second;
            //cout << (int)adjList[topNodeID].size() << " "<< neighborRoadID << endl;
            int d = vDistance[topNodeID] + neighborLength;
            if(!vbVisited[neighborNodeID])
            {
                if(vDistance[neighborNodeID] > d)
                {
                    vDistance[neighborNodeID] = d;
                    queue.update(neighborNodeID, d);
                    vPrevious[neighborNodeID] = topNodeID;
                    vPreviousEdge[neighborNodeID] = neighborRoadID;
                }
            }
        }
		/*for(ivp = adjList[topNodeID].begin(); ivp != adjList[topNodeID].end(); ivp++)
		{
            neighborNodeID = (*ivp).first;
            neighborLength = (*ivp).second;
            //cout << neighborNodeID << " " << neighborLength << endl;
			int d = vDistance[topNodeID] + neighborLength;
			if(!vbVisited[neighborNodeID])
			{
				if(vDistance[neighborNodeID] > d)
				{
					vDistance[neighborNodeID] = d;
					queue.update(neighborNodeID, d);
					vPrevious[neighborNodeID] = topNodeID;
					vPreviousEdge[neighborNodeID] = neighborRoadID;
                    //cout << topNodeID << " " << neighborNodeID << endl;

				}
			}
		}*/
	}
	//cout << vPrevious[20579] << endl;
	vPath.clear();
	vPathEdge.clear();
	vPath.push_back(ID2);
	int p = vPrevious[ID2];
	int e = vPreviousEdge[ID2];
	while(p != -1)
	{
		vPath.push_back(p);
		vPathEdge.push_back(e);
		e = vPreviousEdge[p];
		p = vPrevious[p];
	}

//	if(vPathEdge.size() > 0)
//		vPathEdge.erase(vPathEdge.end()-1);

    reverse(vPath.begin(), vPath.end());
	reverse(vPathEdge.begin(), vPathEdge.end());
	return vDistance[ID2];
}

int Graph::DijkstraPath2(int ID1, int ID2, unordered_set<int>& sRemovedNode, vector<int>& vPath, vector<int>& vPathEdge)
{
    benchmark::heap<2, int, int> queue(adjList.size());
    queue.update(ID1, 0);

    vector<int> vDistance(adjList.size(), INF);
    vector<int> vPrevious(adjList.size(), -1);
    vector<int> vPreviousEdge(adjList.size(), -1);
    vector<bool> vbVisited(adjList.size(), false);
    int topNodeID, neighborNodeID, neighborLength, neighborRoadID;

    vDistance[ID1] = 0;

    while(!queue.empty())
    {
        int topDistance;
        queue.extract_min(topNodeID, topDistance);
        vbVisited[topNodeID] = true;
        if(topNodeID == ID2)
            break;

        for(int i = 0; i < (int)adjList[topNodeID].size(); i++)
        {
            neighborNodeID = adjList[topNodeID][i].first;
            if(sRemovedNode.find(neighborNodeID) != sRemovedNode.end())
                continue;
            neighborLength = adjList[topNodeID][i].second;
            neighborRoadID = adjListEdge[topNodeID][i].second;
            int d = vDistance[topNodeID] + neighborLength;
            if(!vbVisited[neighborNodeID])
            {
                if(vDistance[neighborNodeID] > d)
                {
                    vDistance[neighborNodeID] = d;
                    queue.update(neighborNodeID, d);
                    vPrevious[neighborNodeID] = topNodeID;
                    vPreviousEdge[neighborNodeID] = neighborRoadID;
                }
            }
        }
    }

    vPath.clear();
    vPathEdge.clear();
    vPath.push_back(ID2);
    int p = vPrevious[ID2];
    int e = vPreviousEdge[ID2];
    while(p != -1)
    {
        vPath.push_back(p);
        vPathEdge.push_back(e);
        e = vPreviousEdge[p];
        p = vPrevious[p];
    }

//	if(vPathEdge.size() > 0)
//		vPathEdge.erase(vPathEdge.end()-1);

    reverse(vPath.begin(), vPath.end());
    reverse(vPathEdge.begin(), vPathEdge.end());

    /*cout << ID1 << "...." << ID2 << "...." << vDistance[ID2] << endl;
    for(auto it = sRemovedNode.begin(); it != sRemovedNode.end(); it++)
    {
        cout << *it << endl;
    }*/
    return vDistance[ID2];
}

int Graph::iBoundingAstar(int ID1, int ID2, unordered_set<int>& sRemovedNode, vector<int>& vPath, vector<int>& vPathEdge, int T)
{
    vector<int> vDistance(adjList.size(), INF);
    benchmark::heap<2, int, int> queue(adjList.size());
    vector<bool> vbVisited(adjList.size(), false);
    vector<int> vPrevious(adjList.size(), -1);
    vector<int> vPreviousEdge(adjList.size(), -1);
    int topNodeID, neighborNodeID, neighborLength, neighborRoadID;

    vDistance[ID1] = 0;
    queue.update(ID1, 0);

    while(!queue.empty())
    {
        int topDistance;
        queue.extract_min(topNodeID, topDistance);
        vbVisited[topNodeID] = true;
        if(topNodeID == ID2)
            break;
        for(int i = 0; i < (int)adjList[topNodeID].size(); i++)
        {
            neighborNodeID = adjList[topNodeID][i].first;
            if(sRemovedNode.find(neighborNodeID) != sRemovedNode.end())
                continue;
            neighborLength = adjList[topNodeID][i].second;
            neighborRoadID = adjListEdge[topNodeID][i].second;
            int d = vDistance[topNodeID] + neighborLength;
            if(!vbVisited[neighborNodeID])
            {
                if(vDistance[neighborNodeID] > d)
                {
                    vDistance[neighborNodeID] = d;
                    int lm = Landmark(adjList[topNodeID][i].first, ID2);
                    if( d + lm <= T)
                    {
                        queue.update(neighborNodeID, d + lm);
                        vPrevious[neighborNodeID] = topNodeID;
                        vPreviousEdge[neighborNodeID] = neighborRoadID;
                    }
                }
            }
        }
    }

    vPath.clear();
    vPathEdge.clear();
    vPath.push_back(ID2);
    int p = vPrevious[ID2];
    int e = vPreviousEdge[ID2];
    while(p != -1)
    {
        vPath.push_back(p);
        vPathEdge.push_back(e);
        e = vPreviousEdge[p];
        p = vPrevious[p];
    }
    reverse(vPath.begin(), vPath.end());
    reverse(vPathEdge.begin(), vPathEdge.end());

    return vDistance[ID2];
}
vector<int> Graph::Dijkstra(int ID1, int ID2)
{
	benchmark::heap<2, int, int> queue(adjList.size());
	queue.update(ID1, 0);

	vector<int> vDistance(nodeNum, INF);
	vector<bool> vbVisited(nodeNum, false);
	int topNodeID, neighborNodeID, neighborLength;
	vector<pair<int, int> >::iterator ivp;

	vDistance[ID1] = 0;
	countNum = 0;

	compareNode cnTop;
	while(!queue.empty())
	{
		int topDistance;
		queue.extract_min(topNodeID, topDistance);
		vbVisited[topNodeID] = true;
		countNum += 1;
        //cout << topNodeID << "\t" << vDistance[topNodeID] << endl;
		if(topNodeID == ID2)
			break;
		/*if(topDistance > 700000)
		{
			ID2 = topNodeID;
			cout << ID1+1 << " " << ID2+1 << endl;
			break;
		}*/
		for(ivp = adjList[topNodeID].begin(); ivp != adjList[topNodeID].end(); ivp++)
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
    //cout << "Count: " << countNum << endl;
	return vDistance;
}

vector<int> Graph::vDijkstra(int ID1)
{
    benchmark::heap<2, int, int> queue(adjList.size());
    queue.update(ID1, 0);

    vector<int> vDistance(nodeNum, INF);
    vector<bool> vbVisited(nodeNum, false);
    int topNodeID, neighborNodeID, neighborLength;
    vector<pair<int, int> >::iterator ivp;

    vDistance[ID1] = 0;
    countNum = 0;

    compareNode cnTop;
    while(!queue.empty())
    {
        int topDistance;
        queue.extract_min(topNodeID, topDistance);
        vbVisited[topNodeID] = true;
        countNum += 1;
        //cout << topNodeID << "\t" << vDistance[topNodeID] << endl;

        for(ivp = adjList[topNodeID].begin(); ivp != adjList[topNodeID].end(); ivp++)
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
    //cout << "Count: " << countNum << endl;
    return vDistance;
}

int Graph::AStar(int ID1, int ID2, vector<pair<double, double> >& vCoor)
{
	benchmark::heap<2, int, int> queue(adjList.size());
	vector<int> vDistance(adjList.size(), INF);
	vector<bool> vbVisited(adjList.size(), false);
	int topNodeID, neighborNodeID, neighborLength;
	vector<pair<int, int> >::iterator ivp;
	//cout << ID1 << " " << ID2 << endl;
	queue.update(ID1, EuclideanDistance_2(ID1, ID2, vCoor));

	countNum = 0;
	vDistance[ID1] = 0;

	compareNode cnTop;
	while(!queue.empty())
	{
		int topDistance;
		queue.extract_min(topNodeID, topDistance);
		vbVisited[topNodeID] = true;
		countNum += 1;
        //cout << topNodeID << "\t" << vDistance[topNodeID] << endl;
		if(topNodeID == ID2)
			break;

		for(ivp = adjList[topNodeID].begin(); ivp != adjList[topNodeID].end(); ivp++)
		{
			neighborNodeID = (*ivp).first;
			neighborLength = (*ivp).second;
            //cout << neighborNodeID << "," << neighborLength << endl;
			int d = vDistance[topNodeID] + neighborLength;
			//cout << vbVisited[neighborNodeID];
			if(!vbVisited[neighborNodeID])
			{
				if(vDistance[neighborNodeID] > d)
				{
				    //cout << neighborNodeID << " " << vDistance[neighborNodeID] << endl;
					vDistance[neighborNodeID] = d;
					queue.update(neighborNodeID, d+EuclideanDistance_2(neighborNodeID, ID2, vCoor));
				}
			}
		}
	}
    cout << "Count: " << countNum << endl;
	cout << "Distance:" << vDistance[ID2] << endl;
	return vDistance[ID2];
}

int Graph::AStarLandmark(int ID1, int ID2, vector<pair<double, double> >& vCoor)
{
    benchmark::heap<2, int, int> queue(adjList.size());
    vector<int> vDistance(adjList.size(), INF);
    vector<bool> vbVisited(adjList.size(), false);
    int topNodeID, neighborNodeID, neighborLength;
    vector<pair<int, int> >::iterator ivp;
    int countNum = 0;
    dNode.clear();
    //srand(time(NULL));
    for(int i = 0; i < 20; i++)
    {
        Node[i] = rand() % nodeNum;
        //cout << Node[i] << endl;
        dNode.push_back(Dijkstra(Node[i], 321269));
    }
    //cout << dNode[1][ID2] << endl;
    queue.update(ID1, Landmark(ID1, ID2));
    //queue.update(ID1, EuclideanDistance(ID1, ID2, vCoor));
    vDistance[ID1] = 0;

    compareNode cnTop;
    while(!queue.empty())
    {
        int topDistance;
        queue.extract_min(topNodeID, topDistance);
        vbVisited[topNodeID] = true;
        countNum += 1;
        //cout << "count:" << countNum << "\t" << topNodeID << "\t" << topDistance << "\t" << vDistance[topNodeID] << endl;
        if(topNodeID == ID2)
        {
            //cout << "Distance:" << vDistance[ID2] << endl;
            break;
        }


        for(ivp = adjList[topNodeID].begin(); ivp != adjList[topNodeID].end(); ivp++)
        {
            neighborNodeID = (*ivp).first;
            neighborLength = (*ivp).second;
            int d = vDistance[topNodeID] + neighborLength;
            if(!vbVisited[neighborNodeID]) {
                if (vDistance[neighborNodeID] > d) {
                    vDistance[neighborNodeID] = d;
                    queue.update(neighborNodeID, d + Landmark(neighborNodeID, ID2));
                    //queue.update(neighborNodeID, d + EuclideanDistance(neighborNodeID, ID2, vCoor));
                    //cout << "landmark:" << Landmark(neighborNodeID, ID2) << "\tEuc:"
                      //   << EuclideanDistance(neighborNodeID, ID2, vCoor) << endl;
                }
            }
        }
    }
    //cout << "Count" << countNum << endl;
    return vDistance[ID2];
}

int Graph::AStarLandmarkPath(int ID1, int ID2, vector<int>& vPath, vector<int>& vPathEdge)
{
	benchmark::heap<2, int, int> queue(adjList.size());
	vector<int> vDistance(adjList.size(), INF);
	vector<int> vPrevious(adjList.size(), -1);
	vector<int> vPreviousEdge(adjList.size(), -1);
	vector<bool> vbVisited(adjList.size(), false);
	int topNodeID, neighborNodeID, neighborLength, neighborRoadID;
    vector<pair<int, int> >::iterator ivp;

	/*int latU, lonU;
	if(city == "US")
	{
		lonU = 84000;
		latU = 69000;
	}
	else
	{
		lonU = 83907;
		latU = 111319; 
	}*/
    int countNum = 0;
    dNode.clear();
    //srand(time(NULL));
    for(int i = 0; i < 20; i++)
    {
        Node[i] = rand() % nodeNum;
        //cout << Node[i] << endl;
        dNode.push_back(Dijkstra(Node[i], 321269));
    }
    //cout << dNode[1][ID2] << endl;
    queue.update(ID1, Landmark(ID1, ID2));
	//queue.update(ID1, EuclideanDistance(ID1, ID2, vCoor));
	vDistance[ID1] = 0;

	compareNode cnTop;
	while(!queue.empty())
	{
		int topDistance;
		queue.extract_min(topNodeID, topDistance);
		vbVisited[topNodeID] = true;

		if(topNodeID == ID2)
			break;

        for(ivp = adjList[topNodeID].begin(); ivp != adjList[topNodeID].end(); ivp++)
        {
            neighborNodeID = (*ivp).first;
            neighborLength = (*ivp).second;
			int d = vDistance[topNodeID] + neighborLength;
			if(!vbVisited[neighborNodeID])
			{
				if(vDistance[neighborNodeID] > d)
				{
					vDistance[neighborNodeID] = d;
					queue.update(neighborNodeID, d + Landmark(neighborNodeID, ID2));
					vPrevious[neighborNodeID] = topNodeID;
				}
			}
		}
	}
	
	vPath.clear();
	vPathEdge.clear();
	vPath.push_back(ID2);
	int p = vPrevious[ID2];
	int e = vPreviousEdge[ID2];
	while(p != -1)
	{
		vPath.push_back(p);
		vPathEdge.push_back(e);
		e = vPreviousEdge[p];
		p = vPrevious[p];
	}

	reverse(vPath.begin(), vPath.end());
	reverse(vPathEdge.begin(), vPathEdge.end());

	return vDistance[ID2];
}

inline int Graph::EuclideanDistance(int ID1, int ID2, vector<pair<double, double> >& vCoor)
{
    //cout << ID1 << "," << ID2 << "," << vCoor[ID1].second << endl;
	//int lat=(int)(abs(vCoor[ID1].first - vCoor[ID2].first)*111319);
    int lon=(int)(abs(vCoor[ID1].first - vCoor[ID2].first)*880000);
	int lat=(int)(abs(vCoor[ID1].second - vCoor[ID2].second)*1110000);
	int min, max;
	min = (lat > lon) ? lon : lat;
	max = (lat > lon) ? lat : lon;
	int approx = max*1007 + min*441;
	if(max < (min << 4))
		approx -= max * 40;
	//cout << ID1 << "," << ID2 << "," << approx << endl;
	return (approx + 512) >> 10;
}

inline int Graph::EuclideanDistance_2(int ID1, int ID2, vector<pair<double, double> >& vCoor)
{
    int radius = 6371;
    double lon1 = vCoor[ID1].first*PI/180;
    double lon2 = vCoor[ID2].first*PI/180;
    double lat1 = vCoor[ID1].second*PI/180;
    double lat2 = vCoor[ID2].second*PI/180;
    double deltaLat = lat2 - lat1;
    double deltaLon = lon2 - lon1;
    double x = deltaLon * cos((lat1 + lat2)/2);
    double y = deltaLat;
    double dist = radius * sqrt(x * x + y * y)*10000;
    //cout << vCoor[ID1].first << "," << vCoor[ID1].second << "," << dist << endl;
    //cout << dist << endl;
    return dist;
}

int Graph::Landmark(int ID1, int ID2)
{
    int d1, d2, est;
    int max = -1;
    for(int i = 0; i < 20; i++)
    {
       // d1 = dNode[Node[i]][ID1];
       // d2 = dNode[Node[i]][ID2];

        d1 = dNode[i][ID1];
        d2 = dNode[i][ID2];
        est = abs(d1 - d2);
        if(est > max)
            max = est;
        //estList[i] = est;
        //cout << Node << "," << est << endl;
    }
    //int minValue =  *min_element(estList,estList+20);
    return max;
}

inline int Graph::EuclideanDistanceAdaptive(int ID1, int ID2, int latU, int lonU)
{
	int lat=(int)(abs(vCoor[ID1].first - vCoor[ID2].first)*latU);
	int lon=(int)(abs(vCoor[ID1].second - vCoor[ID2].second)*lonU);
	int min, max;
	min = (lat > lon) ? lon : lat;
	max = (lat > lon) ? lat : lon;
	int approx = max*1007 + min*441;
	if(max < (min << 4))
		approx -= max * 40;
	return (approx + 512) >> 10;
}

int Graph::ISONodes()
{
	srand (time(NULL));
	vbISOF.assign((int)adjList.size(), false);	//forward
	vbISOB.assign((int)adjList.size(), false);	//backward
	vbISOU.assign( (int)adjList.size(), false);	//F & B
	vbISO.assign((int)adjList.size(), false);	//F | B

	ifstream ifISOF("./beijingISOF");
	ifstream ifISOB("./beijingISOB");
	ifstream ifISOU("./beijingISOU");
	ifstream ifISO("./beijingISO");

	if(!ifISOF || !ifISOB || !ifISOU || !ifISO)
	{
		//ISO files do not exist
		//Create new ones
		srand(time(NULL));
		cout << "Identifying ISO Nodes" << endl;
		for(int i = 0; i < 10; i++)
		{
			int nodeID = rand() % adjList.size();
			cout <<nodeID <<endl;
			vector<bool> vbVisitedF;
			vector<bool> vbVisitedB;
			int vnF = BFS(nodeID, true, vbVisitedF);
			int vnB = BFS(nodeID, false, vbVisitedB);
			cout <<vnF <<"\t" << vnB <<endl;

			if(vnF < 1000 || vnB < 1000)
				continue;

			for(int j = 0; j < (int)adjList.size(); j++)
			{
				if(!vbVisitedF[j])
					vbISOF[j] = true;

				if(!vbVisitedB[j])
					vbISOB[j] = true;

				if(!vbVisitedF[j] || !vbVisitedB[j])
					vbISOU[j] = true;

				if(!vbVisitedF[j] && !vbVisitedB[j])
					vbISO[j] = true;
			}
		}

		ofstream ofF("./beijingISOF");
		ofstream ofB("./beijingISOB");
		ofstream ofU("./beijingISOU");
		ofstream of("./beijingISO");

		for(int i = 0; i < (int)adjList.size(); i++)
		{
			if(vbISOF[i])
				ofF << i << endl;

			if(vbISOB[i])
				ofB << i << endl;

			if(vbISOU[i])
				ofU << i << endl;

			if(vbISO[i])
				of << i << endl;
		}

		ofF.close();
		ofB.close();
		ofU.close();
		of.close();

		return 0;
	}
	else
	{
		int nodeID;
		cout << "Loading ISO Nodes" << endl;
		while(ifISOF >> nodeID)
			vbISOF[nodeID] = true;

		while(ifISOB >> nodeID)
			vbISOB[nodeID] = true;

		while(ifISOU >> nodeID)
			vbISOU[nodeID] = true;

		while(ifISO >> nodeID)
			vbISO[nodeID] = true;

		return 0;
	}
}

int Graph::BFS(int nodeID, bool bF, vector<bool>& vbVisited)
{
	vbVisited.assign(adjList.size()+1, false);
	queue<int> Q;
	Q.push(nodeID);
	vbVisited[nodeID] = true;
	int count = 0;
	while(!Q.empty())
	{
		int topNodeID = Q.front();
		Q.pop();
		count++;
		if(bF)
		{
			for(auto& it : adjList[topNodeID])
				if(!vbVisited[it.first])
				{
					vbVisited[it.first] = true;
					Q.push(it.first);
				}
		}
		else
		{
			for(auto& it : adjListR[topNodeID])
				if(!vbVisited[it.first])
				{
					vbVisited[it.first] = true;
					Q.push(it.first);
				}
		}
	}

	return count;
}


void Graph::SPT(int root, int ID2, vector<int>& vSPTDistance, vector<int>& vSPTParent,vector<int>& vSPTHeight, vector<int>& vSPTParentEdge, vector<vector<int> >& vSPT)
{
    int boundDistance = 999999999;
    benchmark::heap<2, int, int> queue(nodeNum);
    queue.update(root, 0);

    vector<bool> vbVisited(nodeNum, false);
    int topNodeID, neighborNodeID, neighborLength, neighborRoadID;
    vector<pair<int, int> >::iterator ivp;
    vSPTHeight[root] = 1;

    vSPTDistance[root] = 0;
    compareNode cnTop;
    while(!queue.empty())
    {
        int topDistance;
        queue.extract_min(topNodeID, topDistance);
        vbVisited[topNodeID] = true;
        if(topNodeID == ID2)
        {
            boundDistance = 1.2 * topDistance;
            //boundDistance = 50442;
            //cout << boundDistance << endl;
        }
        //boundDistance = 503442;
        if(topDistance > boundDistance)
            break;
        for(int i = 0; i < (int)adjList[topNodeID].size(); i++)
        {
            neighborNodeID = adjList[topNodeID][i].first;
            neighborLength = adjList[topNodeID][i].second;
            neighborRoadID = adjListEdge[topNodeID][i].second;
            int d = vSPTDistance[topNodeID] + neighborLength;
            if(!vbVisited[neighborNodeID])
            {
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
    for(int i = 0; i < nodeNum; i++){
        if(vSPTParent[i] != -1) {
            //cout << i << endl;
            vSPT[vSPTParent[i]].push_back(i);
        }
    }
}

vector<int> Graph::makeRMQDFS(int p, vector<vector<int> >& vSPT, vector<int>& vSPTParent){
    stack<int> sDFS;
    sDFS.push(p);
    vector<bool> vbVisited(nodeNum, false);
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
    rEulerSeq.assign(nodeNum,-1);
    //rEulerSeq.assign(EulerSeq.size(),-1);
    for(int i = 0; i < EulerSeq.size(); i++)
    {
        rEulerSeq[EulerSeq[i]] = i;
        //cout <<  EulerSeq[i] << "  " << rEulerSeq[EulerSeq[i]] << " "<< rEulerSeq.size() << endl;
    }
    //cout << rEulerSeq.size() << endl;
    return EulerSeq;
}

vector<vector<int>> Graph::makeRMQ(int p, vector<vector<int> >& vSPT, vector<int>& vSPTHeight, vector<int>& vSPTParent){
    EulerSeq.clear();
    toRMQ.assign(nodeNum,0);
    RMQIndex.clear();
    /*std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    t1 = std::chrono::high_resolution_clock::now();*/
    makeRMQDFS(p, vSPT,vSPTParent);
    /*t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);*/
    //cout << "Euler Time" << time_span.count() << endl;
    RMQIndex.push_back(EulerSeq);

    int m = EulerSeq.size();
    //t1 = std::chrono::high_resolution_clock::now();
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
    /* t2 = std::chrono::high_resolution_clock::now();
     time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);*/
    //cout << "Table Time" << time_span.count() << endl;
    return RMQIndex;
}

int Graph::LCAQuery(int _p, int _q, vector<vector<int> >& vSPT, vector<int>& vSPTHeight, vector<int>& vSPTParent){
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
        //cout << "1: " << vSPTHeight[RMQIndex[k][p]] << " 2: " << vSPTHeight[RMQIndex[k][q]] << endl;
        //cout << "k: " << k << " p: " << p << " q: " << q << endl;
        return RMQIndex[k][p];
    }
    else return RMQIndex[k][q];
}
