#ifndef GRAPH_H
#define GRAPH_H

#include "heap.h"
#include <unordered_map> 
#include <string>
#include <boost/thread/thread.hpp>
#include <unordered_set>
#include <stack>
#include <boost/functional/hash.hpp>

struct Graph{
    vector<vector<vector<int> > > vvGridNode;
    vector<vector<int> > vvGridLeftTop;
    vector<vector<int> > vvGridCentreNode;
    //vector<vector<vector<int> > > vvGridNode;
	int n;//node number
    vector<int> vvNodeGrid; //node belongs to which grid
	vector<vector<pair<int,int>>> Edges;
	vector<vector<pair<int,int>>> EdgesRe;
    vector<vector<pair<int, int> > > adjList;		//neighborID, Distance
    vector<vector<pair<int, int> > > adjListR;
    vector<vector<pair<int, int> > > adjListEdge;	//neighbor,edgeID
    vector<vector<pair<int, int> > > adjListEdgeR;
    struct Edge
    {
        int ID1, ID2, length, edgeID;
    };
    vector<Edge> vEdge;
	vector<pair<double,double>> Locat;
	double minX = 1000, minY = 1000, maxX = 0, maxY = 0;
	Graph(){
		n=0;
		Edges.clear();
		EdgesRe.clear();
		Locat.clear();
	}
	Graph(string filename){
        if(filename == "/Users/angel/CLionProjects/ZIGZAG/beijing")
        {
            ifstream in(filename);
            string line;

            if(in){
                int linenum=-1;

                while(getline(in,line)){
                    int u,v,e;//directed edge(u,v) and corresponding length e
                    linenum+=1;
                    if(linenum==0){
                        vector<string> vstr;
                        boost::split(vstr,line,boost::is_any_of("	"),boost::token_compress_on);
                        n=stoi(vstr[0]);
                        vector<pair<int,int>> v; v.clear();
                        Edges.assign(n, v); EdgesRe.assign(n, v);
                        minY = stod(vstr[1]);
                        maxY = stod(vstr[2]);
                        minX = stod(vstr[3]);
                        maxX = stod(vstr[4]);

                    }else{
                        vector<string> vstr;
                        boost::split(vstr,line,boost::is_any_of("	"),boost::token_compress_on);

                        double lat=stod(vstr[2]);
                        double lon=stod(vstr[3]);
                        Locat.push_back(make_pair(lat,lon));

                        u=linenum-1;
                        int adjnum=stoi(vstr[6]);

                        for(int i=0;i<adjnum;i++){
                            v = stoi(vstr[7+2*i]);
                            e = stoi(vstr[8+2*i]);
                            Edges[u].push_back(make_pair(v, e));
                            EdgesRe[v].push_back(make_pair(u, e));
                        }
                    }
                }
            }else{
                cout<<"no such file"<<endl;
            }
            cout<<"Graph reading finished "<<endl;
        }
        else{
            ifstream in(filename);
            string line;
            if(in){
                vector<pair<int,int>> v; v.clear();
                do
                {
                    getline(in,line);
                    if(line[0]=='p')
                    {
                        vector<string> vs;
                        boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
                        n = stoi(vs[2]);
                        Edges.assign(n, v);
                        EdgesRe.assign(n, v);
                        cout << "Nodenum " << n <<endl;
                    }
                }while(line[0]=='c'|| line[0]=='p');

                adjList.resize(n);
                adjListR.resize(n);
                adjListEdge.resize(n);
                adjListEdgeR.resize(n);
                int edgeCount = 0;
                //string a;
                while(!in.eof()) {
                    int u, v, l;//directed edge(u,v) and corresponding length e
                    vector<string> vs;
                    boost::split(vs, line, boost::is_any_of(" "), boost::token_compress_on);

                    u = stoi(vs[1]) - 1;
                    v = stoi(vs[2]) - 1;
                    l = stoi(vs[3]);

                    bool bExisit = false;
                    for (int i = 0; i < (int) Edges[u].size(); i++) {
                        if (Edges[u][i].first == v) {
                            bExisit = true;
                            break;
                        }
                    }
                    struct Edge e;
                    e.ID1 = u;
                    e.ID2 = v;
                    e.length = l;
                    e.edgeID = edgeCount;

                    if (!bExisit) {
                        vEdge.push_back(e);
                        adjList[u].push_back(make_pair(v, l));
                        adjListR[v].push_back(make_pair(u, l));
                        adjListEdge[u].push_back(make_pair(v, edgeCount));
                        adjListEdgeR[v].push_back(make_pair(u, edgeCount));
                        Edges[u].push_back(make_pair(v, l));
                        EdgesRe[v].push_back(make_pair(u, l));
                        edgeCount++;
                    }
                    getline(in,line);
                }

            }else{
                cout<<"no such file"<<endl;
            }

            in.close();

            string coorFile = filename.substr(0, filename.size()-2)+"co";
            ifstream inCoor(coorFile.c_str());
            if(!inCoor)
            cout << "Cannot open Coordinate " << coorFile << endl;
            cout << "Reading " << coorFile << endl;

            do
            {
                getline(inCoor,line);
            }while(line[0]=='c'|| line[0]=='p');

            double lon, lat;

            while(!inCoor.eof())
            {
                //First lat second lon
                //NY 1 lon 31km. 1 lat 111km
                //NY 1 lon 84km. 1 lat 111km
                //COL 1 lon 86km. 1 lat 111km
                //x: lon
                vector<string> vs;
                boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
                //NY
                lon = stod(vs[2].substr(0,3) + "." + vs[2].substr(3));
                lon = -lon;
                lat = stod(vs[3].substr(0,2) + "." + vs[3].substr(2));

               //COL
                /*lon = stod(vs[2].substr(0,4) + "." + vs[2].substr(4));
                lon = -lon;
                lat = stod(vs[3].substr(0,2) + "." + vs[3].substr(2));*/

                //Manhattan
                /*lon = stod(vs[2].substr(0,10));
                lon = -lon;
                //lat = stod(vs[3].substr(0,2) + "." + vs[3].substr(2));
                lat = stod(vs[3].substr(0,9));*/

                if(lon < minX)
                    minX = lon;
                if(lat < minY)
                    minY = lat;
                if(lon > maxX)
                    maxX = lon;
                if(lat > maxY)
                    maxY = lat;

                //lat = stod(vs[2].substr(0,3) + "." + vs[2].substr(3));

                //COL
                /*lon = stod(vs[2].substr(0,4) + "." + vs[2].substr(4));
                lon = -lon;
                lat = stod(vs[3].substr(0,2) + "." + vs[3].substr(2));*/
                /*if(lat < minX)
                    minX = lat;
                if(lon < minY)
                    minY = lon;
                if(lat > maxX)
                    maxX = lat;
                if(lon > maxY)
                    maxY = lon;*/

                //cout << "lon: " << lon << " lat: " << lat << endl;
                Locat.push_back(make_pair(lat, lon));
                getline(inCoor,line);
            }
            cout << "minX: " << minX << " minY: " << minY << " maxX: " << maxX << " maxY: " << maxY << endl;
            //cout << Locat[0].first << endl;
            cout<<"Graph reading finished "<<endl;
        }
	}

	int Eucli(int s, int t){//Euclidean distance computation
		/*int lat=(int)(abs(Locat[s].first-Locat[t].first)*111319);
		int lon=(int)(abs(Locat[s].second-Locat[t].second)*83907);*/

        //NY
        //COL 1 lon 86km. 1 lat 111km
        int lat=(int)(abs(Locat[s].first-Locat[t].first)*1110000);
        int lon=(int)(abs(Locat[s].second-Locat[t].second)*860000);
        //int lon=(int)(abs(Locat[s].second-Locat[t].first)*310000);
		int min,max;
		min=(lat>lon)?lon:lat;
		max=(lat>lon)?lat:lon;
		int approx=max*1007+min*441;
		if(max<(min<<4))
			approx-=max*40;
		return (approx+512)>>10;
	}

	double Angle(int s, int t){//angle computation
		double lat=Locat[t].first-Locat[s].first;
		double lon=Locat[t].second-Locat[s].second;
        //cout << "lat: " << lat << endl;
        //cout << "lon: " << lon << endl;


		double absAngle=atan(abs(1.3267*lat/lon));
		double trueAngle=0;
		if(lat>=0&&lon>0)
			trueAngle=absAngle;
		else if(lat>0&&lon<=0)
			trueAngle=PI-absAngle;
		else if(lat<=0&&lon<0)
			trueAngle=PI+absAngle;
		else if(lat<0&&lon>=0)
			trueAngle=2*PI-absAngle;

		return trueAngle;
	}

	struct DisQueue{//distance maximum computation
		int nodeid;
		double dis;
		double ang;
		int qID;
		DisQueue(){
		}
		DisQueue(int _nodeid, double _dis, double _ang){
			nodeid = _nodeid;
			dis = _dis;
			ang = _ang;
		}
		DisQueue(int _nodeid, double _dis, double _ang, int _qID){
			nodeid = _nodeid;
			dis = _dis;
			ang = _ang;
			qID = _qID;
		}
		//maximum heap
		bool operator < (const DisQueue _disqueue) const{
			if (dis == _disqueue.dis)
				return nodeid < _disqueue.nodeid;
			return dis < _disqueue.dis;
		}
	};

	void Petal(int s, set<int>& T, bool ForB, vector<vector<int>>& paths){
		//Target set distance-based decompose
		priority_queue<DisQueue> q;
		while(!q.empty())
			q.pop();
		set<int>::iterator itset=T.begin();
		for(;itset!=T.end();itset++)
			q.push(DisQueue(*itset, Eucli(s, *itset), Angle(s, *itset)));

		vector<pair<int,vector<int>>> clusters; clusters.clear();//the farthest node : the target set
		vector<double> cluAng; double angleThresh = PI/3;
		vector<pair<int,double>> NotAddClu; NotAddClu.clear();//nodeID and angle of notAddedNode
		while(!q.empty()){
			double topAng = q.top().ang; int topNode = q.top().nodeid; q.pop();
			bool addClu = false;
			double miniAnDif = 2*PI;//the minimum angle different with current clusters
			for(int i=0;i<cluAng.size();i++){
				miniAnDif = (miniAnDif> abs(topAng-cluAng[i])) ? abs(topAng-cluAng[i]) : miniAnDif;
				if(miniAnDif <= 0.5*angleThresh){
					clusters[i].second.push_back(topNode);
					addClu = true;
					break;
				}
			}
			if(!addClu){
				if(miniAnDif >= angleThresh){
					vector<int> newcluster; newcluster.clear();
					newcluster.push_back(topNode);
					cluAng.push_back(topAng);
					clusters.push_back(make_pair(topNode, newcluster));
				}
				else
					NotAddClu.push_back(make_pair(topNode, topAng));
			}
		}
		for(int i=0;i<NotAddClu.size();i++){//put the "notaddnode" into the cluster with minimum angle difference
			double a = NotAddClu[i].second; int nid = NotAddClu[i].first;
			double miniAndif = 2*PI;
			int clusterSeq=0;
			for(int j=0;j<cluAng.size();j++){
				miniAndif = (miniAndif-abs(cluAng[j]-a)) ? abs(cluAng[j]-a) : miniAndif;
				clusterSeq = j;
			}
			clusters[clusterSeq].second.push_back(nid);
		}//distance_based decomposition finished
		//cout<<"clustering finished and clusters' size "<<clusters.size()<<endl;


		//multiple 1-N
		if(ForB){//Forward 1-N
			benchmark::heap<2, int, int> queue(n);
			vector<bool> closed; closed.assign(n, false);//closed or not
			vector<int> actuDis; actuDis.assign(n, INF);//true distance from s
			vector<int> parent; parent.assign(n, -1);
			queue.update(s, 0);//initiate
			actuDis[s]=0;

			for(int c=0;c<clusters.size();c++){
				int curRepre = clusters[c].first; vector<int> curSet = clusters[c].second;
				set<int> ToBeProcess; ToBeProcess.clear();//remain to be process
				for(int i=0;i<curSet.size();i++){
					if(!closed[curSet[i]])
						ToBeProcess.insert(curSet[i]);
				}
				if(ToBeProcess.size()!=0){//redirect first
					vector<int> nodeinHeap; nodeinHeap.clear(); queue.elementsInHeap(nodeinHeap);
					for(int j=0;j<nodeinHeap.size();j++)
						queue.update(nodeinHeap[j], actuDis[nodeinHeap[j]]+Eucli(s, curRepre));
				}
				while(!queue.empty()){
					int topid; int topvalue; queue.extract_min(topid, topvalue);
					closed[topid] = true;
					if(ToBeProcess.find(topid)!=ToBeProcess.end())
						ToBeProcess.erase(topid);

					vector<pair<int,int>>::iterator it=Edges[topid].begin();
					int NNodeID=0;
					int NWeigh=0;
					for(;it!=Edges[topid].end();it++){
						NNodeID = (*it).first; NWeigh = (*it).second+actuDis[topid];
						if(!closed[NNodeID]){
							if(actuDis[NNodeID]>NWeigh){
								actuDis[NNodeID]=NWeigh;
								queue.update(NNodeID, NWeigh+Eucli(s,curRepre));
								parent[NNodeID]=topid;
							}
						}
					}
					if(ToBeProcess.size()==0) break;
				}
			}
			//return paths
			set<int>::iterator setit=T.begin();
			for(;setit!=T.end();setit++){
				//cout<<s<<" to "<<*setit<<" "<<actuDis[*setit]<<endl;
				vector<int> path; path.clear(); int t=*setit;
				while(t!=s){path.push_back(t); t=parent[t];} path.push_back(s);
				paths.push_back(path);
			}
		}else{//Backward 1-N
			benchmark::heap<2, int, int> queue(n);
			vector<bool> closed; closed.assign(n, false);//closed or not
			vector<int> actuDis; actuDis.assign(n, INF);//true distance from s
			vector<int> parent; parent.assign(n,-1);
			queue.update(s, 0);//initiate
			actuDis[s]=0;

			for(int c=0;c<clusters.size();c++){
				int curRepre = clusters[c].first; vector<int> curSet = clusters[c].second;
				set<int> ToBeProcess; ToBeProcess.clear();//remain to be process
				for(int i=0;i<curSet.size();i++){
					if(!closed[curSet[i]])
						ToBeProcess.insert(curSet[i]);
				}
				if(ToBeProcess.size()!=0){//redirect first
					vector<int> nodeinHeap; nodeinHeap.clear(); queue.elementsInHeap(nodeinHeap);
					for(int j=0;j<nodeinHeap.size();j++)
						queue.update(nodeinHeap[j], actuDis[nodeinHeap[j]]+Eucli(s, curRepre));
				}
				while(!queue.empty()){
					int topid; int topvalue; queue.extract_min(topid, topvalue);
					closed[topid] = true;
					if(ToBeProcess.find(topid)!=ToBeProcess.end())
						ToBeProcess.erase(topid);

					vector<pair<int,int>>::iterator it=EdgesRe[topid].begin();
					int NNodeID=0;
					int NWeigh=0;
					for(;it!=EdgesRe[topid].end();it++){
						NNodeID = (*it).first; NWeigh = (*it).second+actuDis[topid];
						if(!closed[NNodeID]){
							if(actuDis[NNodeID]>NWeigh){
								actuDis[NNodeID]=NWeigh;
								queue.update(NNodeID, NWeigh+Eucli(s,curRepre));
								parent[NNodeID]=topid;
							}
						}
					}
					if(ToBeProcess.size()==0) break;
				}
			}
			//return paths
			set<int>::iterator setit=T.begin();
			for(;setit!=T.end();setit++){
				//cout<<*setit<<" to "<<s<<" "<<actuDis[*setit]<<endl;
				vector<int> path; path.clear(); int t=*setit;
				while(t!=s){path.push_back(t); t=parent[t];} path.push_back(s);
				paths.push_back(path);
			}
		}
	}

	struct PT{
		int dis;//query size
		int x;//nodeID
		bool ForB;//forward(true) or backward(false)
		PT(){
		}
		PT(int _x, int _dis, bool _ForB){
			dis = _dis;
			x = _x;
			ForB = _ForB;
		}
		// maximum heap
		bool operator < (const PT _pt) const{
			if (dis == _pt.dis)
				return x < _pt.x;
			return dis < _pt.dis;
		}
	};

	void ZigzagPetal(vector<pair<int,int>>& Q){
		vector<int> fm,bm;//the start/target id : the sequence number
		vector<int> fseq,bseq;//the start/target id
		vector<set<int>> Fset,Bset;
		fseq.clear(); bseq.clear(); Fset.clear(); Bset.clear(); fm.assign(n,-1); bm.assign(n,-1);

		//load the query data in Q
		vector<pair<int,int>>::iterator qit = Q.begin();
		int ori, des;
		set<int> oriSet, desSet;
		for(;qit!=Q.end();qit++){
			ori = (*qit).first;
			des = (*qit).second;
			if(fm[ori]==-1){// if not origin exist
				fm[ori] = Fset.size();
				desSet.clear(); desSet.insert(des);
				Fset.push_back(desSet);
				fseq.push_back(ori);
			}else{
				Fset[fm[ori]].insert(des);
			}

			if(bm[des]==-1){// if not destination exist
				bm[des] = Bset.size();
				oriSet.clear(); oriSet.insert(ori);
				Bset.push_back(oriSet);
				bseq.push_back(des);
			}else{
				Bset[bm[des]].insert(ori);
			}
		}//query set finish loading

		//maximum priority queue
		priority_queue<PT> Queue;//1-N query size
		while (!Queue.empty())
			Queue.pop();

		for(int i=0;i<fseq.size();i++)
			Queue.push(PT(fseq[i], Fset[i].size(), true));
		for(int j=0;j<bseq.size();j++)
			Queue.push(PT(bseq[j], Bset[j].size(), false));

		while(!Queue.empty()){
			set<int> desSet, oriSet;

			int CurrentQuerySize;
			int NodeID=Queue.top().x;
			bool direction=Queue.top().ForB;
			Queue.pop();

			if(direction){//forward Process
				desSet.clear(); desSet = Fset[fm[NodeID]];
				CurrentQuerySize = desSet.size(); if(CurrentQuerySize==0) continue;
				if(CurrentQuerySize >= Queue.top().dis){//continue process
					vector<vector<int>> paths; vector<int> v; v.clear(); paths.assign(CurrentQuerySize,v);
					Petal(NodeID, desSet, true, paths);
					//redundant deletion
					set<int>::iterator itf=desSet.begin();
					for(;itf!=desSet.end();itf++)
						Bset[bm[*itf]].erase(NodeID);
				}else
					Queue.push(PT(NodeID, CurrentQuerySize, true));
			}else{//backward process
				oriSet.clear(); oriSet = Bset[bm[NodeID]];
				CurrentQuerySize = oriSet.size(); if(CurrentQuerySize==0) continue;
				if(CurrentQuerySize >= Queue.top().dis){//continue process
					vector<vector<int>> paths; vector<int> v; v.clear(); paths.assign(CurrentQuerySize,v);
					Petal(NodeID, oriSet, false, paths);
					//redundant deletion
					set<int>::iterator itb=oriSet.begin();
					for(;itb!=oriSet.end();itb++)
						Fset[fm[*itb]].erase(NodeID);

				}else
					Queue.push(PT(NodeID, CurrentQuerySize, false));
			}
			//cout<<"Q size after one time process "<<Queue.size()<<endl;
		}//all queries processed
	}

	void DisDecomp(int s, vector<int>& T, vector<pair<int,set<int>>>& clusters){//representative node : sub-cluster
		priority_queue<DisQueue> q;
		while(!q.empty())
			q.pop();
		vector<int>::iterator itset=T.begin();
		for(;itset!=T.end();itset++)
			q.push(DisQueue(*itset, Eucli(s, *itset), Angle(s, *itset)));

		vector<double> cluAng; double angleThresh = PI/3;
		vector<pair<int,double>> NotAddClu; NotAddClu.clear();//nodeID and angle of notAddedNode
		while(!q.empty()){
			double topAng = q.top().ang; int topNode = q.top().nodeid; q.pop();
			bool addClu = false;
			double miniAnDif = 2*PI;//the minimum angle different with current clusters
			for(int i=0;i<cluAng.size();i++){
				miniAnDif = (miniAnDif> abs(topAng-cluAng[i])) ? abs(topAng-cluAng[i]) : miniAnDif;
				if(miniAnDif <= 0.5*angleThresh){
					clusters[i].second.insert(topNode);
					addClu = true;
					break;
				}
			}
			if(!addClu){
				if(miniAnDif >= angleThresh){
					set<int> newcluster; newcluster.clear();
					newcluster.insert(topNode);
					cluAng.push_back(topAng);
					clusters.push_back(make_pair(topNode, newcluster));
				}
				else
					NotAddClu.push_back(make_pair(topNode, topAng));
			}
		}
		for(int i=0;i<NotAddClu.size();i++){//put the "notaddnode" into the cluster with minimum angle difference
			double a = NotAddClu[i].second; int nid = NotAddClu[i].first;
			double miniAndif = 2*PI;
			int clusterSeq=0;
			for(int j=0;j<cluAng.size();j++){
				miniAndif = (miniAndif-abs(cluAng[j]-a)) ? abs(cluAng[j]-a) : miniAndif;
				clusterSeq = j;
			}
			clusters[clusterSeq].second.insert(nid);
		}//distance_based decomposition finished
	}

	void DisDecomp2(int s, vector<int>& T, vector<pair<int,set<int>>>& clusters, vector<int>& vQ){//representative node : sub-cluster
		priority_queue<DisQueue> q;
		while(!q.empty())
			q.pop();
	//	vector<int>::iterator itset=T.begin();
		for(int i = 0; i < (int)T.size(); i++)
			q.push(DisQueue(T[i], Eucli(s, T[i]), Angle(s, T[i]), vQ[i]));

		vector<double> cluAng; double angleThresh = PI/3;   
		vector<pair<int,double>> NotAddClu; NotAddClu.clear();//nodeID and angle of notAddedNode
		while(!q.empty()){
			double topAng = q.top().ang;  
			int topNode = q.top().nodeid; 
			int topQID = q.top().qID;
			q.pop();
			bool addClu = false;
			double miniAnDif = 2*PI;//the minimum angle different with current clusters
			for(int i=0;i<cluAng.size();i++){
				miniAnDif = (miniAnDif> abs(topAng-cluAng[i])) ? abs(topAng-cluAng[i]) : miniAnDif;
				if(miniAnDif <= 0.5*angleThresh){
				//	clusters[i].second.insert(topNode);
					clusters[i].second.insert(topQID);
					addClu = true;
					break;
				}
			}
			if(!addClu){
				if(miniAnDif >= angleThresh){
					set<int> newcluster; newcluster.clear();
				//	newcluster.insert(topNode);
					newcluster.insert(topQID);
					cluAng.push_back(topAng);
				//	clusters.push_back(make_pair(topNode, newcluster));
					clusters.push_back(make_pair(topQID, newcluster));
				}
				else
					NotAddClu.push_back(make_pair(topQID, topAng)); 
				//	NotAddClu.push_back(make_pair(topNode, topAng));
			}
		}
		for(int i=0;i<NotAddClu.size();i++){//put the "notaddnode" into the cluster with minimum angle difference
			double a = NotAddClu[i].second; 
			int nid = NotAddClu[i].first;
			double miniAndif = 2*PI;
			int clusterSeq=0;
			for(int j=0;j<cluAng.size();j++){
				miniAndif = (miniAndif-abs(cluAng[j]-a)) ? abs(cluAng[j]-a) : miniAndif;
				clusterSeq = j;
			}
			clusters[clusterSeq].second.insert(nid);
		}//distance_based decomposition finished
	}

	void Astar12N(int s,int rep, set<int>& T, bool direction, vector<vector<int>>& paths){
		benchmark::heap<2, int, int> queue(n);
		vector<bool> closed; closed.assign(n,false); vector<int> actuDis; actuDis.assign(n,INF); vector<int> parent; parent.assign(n,-1);
		queue.update(s, Eucli(s, rep)); actuDis[s]=0;
		int processed=0;

		if(direction){
			while(!queue.empty()){
				int nodeid; int topvalue; queue.extract_min(nodeid, topvalue);
				closed[nodeid]=true;
				if(T.find(nodeid)!=T.end()) processed+=1;
				if(processed==T.size()) break;
				int id; int w;
				vector<pair<int,int>>::iterator it=Edges[nodeid].begin();
				for(;it!=Edges[nodeid].end();it++){
					id=(*it).first; w=(*it).second+actuDis[nodeid];
					if(!closed[id]){
						if(actuDis[id]>w){
							actuDis[id]=w;
							queue.update(id,w+Eucli(id,rep));
							parent[id]=nodeid;
						}
					}
				}
			}

			set<int>::iterator setit=T.begin();
			for(;setit!=T.end();setit++){
				//cout<<s<<" to "<<*setit<<" is "<<actuDis[*setit]<<endl;
				vector<int> path; path.clear(); int t=*setit;
				while(t!=s){path.push_back(t); t=parent[t];} path.push_back(s);
				paths.push_back(path);
			}
		}else{
			while(!queue.empty()){
				int nodeid; int topvalue; queue.extract_min(nodeid, topvalue);
				closed[nodeid]=true;
				if(T.find(nodeid)!=T.end()) processed+=1;
				if(processed==T.size()) break;
				int id; int w;
				vector<pair<int,int>>::iterator it=EdgesRe[nodeid].begin();
				for(;it!=EdgesRe[nodeid].end();it++){
					id=(*it).first; w=(*it).second+actuDis[nodeid];
					if(!closed[id]){
						if(actuDis[id]>w){
							actuDis[id]=w;
							queue.update(id,w+Eucli(id,rep));
							parent[id]=nodeid;
						}
					}
				}
			}
			//return paths
			set<int>::iterator setit=T.begin();
			for(;setit!=T.end();setit++){
				//cout<<*setit<<" to "<<s<<" is "<<actuDis[*setit]<<endl;
				vector<int> path; path.clear(); int t=*setit;
				while(t!=s){path.push_back(t); t=parent[t];} path.push_back(s);
				paths.push_back(path);
			}
		}
	}

	struct PTC{
		int dis;//query size
		int x;//nodeID
		bool ForB;//forward(true) or backward(false)
		int seq;//the sequence
		PTC(){
		}
		PTC(int _x, int _dis, bool _ForB,int _seq){
			dis = _dis;
			x = _x;
			ForB = _ForB;
			seq = _seq;
		}
		// maximum heap
		bool operator < (const PTC _pt) const{
			if (dis == _pt.dis)
				return x < _pt.x;
			return dis < _pt.dis;
		}
	};

	void ZigzagCluster(vector<pair<int,int>>& Q){
		vector<int> fm,bm;//the start/target id : the sequence number
		vector<int> fseq,bseq;//the start/target id
		vector<vector<int>> Fset,Bset;
		fseq.clear(); bseq.clear(); Fset.clear(); Bset.clear(); fm.assign(n,-1); bm.assign(n,-1);

		//load the query data in Q
		vector<pair<int,int>>::iterator qit = Q.begin();
		int ori, des;
		vector<int> oriV, desV;
		for(;qit!=Q.end();qit++){
			ori = (*qit).first;
			des = (*qit).second;
			if(fm[ori]==-1){// if not origin exist
				fm[ori] = Fset.size();
				desV.clear(); desV.push_back(des);
				Fset.push_back(desV);
				fseq.push_back(ori);
			}else{
				Fset[fm[ori]].push_back(des);
			}

			if(bm[des]==-1){// if not destination exist
				bm[des] = Bset.size();
				oriV.clear(); oriV.push_back(ori);
				Bset.push_back(oriV);
				bseq.push_back(des);
			}else{
				Bset[bm[des]].push_back(ori);
			}
		}//query set finish loading

		//decompose all the query sets
		vector<pair<int,int>> fseqC,bseqC; fseqC.clear(); bseqC.clear();
		vector<set<int>> FsetC,BsetC; FsetC.clear(); BsetC.clear();
		vector<vector<int>> fmC,bmC; vector<int> vempty; vempty.clear(); fmC.assign(n,vempty); bmC.assign(n,vempty);
		priority_queue<PTC> QueueC;
		while(!QueueC.empty())
			QueueC.pop();
		for(int i=0;i<fseq.size();i++){
			vector<pair<int,set<int>>> clusters; clusters.clear();
			DisDecomp(fseq[i],Fset[i],clusters);
			for(int j=0;j<clusters.size();j++){
				fmC[fseq[i]].push_back(fseqC.size());
				QueueC.push(PTC(fseq[i], clusters[j].second.size(), true, fseqC.size()));
				fseqC.push_back(make_pair(fseq[i],clusters[j].first));
				FsetC.push_back(clusters[j].second);
			}
		}
		for(int i=0;i<bseq.size();i++){
			vector<pair<int,set<int>>> clusters; clusters.clear();
			DisDecomp(bseq[i], Bset[i], clusters);
			for(int j=0;j<clusters.size();j++){
				bmC[bseq[i]].push_back(bseqC.size());
				QueueC.push(PTC(bseq[i], clusters[j].second.size(), false, bseqC.size()));
				bseqC.push_back(make_pair(bseq[i],clusters[j].first));
				BsetC.push_back(clusters[j].second);
			}
		}

		while(!QueueC.empty()){
			int NodeID=QueueC.top().x; bool direction=QueueC.top().ForB; int sequence=QueueC.top().seq;
			int CurrentSize; QueueC.pop();

			if(direction){
				CurrentSize = FsetC[sequence].size(); if(CurrentSize==0) continue;
				if(CurrentSize>=QueueC.top().dis){
					vector<vector<int>> paths; vector<int> v; v.clear(); paths.assign(CurrentSize,v);
					Astar12N(NodeID, fseqC[sequence].second, FsetC[sequence], direction, paths);
					//delete redundancy
					set<int>::iterator sit=FsetC[sequence].begin();
					for(;sit!=FsetC[sequence].end();sit++){
						for(int i=0;i<bmC[*sit].size();i++){
							if(BsetC[bmC[*sit][i]].find(NodeID)!=BsetC[bmC[*sit][i]].end()){
								BsetC[bmC[*sit][i]].erase(NodeID); break;
							}
						}
					}
				}else
					QueueC.push(PTC(NodeID, CurrentSize, true, sequence));
			}else{
				CurrentSize = BsetC[sequence].size(); if(CurrentSize==0) continue;
				if(CurrentSize>=QueueC.top().dis){
					vector<vector<int>> paths; vector<int> v; v.clear(); paths.assign(CurrentSize,v);
					Astar12N(NodeID, bseqC[sequence].second, BsetC[sequence], direction, paths);
					//delete redundancy
					set<int>::iterator sit=BsetC[sequence].begin();
					for(;sit!=BsetC[sequence].end();sit++){
						for(int i=0;i<fmC[*sit].size();i++){
							if(FsetC[fmC[*sit][i]].find(NodeID)!=FsetC[fmC[*sit][i]].end()){
								FsetC[fmC[*sit][i]].erase(NodeID); break;
							}
						}
					}
				}else
					QueueC.push(PTC(NodeID, CurrentSize, false, sequence));
			}
			//cout<<"Q size after one time process "<<QueueC.size()<<endl;
		}
	}

	bool ReachabilityTest(int s, int t){
		queue<int> queue; vector<bool> colored; colored.assign(n, false);
		if(!queue.empty())
			queue.pop();
		queue.push(s); colored[s]=true;
		while(!queue.empty()){
			int top = queue.front(); queue.pop();
			vector<pair<int,int>>::iterator itE=Edges[top].begin();
			for(;itE!=Edges[top].end();itE++){
				int neighbor = (*itE).first;
				if(!colored[neighbor]){
					colored[neighbor]=true; queue.push(neighbor);
				}
			}
		}
		return colored[t];
	}
	
	int Astar(int s,int rep, vector<int>& path)
	{
		benchmark::heap<2, int, int> queue(n);
		vector<bool> closed;
		vector<int> actuDis, parent; 
		
		closed.assign(n,false); 
		actuDis.assign(n,INF); 
		parent.assign(n,-1);
	
		queue.update(s, Eucli(s, rep)); 
		actuDis[s]=0;

		int nodeid, topvalue, id, w; 
		while(!queue.empty())
		{
			queue.extract_min(nodeid, topvalue);
			closed[nodeid]=true;
			
			if(nodeid == rep) 
				break;

			vector<pair<int,int>>::iterator it=Edges[nodeid].begin();
		
			for(;it!=Edges[nodeid].end();it++)
			{
				id = (*it).first; 
				w = (*it).second + actuDis[nodeid];
				if(!closed[id])
				{
					if(actuDis[id] > w)
					{
						actuDis[id]=w;
						queue.update(id,w+Eucli(id,rep));
						parent[id]=nodeid;
					}
				}
			}
		}

		path.clear(); 
		int t = rep;
		while(t != s)
		{
			path.push_back(t); 
			t=parent[t];
		} 
		path.push_back(s); 

		return actuDis[rep];
	}
	
	void DijR(int s, vector<int>& vDestination, vector<int>& vDistance, bool bF, int radius) 
	{
		benchmark::heap<2, int, int> queue(n);
		vector<bool> closed;
		vector<int> actuDis, parent; 
		
		closed.assign(n,false); 
		actuDis.assign(n,INF); 
		parent.assign(n,-1);

		vDistance.assign(vDestination.size(), INF);
	
		queue.update(s, 0); 
		actuDis[s]=0;
		int count = 0;

		//id, pos
		unordered_map<int, int> mDestination;
		for(int i = 0; i < vDestination.size(); i++)
			mDestination[vDestination[i]] = i;

		int nodeid, topvalue, id, w; 
		while(!queue.empty())
		{
			queue.extract_min(nodeid, topvalue);
			closed[nodeid]=true;
			
			if(topvalue > 2*radius)
				break;

			if(mDestination.find(nodeid) != mDestination.end()) 
			{
				vDistance[mDestination[nodeid]] = actuDis[nodeid];
				count++;
			}

			if(count == mDestination.size()) 
				break;

		
			if(bF)
			{
				for(auto it = Edges[nodeid].begin(); it != Edges[nodeid].end(); it++)
				{
					id = (*it).first; 
					w = (*it).second+actuDis[nodeid];
	
					if(!closed[id])
					{
						if(actuDis[id]>w)
						{
							actuDis[id] = w;
							queue.update(id, w);
							parent[id] = nodeid;
						}
					}
				}
			}
			else
			{
				for(auto it = EdgesRe[nodeid].begin(); it != EdgesRe[nodeid].end(); it++)
				{
					id = (*it).first; 
					w = (*it).second + actuDis[nodeid];
	
					if(!closed[id])
					{
						if(actuDis[id]>w)
						{
							actuDis[id]=w;
							queue.update(id, w);
							parent[id]=nodeid;
						}
					}
				}
			}
		}

		for(int i = 0; i < vDistance.size(); i++)
		{
			if(vDistance[i] != INF)
			{
				int t = vDestination[i];  
				vector<int> path;
				path.reserve(1000);
				while(t != s)
				{
					path.push_back(t); 
					t=parent[t];
				} 
				path.push_back(s);
			}
		}
	}

};

typedef unsigned int NodeID;
typedef pair<NodeID,NodeID> EdgeNew;
typedef unordered_map<NodeID,int> EdgeList;
class RoadNetwork {
public:
    unsigned int numNodes;
    unsigned int numEdges;
    vector<EdgeList> adjListOut;
    vector<EdgeList> adjListInc;

    RoadNetwork(const char *filename);
    int getEdgeWeight(NodeID lnode, NodeID rnode);
    void print();
    RoadNetwork(){};
    ~RoadNetwork();

    EdgeList ougoingEdgesOf(NodeID);
    EdgeList incomingEdgesOf(NodeID);
};

bool operator==(const EdgeNew& le, const EdgeNew& re);

class Path {
public:
    vector<NodeID> nodes;
    unordered_set<EdgeNew,boost::hash<EdgeNew>> edges;
    int length;

    Path() {
        length = -1;
    }
    bool containsEdge(EdgeNew e);
    double overlap_ratio(RoadNetwork *rN, Path &path2);

};
bool operator==(const Path& lp, const Path& rp);
//vector<vector<vector<int> > > vvGridNode;

#endif
