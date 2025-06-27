//
// Created by Angel on 15/9/2023.
//

//Use this one

#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include "graph.h"
#include "ClassificationSystem.h"
#include "CoherenceDecomposition.h"
#include "kspwlo.hpp"
#include <algorithm>
#include <cctype>
#include <boost/algorithm/string.hpp>

#define BUFFER_SIZE 1024


using namespace boost::algorithm;
string trim(const string& str) {
    const auto strBegin = std::find_if_not(str.begin(), str.end(), ::isspace);
    const auto strEnd = std::find_if_not(str.rbegin(), str.rend(), ::isspace).base();
    return (strBegin < strEnd ? std::string(strBegin, strEnd) : "");
}
const string terminator = "END_OF_MESSAGE";
int main() {
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    std::map<std::string, double> timing_sums;
    std::map<std::string, int> timing_counts;

    Classification obj1;


    vector<int> ID1List;
    vector<int> ID2List;
//    string queryFile = "./COL/USA-COL-Q1.txt";
    string queryFile = "./NY/USA-NY-Q1.txt";
    cout << "Reading " << queryFile << endl;
    ifstream inGraph(queryFile);
    if(!inGraph)
        cout << "Cannot open Map " << queryFile << endl;
    int pID1, pID2;
    string line4;
    getline(inGraph,line4);
    //cout << line4 << endl;
    while(!inGraph.eof())
    {
        vector<string> vs4;
        boost::split(vs4,line4,boost::is_any_of(" "),boost::token_compress_on);
        //cout << vs4.size() << endl;
        pID1 = stoi(vs4[0]);
        pID2 = stoi(vs4[1]);
        ID1List.push_back(pID1);
        ID2List.push_back(pID2);
        getline(inGraph, line4);
    }
    inGraph.close();

//    string graphFile = "./USA-road-d.COL.gr";
    string graphFile = "./USA-road-d.NY.gr";
    Graph G=Graph(graphFile);
    //CoherenceDecomposition cd = CoherenceDecomposition(&G, 4, mm, nn);



    map<pair<int, int>, int> trueLabels;

    string predictionStr;
    double totalTime = 0;
    vector<int> averageLength;
    float totalMaxSim = 0;
    for(int m = 0; m < ID1List.size(); m++){
    //for(int m = 0; m < 1; m++){
        int ID1 = ID1List[m]-1;
        int ID2 = ID2List[m]-1;
        cout << "ID1: " << ID1 << " ID2: " << ID2 << endl;
        int xLevel;
        int yLevel;
        int k = 5;
        double t = 0.7;

        vector<pair<string,string>> edge_attribute;
        vector<pair<string,string>> edge_index;
        vector<pair<string,string>> graph_data;
        vector<vector<int> > staMatrix;

        CoherenceDecomposition cd = CoherenceDecomposition(&G, 4, ID1, ID2, xLevel, yLevel);
        //cout << "FirstPath!" << endl;
        vector<int> vSPTDistance(G.n, INF);
        vector<int> vSPTParent(G.n, -1);
        vector<int> vSPTHeight(G.n, -1);
        vector<int> vSPTParentEdge(G.n, -1);
        vector<int> vSPTChildren(G.n, -1);
        vector<int> vTmp;
        vector<vector<int>> vSPT(G.n, vTmp);

        vector<vector<int> > vvPathCandidate;	 //nodes
        vector<vector<int> > vvPathCandidateEdge;
        vector<int> kResults;
        vector<vector<int> > vkPath;

        t1 = std::chrono::high_resolution_clock::now();
        int SP = obj1.FirstPath(&G, ID1,ID2,edge_attribute,edge_index,graph_data,staMatrix, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPTChildren, vSPT, vvPathCandidate, vvPathCandidateEdge, kResults, vkPath);

        string edge_attribute1;
        string edge_index1;
        for (int i = 0; i < edge_index.size(); ++i) {
            edge_index1 = edge_index1 + "[" + edge_index[i].first + "," + edge_index[i].second + "]" + " ";
            edge_attribute1 = edge_attribute1 + "[" + edge_attribute[i].first + "," + edge_attribute[i].second + "]" + " ";
        }
        string graph_data1;
        for (int i = 0; i < graph_data.size(); ++i) {
            graph_data1 = graph_data1 + "[" + graph_data[i].first + "," + graph_data[i].second + "]" + " ";
        }

        string outFilename = "./NY_Result_changeKSP_fileNew_addKSPLost";
//        string outFilename = "./COL_Result_changeKSP_fileNew_addKSPLost2";
        string tabular_data1;
        string tabular_data2;

        ifstream inOutFile(outFilename);
        string line;
        if(inOutFile){
            getline(inOutFile,line);
//            cout << line << endl;
            while(!inOutFile.eof()) {
                //cout << line << endl;
                getline(inOutFile, line);
//                cout << line << endl;
                vector<string> vs;
                boost::split(vs,line,boost::is_any_of(","),boost::token_compress_on);
                int xLevel1 = stoi(vs[4]);
                int yLevel1 = stoi(vs[5]);
//                cout << "xLevel:" << xLevel << " yLevel: " << yLevel << " xLevel1: " << xLevel1 << " yLevel1: " << yLevel1 <<endl;
                if(xLevel == xLevel1 && yLevel == yLevel1){
                    int coverNumber = stoi(vs[6]);
                    int CEONumber = stoi(vs[7]);
                    vector<double> KSP;
                    for (int i = 11; i < 15; ++i) {
                        KSP.push_back(stod(vs[i]));
                        //cout << KSP[0] << endl;
                    }

//                    NY
                    tabular_data1 = to_string(k) + "," + to_string(t) + "," + to_string(coverNumber) + "," +
                                   to_string(CEONumber) + "," + to_string(SP) + ",";
//                    COL
//                    tabular_data1 = to_string(coverNumber) + "," +
//                                    to_string(CEONumber) + "," + to_string(SP) + ",";


                    for (int i = 0; i < 3; ++i) {
                        tabular_data1 = tabular_data1 + to_string(KSP[i]) + ",";
                    }
                    tabular_data1 = tabular_data1 + to_string(KSP[3]);

                    for (int m = 0; m < staMatrix.size(); m++) {
                        for(int n = 0; n < staMatrix[m].size(); n++) {
                            if(m == 0 && n == 0){
                                tabular_data2 = to_string(staMatrix[m][n]) + ",";
                            }
                            else if(m == 4 && n == 6){
                                tabular_data2 = tabular_data2 + to_string(staMatrix[m][n]) + " ";
                            }
                            else{
                                tabular_data2 = tabular_data2 + to_string(staMatrix[m][n]) + ",";
                            }
                        }
                    }
                    break;
                }
                else{
                    for (int i = 0; i < 8; ++i) {
                        getline(inOutFile, line);
                    }
                }
            }
        }

        std::string combined_data = edge_attribute1 + "|" + edge_index1 + "|" + graph_data1 + "|" + tabular_data1 + "|" + tabular_data2 + "|" + terminator;

        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            std::cerr << "Could not create socket." << std::endl;
            return -1;
        }

        sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(65432);
        server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

        if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Connection failed." << std::endl;
            close(sock);
            return -1;
        }

        size_t totalSent = 0;
        while (totalSent < combined_data.size()) {
            size_t toSend = std::min(combined_data.size() - totalSent, (size_t)BUFFER_SIZE);
            ssize_t sent = send(sock, combined_data.c_str() + totalSent, toSend, 0);
            if (sent == -1) {
                std::cerr << "Failed to send data." << std::endl;
                close(sock);
                return -1;
            }
            totalSent += sent;
        }

        // Receive the processed result
        char buffer[1024] = {0};
        int bytes_received = recv(sock, buffer, sizeof(buffer) - 1, 0);  // 防止越界

        if (bytes_received <= 0) {
            cerr << "Error: failed to receive data from server." << endl;
            close(sock);
            return -1;
        }

        buffer[bytes_received] = '\0';

        string response(buffer);
        cout << "Full response: " << response << endl;

        predictionStr = buffer;



        int predictedLabel = stoi(predictionStr);


        if (predictedLabel == 0) {
            cout << 0 << endl;
            float maxSim = 0;
            float AveSim = 0;
            float minSim = 0;
            int pathLength = 0;
            //DKSP
            vector<int> kResults;
            vector<vector<int> > vkPath;
            obj1.eKSPCompare(&G, ID1, ID2, k, kResults, vkPath, t, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPTChildren, vSPT, vvPathCandidate, vvPathCandidateEdge,maxSim);
            t2 = std::chrono::high_resolution_clock::now();
            time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            //DKSP

            //OP
//            RoadNetwork *rN = 0;
//            rN = new RoadNetwork(graphFile.c_str());
//            NodeID source = ID1;
//            NodeID target = ID2;
//            double theta = t;
//            vector<Path> result;
//            result = onepass(rN,source,target,k,theta,AveSim,minSim,maxSim);
//            t2 = std::chrono::high_resolution_clock::now();
//            time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            //OP

            if(time_span.count() > 5){
//                DS
//                obj1.DynamicSimilarity(&G, ID1, ID2, k, kResults, vkPath, t, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPTChildren, vSPT, vvPathCandidate, vvPathCandidateEdge, maxSim);
//                t2 = std::chrono::high_resolution_clock::now();
//                totalMaxSim += maxSim;
//                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//                for(auto& d : kResults){
//                    cout << d << "\t";
//                    pathLength += d;
//                }
//                cout << endl;
//                int tempAverage = pathLength / kResults.size();
//                cout << " AveLength:" << tempAverage << endl;
//                averageLength.push_back(tempAverage);
//                cout << "Simple Time:" << time_span.count() << endl;
//                totalTime += time_span.count();
//                cout << endl;
//                cout << endl;
//                cout << endl;

                //esxc

                //If it is OP close this part
                RoadNetwork *rN = 0;
                rN = new RoadNetwork(graphFile.c_str());
                //If it is OP close this part
                pair<vector<Path>, double> completeResult;
                NodeID source = ID1;
                NodeID target = ID2;
                double theta = t;
                vector<Path> result;
                completeResult = esx_complete(rN,source,target,k,theta,AveSim,minSim,maxSim);
                t2 = std::chrono::high_resolution_clock::now();
                result = completeResult.first;
                cout << source << "\t" << target << "\t["<< result[0].length;
                pathLength += result[0].length;
                for(unsigned int j = 1;j<result.size();j++) {
                    //totalResultLengthSize += 1;
                    cout << "," << result[j].length;
                    pathLength += result[j].length;
                }
                cout << "]" << endl;
                totalMaxSim += maxSim;
                int tempAverage = pathLength / result.size();
                cout << " AveLength:" << tempAverage << endl;
                averageLength.push_back(tempAverage);
                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
                cout << "Simple Time:" << time_span.count() << endl;
                totalTime += time_span.count();
                cout << endl;
                cout << endl;
                cout << endl;

                //svp
//                // If it is OP close this part
////                RoadNetwork *rN = 0;
////                rN = new RoadNetwork(graphFile.c_str());
//                //If it is OP close this part
//                pair<vector<Path>, double> completeResult;
//                NodeID source = ID1;
//                NodeID target = ID2;
//                double theta = t;
//                vector<Path> result;
//                //t1 = std::chrono::high_resolution_clock::now();
//                completeResult = svp_plus_complete(rN,source,target,k,theta,AveSim,minSim,maxSim);
//                t2 = std::chrono::high_resolution_clock::now();
//                result = completeResult.first;
//                cout << source << "\t" << target << "\t[" << result[0].length;
//                pathLength += result[0].length;
//                for(unsigned int j = 1;j<result.size();j++) {
//                    //totalResultLengthSize += 1;
//                    cout << "," << result[j].length;
//                    pathLength += result[j].length;
//                }
//                cout << "]" << endl;
//                totalMaxSim += maxSim;
//                int tempAverage = pathLength / result.size();
//                cout << " AveLength:" << tempAverage << endl;
//                averageLength.push_back(tempAverage);
//                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//                cout << "Simple Time:" << time_span.count() << endl;
//                totalTime += time_span.count();
//                cout << endl;
//                cout << endl;
//                cout << endl;

                //
            }
            else{
                //DKSP
                for(auto& d : kResults){
                    cout << d << "\t";
                    pathLength += d;
                }
                cout << endl;
                totalMaxSim += maxSim;
                int tempAverage = pathLength / kResults.size();
                cout << " AveLength:" << tempAverage << endl;
                averageLength.push_back(tempAverage);
                cout << "Simple Time:" << time_span.count() << endl;
                totalTime += time_span.count();
                cout << endl;
                cout << endl;
                cout << endl;
                //DKSP

                //OP
//                cout << source << "\t" << target << "\t[" << result[0].length;
//                pathLength += result[0].length;
//                for(unsigned int j = 1;j<result.size();j++) {
//                    //totalResultLengthSize += 1;
//                    cout << "," << result[j].length;
//                    pathLength += result[j].length;
//                }
//                cout << "]" << endl;
//                totalMaxSim += maxSim;
//                int tempAverage = pathLength / result.size();
//                cout << " AveLength:" << tempAverage << endl;
//                averageLength.push_back(tempAverage);
//                time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//                cout << "Simple Time:" << time_span.count() << endl;
//                totalTime += time_span.count();
//                cout << endl;
//                cout << endl;
//                cout << endl;
                //OP


            }
        }
        else if (predictedLabel == 1) {
            cout << 1 << endl;
            int pathLength = 0;
            float maxSim = 0;
            float AveSim = 0;
            float minSim = 0;

            //DS
//            obj1.DynamicSimilarity(&G, ID1, ID2, k, kResults, vkPath, t, vSPTDistance, vSPTParent, vSPTHeight, vSPTParentEdge, vSPTChildren, vSPT, vvPathCandidate, vvPathCandidateEdge,maxSim);
//            t2 = std::chrono::high_resolution_clock::now();
//            time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//            for(auto& d : kResults){
//                cout << d << "\t";
//                pathLength += d;
//            }
//            cout << endl;
//            totalMaxSim += maxSim;
//            int tempAverage = pathLength / kResults.size();
//            cout << " AveLength:" << tempAverage << endl;
//            averageLength.push_back(tempAverage);
//            cout << "Simple Time:" << time_span.count() << endl;
//            totalTime += time_span.count();
//            cout << endl;
//            cout << endl;
//            cout << endl;


            //esx
            RoadNetwork *rN = 0;
            rN = new RoadNetwork(graphFile.c_str());
            pair<vector<Path>, double> completeResult;
            NodeID source = ID1;
            NodeID target = ID2;
            double theta = t;
            vector<Path> result;
            completeResult = esx_complete(rN,source,target,k,theta,AveSim,minSim,maxSim);
            t2 = std::chrono::high_resolution_clock::now();
            result = completeResult.first;
            cout << source << "\t" << target << "\t[" << result[0].length;
            pathLength += result[0].length;
            for(unsigned int j = 1;j<result.size();j++) {
                cout << "," << result[j].length;
                pathLength += result[j].length;
            }
            cout << "]" << endl;
            totalMaxSim += maxSim;
            int tempAverage = pathLength / result.size();
            cout << " AveLength:" << tempAverage << endl;
            averageLength.push_back(tempAverage);
            time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            cout << "Complex Time:" << time_span.count() << endl;
            totalTime += time_span.count();
            cout << endl;
            cout << endl;
            cout << endl;

            //svp
//            RoadNetwork *rN = 0;
//            rN = new RoadNetwork(graphFile.c_str());
//            pair<vector<Path>, double> completeResult;
//            NodeID source = ID1;
//            NodeID target = ID2;
//            double theta = t;
//            completeResult = svp_plus_complete(rN,source,target,k,theta,AveSim,minSim,maxSim);
//            t2 = std::chrono::high_resolution_clock::now();
//            vector<Path> result;
//            result = completeResult.first;
//            cout << source << "\t" << target << "\t[" << result[0].length;
//            pathLength += result[0].length;
//            for(unsigned int j = 1; j<result.size();j++) {
//                cout << "," << result[j].length;
//                pathLength += result[j].length;
//            }
//            cout << "]" << endl;
//            totalMaxSim += maxSim;
//            int tempAverage = pathLength / result.size();
//            cout << " AveLength:" << tempAverage << endl;
//            averageLength.push_back(tempAverage);
//            time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//            cout << "Complex Time:" << time_span.count() << endl;
//            totalTime += time_span.count();
//            cout << endl;
//            cout << endl;
//            cout << endl;
        }
        else
            cout << "error" << endl;
        close(sock);
    }
    double sumLength = 0;
    double meanLength = 0;
    for(int m = 0; m < averageLength.size(); m++)
    {
        sumLength += averageLength[m];
    }
    meanLength = sumLength / (averageLength.size());
    cout << "sumLength: " << sumLength << "aveSize: " << averageLength.size() << endl;
    totalMaxSim = totalMaxSim / averageLength.size();

    cout << "Hybird Time : " << totalTime/ID1List.size() << " Average Length: " << meanLength << " maxSim: " << totalMaxSim << endl;



    return 0;
}
