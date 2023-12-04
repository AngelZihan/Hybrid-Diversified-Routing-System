#include "graph.h"
   
int main()
{

    //string filename="/Users/angel/CLionProjects/Project2/USA-road-d.NY.gr";
    //string filename = "DFSTest";
	//string filename = "PortoAlegre_Edgelist.csv";
	//string filename = "Tianjin_Edgelist.csv";
	//string filename = "PortoAlegre_Edgelist.csv";
//	string filename = "/media/TraminerData/s4606021/Project/USA-road-d.NY.gr";
    string filename = "./USA-road-d.NY.gr";
   //string filename = "./USA-road-d.COL.gr";
    //string filename = "./Manhattan.gr";
	//string filename = "/media/TraminerData/s4606021/Project/USA-road-d.COL.gr";	
	//string filename = "/media/TraminerData/s4606021/Project/Manhattan.csv";
	//string filename = "/media/TraminerData/s4606021/Project/USA-road-d.FLA.gr";
    //string graphFile = "/Users/angel/CLionProjects/Project2/Test-Map";
    //string graphFile = "/Users/angel/CLionProjects/Project2/USA-road-d.NY.gr";
    //string coorFile = "/Users/angel/CLionProjects/Project2/USA-road-d.NY.co";

    Graph g = Graph(filename);
    cout << "Graph loading finish" << endl;

    srand (time(NULL));

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    vector<int> vPath, vPathEdge;

    srand (time(NULL));


	int ID1, ID2;


	vector<int> ID1List;
	vector<int> ID2List;
	//string queryFilename = "Porto/Porto-Q3";
	//string queryFilename = "Tianjin/Tianjin-Q1";
	//string queryFilename = "Manhattan/Manhattan-Q2";
	//string queryFilename = "/home/s4606021/USA-query/USA-COL-Q1.txt";	
	string queryFilename = "./NY/USA-NY-Q1.txt";
    //string queryFilename = "./COL/USA-COL-Q2.txt";
    //string queryFilename = "./Manhattan/Manhattan-Q2";
    cout << "Reading " << queryFilename << endl;
	//string queryFilename = "/home/s4606021/Project3/USA-COL-Q3.txt";
	//string queryFilename = "/home/s4606021/Project3/USA-NY-Q3.txt";
	//string queryFilename = "USA-COL-Q2.txt";
	ifstream inGraph(queryFilename);	
    if(!inGraph)
        cout << "Cannot open Map " << queryFilename << endl;
    //cout << "Reading " << queryFilename << endl;
	int pID1, pID2;
	string line;
	getline(inGraph,line);
	while(!inGraph.eof())
	{
		vector<string> vs;
		boost::split(vs,line,boost::is_any_of(" "),boost::token_compress_on);
		pID1 = stoi(vs[0]) - 1;
		pID2 = stoi(vs[1]) - 1;
	//	cout << pID1 << endl;
		ID1List.push_back(pID1);
		ID2List.push_back(pID2);
		getline(inGraph, line);
	}
	inGraph.close();



	double sumTime = 0;
    int sumCount = 0;
    int sumPop = 0;
	int sumLength = 0;
	double sumPercentage = 0;
	float sumSimTime = 0;
    double meanTime;
    int meanCount;
    int meanPop;
	int meanLength;
	double meanPercentage;
	float meanSimTime;

    vector<double> eKSPTime;
    vector<int> eKSPCount;
    vector<int> eKSPPop;
	vector<int> eKSPAverageLength;
	vector<float> eKSPSimTime;


    vector<double> eKSPCompareTime;
    vector<int> eKSPCompareCount;
    vector<int> eKSPComparePop;
	vector<int> eKSPCompareAverageLength;
	vector<double> eKSPComparePercentage;
	vector<float> eKSPCompareSimTime;
	float eKSPCompareTimeStd = 0;
	float eKSPCompareLengthStd = 0;
	float eKSPCompareAveSim = 0;
	float eKSPCompareminSim = 0;
	float eKSPComparemaxSim = 0;

    vector<double> DynamicSimilarityTime;
    vector<int> DynamicSimilarityCount;
    vector<int> DynamicSimilarityPop;
	vector<int> DynamicSimilarityAverageLength;
	float DynamicSimilarityTimeStd = 0;
	float DynamicLengthStd = 0;
	float DynamicAveSim = 0;
	float DynamicSimilarityminSim = 0;
	float DynamicSimilaritymaxSim = 0;


    vector<double> EdgesBlockingTime;
    vector<int> EdgesBlockingCount;
    vector<int> EdgesBlockingPop;
	vector<int> EdgesBlockingAverageLength;
	float EdgesBlockingTimeStd = 0;
	float EdgesBlockingLengthStd = 0;
	float EdgesBlockingAveSim = 0;
	float EdgesBlockingminSim = 0;
	float EdgesBlockingmaxSim = 0;


    vector<double> eKSPPruneTime;
    vector<int> eKSPPruneCount;
    vector<int> eKSPPrunePop;
	vector<int> eKSPPruneAverageLength;
	vector<double> eKSPPrunePercentage;

	vector<double> eKSPIntervalTime;
	vector<int> eKSPIntervalCount;
	vector<int> eKSPIntervalPop;




    // For Generate the 3000 additional  dataset
    ifstream inFile("./additionalDataset.txt");
    ofstream outFile("./processedDataset.txt");  // Output file

    if (!inFile.is_open()) {
        cerr << "Error opening file" << endl;
        return 1;
    }

    string line1;

    // Skip the header line
    if (!getline(inFile, line1)) {
        cerr << "Error or empty file" << endl;
        return 1;
    }
    outFile << line1 << ",result" << endl;
    while (getline(inFile, line1)) {
        stringstream ss(line1);
        string item;

        // Read ID1, ID2, k, and t from the line
        int ID1, ID2, k;
        double t;
        getline(ss, item, ',');
        ID1 = stoi(item);
        getline(ss, item, ',');
        ID2 = stoi(item);
        getline(ss, item, ',');
        k = stoi(item);
        getline(ss, item, ',');
        t = stod(item);

        // Other variables for eKSPCompare
        vector<int> kResults;
        int countNumber = 0;
        int popPath = 0;
        float percentage, SimTime, AveSim, minSim = 1, maxSim = 0;
        vector<vector<int>> vkPath;

        auto t1 = std::chrono::high_resolution_clock::now();
        g.eKSPCompare(ID1, ID2, k, kResults, vkPath, t, countNumber, popPath, percentage, SimTime, AveSim, minSim, maxSim);
        auto t2 = std::chrono::high_resolution_clock::now();

        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

        // Check time_span and add result
        string result = time_span.count() > 5 ? "complex!!!" : "simple!!!";

        // Write the modified data to the output file
        outFile << line1 << "," << result << endl;
    }

    inFile.close();
    outFile.close();
    // End for Generate the 3000 additional  dataset



    /*for (int i = 0; i < ID1List.size(); ++i) {
    //for (int i = 0; i < 1; ++i) {
		ID1 = ID1List[i];
		ID2 = ID2List[i];
		cout << "ID1:" << ID1 << "\tID2:" << ID2 << endl;
        int k = 5;
		double t = 0.7;
		cout << "k: " << k << " t: " << t << endl;
        vector<int> kResults;
        int countNumber = 0;
        int popPath = 0;
		vector<float> sim;
		float percentage;
		float SimTime;
		float AveSim;
		float minSim = 1;
		float maxSim = 0;
        kResults.reserve(k);
        vector<vector<int> >vkPath;
        g.FindRepeatedPath(vkPath);

		
		kResults.clear();
		AveSim = 0;
        vkPath.clear();
		sim.clear();
        t1 = std::chrono::high_resolution_clock::now();
        g.eKSPCompare(ID1, ID2, k, kResults, vkPath, t, countNumber, popPath, percentage, SimTime, AveSim,minSim,maxSim);
        t2 = std::chrono::high_resolution_clock::now();
		// << "Ave: " << AveSim << " maxSim:" << maxSim << endl;
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
		if(popPath != -1){
			eKSPCompareTime.push_back(time_span.count());
			eKSPCompareCount.push_back(countNumber);
			eKSPComparePop.push_back(popPath);
			eKSPComparePercentage.push_back(percentage);
			eKSPCompareSimTime.push_back(SimTime);
			eKSPComparemaxSim += maxSim;
			eKSPCompareminSim += minSim;
			eKSPCompareAveSim += AveSim;
			cout << " minSim: " << minSim << " maxSim: " << maxSim << endl;
			cout << "eKSPCompare Time:" << time_span.count() << endl;
			int eKSPComparePathAverageLength = 0;
			for(auto& d : kResults){
				cout << d << "\t";
				eKSPComparePathAverageLength += d;
			}
			int tempAverage = eKSPComparePathAverageLength /kResults.size();
			eKSPCompareAverageLength.push_back(tempAverage);
			float tempStd = 0;
			for(auto& d : kResults){
				tempStd += pow(abs(d-tempAverage),2);
			}
			eKSPCompareLengthStd += sqrt(tempStd / kResults.size());
			cout << endl;
			cout << "Path Average Length: " << eKSPComparePathAverageLength / kResults.size() << endl;
			cout << endl;
			cout << endl;
		}


//		kResults.clear();
//        AveSim = 0;
//		vkPath.clear();
//		sim.clear();
//        t1 = std::chrono::high_resolution_clock::now();
//        g.DynamicSimilarity(ID1, ID2, k, kResults, vkPath, t, countNumber, popPath,AveSim,minSim,maxSim);
//        t2 = std::chrono::high_resolution_clock::now();
//        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//		if(popPath != -1){
//			DynamicSimilarityminSim += minSim;
//			DynamicSimilaritymaxSim += maxSim;
//			DynamicSimilarityTime.push_back(time_span.count());
//			DynamicSimilarityCount.push_back(countNumber);
//			DynamicSimilarityPop.push_back(popPath);
//			DynamicAveSim += AveSim;
//			cout << " minSim: " << minSim << " maxSim: " << maxSim << endl;
//			cout << "DynamicSimilarity Time:" << time_span.count() << endl;
//			int DynamicSimilarityPathAverageLength = 0;
//			for(auto& d : kResults){
//				cout << d << "\t";
//				DynamicSimilarityPathAverageLength += d;
//			}
//			//DynamicSimilarityAverageLength.push_back(DynamicSimilarityPathAverageLength / kResults.size());
//			int tempAverage = DynamicSimilarityPathAverageLength /kResults.size();
//			DynamicSimilarityAverageLength.push_back(tempAverage);
//			float tempStd = 0;
//			for(auto& d : kResults){
//				tempStd += pow(abs(d-tempAverage),2);
//			}
//			DynamicLengthStd += sqrt(tempStd / kResults.size());
//			cout << endl;
//			cout << "Path Average Length: " << DynamicSimilarityPathAverageLength / kResults.size() << endl;
//			cout << endl;
//			cout << endl;
//		}



//		kResults.clear();
//        AveSim = 0;
//		vkPath.clear();
//		sim.clear();
//        t1 = std::chrono::high_resolution_clock::now();
//        g.EdgesBlocking(ID1, ID2, k, kResults, vkPath, t, countNumber, popPath,AveSim, minSim,maxSim);
//        t2 = std::chrono::high_resolution_clock::now();
//        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//		if(popPath != -1){
//			EdgesBlockingTime.push_back(time_span.count());
//			EdgesBlockingCount.push_back(countNumber);
//			EdgesBlockingminSim += minSim;
//			EdgesBlockingmaxSim += maxSim;
//			EdgesBlockingPop.push_back(popPath);
//			EdgesBlockingAveSim += AveSim;
//			cout << "EdgesBlocking Time:" << time_span.count() << endl;
//			int EdgesBlockingPathAverageLength = 0;
//			for(auto& d : kResults){
//				cout << d << "\t";
//				EdgesBlockingPathAverageLength += d;
//			}
//			int tempAverage = EdgesBlockingPathAverageLength /kResults.size();
//			EdgesBlockingAverageLength.push_back(tempAverage);
//			float tempStd = 0;
//			for(auto& d : kResults){
//				tempStd += pow(abs(d-tempAverage),2);
//			}
//			EdgesBlockingLengthStd += sqrt(tempStd / kResults.size());
//			cout << endl;
//			cout << "Path Average Length: " << EdgesBlockingPathAverageLength / kResults.size() << endl;
//			cout << endl;
//			cout << endl;
//		}

		
//		kResults.clear();
//        vkPath.clear();
//        t1 = std::chrono::high_resolution_clock::now();
//        g.eKSPPrune(ID1, ID2, k, kResults, vkPath, t, countNumber, popPath);
//        t2 = std::chrono::high_resolution_clock::now();
//        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
//		if(popPath != -1){
//			eKSPPruneTime.push_back(time_span.count());
//			eKSPPruneCount.push_back(countNumber);
//			eKSPPrunePop.push_back(popPath);
//			//eKSPPrunePercentage.push_back(percentage);
//			cout << "eKSPPrune Time:" << time_span.count() << endl;
//			int eKSPPrunePathAverageLength = 0;
//			for(auto& d : kResults){
//				cout << d << "\t";
//				eKSPPrunePathAverageLength += d;
//			}
//			eKSPPruneAverageLength.push_back(eKSPPrunePathAverageLength / kResults.size());
//			cout << endl;
//			cout << "Path Average Length: " << eKSPPrunePathAverageLength / kResults.size() << endl;
//			cout << endl;
//			cout << endl;
//		}


    }*/





    /*sumTime = 0;
    sumCount = 0;
    sumPop = 0;
	sumLength = 0;
	sumSimTime = 0;
    for (int i = 0; i < eKSPTime.size(); i++)
    {
        sumTime += eKSPTime[i];
        sumCount += eKSPCount[i];
        sumPop += eKSPPop[i];
		sumLength += eKSPAverageLength[i];
		sumSimTime += eKSPSimTime[i];
    }
    meanTime = sumTime / eKSPTime.size();
    meanCount = sumCount / eKSPCount.size();
    meanPop = sumPop / eKSPPop.size();
	meanLength = sumLength / eKSPAverageLength.size();
	meanSimTime = sumSimTime / eKSPSimTime.size();
	cout << "Pair Size:" << eKSPTime.size() << endl;
    cout  <<"eKSP Average Time: " << meanTime << "\teKSP Average Count: " << meanCount << "\teKSP Average Pop: " << meanPop << " \teKSP Average Length: " << meanLength << " \t eKSP SimTime: " << meanSimTime << endl;*/


	sumTime = 0;
    sumCount = 0;
    sumPop = 0;
	sumLength = 0;
	sumPercentage = 0;
	sumSimTime = 0;
    for (int i = 0; i < eKSPCompareTime.size(); i++)
    {
        sumTime += eKSPCompareTime[i];
        sumCount += eKSPCompareCount[i];
        sumPop += eKSPComparePop[i];
		sumLength += eKSPCompareAverageLength[i];
		sumPercentage += eKSPComparePercentage[i];
		sumSimTime += eKSPCompareSimTime[i];
    }
    //meanTime = sumTime / eKSPCompareTime.size();
    int lossSize;
    lossSize = ID1List.size() - eKSPCompareTime.size();
    for(int i = 0; i < lossSize; i++){
        sumTime += 500;
    }
    meanTime = sumTime / ID1List.size();
    meanCount = sumCount / eKSPCompareCount.size();
    meanPop = sumPop / eKSPComparePop.size();
	meanLength = sumLength / eKSPCompareAverageLength.size();
	meanPercentage = sumPercentage / eKSPComparePercentage.size();
	meanSimTime = sumSimTime / eKSPCompareSimTime.size();
	for(int j = 0; j < eKSPCompareTime.size(); j++)
	{
		float tempTime = fabs(eKSPCompareTime[j] - meanTime);
		//int tempLength = abs(eKSPCompareAverageLength[j] - meanLength);
		//cout << tempTime << " tempLength: " << tempLength << endl;
		eKSPCompareTimeStd += pow(tempTime,2);
		//eKSPCompareLengthStd += pow(tempLength, 2);
		//cout << eKSPCompareLengthStd << endl;
	}
	eKSPCompareTimeStd = sqrt(eKSPCompareTimeStd / eKSPCompareTime.size());
	//eKSPCompareLengthStd = sqrt(eKSPCompareLengthStd / eKSPCompareTime.size());
	eKSPCompareLengthStd = eKSPCompareLengthStd /eKSPCompareTime.size();
	//cout << eKSPCompareAveSim << endl;
	eKSPCompareAveSim = eKSPCompareAveSim / eKSPCompareTime.size();
	eKSPCompareminSim = eKSPCompareminSim / eKSPCompareTime.size();
	eKSPComparemaxSim = eKSPComparemaxSim / eKSPCompareTime.size();
	cout << "eKSPAveSim: " << eKSPCompareAveSim << " size: " << eKSPCompareTime.size() << endl;
	cout << "Pair Size:" << eKSPCompareTime.size() << endl;
    cout  <<"eKSPCompare Average Time: " << meanTime << "\teKSPCompare Average Count: " << meanCount << "\teKSPCompare Average Pop: " << meanPop << " \teKSPCompare Average Length: " << meanLength<< " \teKSPCompare Average Percentage: " << meanPercentage  << " \teKSPCompare SimTime: " << meanSimTime << "\tTime Std:" << eKSPCompareTimeStd << "\tLength Std:" << eKSPCompareLengthStd << " \tAveSim: " << eKSPCompareAveSim << " \teKSPCompareminSim: " << eKSPCompareminSim << " \teKSPComparemaxSim: " << eKSPComparemaxSim << endl;


	/*sumTime = 0;
    sumCount = 0;
    sumPop = 0;
	sumLength = 0;
	sumPercentage = 0;
	sumSimTime = 0;
    for (int i = 0; i < DynamicSimilarityTime.size(); i++)
    {
        sumTime += DynamicSimilarityTime[i];
        sumCount += DynamicSimilarityCount[i];
        sumPop += DynamicSimilarityPop[i];
		sumLength += DynamicSimilarityAverageLength[i];
    }
    meanTime = sumTime / DynamicSimilarityTime.size();
    meanCount = sumCount / DynamicSimilarityCount.size();
    meanPop = sumPop / DynamicSimilarityPop.size();
	meanLength = sumLength / DynamicSimilarityAverageLength.size();
	for(int j = 0; j < DynamicSimilarityTime.size(); j++)
	{
		float tempTime = fabs(DynamicSimilarityTime[j] - meanTime);
	//	cout << tempTime << endl;
		//int tempLength = abs(cTKSPDAverageLength[j] - meanLength);
		DynamicSimilarityTimeStd += pow(tempTime,2);
	}
	cout << DynamicSimilarityTimeStd << endl;
	DynamicSimilarityTimeStd = sqrt(DynamicSimilarityTimeStd / DynamicSimilarityTime.size());
	//cout << DynamicSimilarityTimeStd << endl;
	DynamicLengthStd = DynamicLengthStd / DynamicSimilarityTime.size();
	//cout << DynamicAveSim<< endl;
	DynamicAveSim = DynamicAveSim / DynamicSimilarityTime.size();
	DynamicSimilarityminSim = DynamicSimilarityminSim / DynamicSimilarityTime.size();

	DynamicSimilaritymaxSim = DynamicSimilaritymaxSim / DynamicSimilarityTime.size();

	cout << "DynamicAveSim: " << DynamicAveSim << " size: " << DynamicSimilarityTime.size() << endl;

	cout << "Pair Size:" << DynamicSimilarityTime.size() << endl;
    cout  <<"DynamicSimilarity Average Time: " << meanTime << "\tDynamicSimilarity Average Count: " << meanCount << "\tDynamicSimilarity Pop: " << meanPop << " \tDynamicSimilarity Average Length: " << meanLength << " \tTimeStd: " << DynamicSimilarityTimeStd << " \tLengthStd: "<< DynamicLengthStd << " \tAveSim: " << DynamicAveSim << " \tminSim: " << DynamicSimilarityminSim << " \tmaxSim: " << DynamicSimilaritymaxSim <<endl;
*/

/*	sumTime = 0;
    sumCount = 0;
    sumPop = 0;
	sumLength = 0;
	sumPercentage = 0;
	sumSimTime = 0;
    for (int i = 0; i < EdgesBlockingTime.size(); i++)
    {
        sumTime += EdgesBlockingTime[i];
        sumCount += EdgesBlockingCount[i];
        sumPop += EdgesBlockingPop[i];
		sumLength += EdgesBlockingAverageLength[i];
    }
    meanTime = sumTime / EdgesBlockingTime.size();
    meanCount = sumCount / EdgesBlockingCount.size();
    meanPop = sumPop / EdgesBlockingPop.size();
	meanLength = sumLength / EdgesBlockingAverageLength.size();
	for(int j = 0; j < EdgesBlockingTime.size(); j++)
	{
		float tempTime = fabs(EdgesBlockingTime[j] - meanTime);
		EdgesBlockingTimeStd += pow(tempTime,2);
	}
	//cout << EdgesBlockingTimeStd << endl;
	EdgesBlockingTimeStd = sqrt(EdgesBlockingTimeStd / EdgesBlockingTime.size());
	//cout << EdgesBlockingTimeStd << endl;
	EdgesBlockingLengthStd = EdgesBlockingLengthStd / EdgesBlockingTime.size();
	cout << EdgesBlockingAveSim << endl;
	EdgesBlockingAveSim = EdgesBlockingAveSim / EdgesBlockingTime.size();
	EdgesBlockingminSim = EdgesBlockingminSim / EdgesBlockingTime.size();

	EdgesBlockingmaxSim = EdgesBlockingmaxSim / EdgesBlockingTime.size();
	cout << "EdgesBlockingAveSim: " << EdgesBlockingAveSim << " size: " << EdgesBlockingTime.size() << endl;

	cout << "Pair Size:" << EdgesBlockingTime.size() << endl;
    cout  << "EdgesBlocking Average Time: " << meanTime << "\tEdgesBlocking Average Count: " << meanCount << "\tEdgesBlocking Pop: " << meanPop << " \tEdgesBlockingAverage Length: " << meanLength << " \tTimeStd: " << EdgesBlockingTimeStd << " \tLengthStd: "<< EdgesBlockingLengthStd << "\tAveSim: " << EdgesBlockingAveSim << " \tminSim: " << EdgesBlockingminSim << " \tmaxSim: " << EdgesBlockingmaxSim << endl;*/


	/*sumTime = 0;
    sumCount = 0;
    sumPop = 0;
	sumLength = 0;
    for (int i = 0; i < eKSPPruneTime.size(); i++)
    {
        sumTime += eKSPPruneTime[i];
        sumCount += eKSPPruneCount[i];
        sumPop += eKSPPrunePop[i];
		sumLength += eKSPPruneAverageLength[i];
    }
    meanTime = sumTime / eKSPPruneTime.size();
    meanCount = sumCount / eKSPPruneCount.size();
    meanPop = sumPop / eKSPPrunePop.size();
	meanLength = sumLength / eKSPPruneAverageLength.size();
	cout << "Pair Size:" << eKSPPruneTime.size() << endl;
    cout  <<"eKSPPrune Average Time: " << meanTime << "\teKSPPrune Average Count: " << meanCount << "\teKSPPrune Average Pop: " << meanPop << " \teKSPPrune Average Length: " << meanLength<< endl;*/

	/*sumTime = 0;
	sumCount = 0;
	sumPop = 0;
	for (int i = 0; i < eKSPIntervalTime.size(); i++)
	{
		sumTime += eKSPIntervalTime[i];
		sumCount += eKSPIntervalCount[i];
		sumPop += eKSPPop[i];
	}
	meanTime = sumTime / eKSPIntervalTime.size();
	meanCount = sumCount / eKSPIntervalCount.size();
	meanPop = sumPop / eKSPIntervalPop.size();
	cout << "eKSPInterval Average Time: " << meanTime << "\teKSPInterval Average Count:" << meanCount << "\teKSPInterval Average Pop: " << meanPop << endl;*/
	//cout << ebool << endl;
    return 0;
}






void Graph::FindRepeatedPath(vector<vector<int> >& vvPath)
{
    for(int i = 0; i < (int)vvPath.size()-1; i++)
    {
        vector<int> vSame;
        for(int j = i + 1; j < (int)vvPath.size(); j++)
        {
            if(vvPath[i].size() != vvPath[j].size())
                continue;

            bool bSame = true;
            for(int k = 0; k < vvPath[i].size(); k++)
            {
                if(vvPath[i][k] != vvPath[j][k])
                {
                    bSame = false;
                    break;
                }
            }

            if(bSame)
                vSame.push_back(j);
        }

        if(!vSame.empty())
        {
            cout << "Same path of " << i << ":" << endl;
            for(auto& sID : vSame)
                cout << sID << "\t";
            cout << endl;
        }
    }
}
