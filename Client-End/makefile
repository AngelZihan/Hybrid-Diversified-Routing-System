CXX=g++ -std=c++11
OPT=-O3

classification: main.o SPT.o FirstPath_addMatrix.o DKSP.o esx.o graph.o astar.o dijkstra.o onepass.o completeness.o DynamicSimilarity.o svp_plus.o -lpthread -lboost_system -lboost_thread
	$(CXX) -g -o classification main.o SPT.o FirstPath_addMatrix.o DKSP.o esx.o graph.o astar.o dijkstra.o onepass.o completeness.o DynamicSimilarity.o svp_plus.o -lpthread -lboost_system -lboost_thread

main.o:main.cpp
	$(CXX) -g -c $(OPT) main.cpp
SPT.o:SPT.cpp
	$(CXX) -g -c $(OPT) SPT.cpp
FirstPath.o:FirstPath_addMatrix.cpp.cpp
	$(CXX) -g -c $(OPT) FirstPath_addMatrix.cpp
DKSP.o:DKSP.cpp
	$(CXX) -g -c $(OPT) DKSP.cpp
esx.o:esx.cpp
	$(CXX) -g -c $(OPT) esx.cpp
graph.o:graph.cpp
	$(CXX) -g -c $(OPT) graph.cpp
astar.o:astar.cpp
	$(CXX) -g -c $(OPT) astar.cpp
dijkstra.o:dijkstra.cpp
	$(CXX) -g -c $(OPT) dijkstra.cpp
onepass.o:onepass.cpp
	$(CXX) -g -c $(OPT) onepass.cpp
completeness.o:completeness.cpp
	$(CXX) -g -c $(OPT) completeness.cpp
DynamicSimilarity.o:DynamicSimilarity.cpp
	$(CXX) -g -c $(OPT) DynamicSimilarity.cpp
svp_plus.o:svp_plus.cpp
	$(CXX) -g -c $(OPT) svp_plus.cpp
clean:
	rm *.o
	rm classification
