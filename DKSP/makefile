CXX=g++ -std=c++11
OPT=-O3

DKSP: main.o graph.o tools.o eKSP.o eKSPCompare.o eKSPPrune.o DynamicSimilarity.o EdgesBlocking.o -lpthread -lboost_system -lboost_thread 
	$(CXX) -g -o DKSP main.o graph.o tools.o eKSP.o eKSPCompare.o eKSPPrune.o DynamicSimilarity.o EdgesBlocking.o -lpthread -lboost_system -lboost_thread

graph.o:graph.cpp
	$(CXX) -g -c $(OPT) graph.cpp
main.o:main.cpp
	$(CXX) -g -c $(OPT) main.cpp
tools.o:tools.cpp
	$(CXX) -g -c $(OPT) tools.cpp
eKSP.o:eKSP.cpp
	$(CXX) -g -c $(OPT) eKSP.cpp
eKSPCompare.o: eKSPCompare.cpp
	$(CXX) -g -c $(OPT) eKSPCompare.cpp
eKSPPrune.o: eKSPPrune.cpp
	$(CXX) -g -c $(OPT) eKSPPrune.cpp
DynamicSimilarity.o: DynamicSimilarity.cpp
	$(CXX) -g -c $(OPT) DynamicSimilarity.cpp
EdgesBlocking.o:EdgesBlocking.cpp
	$(CXX) -g -c $(OPT) EdgesBlocking.cpp
	
clean:
	rm *.o
	rm DKSP
