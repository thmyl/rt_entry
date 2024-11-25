#include "graph.h"

Graph::Graph(Entry entry_){
    entry = entry_;
}

Graph::~Graph(){
    entry.CleanUp();
}

void Graph::Search_entry(){
    entry.Search();
}