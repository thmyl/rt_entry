#pragma once
#include "entry.h"

class Graph{
public:
	Entry entry;
	Graph(){}
	Graph(Entry entry_);
	~Graph();
	void Search_entry();
};