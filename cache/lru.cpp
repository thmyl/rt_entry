#include "lru.h"
#include <iostream>
#include <cassert>

LRUList::LRUList(int num_pages) : num_pages(num_pages) {
    // 创建哨兵节点
    head = new LRUNode(-1, -1, -1);
    tail = new LRUNode(-1, -1, -1);
    head->next = tail;
    tail->prev = head;
    
    // 创建节点索引数组
    nodes = new LRUNode*[num_pages];
    
    // 初始化所有page节点并加入链表
    for (int i = 0; i < num_pages; i++) {
        nodes[i] = new LRUNode(i, -1, -1);  // 初始时未绑定任何cluster的page
        // 插入到链表尾部（head后面）
        insert_to_head(nodes[i]);
    }
}

LRUList::~LRUList() {
    // 删除所有节点
    for (int i = 0; i < num_pages; i++) {
        delete nodes[i];
    }
    delete[] nodes;
    delete head;
    delete tail;
}

void LRUList::touch(int page_id) {
    assert(page_id >= 0 && page_id < num_pages);
    
    LRUNode* node = nodes[page_id];
    
    // 从当前位置移除
    remove_node(node);
    
    // 插入到头部
    insert_to_head(node);
}

int LRUList::insert(int cluster_id, int local_page_id, int& old_cluster_id, int& old_local_page_id) {
    // 获取尾部节点（最久未使用的page）
    LRUNode* victim = tail->prev;
    assert(victim != head);  // 确保不是哨兵节点
    
    // 记录被淘汰的信息
    old_cluster_id = victim->cluster_id;
    old_local_page_id = victim->local_page_id;
    int victim_page_id = victim->page_id;
    
    // 从链表中移除
    remove_node(victim);
    
    // 更新节点信息
    victim->cluster_id = cluster_id;
    victim->local_page_id = local_page_id;
    
    // 插入到头部
    insert_to_head(victim);
    
    return victim_page_id;
}

void LRUList::remove_node(LRUNode* node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
}

void LRUList::insert_to_head(LRUNode* node) {
    node->next = head->next;
    node->prev = head;
    head->next->prev = node;
    head->next = node;
}

void LRUList::print_status() const {
    std::cout << "LRU Status (from MRU to LRU): ";
    LRUNode* curr = head->next;
    int count = 0;
    while (curr != tail && count < 10) {  // 只打印前10个
        std::cout << "[P" << curr->page_id << ":C" << curr->cluster_id 
                  << ",L" << curr->local_page_id << "] ";
        curr = curr->next;
        count++;
    }
    if (curr != tail) {
        std::cout << "...";
    }
    std::cout << std::endl;
}

