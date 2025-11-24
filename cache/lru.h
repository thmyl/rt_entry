#pragma once

#include <cstdint>

/**
 * LRU 双向链表节点
 */
struct LRUNode {
    int page_id;           // 当前节点对应的page在cache中的序号
    int cluster_id;        // 当前page所属的cluster的id
    int local_page_id;     // 在cluster内部的page_id
    LRUNode* next;         // 指向下一节点
    LRUNode* prev;         // 指向上一节点
    
    LRUNode() : page_id(-1), cluster_id(-1), local_page_id(-1), next(nullptr), prev(nullptr) {}
    LRUNode(int pid, int cid, int lpid) 
        : page_id(pid), cluster_id(cid), local_page_id(lpid), next(nullptr), prev(nullptr) {}
};

/**
 * LRU 链表管理器
 */
class LRUList {
private:
    LRUNode* head;         // 链表头节点（哨兵节点）
    LRUNode* tail;         // 链表尾节点（哨兵节点）
    LRUNode** nodes;       // page_id到节点的快速索引数组
    int num_pages;         // cache中的总page数量
    
public:
    /**
     * 构造函数
     * @param num_pages cache中的总page数量
     */
    LRUList(int num_pages);
    
    /**
     * 析构函数
     */
    ~LRUList();
    
    /**
     * 将指定page移动到链表头部（最近使用）
     * @param page_id page在cache中的序号
     */
    void touch(int page_id);
    
    /**
     * 插入新的page（从尾部淘汰，插入到头部）
     * @param cluster_id 要插入的page所属的cluster_id
     * @param local_page_id 在cluster内部的page_id
     * @param old_cluster_id 输出参数：被淘汰的page的cluster_id
     * @param old_local_page_id 输出参数：被淘汰的page的local_page_id
     * @return 被淘汰并重用的page在cache中的序号
     */
    int insert(int cluster_id, int local_page_id, int& old_cluster_id, int& old_local_page_id);
    
    /**
     * 获取指定page_id的节点信息
     */
    const LRUNode* get_node(int page_id) const { return nodes[page_id]; }
    
    /**
     * 打印LRU链表状态（用于调试）
     */
    void print_status() const;
    
private:
    /**
     * 从链表中移除节点
     */
    void remove_node(LRUNode* node);
    
    /**
     * 将节点插入到链表头部
     */
    void insert_to_head(LRUNode* node);
};

