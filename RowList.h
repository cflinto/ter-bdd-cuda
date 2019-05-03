#ifndef ROWLIST_H_INCLUDED
#define ROWLIST_H_INCLUDED

#include <vector>
#include <tuple>
#include <iostream>

class Table;

class RowList
{
    public:
    
        RowList(Table *table) : m_table(table) {};
        virtual ~RowList();
        
        Table *getTable(void) const;
        
        std::vector<int> index;
    
    protected:
    
        Table *m_table;
};

#endif // ROWLIST_H_INCLUDED
