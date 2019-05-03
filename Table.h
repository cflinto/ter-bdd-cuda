#ifndef TABLE_H_INCLUDED
#define TABLE_H_INCLUDED

#include <stdlib.h>
#include <string>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <iostream>

#include <defines.h>

class Row;
class RowList;

std::vector<std::string> getCSVitems(std::string line, char separator);

class Table
{
    public:
    
        Table();
        virtual ~Table();
        
        void allocateTable(int columnNum, int rowNum, std::string name);
        
        void loadTestTable(int columnNum, int rowNum);
        void loadFromCSV(std::string name, char separator = ',');
        
        virtual int &getData(int column, int row) const = 0;
        int *getArray(void);
        Row getRowFromIndex(int row);
        
        int getRowNum(void);
        int getColumnNum(void);
        
        std::string getName(void);
    
    protected:
    
        std::string m_name;
        int m_columnNum;
        int m_rowNum;
        int *m_array;
        bool m_arrayInit;
};

class TableColumnFirst : public Table
{
    virtual int &getData(int column, int row) const
    {
        return m_array[m_rowNum*column+row];
    }
};

class TableRowFirst : public Table
{
    virtual int &getData(int column, int row) const
    {
        return m_array[m_columnNum*row+column];
    }
};

#endif // TABLE_H_INCLUDED
