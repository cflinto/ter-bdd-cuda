#include "Row.h"

Row::~Row()
{
    
}

int Row::getIndex(void) const
{
    return m_index;
}

Table *Row::getTable(void) const
{
    return m_table;
}

Row Table::getRowFromIndex(int row)
{
    return Row(this, row);
}

std::ostream& operator<< (std::ostream &out, Row const& row)
{
    for(int i=0;i<row.getTable()->getColumnNum();++i)
    {
        out << std::string("\t") << myString::to_string(row.getTable()->getData(i, row.getIndex()));
    }

    return out;
}

std::ostream& operator<< (std::ostream &out, RowList const& data)
{
    out << "Table " << data.getTable()->getName() << " (" << data.index.size()  << " elements) :" << std::endl;
    for(std::vector<int>::const_iterator it = data.index.begin(); it != data.index.end(); ++it)
    {
        out << "Row " << (*it) << "\t:" << data.getTable()->getRowFromIndex(*it) << std::endl;
    }
    out << std::endl;
    out << "(" << data.index.size()  << " elements) :" << std::endl;

    return out;
}
