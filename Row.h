#ifndef ROW_H_INCLUDED
#define ROW_H_INCLUDED

#include <string>
#include <sstream>

#include "Table.h"
#include "RowList.h"


// if to_string is not defined
namespace myString
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

class Row
{
    public:
    
        Row(Table *table, int index) : m_table(table), m_index(index) {};
        virtual ~Row();
        
        int getIndex(void) const;
        Table *getTable(void) const;
    
    protected:
    
        Table *m_table;
        int m_index;
};

std::ostream& operator<< (std::ostream &out, Row const& row);
std::ostream& operator<< (std::ostream &out, RowList const& data);

#endif // ROW_H_INCLUDED
