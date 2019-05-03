#include "Table.h"

std::vector<std::string> getCSVitems(std::string line, char separator)
{
    std::vector<std::string> item;
    std::stringstream lineStream(line);
    std::string cell;

    while(std::getline(lineStream, cell, separator))
    {
        item.push_back(cell);
    }
    
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        item.push_back("");
    }

    return item;
}

Table::Table()
{
    m_arrayInit = false;
}

Table::~Table()
{
    delete[] m_array;
}

void Table::allocateTable(int columnNum, int rowNum, std::string name)
{
    m_rowNum = rowNum;
    m_columnNum = columnNum+1;
    m_name = name;
    
    m_array = new int[m_rowNum*m_columnNum];
}

void Table::loadTestTable(int columnNum, int rowNum)
{
    allocateTable(columnNum, rowNum, "test_set");
    
    srand(0);
    
    for(int column=0;column<columnNum;++column) // don't fill the last column
    {
        for(int row=0;row<m_rowNum;++row)
        {
            getData(column, row) = rand()%1000000;
        }
    }
}

void Table::loadFromCSV(std::string name, char separator)
{
    std::ifstream file;
    file.open (name);

    if(!file)
    {
        std::cerr << "Unable to open file : " << name << std::endl;
        return;
    }

    std::string line;
    std::vector<std::string> CSVitems;
    
    int rowNum, columnNum;
    
    for(rowNum = -1;std::getline(file, line);++rowNum); // count rows in file
    
    if(rowNum == -1)
    { // no row found
        std::cerr << "No row found in file : " << name << std::endl;
        return;
    }
    
    file.clear(); // returning to the beginning of the stream
    file.seekg(0, std::ios::beg);


    if(!std::getline(file, line))
    {
        std::cerr << "Unable to read first line of file : " << name << std::endl;
        return;
    }

    CSVitems = getCSVitems(line, separator);

    columnNum = CSVitems.size();

    // all the metadata has been found, we can allocate the table
    allocateTable(columnNum, rowNum, name);

    int row, column;
    for(row=0;std::getline(file, line);++row)
    {
        CSVitems = getCSVitems(line, separator);
        
        column=0;
        for(std::vector<std::string>::iterator it = CSVitems.begin();
                it != CSVitems.end(); ++it,++column)
        { // filling in the data
            try
            {
                int value = std::stoi(*it);
                getData(column, row) = value;
            }
            catch(const std::invalid_argument& ia)
            {
                std::cerr << "Warning, a cell could not be parsed : row ";
                std::cerr << row << ", column " << column << std::endl;
                std::cerr << "its value was " << (*it) << std::endl;
            }
        }
    }

    file.close();
}

int *Table::getArray(void)
{
    return m_array;
}

int Table::getRowNum(void)
{
    return m_rowNum;
}

int Table::getColumnNum(void)
{
    return m_columnNum;
}

std::string Table::getName(void)
{
    return m_name;
}
