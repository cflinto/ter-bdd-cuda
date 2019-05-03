/* Request.h
 * 
 * Handles queries on a table.
 */
#ifndef REQUEST_H_INCLUDED
#define REQUEST_H_INCLUDED

#define DEFAULT_ROW_LIST_ALLOCATION 1024
#define ROW_LIST_ALLOCATION_INCREMENT 1024

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#include <stdlib.h>
#include <iostream>
#include <atomic>
#include <omp.h>

#include "RowList.h"
#include "Table.h"

typedef struct
{
    int *list;
    int size;
} integerList;

class ITERATIVE{};
class PARALLEL{};
class MIN{};

class OMP_CRITICAL{};
class NONE{};
class COMPARE_EXCHANGE{};

extern double elapsedTime;

class NoConstraint
{
    public:
        static inline bool respects(const Table *table, const int idRow)
        {
            return true;
        }
};

template <int keepNum>
class KeepOneOutOf
{
    public:
        static inline bool respects(const Table *table, const int idRow)
        {
            return idRow%keepNum == 0;
        }
};

template <int min, int max>
class KeepRange
{
    public:
        static inline bool respects(const Table *table, const int idRow)
        {
            return idRow >= min && idRow <= max;
        }
};

template <int column, int value>
class ColumnInferior
{
    public:
        static inline bool respects(const Table *table, const int idRow)
        {
            return table->getData(column, idRow) < value;
        }
};

template <int column, int value>
class ColumnSuperior
{
    public:
        static inline bool respects(const Table *table, const int idRow)
        {
            return table->getData(column, idRow) > value;
        }
};


template<class T>
bool inline loadBrush_sub_impl(const Table *table, const int idRow)
{
    return T::respects(table, idRow);
}

template<class T = void, class... Targs>
bool inline checkOneConstraint(const Table *table, const int idRow);

template<class T, class... Targs>
bool inline checkOneConstraint(const Table *table, const int idRow)
{
    return loadBrush_sub_impl<T>(table, idRow) && checkOneConstraint<Targs...>(table, idRow);
}

template<>
bool inline checkOneConstraint<>(const Table *table, const int idRow)
{
    return true;
}

template </*class firstConstraint, */class... otherConstraints>
class CombineConstraints
{
    public:
    
        static inline bool respects(const Table *table, const int idRow)
        {
            return checkOneConstraint<otherConstraints...>(table, idRow);
        }
};

template <class lockMethod, class constraint>
integerList selectWhere(Table *table)
{
    integerList result;
    result.list = new int[table->getRowNum()];
    //result.size = 0;
    std::atomic<int> atomicSize = 0;
    
    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        std::atomic<int> idWrite=0;
        #pragma omp for
        for(int row=0;row<table->getRowNum();++row)
        {
            if(!constraint::respects(table, row))
            {
                continue;
            }
            
            // methode 1
            if (std::is_same<lockMethod, OMP_CRITICAL>::value)
            {
                #pragma omp critical
                {
                    result.list[atomicSize] = row;
                    ++atomicSize;
                }
            }
            
            // methode 2
            if (std::is_same<lockMethod, NONE>::value)
            {
                result.list[atomicSize++] = row;
            }
            
            
            // methode 3
            if (std::is_same<lockMethod, COMPARE_EXCHANGE>::value)
            {
                int idWrite = atomicSize;
                int toSet = idWrite+1;
                while(!atomicSize.compare_exchange_strong(idWrite, toSet)){
                    toSet = idWrite+1;
                }
                result.list[idWrite] = row;
            }
            
        }
    }
    elapsedTime = omp_get_wtime() - start_time;
    
    result.size = atomicSize;
    
    return result;
}

template <class lockMethod, class constraint>
RowList selectWhereAgregate(Table *table)
{
    integerList list = selectWhere<lockMethod, constraint>(table);
    
    RowList ret(table);
    for(int i=0;i<list.size;++i)
    {
        ret.index.push_back(list.list[i]);
    }
    
    free(list.list); // we don't neet the row list anymore, since it is now stored in the RowList
    
    return ret;
}

template <class lockMethod, class constraint>
RowList selectWhereAgregate(Table *table, integerList list)
{
    RowList ret(table);
    for(int i=0;i<list.size;++i)
    {
        ret.index.push_back(list.list[i]);
    }
    
    free(list.list); // we don't neet the row list anymore, since it is now stored in the RowList
    
    return ret;
}


#endif // REQUEST_H_INCLUDED
