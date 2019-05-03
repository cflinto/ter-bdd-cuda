#define CONSTRAINT_KEEP_EVERYTHING -1
#define CONSTRAINT_LOW_SEVEN_COLUMN_RANGE 0
#define CONSTRAINT_MIDDLE_SEVEN_COLUMN_RANGE 1
#define CONSTRAINT_HIGH_SEVEN_COLUMN_RANGE 2
#define CONSTRAINT_1_OUT_OF_100 3
#define CONSTRAINT_1_OUT_OF_1000 4
#define CONSTRAINT_1_OUT_OF_10000 5
#define CONSTRAINT_FROM_0_TO_100000 6

#define LOCK_OMP_CRITICAL 0
#define LOCK_NONE 1
#define LOCK_COMPARE_EXCHANGE 2

#include <iostream>
#include <unistd.h>
#include <string.h>

#include "Table.h"
#include "Row.h"
#include "Request.h"

double elapsedTime;
bool verbose = false;
bool printTime = false;
bool rowFirst = false;
int constraintType = CONSTRAINT_LOW_SEVEN_COLUMN_RANGE;
int lockMode = LOCK_OMP_CRITICAL;
int chunkNum = -1;

int main(int argc, char *argv[])
{
    int rowNum = 1000000;
    int columnNum = 3;
    char opt;
    while ((opt = getopt(argc, argv, "n:c:vtrm:l:x:s:p:")) != -1)
    {
        switch (opt)
        {
            case 'n':
                rowNum = atoi(optarg);
                break;
            case 'c':
                columnNum = atoi(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            case 't':
                printTime = true;
                break;
            case 'r':
                rowFirst = true;
                break;
            case 'm':
                if(strcmp(optarg, "none") == 0)
                {
                    constraintType = CONSTRAINT_KEEP_EVERYTHING;
                }
                else if(strcmp(optarg, "constaint_low") == 0)
                {
                    constraintType = CONSTRAINT_LOW_SEVEN_COLUMN_RANGE;
                }
                else if(strcmp(optarg, "constaint_medium") == 0)
                {
                    constraintType = CONSTRAINT_MIDDLE_SEVEN_COLUMN_RANGE;
                }
                else if(strcmp(optarg, "constaint_high") == 0)
                {
                    constraintType = CONSTRAINT_HIGH_SEVEN_COLUMN_RANGE;
                }
                else if(strcmp(optarg, "1/100") == 0)
                {
                    constraintType = CONSTRAINT_1_OUT_OF_100;
                }
                else if(strcmp(optarg, "1/1000") == 0)
                {
                    constraintType = CONSTRAINT_1_OUT_OF_1000;
                }
                else if(strcmp(optarg, "1/10000") == 0)
                {
                    constraintType = CONSTRAINT_1_OUT_OF_10000;
                }
                else if(strcmp(optarg, "1-100000") == 0)
                {
                    constraintType = CONSTRAINT_FROM_0_TO_100000;
                }
                else
                {
                    std::cerr << "Option -m : unrecognized type" << std::endl;
                }
                break;
            case 'l':
                if(strcmp(optarg, "none") == 0)
                {
                    lockMode = LOCK_NONE;
                }
                else if(strcmp(optarg, "omp_critical") == 0)
                {
                    lockMode = LOCK_OMP_CRITICAL;
                }
                else if(strcmp(optarg, "compare_exchange") == 0)
                {
                    lockMode = LOCK_COMPARE_EXCHANGE;
                }
                else
                {
                    lockMode = LOCK_NONE;
                    std::cerr << "wrong lock mode" << std::endl;
                }
                break;
            case 'x':
                // chunk number, had to be BEFORE -s parameter
                chunkNum = std::stoi(optarg);
                break;
            case 's':
                // schedule
                if(strcmp(optarg, "static") == 0)
                {
                    omp_set_schedule(omp_sched_static, chunkNum);
                }
                else if(strcmp(optarg, "dynamic") == 0)
                {
                    omp_set_schedule(omp_sched_dynamic, chunkNum);
                }
                else if(strcmp(optarg, "guided") == 0)
                {
                    omp_set_schedule(omp_sched_guided, chunkNum);
                }
                else
                {
                    std::cerr << "wrong schedule type" << std::endl;
                }
                break;
            case 'p':
                // thread number
                omp_set_num_threads(std::stoi(optarg));
                break;
            default:
                fprintf(stderr, "Usage: %s [-m max scenarii] [-n] name\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    
    Table *table;
    
    if(rowFirst)
    {
        table = new TableRowFirst;
    }
    else
    {
        table = new TableColumnFirst;
    }
    
    //table.loadFromCSV("data/frogs.csv");
    table->loadTestTable(columnNum, rowNum);
    
    integerList list;

    if(lockMode == LOCK_OMP_CRITICAL)
    {
        switch(constraintType)
        {
            case CONSTRAINT_KEEP_EVERYTHING:
                list = selectWhere<OMP_CRITICAL, NoConstraint>(table);
                break;
            case CONSTRAINT_LOW_SEVEN_COLUMN_RANGE:
                list = selectWhere<OMP_CRITICAL, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 720000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 720000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 720000>
                > >(table);
                break;
            case CONSTRAINT_MIDDLE_SEVEN_COLUMN_RANGE:
                list = selectWhere<OMP_CRITICAL, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 320000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 320000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 320000>
                > >(table);
                break;
            case CONSTRAINT_HIGH_SEVEN_COLUMN_RANGE:
                list = selectWhere<OMP_CRITICAL, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 100000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 100000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 100000>
                > >(table);
                break;
            case CONSTRAINT_1_OUT_OF_100:
                list = selectWhere<OMP_CRITICAL, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<100>>>(table);
                break;
            case CONSTRAINT_1_OUT_OF_1000:
                list = selectWhere<OMP_CRITICAL, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<1000>>>(table);
                break;
            case CONSTRAINT_1_OUT_OF_10000:
                list = selectWhere<OMP_CRITICAL, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<10000>>>(table);
                break;
            case CONSTRAINT_FROM_0_TO_100000:
                list = selectWhere<OMP_CRITICAL, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepRange<0, 100000>>>(table);
                break;
            default:
                std::cerr << "Unknown constraint type" << std::endl;
                return 1;
        }
    }
    
    if(lockMode == LOCK_NONE)
    {
        switch(constraintType)
        {
            case CONSTRAINT_KEEP_EVERYTHING:
                list = selectWhere<NONE, NoConstraint>(table);
                break;
            case CONSTRAINT_LOW_SEVEN_COLUMN_RANGE:
                list = selectWhere<NONE, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 720000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 720000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 720000>
                > >(table);
                break;
            case CONSTRAINT_MIDDLE_SEVEN_COLUMN_RANGE:
                list = selectWhere<NONE, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 320000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 320000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 320000>
                > >(table);
                break;
            case CONSTRAINT_HIGH_SEVEN_COLUMN_RANGE:
                list = selectWhere<NONE, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 100000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 100000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 100000>
                > >(table);
                break;
            case CONSTRAINT_1_OUT_OF_100:
                list = selectWhere<NONE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<100>>>(table);
                break;
            case CONSTRAINT_1_OUT_OF_1000:
                list = selectWhere<NONE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<1000>>>(table);
                break;
            case CONSTRAINT_1_OUT_OF_10000:
                list = selectWhere<NONE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<10000>>>(table);
                break;
            case CONSTRAINT_FROM_0_TO_100000:
                list = selectWhere<NONE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepRange<0, 100000>>>(table);
                break;
            default:
                std::cerr << "Unknown constraint type" << std::endl;
                return 1;
        }
    }
    
    if(lockMode == LOCK_COMPARE_EXCHANGE)
    {
        switch(constraintType)
        {
            case CONSTRAINT_KEEP_EVERYTHING:
                list = selectWhere<COMPARE_EXCHANGE, NoConstraint>(table);
                break;
            case CONSTRAINT_LOW_SEVEN_COLUMN_RANGE:
                list = selectWhere<COMPARE_EXCHANGE, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 720000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 720000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 720000>
                > >(table);
                break;
            case CONSTRAINT_MIDDLE_SEVEN_COLUMN_RANGE:
                list = selectWhere<COMPARE_EXCHANGE, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 320000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 320000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 320000>
                > >(table);
                break;
            case CONSTRAINT_HIGH_SEVEN_COLUMN_RANGE:
                list = selectWhere<COMPARE_EXCHANGE, CombineConstraints<
                    ColumnSuperior<0, 1000>, ColumnInferior<0, 100000>,
                    ColumnSuperior<1, 1000>, ColumnInferior<1, 100000>,
                    ColumnSuperior<2, 1000>, ColumnInferior<2, 100000>
                > >(table);
                break;
            case CONSTRAINT_1_OUT_OF_100:
                list = selectWhere<COMPARE_EXCHANGE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<100>>>(table);
                break;
            case CONSTRAINT_1_OUT_OF_1000:
                list = selectWhere<COMPARE_EXCHANGE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<1000>>>(table);
                break;
            case CONSTRAINT_1_OUT_OF_10000:
                list = selectWhere<COMPARE_EXCHANGE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepOneOutOf<10000>>>(table);
                break;
            case CONSTRAINT_FROM_0_TO_100000:
                list = selectWhere<COMPARE_EXCHANGE, CombineConstraints<
                    ColumnSuperior<0, 0>, ColumnInferior<0, 1000000>,
                    ColumnSuperior<1, 0>, ColumnInferior<1, 1000000>,
                    ColumnSuperior<2, 0>, ColumnInferior<2, 1000000>,
                    KeepRange<0, 100000>>>(table);
                break;
            default:
                std::cerr << "Unknown constraint type" << std::endl;
                return 1;
        }
    }
    
    if(verbose)
    {
        RowList rowList(table);
        rowList = selectWhereAgregate<NONE, NoConstraint>(table, list);
        std::cout << rowList;
    }
    
    if(printTime)
    {
        std::cout << elapsedTime;
    }
    
    return 0;
}
