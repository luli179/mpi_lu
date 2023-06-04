#include<iostream>
#include<mpi.h>
#include <sys/time.h>                                                                                           
#include <unistd.h>
using namespace std;

int mpi_size;
int mpi_rank;
const int N = 1024;
float matrix[N][N];
const unsigned long Converter = 1000 * 1000; // 1s == 1000 * 1000 us
//print function
void matrix_print()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << matrix[i][j] << ' ';
        }
        cout << endl;
    }
}
//initial function
void setmatrix()
{
    srand(2);
    for (int i = 0; i < N; i++)
    {
        matrix[i][i] = 1.0;
        for (int j = 0; j < N; j++)
        {
            if (i < j)
                matrix[i][j] = rand() % 4;
            if (i > j)
                matrix[i][j] = 0.0;
        }
    }
    int i = 0, j = 0;
    for (int k = 0; k < N; k++)
    {
        i = rand() % N;
        j = rand() % N;
        for (int m = 0; m < N; m++)
        {
            matrix[i][m] += matrix[j][m];
        }
    }
}
void mpi_lu()
{
    for (int k = 0; k < N; k++)
    {
        if (mpi_rank == k % mpi_size)
        {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] /= matrix[k][k];//除法操作
            matrix[k][k] = 1.0;
        }
        MPI_Bcast(matrix[k], N, MPI_FLOAT, k % mpi_size, MPI_COMM_WORLD);
        for (int i = k + 1; i < N; i++)
        {
            if (mpi_rank == i % mpi_size)
            {
                for (int j = k + 1; j < N; j++)
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];//消去操作
                matrix[i][k] = 0.0;
            }
        }
    }
}
int main()
{
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    //initial
    if (mpi_rank == 0)
    {
        setmatrix();
    }
    MPI_Bcast(matrix[0], N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //start time
    struct timeval val;
    int ret = gettimeofday(&val, NULL);
    if (ret == -1)
    {
        printf("Error: gettimeofday()\n");
        return ret;
    }
    mpi_lu();
    MPI_Barrier(MPI_COMM_WORLD);
    //end time
    struct timeval newVal;
    ret = gettimeofday(&newVal, NULL);
    if (ret == -1)
    {
        printf("Error: gettimeofday()\n");
        return ret;
    }
    if (mpi_rank == 0)
    {
        //matrix_print();
        //print time
        printf("start: sec --- %ld, usec --- %ld\n", val.tv_sec, val.tv_usec);
        printf("end:   sec --- %ld, usec --- %ld\n", newVal.tv_sec, newVal.tv_usec);
        //time sub
        unsigned long diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
        printf("diff:  sec --- %ld, usec --- %ld\n", diff / Converter, diff % Converter);
    }
    MPI_Finalize();
    return 0;
}