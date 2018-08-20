#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*
 * Jonathan Land 
 * 3/1/2018
 */

//prototype methods for eigen calculation, via power method 
double retNormalization(int, double*);
void normalize(int, double, double*, double*);
void multiply(int, double*, double*); 

//iteration epsilon value, as stated in assignment (p1)
#define EPSILON        (1.0e-10)

//buffer (file line feed) of string in order to allocate memory
#define MAX            (250)

//pointers to primary/initial and secondary array in matrix
int* ia;
int* ja;
double* mat;

int main(int argc,char *argv[])
{
    int ia_index  = 0;
    int ja_index  = 0;
    int mat_index = 0;
    double* x;
    double* ans;
    
    int i;
    int nonZero = 0;
    int fileLine = 0;
    int size = 0;
    char str[MAX];

	/* C code for checking, reading/structuring of matrix file */
	
    FILE *file;

	//if greater than 1 there is value in argv, so we take v
    if (argc > 1) omp_set_num_threads(atoi(argv[1]));

	//have to plan for what happens if there are issues with the file
	//or if there is nothing to read in the file
    if (argc == 3)
    {
        if ((file = fopen(argv[2], "r")) == NULL)
        {
        	//program exits if file errors 
            printf("Problem opening file"); 
            return 0;
        }
    }
    
    else
    {
       if ((file = fopen("mat.big3", "r")) == NULL)
       {
       		//Program exits if file errors and  returns 0.
           printf("Problem opening file"); 
           return 0;
       }
    }
    
    //pass in the max length here and then increase line by line of the file
    //start reading the file and look in matrix file
    while(fgets(str, MAX, file))
    {
        fileLine++;
     
    //each time compiler reads this line of the matrix, then
	if (fileLine == 2)//
        {
        	//take size whatever is in the matrix file
            size = atoi(str);
            
            //and allocates initial array of size of matrix file
            if (size > 0)
            {
               ia = (int*) malloc((size + 1) * sizeof(int));
            }
        }
        
        //each time compiler reads this line of the matrix, then
        if (fileLine == 4)
        {
        	//gets nonzero and if greater than 1, then allocating in another
        	//array of integer of size non-zero, and also allocate an array of doubles
            nonZero = atoi(str);
            if (nonZero > 0)
            {
               ja = (int*) malloc((nonZero) * sizeof(int));
               mat = (double*) malloc((nonZero) * sizeof(double));
            }
        }
 
 		//start at line 5 in matrix file and loads into initial array (ia)
        if (fileLine > 5 && fileLine <= (size + 6))
        {
            ia[ia_index] = atoi(str);
            ia_index++;
        }
        
        //starting at line in the file which is the size of initial array, plus 7 
        //and loading values into the ja array (secondary array)
        if (fileLine > (size + 7) && fileLine <= (size + nonZero + 7))
        {
            ja[ja_index] = atoi(str);
            ja_index++;
        }
        
        //after secondary array loads, starts at 0, and loading three different
        //arrays from the data in that file
        if (fileLine > (size + nonZero + 8) && fileLine <= (size+( 2 * nonZero) + 8))
        {
            mat[mat_index] = atof(str);
            mat_index++;
        }
    }

    //2 vectors for normalization calculations
    
    //allocate space for ans vector
    ans = (double*) malloc(size * sizeof(double)); 
    //allocate space for x vector 
    x = (double*) malloc(size * sizeof(double)); 

    //initialize elements of x[] = 1
    for (i = 0; i < size; i++) x[i] = .5; 

    double tNorm = 0;
    double norm  = 0;
    int cnt = 1;

	//timing vars
    double start, end;         
    //total time spent on multiply
    double multiply_time = 0; 
    //total time spent on normalizing after multiplication
    double normalize_time = 0; 
    //Total time spend on getting eigenvalue
    double norm_time = 0;

	//timing tests for methods
    while (1)
    {
    
    	//timing for multiply method
       double start = omp_get_wtime();
       multiply(size, x, ans);
       double end = omp_get_wtime();
       multiply_time += end - start;
       
       //timing for normalization method
       start = omp_get_wtime();
       norm = retNormalization(size, ans);
       end = omp_get_wtime();
       norm_time +=  end - start;

       if (cnt > 1)
       {
           if (fabs(tNorm - norm) < EPSILON) break;
       }
       tNorm = norm;
      
       //timing to normalize
       start = omp_get_wtime();
       normalize(size, norm, ans, x);
       end = omp_get_wtime();
       normalize_time += end - start;
       cnt++;
       //limit number of iterations
       if (cnt == 10) break;
    }

    printf("\n>>Iterations: %d \n>>Eigenvalue: %.12f\n", cnt, norm);
    printf(">>Number Of Threads: %d\n", omp_get_max_threads());
    printf(">>Timings:    Multiply \t Normalize\t Eigenvalue\n");
    printf("              %lf \t %lf \t %lf\n", multiply_time, normalize_time, norm_time);
    printf("\n");

    //now free mem and close the file
    free(ia); free(ja); free(mat);
    fclose(file);
    
    return 0;
}

//2 arg method to calculate norm
double retNormalization(int size, double* ans)
{
 	//need to declare this outside the pragmas to avoid race conditions
    double sum = 0.0;
    int i;
    double lSum = 0.0; //if needed
    
//main pragma for parallelization which sums the values of each using a number of clauses, esp. reduction 

//have to make sure each thread has own var, but that they can still share data

//default none: requires that each data var is visible to the parallelized block
//first private: scope of data vars private to each thread so no race conditions with sum
//shared: size, ans shared across all threads
//reduction: we have to have a way to "combine" all the private copies owned by those processes
// with the shared reduction var (sum). 
#pragma omp parallel default(none) private(i) firstprivate(lSum) shared(size, ans) reduction(+:sum)
{
//distribute loop iterations to team of threads
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        sum += ans[i] * ans[i];
        //lSum += ans[i] * ans[i];

    }
    //sum = sum + lSum;
}
    return sqrt(sum);	
}

//method to normalize the vector 
void normalize(int size, double norm, double* ans, double* x)
{
    int i;
#pragma omp parallel for default(none) private(i) shared(norm, size, x, ans)
    for (i = 0; i < size; i++)
    {
    	//where the normalization is happening
        x[i] = ans[i]/norm;
    }
}

//method to calculate matrix by vector product
void multiply(int size, double* x, double* ans)
{
    int row, col;
#pragma omp parallel for default(none) private(row, col) shared(ia, ja, mat, size, x, ans)
    for (row = 0; row < size; row++)
    {
       double localV = 0;
       for (col = ia[row]; col < ia[row+1]; col++)
       {
           int j = ja[col-1]-1;
           localV += mat[col-1] * x[j];
       }
       ans[row] = localV;
    }
}
