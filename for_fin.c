#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double t1, t2;
double eps;
double A [N][N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
	int it;
    omp_set_num_threads((int)strtol(as[1], NULL, 10));

	t1 = omp_get_wtime();	

	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		//printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify();

	t2 = omp_get_wtime();
	printf("%.3lf\n", t2-t1);

	return 0;
}


void init()
{ 
#pragma omp parallel shared(A) private(i, j, k)
    {
#pragma omp for
        for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
        for(k=0; k<=N-1; k++)
        {
            if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
            A[i][j][k]= 0.;
            else A[i][j][k]= ( 4. + i + j + k) ;
        }
    }
} 

void relax()
{
#pragma omp parallel shared(A, eps) private(i, j, k) 
    {
    	for(i=1; i<=N-2; i++)
#pragma omp for
	    for(j=1; j<=N-2; j++)
	    for(k=1; k<=N-2; k++)
	    {
		    A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
        }

#pragma omp for
        for(i=1; i<=N-2; i++)
	    for(j=1; j<=N-2; j++)
	    for(k=1; k<=N-2; k++)
	    {
		    A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
        }
        
#pragma omp for reduction(max:eps) 
	    for(i=1; i<=N-2; i++)
	    for(j=1; j<=N-2; j++)
	    for(k=1; k<=N-2; k++)
	    {
		    double vr=A[i][j][k];
		    A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
		    eps=Max(eps,fabs(vr-A[i][j][k]));
        }
    }
}


void verify()
{
	double s;

	s=0.;

#pragma omp parallel shared(A) private(i, j, k)
    {
#pragma omp for reduction(+:s)
        for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
        for(k=0; k<=N-1; k++)
        {
            s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
        }
    }

	//printf("  SS = %lf\n",s);

}
