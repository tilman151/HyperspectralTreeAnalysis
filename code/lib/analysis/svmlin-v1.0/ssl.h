/*    Copyright 2006 Vikas Sindhwani (vikass@cs.uchicago.edu)
      SVM-lin: Fast SVM Solvers for Supervised and Semi-supervised Learning

      This file is part of SVM-lin.      

      SVM-lin is free software; you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation; either version 2 of the License, or
      (at your option) any later version.
 
      SVM-lin is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with SVM-lin (see gpl.txt); if not, write to the Free Software
      Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/
#ifndef _svmlin_H
#define _svmlin_H
#include <vector>
#include <ctime>

using namespace std;

/* OPTIMIZATION CONSTANTS */
#define CGITERMAX 10000 /* maximum number of CGLS iterations */
#define SMALL_CGITERMAX 10 /* for heuristic 1 in reference [2] */
#define EPSILON   1e-6 /* most tolerances are set to this value */
#define BIG_EPSILON 0.01 /* for heuristic 2 in reference [2] */
#define RELATIVE_STOP_EPS 1e-9 /* for L2-SVM-MFN relative stopping criterion */
#define MFNITERMAX 50 /* maximum number of MFN iterations */
#define TSVM_ANNEALING_RATE 1.5 /* rate at which lambda_u is increased in TSVM */
#define TSVM_LAMBDA_SMALL 1e-5 /* lambda_u starts from this value */
#define DA_ANNEALING_RATE 1.5 /* annealing rate for DA */
#define DA_INIT_TEMP 10 /* initial temperature relative to lambda_u */
#define DA_INNER_ITERMAX 100 /* maximum fixed temperature iterations for DA */
#define DA_OUTER_ITERMAX 30 /* maximum number of outer loops for DA */

#define VERBOSE_CGLS 0

/* Data: Input examples are stored in sparse (Compressed Row Storage) format */
struct sparseData
{
  size_t  m; /* number of examples */
  size_t  l; /* number of labeled examples */
  size_t  u; /* number of unlabeled examples l+u = m */
  size_t  n; /* number of features */ 
  size_t  nz; /* number of non-zeros */
  double *val; /* data values (nz elements) [CRS format] */
  size_t *rowptr; /* n+1 vector [CRS format] */
  size_t *colind; /* nz elements [CRS format] */ 
  double *Y;   /* labels */
  double *C;   /* cost associated with each example */
};

struct vector_double /* defines a vector of doubles */
{
  size_t  d; /* number of elements */
  double *vec; /* ptr to vector elements*/
};



struct vector_size_t  /* defines a vector of size_t s for index subsets */
{
  size_t  d; /* number of elements */
  size_t  *vec; /* ptr to vector elements */
};

enum { RLS, SVM, TSVM, DA_SVM }; /* currently implemented algorithms */

struct options 
{
  /* user options */
  size_t  algo; /* 1 to 4 for RLS,SVM,TSVM,DASVM */
  double lambda; /* regularization parameter */
  double lambda_u; /* regularization parameter over unlabeled examples */
  size_t  S; /* maximum number of TSVM switches per fixed-weight label optimization */
  double R; /* expected fraction of unlabeled examples in positive class */
  double Cp; /* cost for positive examples */
  double Cn; /* cost for negative examples */
  /*  size_t ernal optimization options */    
  double epsilon; /* all tolerances */
  size_t  cgitermax;  /* max iterations for CGLS */
  size_t  mfnitermax; /* max iterations for L2_SVM_MFN */
  
};

class timer { /* to output run time */
protected:
  double start, finish;
public:
  vector<double> times;
  void record() {
    times.push_back(time());
  }
  void reset_vectors() {
    times.erase(times.begin(), times.end());
  }
  void restart() { start = clock(); }
  void stop() { finish = clock(); }
  double time() const { return ((double)(finish - start))/CLOCKS_PER_SEC; }
};
class Delta { /* used in line search */
 public: 
   Delta() {delta=0.0; index=0;s=0;};  
   double delta;   
   size_t  index;
   size_t  s;   
};
inline bool operator<(const Delta& a , const Delta& b) { return (a.delta < b.delta);};

void initialize(struct vector_double *A, size_t  k, double a);  
/* initializes a vector_double to be of length k, all elements set to a */
void initialize(struct vector_size_t  *A, size_t  k); 
/* initializes a vector_size_t  to be of length k, elements set to 1,2..k. */
void SetData(struct sparseData *Data, size_t  m,size_t  n, size_t  l,size_t  u, size_t  nz, 
	     double *VAL, size_t  *R, size_t  *C, double *Y, double *COSTS); /* sets data fields */
void GetLabeledData(struct sparseData *Data_Labeled, const struct sparseData *Data); 
/* extracts labeled data from Data and copies it size_t o Data_Labeled */
void Write(const char *file_name, const struct vector_double *somevector);
/* writes a vector size_t o filename, one element per line */
void Clear(struct sparseData *a); /* deletes a */
void Clear(struct vector_double *a); /* deletes a */
void Clear(struct vector_size_t  *a); /* deletes a */
double norm_square(const vector_double *A); /* returns squared length of A */

/* ssl_train: takes data, options, uninitialized weight and output
   vector_doubles, routes it to the algorithm */
/* the learnt weight vector and the outputs it gives on the data matrix are saved */
void ssl_train(struct sparseData *Data, 
	       struct options *Options,
	       struct vector_double *W, /* weight vector */
	       struct vector_double *O); /* output vector */

/* Main svmlin Subroutines */
/*ssl_predict: reads test inputs from input_file_name, a weight vector, and an 
 uninitialized outputs vector. Performs */
void ssl_predict(char *inputs_file_name, const struct vector_double *Weights, 
		 struct vector_double *Outputs);
/* ssl_evaluate: if test labels are given in the vector True, and predictions in vector Output,
   this code prsize_t s out various performance statistics. Currently only accuracy. */
void ssl_evaluate(struct vector_double *Outputs,struct vector_double *True);
 
/* svmlin algorithms and their subroutines */
 
/* Conjugate Gradient for Sparse Linear Least Squares Problems */
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_{i in Subset} Data->C[i] (Y[i]- w' x_i)^2 */
/* over a subset of examples x_i specified by vector_size_t  Subset */
size_t  CGLS(const struct sparseData *Data, 
	 const struct options *Options, 
	 const struct vector_size_t  *Subset,
	 struct vector_double *Weights,
	 struct vector_double *Outputs);

/* Linear Modified Finite Newton L2-SVM*/
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_i Data->C[i] max(0,1 - Y[i] w' x_i)^2 */
size_t  L2_SVM_MFN(const struct sparseData *Data, 
	       struct options *Options, 
	       struct vector_double *Weights,
	       struct vector_double *Outputs,
	       size_t  ini); /* use ini=0 if no good starting guess for Weights, else 1 */
double line_search(double *w, 
                   double *w_bar,
                   double lambda,
                   double *o, 
                   double *o_bar, 
                   double *Y, 
                   double *C,
                   size_t  d,
                   size_t  l);

/* Transductive L2-SVM */
/* Solves : min_(w, Y[i],i in UNlabeled) 0.5*Options->lamda*w'*w + 0.5*(1/Data->l)*sum_{i in labeled} max(0,1 - Y[i] w' x_i)^2 + 0.5*(Options->lambda_u/Data->u)*sum_{i in UNlabeled} max(0,1 - Y[i] w' x_i)^2 
 subject to: (1/Data->u)*sum_{i in UNlabeled} max(0,Y[i]) = Options->R */
size_t    TSVM_MFN(const struct sparseData *Data, 
	      struct options *Options, 
	      struct vector_double *Weights,
	      struct vector_double *Outputs);
size_t  switch_labels(double* Y, double* o, size_t * JU, size_t  u, size_t  S);

/* Deterministic Annealing*/
size_t  DA_S3VM(struct sparseData *Data, 
	   struct options *Options, 
	   struct vector_double *Weights,
	   struct vector_double *Outputs);
void optimize_p(const double* g, size_t  u, double T, double r, double*p);
size_t  optimize_w(const struct sparseData *Data, 
	       const  double *p,
	       struct options *Options, 
	       struct vector_double *Weights,
	       struct vector_double *Outputs,
	       size_t  ini);
double transductive_cost(double normWeights,double *Y, double *Outputs, size_t  m, double lambda,double lambda_u);
double entropy(const  double *p, size_t  u); 
double KL(const  double *p, const  double *q, size_t  u); /* KL-divergence */

#endif
