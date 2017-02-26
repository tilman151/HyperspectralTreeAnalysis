#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ssl.h"
#include "mex.h"

void parse_command_line(double* param);

 
struct options Options;
struct sparseData Data;
struct vector_double Weights;
struct vector_double Outputs;
 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
  double *param;
  
  fprintf(stdout, " SVM_lin (v1.0)  (running thru Matlab)\n\n"); 

  if (mxGetM(prhs[0]) != 7)
    mexErrMsgTxt("Options vector must have 7 elements.");
  if (nrhs < 3) {
    mexErrMsgTxt("At least 4 input arguments required.");
  }
  param=mxGetPr(prhs[0]);
  parse_command_line(param);
  Data.colind  = mxGetIr(prhs[1]);
  Data.rowptr  = mxGetJc(prhs[1]);
  Data.val  = mxGetPr(prhs[1]);
  Data.n = mxGetM(prhs[1]);
  Data.m = mxGetN(prhs[1]);
  Data.nz= mxGetNzmax(prhs[1]);
 
  if(nrhs==4)
    {
      Data.Y = mxGetPr(prhs[2]);
      Data.C = mxGetPr(prhs[3]);
      if (mxGetM(prhs[3]) != Data.m)
	mexErrMsgTxt("Number of examples and costs do not match.");
      if (mxGetM(prhs[2]) != Data.m)
	mexErrMsgTxt("Number of examples and labels do not match.");
       Data.u=0;
       Data.l=0;
      for(int i=0;i<Data.m;i++)
	{
	  if (Data.Y[i]==0.0)
	    Data.u++;
	  else
	    Data.l++;
	}	
      plhs[0]= mxCreateDoubleMatrix(Data.n,1,mxREAL);
      double* w = mxGetPr(plhs[0]);
      plhs[1]= mxCreateDoubleMatrix(Data.m,1,mxREAL);
      double* o = mxGetPr(plhs[1]);
      ssl_train(&Data,&Options,&Weights,&Outputs);
      for(int i=0;i<Data.n;i++)
	w[i]=Weights.vec[i];
      for(int i=0;i<Data.m;i++)
	o[i]=Outputs.vec[i];
      delete[] Weights.vec;
      delete[] Outputs.vec;
    }
  else
    {  
      double *w = mxGetPr(prhs[4]);
      if (mxGetM(prhs[4]) != Data.n)
	mexErrMsgTxt("Number of features and weights do not match.");
      plhs[0]= mxCreateDoubleMatrix(Data.m,1,mxREAL); 
      double *o = mxGetPr(plhs[0]);
      double t;
      size_t *ir = Data.rowptr;
      size_t *jc = Data.colind;
      double *pr = Data.val;

      for(register int i=0; i < Data.m; i++)
	{
	  t=0.0;
	  for(register int j=ir[i]; j < ir[i+1]; j++)
	    t+=pr[j]*w[jc[j]];
	  o[i]=t;
	}       
      Outputs.d=Data.m;
      Outputs.vec=new double[Data.m];
      for(register int i=0; i < Data.m; i++)
	Outputs.vec[i]=o[i];
      
      if(mxGetM(prhs[2])>0) /* evaluate */
	{	
	  struct vector_double Labels;
	  Labels.vec = mxGetPr(prhs[2]);
	  Labels.d = mxGetM(prhs[2]);
	  ssl_evaluate(&Outputs,&Labels);
	}
      delete[] Outputs.vec;
    }
}

void parse_command_line(double *param)
{
  Options.algo = (int) param[0];
  Options.lambda=param[1];
  Options.lambda_u=param[2];
  Options.S= (int) param[3];
  Options.R= param[4];
  Options.epsilon=EPSILON;
  Options.cgitermax= (int) CGITERMAX;
  Options.mfnitermax=(int) MFNITERMAX;
  Options.Cp = param[5];
  Options.Cn = param[6];
  
}
