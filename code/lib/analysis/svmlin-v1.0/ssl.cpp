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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <ctype.h>
#include "ssl.h"

#define VERBOSE 1
#define LOG2(x) 1.4426950408889634*log(x) 
// for compatibility issues, not using log2

using namespace std;

void ssl_train(struct sparseData *Data, 
	       struct options *Options, 
	       struct vector_double *Weights,
	       struct vector_double *Outputs)
{
  // initialize 
  initialize(Weights,Data->n,0.0);
  initialize(Outputs,Data->m,0.0);
  vector_size_t     *Subset  = new vector_size_t [1];
  initialize(Subset,Data->m);
  // call the right algorithm
  size_t  optimality = 0;
  switch(Options->algo)
    {
    case -1:
      cout << "Regularized Least Squares Regression (CGLS)" << endl;
      optimality=CGLS(Data,Options,Subset,Weights,Outputs);
      break;
    case RLS:
      cout << "Regularized Least Squares Classification (CGLS)" << endl;
      optimality=CGLS(Data,Options,Subset,Weights,Outputs);
      break;
    case SVM:
      cout << "Modified Finite Newton L2-SVM (L2-SVM-MFN)" << endl;
      optimality=L2_SVM_MFN(Data,Options,Weights,Outputs,0);
      break;
    case TSVM:
      cout << "Transductive L2-SVM (TSVM)" << endl;
      optimality=TSVM_MFN(Data,Options,Weights,Outputs);
      break;
    case DA_SVM:
      cout << "Deterministic Annealing Semi-supervised L2-SVM (DAS3VM)" << endl;
      optimality=DA_S3VM(Data,Options,Weights,Outputs);
      break;
    default:
      ;
    }
  return;
} 
size_t  CGLS(const struct sparseData *Data, 
	 const struct options *Options, 
	 const struct vector_size_t  *Subset, 
	 struct vector_double *Weights,
	 struct vector_double *Outputs)
{
  //if(VERBOSE_CGLS)
    //cout << "CGLS starting..." << endl;
  /* Disassemble the structures */
  timer tictoc;
  tictoc.restart();
  size_t  active = Subset->d;
  size_t  *J = Subset->vec;
  double *val = Data->val;
  size_t  *row = Data->rowptr;
  size_t  *col = Data->colind;
  double *Y = Data->Y;
  double *C = Data->C;
  size_t  n  = Data->n;
  double lambda = Options->lambda;
  size_t  cgitermax = Options->cgitermax;
  double epsilon = Options->epsilon;
  double *beta = Weights->vec;
  double *o  = Outputs->vec; 
  // initialize z 
  double *z = new double[active];
  double *q = new double[active];
  size_t  ii=0;
  for(size_t  i = active ; i-- ;){
    ii=J[i];      
    z[i]  = C[ii]*(Y[ii] - o[ii]);
  }
  double *r = new double[n];
  for(size_t  i = n ; i-- ;)
    r[i] = 0.0;
  for(register size_t  j=0; j < active; j++)
    {
      ii=J[j];
      for(register size_t  i=row[ii]; i < row[ii+1]; i++)
	r[col[i]]+=val[i]*z[j];
    }
  double *p = new double[n];   
  double omega1 = 0.0;
  for(size_t  i = n ; i-- ;)
    {
      r[i] -= lambda*beta[i];
      p[i] = r[i];
      omega1 += r[i]*r[i];
    }   
  double omega_p = omega1;
  double omega_q = 0.0;
  double inv_omega2 = 1/omega1;
  double scale = 0.0;
  double omega_z=0.0;
  double gamma = 0.0;
  size_t  cgiter = 0;
  size_t  optimality = 0;
  double epsilon2 = epsilon*epsilon;   
  // iterate
  while(cgiter < cgitermax)
    {
      cgiter++;
      omega_q=0.0;
      double t=0.0;
      register size_t  i,j; 
      // #pragma omp parallel for private(i,j)
      for(i=0; i < active; i++)
	{
	  ii=J[i];
	  t=0.0;
	  for(j=row[ii]; j < row[ii+1]; j++)
	    t+=val[j]*p[col[j]];
	  q[i]=t;
	  omega_q += C[ii]*t*t;
	}       
      gamma = omega1/(lambda*omega_p + omega_q);    
      inv_omega2 = 1/omega1;     
      for(size_t  i = n ; i-- ;)
	{
	  r[i] = 0.0;
	  beta[i] += gamma*p[i];
	} 
      omega_z=0.0;
      for(size_t  i = active ; i-- ;)
	{
	  ii=J[i];
	  o[ii] += gamma*q[i];
	  z[i] -= gamma*C[ii]*q[i];
	  omega_z+=z[i]*z[i];
	} 
      for(register size_t  j=0; j < active; j++)
	{
	  ii=J[j];
	  t=z[j];
	  for(register size_t  i=row[ii]; i < row[ii+1]; i++)
	    r[col[i]]+=val[i]*t;
	}
      omega1 = 0.0;
      for(size_t  i = n ; i-- ;)
	{
	  r[i] -= lambda*beta[i];
	  omega1 += r[i]*r[i];
	}
      if(VERBOSE_CGLS)
	//cout << "..." << cgiter << " ( " << omega1 << " )" ; 
      if(omega1 < epsilon2*omega_z)
	{
	  optimality=1;
	  break;
	}
      omega_p=0.0;
      scale=omega1*inv_omega2;
      for(size_t  i = n ; i-- ;)
	{
	  p[i] = r[i] + p[i]*scale;
	  omega_p += p[i]*p[i]; 
	} 
    }            
  if(VERBOSE_CGLS)
    //cout << "...Done." << endl;
  tictoc.stop();
  //cout << "CGLS converged in " << cgiter << " iteration(s) and " << tictoc.time() << " seconds." << endl;
  delete[] z;
  delete[] q;
  delete[] r;
  delete[] p;
  return optimality;
}
size_t  L2_SVM_MFN(const struct sparseData *Data, 
	       struct options *Options, 
	       struct vector_double *Weights,
	       struct vector_double *Outputs,
	       size_t  ini)
{ 
  /* Disassemble the structures */  
  timer tictoc;
  tictoc.restart();
  double *val = Data->val;
  size_t  *row = Data->rowptr;
  size_t  *col = Data->colind;
  double *Y = Data->Y;
  double *C = Data->C;
  size_t  n  = Data->n;
  size_t  m  = Data->m;
  double lambda = Options->lambda;
  double epsilon;
  double *w = Weights->vec;
  double *o = Outputs->vec; 
  double F_old = 0.0;
  double F = 0.0;
  double diff=0.0;
  vector_size_t  *ActiveSubset = new vector_size_t [1];
  ActiveSubset->vec = new size_t [m];
  ActiveSubset->d = m;
  // initialize
  if(ini==0) {
    epsilon=BIG_EPSILON; 
    Options->cgitermax=SMALL_CGITERMAX; 
    Options->epsilon=BIG_EPSILON;
  }
  else {epsilon = Options->epsilon;}  
  for(size_t  i=0;i<n;i++) F+=w[i]*w[i];
  F=0.5*lambda*F;        
  size_t  active=0;
  size_t  inactive=m-1; // l-1      
  for(size_t  i=0; i<m ; i++)
    { 
      diff=1-Y[i]*o[i];
      if(diff>0)
	{
	  ActiveSubset->vec[active]=i;
	  active++;
	  F+=0.5*C[i]*diff*diff;
	}
      else
	{
	  ActiveSubset->vec[inactive]=i;
	  inactive--;
	}   
    }
  ActiveSubset->d=active;        
  size_t  iter=0;
  size_t  opt=0;
  size_t  opt2=0;
  vector_double *Weights_bar = new vector_double[1];
  vector_double *Outputs_bar = new vector_double[1];
  double *w_bar = new double[n];
  double *o_bar = new double[m];
  Weights_bar->vec=w_bar;
  Outputs_bar->vec=o_bar;
  Weights_bar->d=n;
  Outputs_bar->d=m;
  double delta=0.0;
  double t=0.0;
  size_t  ii = 0;
  while(iter<MFNITERMAX)
    {
      iter++;
      //cout << "L2_SVM_MFN Iteration# " << iter << " (" << active << " active examples, " << " objective_value = " << F << ")" << endl;
      for(size_t  i=n; i-- ;) 
	w_bar[i]=w[i];
      for(size_t  i=m; i-- ;)  
	o_bar[i]=o[i];
      //cout << " " ;
      opt=CGLS(Data,Options,ActiveSubset,Weights_bar,Outputs_bar);
      for(register size_t  i=active; i < m; i++) 
	{
	  ii=ActiveSubset->vec[i];   
	  t=0.0;
	  for(register size_t  j=row[ii]; j < row[ii+1]; j++)
	    t+=val[j]*w_bar[col[j]];
	  o_bar[ii]=t;
	}
      if(ini==0) {Options->cgitermax=CGITERMAX; ini=1;};
      opt2=1;
      for(size_t  i=0;i<m;i++)
	{ 
	  ii=ActiveSubset->vec[i];
	  if(i<active)
	    opt2=(opt2 && (Y[ii]*o_bar[ii]<=1+epsilon));
	  else
	    opt2=(opt2 && (Y[ii]*o_bar[ii]>=1-epsilon));  
	  if(opt2==0) break;
	}      
      if(opt && opt2) // l
	{
	  if(epsilon==BIG_EPSILON) 
	    {
	      epsilon=EPSILON;
	      Options->epsilon=EPSILON;
	      //cout << "  epsilon = " << BIG_EPSILON << " case converged (speedup heuristic 2). Continuing with epsilon=" <<  EPSILON << endl;
	      continue;
	    }
	  else
	    {
	      for(size_t  i=n; i-- ;) 
		w[i]=w_bar[i];      
	      for(size_t  i=m; i-- ;)
		o[i]=o_bar[i]; 
	       delete[] ActiveSubset->vec;
	       delete[] ActiveSubset;
	       delete[] o_bar;
	       delete[] w_bar;
	       delete[] Weights_bar;
	       delete[] Outputs_bar;
	       tictoc.stop();
	       //cout << "L2_SVM_MFN converged (optimality) in " << iter << " iteration(s) and "<< tictoc.time() << " seconds. \n" << endl;
	       return 1;      
	    }
	}
      //cout << " " ;
      delta=line_search(w,w_bar,lambda,o,o_bar,Y,C,n,m); 
      //cout << "LINE_SEARCH delta = " << delta << endl;     
      F_old=F;
      F=0.0;
      for(size_t  i=n; i-- ;){ 
	w[i]+=delta*(w_bar[i]-w[i]);
	F+=w[i]*w[i];
      }
      F=0.5*lambda*F;      
      active=0;
      inactive=m-1;  
      for(size_t  i=0; i<m ; i++)
	{
	  o[i]+=delta*(o_bar[i]-o[i]);
	  diff=1-Y[i]*o[i];
	  if(diff>0)
	    {
	      ActiveSubset->vec[active]=i;
	      active++;
	      F+=0.5*C[i]*diff*diff;
	    }
	  else
	    {
	      ActiveSubset->vec[inactive]=i;
	      inactive--;
	    }   
	}
      ActiveSubset->d=active;      
      if(fabs(F-F_old)<RELATIVE_STOP_EPS*fabs(F_old))
	{
	  //cout << "L2_SVM_MFN converged (rel. criterion) in " << iter << " iterations and "<< tictoc.time() << " seconds. \n" << endl;
	  return 2;
	}
    }
  delete[] ActiveSubset->vec;
  delete[] ActiveSubset;
  delete[] o_bar;
  delete[] w_bar;
  delete[] Weights_bar;
  delete[] Outputs_bar;
  tictoc.stop();
  //cout << "L2_SVM_MFN converged (max iter exceeded) in " << iter << " iterations and "<< tictoc.time() << " seconds. \n" << endl;
  return 0;
}

double line_search(double *w, 
                   double *w_bar,
                   double lambda,
                   double *o, 
                   double *o_bar, 
                   double *Y, 
                   double *C,
                   size_t  d, /* data dimensionality -- 'n' */
                   size_t  l) /* number of examples */                  
{                       
   double omegaL = 0.0;
   double omegaR = 0.0;
   double diff=0.0;   
   for(size_t  i=d; i--; )
       {
         diff=w_bar[i]-w[i];  
         omegaL+=w[i]*diff;
         omegaR+=w_bar[i]*diff;
   }
   omegaL=lambda*omegaL;
   omegaR=lambda*omegaR;
   double L=0.0;
   double R=0.0;
   size_t  ii=0;
   for(size_t  i=0;i<l;i++)
       {
         if(Y[i]*o[i]<1)
	   {
	     diff=C[i]*(o_bar[i]-o[i]);  
	     L+=(o[i]-Y[i])*diff;
	     R+=(o_bar[i]-Y[i])*diff;
	   }
       }
   L+=omegaL;
   R+=omegaR;
   Delta* deltas=new Delta[l];    
   size_t  p=0;
   for(size_t  i=0;i<l;i++)
     { 
       diff=Y[i]*(o_bar[i]-o[i]);
       
       if(Y[i]*o[i]<1)
	 {
	   if(diff>0)
	     {
	       deltas[p].delta=(1-Y[i]*o[i])/diff;
	       deltas[p].index=i;
	       deltas[p].s=-1;
	       p++;
	     }
	 }
       else
	 {
	   if(diff<0)
	     {
	       deltas[p].delta=(1-Y[i]*o[i])/diff;
	       deltas[p].index=i;
	       deltas[p].s=1;      
	       p++;
	     }
	 }
     }
   sort(deltas,deltas+p);            
   double delta_prime=0.0;  
   for(size_t  i=0;i<p;i++)
     {
       delta_prime = L + deltas[i].delta*(R-L);       
       if(delta_prime>=0)
	 break;
       ii=deltas[i].index;   
       diff=(deltas[i].s)*C[ii]*(o_bar[ii]-o[ii]);
       L+=diff*(o[ii]-Y[ii]);
       R+=diff*(o_bar[ii]-Y[ii]);
     }   
   delete [] deltas;
   return (-L/(R-L));
} 
size_t  TSVM_MFN(const struct sparseData *Data, 
	      struct options *Options, 
	      struct vector_double *Weights,
	      struct vector_double *Outputs)
{
  /* Setup labeled-only examples and train L2_SVM_MFN */
  timer tictoc;
  tictoc.restart();  
  struct sparseData *Data_Labeled = new sparseData[1];
  struct vector_double *Outputs_Labeled = new vector_double[1];
  initialize(Outputs_Labeled,Data->l,0.0);
  //cout << "Initializing weights, unknown labels" << endl; 
  GetLabeledData(Data_Labeled,Data); /* gets labeled data and sets C=1/l */
  L2_SVM_MFN(Data_Labeled, Options, Weights,Outputs_Labeled,0);
  Clear(Data_Labeled);
  /* Use this weight vector to classify R*u unlabeled examples as
       positive*/   
  size_t  p=0,q=0; 
  double t=0.0;
  size_t  *JU = new size_t [Data->u];
  double *ou = new double[Data->u];
  double lambda_0 = TSVM_LAMBDA_SMALL;
  for(size_t  i=0;i<Data->m;i++)
    {
      if(Data->Y[i]==0.0)
	{
	  t=0.0;
	  for(size_t  j=Data->rowptr[i]; j < Data->rowptr[i+1]; j++)
	    t+=Data->val[j]*Weights->vec[Data->colind[j]];
	  Outputs->vec[i]=t;
	  Data->C[i]=lambda_0*1.0/Data->u;
	  JU[q]=i;
	  ou[q]=t;
	  q++;
	}
      else
	{                
	  Outputs->vec[i]=Outputs_Labeled->vec[p];
	  Data->C[i]=1.0/Data->l;
	  p++;   
	}
    }
  nth_element(ou,ou+size_t ((1-Options->R)*Data->u-1),ou+Data->u);
  double thresh=*(ou+size_t ((1-Options->R)*Data->u)-1);
  delete [] ou;
  for(size_t  i=0;i<Data->u;i++)
    {  
      if(Outputs->vec[JU[i]]>thresh)
	Data->Y[JU[i]]=1.0;
      else
	Data->Y[JU[i]]=-1.0;
    }
  for(size_t  i=0;i<Data->n;i++)
    Weights->vec[i]=0.0;
  for(size_t  i=0;i<Data->m;i++)
    Outputs->vec[i]=0.0;
  L2_SVM_MFN(Data,Options,Weights,Outputs,0); 
  size_t  num_switches=0;
  size_t  s=0;
  size_t  last_round=0;
  while(lambda_0 <= Options->lambda_u)
    {   
      size_t  iter2=0;
      while(1){
	s=switch_labels(Data->Y,Outputs->vec,JU,Data->u,Options->S);
	if(s==0) break;
	iter2++;
	//cout << "****** lambda_0 = " << lambda_0 << " iteration = " << iter2 << " ************************************"  << endl;
	//cout << "Optimizing unknown labels. switched " << s << " labels." << endl; 
	num_switches+=s;
	//cout << "Optimizing weights" << endl;
	L2_SVM_MFN(Data,Options,Weights,Outputs,1); 
      }
      if(last_round==1) break;
      lambda_0=TSVM_ANNEALING_RATE*lambda_0;
      if(lambda_0 >= Options->lambda_u) {lambda_0 = Options->lambda_u; last_round=1;} 
      for(size_t  i=0;i<Data->u;i++)
	Data->C[JU[i]]=lambda_0*1.0/Data->u;       
      //cout << endl << "****** lambda0 increased to " << lambda_0*100/Options->lambda_u << "%" << " of lambda_u = " << Options->lambda_u << " ************************" << endl;
      //cout << "Optimizing weights" << endl;
      L2_SVM_MFN(Data,Options,Weights,Outputs,1); 
    }
  //cout <<  "Total Number of Switches = " << num_switches << endl;
  /* reset labels */
  for(size_t  i=0;i<Data->u;i++) Data->Y[JU[i]] = 0.0;
  double F = transductive_cost(norm_square(Weights),Data->Y,Outputs->vec,Outputs->d,Options->lambda,Options->lambda_u);
  //cout << "Objective Value = " << F << endl;
  delete [] JU;
  tictoc.stop();
  //cout << "Transductive L2_SVM_MFN took " << tictoc.time() << " seconds." << endl;
  return num_switches;
}
size_t  switch_labels(double* Y, double* o, size_t * JU, size_t  u, size_t  S)
{     
  size_t  npos=0;
  size_t  nneg=0;
  for(size_t  i=0;i<u;i++)
    {
      if((Y[JU[i]]>0) && (o[JU[i]]<1.0)) npos++;   
      if((Y[JU[i]]<0) && (-o[JU[i]]<1.0)) nneg++;        
    }     
  Delta* positive=new Delta[npos];
  Delta* negative=new Delta[nneg];  
  size_t  p=0;
  size_t  n=0;
  size_t  ii=0;
  for(size_t  i=0;i<u;i++)
    {
      ii=JU[i];
      if((Y[ii]>0.0) && (o[ii]<1.0)) {
	positive[p].delta=o[ii]; 
	positive[p].index=ii;
	positive[p].s=0;
	p++;};   
      if((Y[ii]<0.0) && (-o[ii]<1.0)) 
	{
	  negative[n].delta=-o[ii]; 
	  negative[n].index=ii;
	  negative[n].s=0;
	  n++;};   
    }
  sort(positive,positive+npos);
  sort(negative,negative+nneg);  
  size_t  s=-1;
  //  cout << "Switching Labels: ";
  while(1)
    {
      s++;    
      if((s>=S) || (positive[s].delta>=-negative[s].delta) || (s>=npos) || (s>=nneg))
	break;
      Y[positive[s].index]=-1.0;
      Y[negative[s].index]= 1.0;
      //  cout << positive[s].index << " " << negative[s].index << "...";
    }
  // cout << "...switched " << s  << " labels." << endl;  
  delete [] positive;
  delete [] negative;
  return s;
}

size_t  DA_S3VM(struct sparseData *Data, 
	   struct options *Options, 
	   struct vector_double *Weights,
	   struct vector_double *Outputs)
{
  timer tictoc;
  tictoc.restart();
  double T = DA_INIT_TEMP*Options->lambda_u;
  size_t  iter1 = 0, iter2 =0;
  double *p = new double[Data->u];
  double *q = new double[Data->u];
  double *g = new double[Data->u];
  double F,F_min;
  double *w_min = new double[Data->n];
  double *o_min = new double[Data->m];
  double *w = Weights->vec;
  double *o = Outputs->vec;
  double kl_divergence = 1.0;
  /*initialize */
  //cout << "Initializing weights, p" << endl; 
  for(size_t  i=0;i<Data->u; i++)
    p[i] = Options->R;
  /* record which examples are unlabeled */
  size_t  *JU = new size_t [Data->u];
  size_t  j=0;
  for(size_t  i=0;i<Data->m;i++)
    {
      if(Data->Y[i]==0.0)
	{JU[j]=i;j++;}
    }  
  double H = entropy(p,Data->u);
  optimize_w(Data,p,Options,Weights,Outputs,0);  
  F = transductive_cost(norm_square(Weights),Data->Y,Outputs->vec,Outputs->d,Options->lambda,Options->lambda_u);
  F_min = F;
  for(size_t  i=0;i<Weights->d;i++)
    w_min[i]=w[i];
  for(size_t  i=0;i<Outputs->d;i++)
    o_min[i]=o[i];
  while((iter1 < DA_OUTER_ITERMAX) && (H > Options->epsilon))
    {
      iter1++;
      iter2=0;
      kl_divergence=1.0;
      while((iter2 < DA_INNER_ITERMAX) && (kl_divergence > Options->epsilon)) 
	{
	  iter2++;
	  for(size_t  i=0;i<Data->u;i++)
	    {
	      q[i]=p[i];
	      g[i] = Options->lambda_u*((o[JU[i]] > 1 ? 0 : (1 - o[JU[i]])*(1 - o[JU[i]])) - (o[JU[i]]< -1 ? 0 : (1 + o[JU[i]])*(1 + o[JU[i]]))); 
	    }
	  //cout << "Optimizing p. " ;
	  optimize_p(g,Data->u,T,Options->R,p);
	  kl_divergence=KL(p,q,Data->u);
	  //cout << "Optimizing weights" << endl;
	  optimize_w(Data,p,Options,Weights,Outputs,1);
	  F = transductive_cost(norm_square(Weights),Data->Y,Outputs->vec,Outputs->d,Options->lambda,Options->lambda_u);
	  if(F < F_min)
	    {
	      F_min = F;
	      for(size_t  i=0;i<Weights->d;i++)
		w_min[i]=w[i];
	      for(size_t  i=0;i<Outputs->d;i++)
		o_min[i]=o[i];
	    }
	  fprintf(stdout,"***** outer_iter = %zd  T = %g  inner_iter = %zd  kl = %g  cost = %g *****\n",iter1,T,iter2,kl_divergence,F); 
	}
      H = entropy(p,Data->u); 
      fprintf(stdout,"***** Finished outer_iter = %zd T = %g  Entropy = %g ***\n", iter1,T,H);
      T = T/DA_ANNEALING_RATE;
    }
  for(size_t  i=0;i<Weights->d;i++)
    w[i]=w_min[i];
  for(size_t  i=0;i<Outputs->d;i++)
    o[i]=o_min[i];
  /* may want to reset the original Y */ 
  delete [] p; 
  delete [] q;
  delete [] g;
  delete [] JU;
  delete [] w_min;
  delete [] o_min;
  tictoc.stop();
  //cout << endl << "(min) Objective Value = " << F_min << endl;
  //cout << "DA-SVM took " << tictoc.time() << " seconds." << endl;
  return 1;
}
size_t  optimize_w(const struct sparseData *Data, 
	       const double *p,
	       struct options *Options, 
	       struct vector_double *Weights,
	       struct vector_double *Outputs,
	       size_t  ini)
{ 
  timer tictoc;
  tictoc.restart();
  double *val = Data->val;
  size_t  *row = Data->rowptr;
  size_t  *col = Data->colind;
  size_t  n  = Data->n;
  size_t  m  = Data->m;
  size_t  u  = Data->u;
  double lambda = Options->lambda;
  double epsilon;
  double *w = Weights->vec;
  double *o = new double[m+u];
  double *Y = new double[m+u];
  double *C = new double[m+u];
  size_t  *labeled_indices = new size_t [m];
  double F_old = 0.0;
  double F = 0.0;
  double diff=0.0;
  double lambda_u_by_u = Options->lambda_u/u;
  vector_size_t  *ActiveSubset = new vector_size_t [1];
  ActiveSubset->vec = new size_t [m];
  ActiveSubset->d = m;
  // initialize
  if(ini==0) 
    {
      epsilon=BIG_EPSILON; 
      Options->cgitermax=SMALL_CGITERMAX;
      Options->epsilon=BIG_EPSILON;}
  else {epsilon = Options->epsilon;}  

  for(size_t  i=0;i<n;i++) F+=w[i]*w[i];
  F=lambda*F;        
  size_t  active=0;
  size_t  inactive=m-1; // l-1      
  size_t  j = 0;
  double temp1;
  double temp2;
  for(size_t  i=0; i<m ; i++)
    { 
      o[i]=Outputs->vec[i];
      if(Data->Y[i]==0.0)
	{
	  labeled_indices[i]=0;
	  o[m+j]=o[i];
	  Y[i]=1;
	  Y[m+j]=-1;
	  C[i]=lambda_u_by_u*p[j];
	  C[m+j]=lambda_u_by_u*(1-p[j]);
	  ActiveSubset->vec[active]=i;
	  active++; 
	  diff = 1 - fabs(o[i]);
	  if(diff>0)
	    {
	      Data->Y[i] = 2*p[j]-1;
	      Data->C[i] = lambda_u_by_u;
	      temp1 = (1 - o[i]);
	      temp2 = (1 + o[i]);
	      F+=lambda_u_by_u*(p[j]*temp1*temp1 + (1-p[j])*temp2*temp2);
	    }
	  else
	    {
	      if(o[i]>0)
		{
		  Data->Y[i] = -1.0;
		  Data->C[i] = C[m+j];
		}
	      else
		{
		  Data->Y[i] = 1.0;
		  Data->C[i] = C[i];		 
		}
	      temp1 = (1-Data->Y[i]*o[i]);
	      F+= Data->C[i]*temp1*temp1;
	    }
	  j++;
	} 
      else
	{
	  labeled_indices[i]=1;
	  Y[i]=Data->Y[i];
	  C[i]=1.0/Data->l;
	  Data->C[i]=1.0/Data->l;
	  diff=1-Data->Y[i]*o[i];
	  if(diff>0)
	    {
	      ActiveSubset->vec[active]=i;
	      active++;
	      F+=Data->C[i]*diff*diff;
	    }
	  else
	    {
	      ActiveSubset->vec[inactive]=i;
	      inactive--;
	    }
	}
    }
  F=0.5*F;
  ActiveSubset->d=active;
  size_t  iter=0;
  size_t  opt=0;
  size_t  opt2=0;
  vector_double *Weights_bar = new vector_double[1];
  vector_double *Outputs_bar = new vector_double[1];
  double *w_bar = new double[n];
  double *o_bar = new double[m+u];
  Weights_bar->vec=w_bar;
  Outputs_bar->vec=o_bar;
  Weights_bar->d=n;
  Outputs_bar->d=m; /* read only the top m ; bottom u will be copies */
  double delta=0.0;
  double t=0.0;
  size_t  ii = 0;
  while(iter<MFNITERMAX)
    {
      iter++;
      //cout << "L2_SVM_MFN Iteration# " << iter << " (" << active << " active examples, " << " objective_value = " << F << ")" << endl;
      for(size_t  i=n; i-- ;) 
	w_bar[i]=w[i];
      
      for(size_t  i=m+u; i-- ;)  
	o_bar[i]=o[i];     
      //cout << " " ;
      opt=CGLS(Data,Options,ActiveSubset,Weights_bar,Outputs_bar);
      for(register size_t  i=active; i < m; i++) 
	{
	  ii=ActiveSubset->vec[i];   
	  t=0.0;
	  for(register size_t  j=row[ii]; j < row[ii+1]; j++)
	    t+=val[j]*w_bar[col[j]];
	  o_bar[ii]=t;
	}
      // make o_bar consistent in the bottom half      
      size_t  j=0;
      for(size_t  i=0; i<m;i++)
	{
	  if(labeled_indices[i]==0)
	    {o_bar[m+j]=o_bar[i]; j++;};
	}
      if(ini==0) {Options->cgitermax=CGITERMAX; ini=1;};
      opt2=1;
      for(size_t  i=0; i < m ;i++)
	{ 
	  ii=ActiveSubset->vec[i];
	  if(i<active)
	    {
	      if(labeled_indices[ii]==1)
		opt2=(opt2 && (Data->Y[ii]*o_bar[ii]<=1+epsilon));
	      else
		{
		  if(fabs(o[ii])<1)
		    opt2=(opt2 && (fabs(o_bar[ii])<=1+epsilon));
		  else
		    opt2=(opt2 && (fabs(o_bar[ii])>=1-epsilon));
		}
	    }
	  else
	    opt2=(opt2 && (Data->Y[ii]*o_bar[ii]>=1-epsilon));  
	  if(opt2==0) break;
	}
      if(opt && opt2) // l
	{
	  if(epsilon==BIG_EPSILON) 
	    {
	      epsilon=EPSILON;
	      Options->epsilon=EPSILON;
	      //cout << "  epsilon = " << BIG_EPSILON << " case converged (speedup heuristic 2). Continuing with epsilon=" <<  EPSILON << endl;
	      continue;
	    }
	  else
	    {
	      for(size_t  i=n; i-- ;) 
		w[i]=w_bar[i];      
	      for(size_t  i=m; i-- ;)
		Outputs->vec[i]=o_bar[i]; 
	      for(size_t  i=m; i-- ;)
		{
		  if(labeled_indices[i]==0)
		    Data->Y[i]=0.0;
		}
	       delete[] ActiveSubset->vec;
	       delete[] ActiveSubset;
	       delete[] o_bar;
	       delete[] w_bar;
	       delete[] o;
	       delete[] Weights_bar;
	       delete[] Outputs_bar;
	       delete[] Y;
	       delete[] C;
	       delete[] labeled_indices;
	       tictoc.stop();
	       //cout << "L2_SVM_MFN converged in " << iter << " iteration(s) and "<< tictoc.time() << " seconds. \n" << endl;
	       return 1;      
	    }
	}  
      
      delta=line_search(w,w_bar,lambda,o,o_bar,Y,C,n,m+u); 
      //cout << " LINE_SEARCH delta = " << delta << endl;     
      F_old=F;
      F=0.0;
      for(size_t  i=0;i<n;i++) {w[i]+=delta*(w_bar[i]-w[i]);  F+=w[i]*w[i];}
      F=lambda*F;
      j=0;
      active=0;
      inactive=m-1;
      for(size_t  i=0; i<m ; i++)
	{ 
	   o[i]+=delta*(o_bar[i]-o[i]);
	   if(labeled_indices[i]==0)
	    {
	      o[m+j]=o[i];
	      ActiveSubset->vec[active]=i;
	      active++; 
	      diff = 1 - fabs(o[i]);
	      if(diff>0)
		{
		  Data->Y[i] = 2*p[j]-1;
		  Data->C[i] = lambda_u_by_u;
		  temp1 = (1 - o[i]);
		  temp2 = (1 + o[i]);
		  F+=lambda_u_by_u*(p[j]*temp1*temp1 + (1-p[j])*temp2*temp2);
		}
	      else
		{
		  if(o[i]>0)
		    {
		      Data->Y[i] = -1;
		      Data->C[i] = C[m+j];
		    }
		  else
		    {
		      Data->Y[i] = +1;
		      Data->C[i] = C[i];
		    }
		  temp1=(1-Data->Y[i]*o[i]);
		  F+= Data->C[i]*temp1*temp1;
		}
	      j++;
	    }
	  else
	    {
	      diff=1-Data->Y[i]*o[i];
	      if(diff>0)
		{
		  ActiveSubset->vec[active]=i;
		  active++;
		  F+=Data->C[i]*diff*diff;
		}
	      else
		{
		  ActiveSubset->vec[inactive]=i;
		  inactive--;
		}
	    }
	}
      F=0.5*F;
      ActiveSubset->d=active;
      if(fabs(F-F_old)<EPSILON)
	break;
    }
  for(size_t  i=m; i-- ;)
    {
      Outputs->vec[i]=o[i];
      if(labeled_indices[i]==0)
	Data->Y[i]=0.0;
    }
  delete[] ActiveSubset->vec;
  delete[] labeled_indices;
  delete[] ActiveSubset;
  delete[] o_bar;
  delete[] w_bar;
  delete[] o;
  delete[] Weights_bar;
  delete[] Outputs_bar;
  delete[] Y;
  delete[] C;
  tictoc.stop();
  //cout << "L2_SVM_MFN converged in " << iter << " iterations and "<< tictoc.time() << " seconds. \n" << endl;
  return 0;
}
void optimize_p(const double* g, size_t  u, double T, double r, double* p)
{
  size_t  iter=0;
  double epsilon=1e-10;
  size_t  maxiter=500; 
  double nu_minus=g[0];
  double nu_plus=g[0];
  for(size_t  i=0;i<u;i++)
    {
      if(g[i]<nu_minus) nu_minus=g[i]; 
      if(g[i]>nu_plus) nu_plus=g[i];
    };
  
  double b=T*log((1-r)/r);
  nu_minus-=b;
  nu_plus-=b;
  double nu=(nu_plus+nu_minus)/2;
  double Bnu=0.0;
  double BnuPrime=0.0;
  double s=0.0;
  double tmp=0.0;
  for(size_t  i=0;i<u;i++)
    {
      s=exp((g[i]-nu)/T);
      if(!(isinf(s)))
	{
	  tmp=1.0/(1.0+s);
	  Bnu+=tmp;
	  BnuPrime+=s*tmp*tmp;    
	}
    }
  Bnu=Bnu/u;
  Bnu-=r;
  BnuPrime=BnuPrime/(T*u);
  double nuHat=0.0;
  while((fabs(Bnu)>epsilon) && (iter < maxiter))
    {
      iter++;
      if(fabs(BnuPrime)>0.0)
        nuHat=nu-Bnu/BnuPrime;
      if((fabs(BnuPrime) > 0.0) | (nuHat>nu_plus)  | (nuHat < nu_minus))
        nu=(nu_minus+nu_plus)/2.0;
      else
        nu=nuHat; 
      Bnu=0.0;
      BnuPrime=0.0;
      for(size_t  i=0;i<u;i++)
	{
	  s=exp((g[i]-nu)/T);
	  if(!(isinf(s)))
	    {
	      tmp=1.0/(1.0+s);
	      Bnu+=tmp;
	      BnuPrime+=s*tmp*tmp;    
	    }
	}
      Bnu=Bnu/u;
      Bnu-=r;
      BnuPrime=BnuPrime/(T*u);
      if(Bnu<0)
        nu_minus=nu;
      else
        nu_plus=nu;
      if(fabs(nu_minus-nu_plus)<epsilon)
	break;   
    }
  if(fabs(Bnu)>epsilon)
    cout << "Warning (Root): root not found to required precision" << endl; 
  
  for(size_t  i=0;i<u;i++)
    {
      s=exp((g[i]-nu)/T);
      if(isinf(s)) p[i]=0.0;
      else p[i]=1.0/(1.0+s);  
    }
  //cout << " root (nu) = " << nu << " B(nu) = " << Bnu << endl; 
}
double transductive_cost(double normWeights,double *Y, double *Outputs, size_t  m, double lambda,double lambda_u)
{
  double F1=0.0,F2=0.0, o=0.0, y=0.0; 
  size_t  u=0,l=0;
  for(size_t  i=0;i<m;i++)
    {
      o=Outputs[i];
      y=Y[i];
      if(y==0.0)
	{F1 += fabs(o) > 1 ? 0 : (1 - fabs(o))*(1 - fabs(o)); u++;}
      else
	{F2 += y*o > 1 ? 0 : (1-y*o)*(1-y*o); l++;}   
    }
  double F;
  F = 0.5*(lambda*normWeights + lambda_u*F1/u + F2/l);
  return F;
}

double entropy(const double *p, size_t  u)
{
  double h=0.0;
  double q=0.0; 
  for(size_t  i=0;i<u;i++)
    {
      q=p[i];
      if(q>0 && q<1)
	h+= -(q*LOG2(q) + (1-q)*LOG2(1-q));
    }
  return h/u;
}
double KL(const double *p, const double *q, size_t  u)
{
  double h=0.0;
  double p1=0.0;
  double q1=0.0;
  double g=0.0;
  for(size_t  i=0;i<u;i++)
    {
      p1=p[i];
      q1=q[i];
      if(p1>1-1e-8) p1-=1e-8;
      if(p1<1-1e-8) p1+=1e-8;
      if(q1>1-1e-8) q1-=1e-8;
      if(q1<1-1e-8) q1+=1e-8;
      g= (p1*LOG2(p1/q1) + (1-p1)*LOG2((1-p1)/(1-q1)));
      if(fabs(g)<1e-12 || isnan(g)) g=0.0;
      h+=g;
    }
    return h/u;   
}
/* Prediction */
/* the test file can potentially be huge..dont need to load it.*/
void ssl_predict(char *inputs_file_name, const struct vector_double *Weights, struct vector_double *Outputs)
{
  FILE *fpin;
  double *w = Weights->vec;
  size_t  n = Weights->d;
  fpin = fopen(inputs_file_name, "r");
  size_t  m=0;
  if (fpin == NULL)
    {
      std::cerr << "Cannot open input file: " << inputs_file_name << "\n";
      exit(1);
    }
 /* read and predict */
  while (1)
    {
      size_t  c = fgetc(fpin);
      switch(c)
	{
	case '\n':
	  ++m; 
	  break;
	case EOF:
	  goto out;
	default:
	  ;
	}
    }
 out:
  initialize(Outputs,m,0.0);
  rewind(fpin);
  size_t  colind;
  double val;
  double t=0.0;
  for(size_t  i=0;i<m;i++)
    {
      t=0.0;
      while(1)
	{
	  size_t  c;
	  do {
	    c = getc(fpin);
	    if(c=='\n') goto out2;
	  } while(isspace(c));
	  ungetc(c,fpin);
	  fscanf(fpin,"%zd:%lf",&colind,&val);
	  colind--;
	  if(colind<n) 
	    t+=val*w[colind];
	}
    out2: 
      Outputs->vec[i]=t + w[n-1];
    }
}
/* Evaluation -- confusion matrix, accuracy, precision, recall, prbep */
/* roc area */
void ssl_evaluate(struct vector_double *Outputs, struct vector_double *TrueLabels)
{ 
  double accuracy=0.0;
  for(size_t  i=0;i<Outputs->d;i++)
    accuracy+=(Outputs->vec[i]*TrueLabels->vec[i])>0;
  //cout << "Accuracy = " << accuracy*100.0/Outputs->d << " %" << endl;
}

/********************** UTILITIES ********************/
double norm_square(const vector_double *A)
{
  double x=0.0, t=0.0;
  for(size_t  i=0;i<A->d;i++)
    {
      t=A->vec[i];
      x+=t*t;
    }
  return x;
} 
void initialize(struct vector_double *A, size_t  k, double a)
 {
  double *vec = new double[k];
  for(size_t  i=0;i<k;i++)
    vec[i]=a;
  A->vec = vec;
  A->d   = k;
  return;
}
void initialize(struct vector_size_t  *A, size_t  k)
{  
  size_t  *vec = new size_t [k];
  for(size_t  i=0;i<k;i++)
    vec[i]=i; 
  A->vec = vec;
  A->d   = k;
  return;
}
void Write(const char *file_name, 
	   const struct vector_double *somevector)
{
  FILE* fp = fopen(file_name,"w");
  for(size_t  i=0;i<somevector->d;i++)
    fprintf(fp,"%g\n",somevector->vec[i]);
  return;
} 
void GetLabeledData(struct sparseData *D, const struct sparseData *Data)
{
  size_t  *J = new size_t [Data->l];
  D->C   = new double[Data->l];
  D->Y   = new double[Data->l];
  size_t  nz=0;
  size_t  k=0;
  size_t  rowptrs_=Data->l;
  for(size_t  i=0;i<Data->m;i++)
    {
      if(Data->Y[i]!=0.0)
	{
	  J[k]=i;
	  D->Y[k]=Data->Y[i];
	  D->C[k]=1.0/Data->l;
	  nz+=(Data->rowptr[i+1] - Data->rowptr[i]);
	  k++;
	}
    }  
  D->val    = new double[nz];
  D->colind = new size_t [nz]; 
  D->rowptr = new size_t [rowptrs_+1];
  nz=0;
  for(size_t  i=0;i<Data->l;i++)
    {
      D->rowptr[i]=nz;
      for(size_t  j=Data->rowptr[J[i]];j<Data->rowptr[J[i]+1];j++)
	{
	  D->val[nz] = Data->val[j];
	  D->colind[nz] = Data->colind[j];
	  nz++;	
	}
    }
  D->rowptr[rowptrs_]=nz;
  D->nz=nz;
  D->l=Data->l;
  D->m=Data->l;
  D->n=Data->n;
  D->u=0;
  delete [] J;
}
void SetData(struct sparseData *a, size_t  m,size_t  n, size_t  l,size_t  u, size_t  nz, double *VAL, size_t  *R, size_t  *C, double *Y, double *COSTS)
{
  a->m=m;
  a->u=u;
  a->l=m-u;
  a->n=n;
  a->nz=nz;
  a->val=VAL;
  a->rowptr=R;
  a->colind=C;
  a->Y=Y;
  a->C=COSTS;
  return;
}
void Clear(struct sparseData *a)
{     
  delete [] a->val;
  delete [] a->rowptr;
  delete [] a->colind;
  delete [] a->Y;
  delete [] a->C;
  delete [] a;
  return;
}
void Clear(struct vector_double *c)
{ delete[] c->vec; delete [] c; return;}
void Clear(struct vector_size_t  *c)
{ delete[] c->vec; delete [] c; return;}
void Clear(struct options *opt)
{ delete[] opt; delete [] opt; return;}
