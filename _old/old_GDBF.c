/* ###########################################################################################################################
## Organization         : ENSEA
##                      :
## File name            : LaGDBF.c
## Language             : C (ANSI)
## Short description    : Layered - GDBF algorithm
##                      :
##                      :
##                      :
## History              : 13/02/2016 created by LE TRUNG Khoa (ENSEA)
##                      :
## COPYRIGHT            : ENSEA
## ######################################################################################################################## */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#define arrondi(x) ((ceil(x)-x)<(x-floor(x))?(int)ceil(x):(int)floor(x))
#define min(x,y) ((x)<(y)?(x):(y))
#define	max(x,y) ((x)<(y)?(y):(x))
#define SQR(A) ((A)*(A))
#define BPSK(x) (1-2*(x))
#define PI 3.1415926536

#define GDBF 1
#define PGDBF 0
#define PGDBF1 0
#define RAND_SEED    95732   // random seed





int GaussianElimination_MRB(int *Perm,int **MatOut,int **Mat,int M,int N)
{
	int k,n,m,m1,buf,ind,indColumn,nb,*Index,dep,Rank;

	Index=(int *)calloc(N,sizeof(int));

	// Triangularization
	indColumn=0;nb=0;dep=0;
	for (m=0;m<M;m++)
	{
		if (indColumn==N) { dep=M-m; break; }

		for (ind=m;ind<M;ind++) { if (Mat[ind][indColumn]!=0) break; }
		// If a "1" is found on the column, permutation of rows
		if (ind<M)
		{
			for (n=indColumn;n<N;n++) { 
				buf=Mat[m][n]; 
				Mat[m][n]=Mat[ind][n]; 
				Mat[ind][n]=buf; 
			}
		// bottom of the column ==> 0
			for (m1=m+1;m1<M;m1++)
			{
				if (Mat[m1][indColumn]==1) { for (n=indColumn;n<N;n++) Mat[m1][n]=Mat[m1][n]^Mat[m][n]; }
			}
			Perm[m]=indColumn;
		}
		// else we "mark" the column.
		else { Index[nb++]=indColumn; m--; }

		indColumn++;
	}

	Rank=M-dep;

	for (n=0;n<nb;n++) Perm[Rank+n]=Index[n];

	// Permutation of the matrix
	for (m=0;m<M;m++) { for (n=0;n<N;n++) MatOut[m][n]=Mat[m][Perm[n]]; }

	// Diagonalization
	for (m=0;m<(Rank-1);m++)
	{
		for (n=m+1;n<Rank;n++)
		{
			if (MatOut[m][n]==1) { for (k=n;k<N;k++) MatOut[m][k]=MatOut[n][k]^MatOut[m][k]; }
		}
	}
	free(Index);
	return(Rank);
}

//#####################################################################################################
int main(int argc, char * argv[])
{
  // Variables Declaration
  FILE *f,*fout; //TBD
  int Graine,NbMonteCarlo,NbIter,nbtestedframes,NBframes;
  float alpha_max, alpha_min,alpha_step,alpha,PoPGDBF;

  char *FileName,*FileMatrix,*FileBaseMatrix,*FileResult,*FileSimu,*name; //TBD
  FileName=(char *)malloc(200);
  FileMatrix=(char *)malloc(200);
  FileBaseMatrix=(char *)malloc(200);
  FileResult=(char *)malloc(200);
  FileSimu=(char *)malloc(200);
  name=(char *)malloc(200);
  // ----------------------------------------------------
  // lecture des param de la ligne de commande
  // ----------------------------------------------------
  //--------------Simulation input parameters for GDBF-------------------------
  NbMonteCarlo=atoi(argv[1]);	// Maximum nb of codewords sent
  NbIter=atoi(argv[2]); 	    // Maximum nb of iterations
  strcpy(FileMatrix,argv[3]); 	// Matrix file
  strcpy(FileBaseMatrix,argv[4]); 	// Base Matrix file
  strcpy(FileResult,argv[5]); 	// Results file
  alpha=atof(argv[6]);
  NBframes=atoi(argv[7]);       // Simulation stops when NBframes in error
  Graine=atoi(argv[8]);		    // Seed Initialization for Multiple Simulations

  PoPGDBF= 0.7;

  alpha_max= atof(argv[9]);		    //Channel Crossover Probability Max and Min
  alpha_min= atof(argv[10]);
  alpha_step= atof(argv[11]);

  // ----------------------------------------------------
  // Load Matrix
  // ----------------------------------------------------
  int *ColumnDegree,*RowDegree,**Mat; //TBD
  int M,N,m,n,k,i,j;
  
  strcpy(FileName,FileResult);
  strcat(FileName,".res");
  fout=fopen(FileName,"a+");

  if (fout==NULL) {
  	  printf("Output File %s error!.. Abort", FileName);
  	  exit(-1);
  }
  
  strcpy(FileName,FileMatrix);
  strcat(FileName,"_size");
  f=fopen(FileName,"r");
  fscanf(f,"%d",&M);fscanf(f,"%d",&N);
  ColumnDegree=(int *)calloc(N,sizeof(int));
  RowDegree=(int *)calloc(M,sizeof(int));
  fclose(f);
  
  strcpy(FileName,FileMatrix);strcat(FileName,"_RowDegree");
  f=fopen(FileName,"r");
  for (m=0;m<M;m++) 
	  fscanf(f,"%d",&RowDegree[m]);
  fclose(f);
  
  Mat=(int **)calloc(M,sizeof(int *));
  for (m=0;m<M;m++) {
	  Mat[m]=(int *)calloc(RowDegree[m],sizeof(int));
  }
  
  strcpy(FileName,FileMatrix);
  f=fopen(FileName,"r");
  for (m=0;m<M;m++) { 
	for (k=0;k<RowDegree[m];k++) {
		fscanf(f,"%d",&Mat[m][k]); 
	}
  }
  fclose(f);
  
  for (m=0;m<M;m++) { 
	for (k=0;k<RowDegree[m];k++) 
		ColumnDegree[Mat[m][k]]++; 
  }

  printf("Matrix Loaded \n");

  // ----------------------------------------------------
  // Build Graph
  // ----------------------------------------------------
  int NbBranch;
  int **NtoB,*Interleaver; //TBD
  int *ind;
  int numColumn,numBranch;
  NbBranch=0; for (m=0;m<M;m++) NbBranch=NbBranch+RowDegree[m];
  NtoB=(int **)calloc(N,sizeof(int *)); for (n=0;n<N;n++) NtoB[n]=(int *)calloc(ColumnDegree[n],sizeof(int));
  Interleaver=(int *)calloc(NbBranch,sizeof(int));
  ind=(int *)calloc(N,sizeof(int));
  numBranch=0;for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) { numColumn=Mat[m][k]; NtoB[numColumn][ind[numColumn]++]=numBranch++; } }
  free(ind);
  numBranch=0;for (n=0;n<N;n++) { for (k=0;k<ColumnDegree[n];k++) Interleaver[numBranch++]=NtoB[n][k]; }

  printf("Graph Build \n");


  // ----------------------------------------------------
  // Load Base Matrix
  // ----------------------------------------------------
  int **Basemat,Nbcol_Basemat,Nbrow_Basemat,Circulant;

  strcpy(FileName,FileBaseMatrix);strcat(FileName,"_size");
  f=fopen(FileName,"r");fscanf(f,"%d",&Nbrow_Basemat);fscanf(f,"%d",&Nbcol_Basemat);fscanf(f,"%d",&Circulant);fclose(f);
  printf("NbCol: %d, NbRow: %d, Cir: %d\n",Nbcol_Basemat,Nbrow_Basemat,Circulant);


  Basemat=(int **)calloc(Nbrow_Basemat,sizeof(int *));for (m=0;m<Nbrow_Basemat;m++) Basemat[m]=(int *)calloc(Nbcol_Basemat,sizeof(int));

  strcpy(FileSimu,FileBaseMatrix);strcat(FileSimu,"_mat");
  f=fopen(FileSimu,"r");  //if(!f)   printf("Unable to open\n");
  for (m=0;m<Nbrow_Basemat;m++) { for (k=0;k<Nbcol_Basemat;k++) fscanf(f,"%d",&Basemat[m][k]); }
  fclose(f);

    printf("Base Matrix:\n");
    for (m=0;m<Nbrow_Basemat;m++) { for (k=0;k<Nbcol_Basemat;k++) printf("%d  ",Basemat[m][k]); printf("\n");}

  // ----------------------------------------------------
  // Decoder
  // ----------------------------------------------------
  int *CtoV,*VtoC,*Codeword,*Receivedword,*Decide,*U,l,kk,*CNvalue,*VNvalue1,*VNvalue2,*energy;;
  int iter,numB,Laynb,Colnb;
  int * Flag = NULL;
  int flip_coin;
  //CtoV=(int *)calloc(NbBranch,sizeof(int));
  //VtoC=(int *)calloc(NbBranch,sizeof(int));
  Codeword=NULL;
  Receivedword=NULL;
  Decide=NULL;
  CNvalue = NULL;
  VNvalue1 = NULL;
  VNvalue2= NULL;

  Codeword=(int *)calloc(N,sizeof(int));
  Receivedword=(int *)calloc(N,sizeof(int));
  Decide=(int *)calloc(N,sizeof(int));
  Flag = (int *) calloc (Circulant, sizeof(int));
  U=(int *)calloc(N,sizeof(int));
  CNvalue=(int *)calloc(Circulant,sizeof(int));
  VNvalue1=(int *)calloc(Circulant,sizeof(int));
  VNvalue2=(int *)calloc(Circulant,sizeof(int));

  energy=(int *)calloc(N,sizeof(int));
  srand(time(0)+Graine*31+113);

  // ----------------------------------------------------
  // Gaussian Elimination for the Encoding Matrix (Full Representation)
  // ----------------------------------------------------
  int **MatFull,**MatG,*PermG; //TBD
  int rank;
  MatG=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) MatG[m]=(int *)calloc(N,sizeof(int));
  MatFull=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) MatFull[m]=(int *)calloc(N,sizeof(int));
  PermG=(int *)calloc(N,sizeof(int)); for (n=0;n<N;n++) PermG[n]=n;
  for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) { MatFull[m][Mat[m][k]]=1; } }
  rank=GaussianElimination_MRB(PermG,MatG,MatFull,M,N);
  //for (m=0;m<N;m++) printf("%d\t",PermG[m]);printf("\n");

  // Variables for Statistics
  int IsCodeword,nb,Syndrome,EnergyMax;
  int NiterMoy,NiterMax;
  int Dmin;
  int NbTotalErrors,NbBitError;
  int NbUnDetectedErrors,NbError;

    #if GDBF
    printf("-------------------------La-GDBF--------------------------------------------------\n");
    #endif // GDBF

    #if PGDBF
    printf("-------------------------La-P-GDBF--------------------------------------------------\n");
    #endif // PGDBF

	#if PGDBF1
    printf("-------------------------La-P-GDBF1--------------------------------------------------\n");
	#endif // PGDBF

    fflush (stdout);

  printf("alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\n");
  fprintf(fout,"alpha\t\t\tNbEr(BER)\t\t\t\tNbFer(FER)\t\t\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\n");
 
  for(alpha=alpha_max;alpha>alpha_min;alpha-=alpha_step)
  {
  NiterMoy=0;NiterMax=0;
  Dmin=1e5;
  NbTotalErrors=0;NbBitError=0;
  NbUnDetectedErrors=0;NbError=0;

  //--------------------------------------------------------------
  for (nb=0,nbtestedframes=0;nb<NbMonteCarlo;nb++)
  {
    //encoding
    for (k=0;k<rank;k++) U[k]=0;
	for (k=rank;k<N;k++){ 
		U[k]=floor(((double)(rand()) / RAND_MAX)*2);
	}
	for (k=rank-1;k>=0;k--) { for (l=k+1;l<N;l++) U[k]=U[k]^(MatG[k][l]*U[l]); }
	for (k=0;k<N;k++) Codeword[PermG[k]]=U[k];
	// All zero codeword
	//for (n=0;n<N;n++) { Codeword[n]=0; }

    // Add Noise
    for (n=0;n<N;n++)  if ((((double)(rand()) / RAND_MAX))<alpha) Receivedword[n]=1-Codeword[n]; else  Receivedword[n]=Codeword[n];
	//0 2 21 80 142 - Tanner
    /*Receivedword[0]=1-Codeword[0];
    Receivedword[2]=1-Codeword[2];
    Receivedword[21]=1-Codeword[21];
    Receivedword[80]=1-Codeword[80];
    Receivedword[142]=1-Codeword[142];*/

    //0 2 21 47 80 142
    /*Receivedword[0]=1-Codeword[0];
      Receivedword[2]=1-Codeword[2];
      Receivedword[21]=1-Codeword[21];
      Receivedword[80]=1-Codeword[80];
      Receivedword[142]=1-Codeword[142];
	  Receivedword[47]=1-Codeword[47];*/

    //0 15 77 90 101 104 119 142
    /*	Receivedword[0]=1-Codeword[0];
        Receivedword[15]=1-Codeword[15];
        Receivedword[77]=1-Codeword[77];
        Receivedword[90]=1-Codeword[90];
        Receivedword[101]=1-Codeword[101];
    	Receivedword[104]=1-Codeword[104];
    	Receivedword[119]=1-Codeword[119];
    	Receivedword[142]=1-Codeword[142]; */
    //============================================================================
 	// Decoder
	//============================================================================
	//for (k=0;k<NbBranch;k++) {CtoV[k]=0;}
	for (k=0;k<N;k++) Decide[k]=Receivedword[k];

	int oldBatch = -1;

	for (iter=0;iter<NbIter;iter++) {
          Syndrome=0;   
		  IsCodeword = 0;
          for(n=0;n<N;n++)  
			  //energy[n]= Decide[n]^Receivedword[n];
			  if (Decide[n]^Receivedword[n])
				  energy[n]= 1;
			  else
				  energy[n]= -1;
		  
	      for(Laynb=0;Laynb<Nbrow_Basemat;Laynb++)
          {
              for(i=0;i<Circulant;i++)  CNvalue[i]=0;

              for(Colnb=0;Colnb<Nbcol_Basemat;Colnb++)
              {
                  if(Basemat[Laynb][Colnb]!=-1)
                  {
                      j= Colnb * Circulant;
                      for(i=0;i<Circulant;i++)  
						  VNvalue1[i]= Decide[j+i];
                      for(i=0,j=Basemat[Laynb][Colnb];i<Circulant;i++)  //Shift Right
                      {
                          VNvalue2[i]= VNvalue1[j];
                          j++; 
						  if(j==Circulant) 
							  j=0;
                      }
                      for(i=0;i<Circulant;i++)      
						  CNvalue[i] ^= VNvalue2[i];
						  /*if (VNvalue2[i]) //Parity Check
							  CNvalue[i]+=-1; 
						  else
							  CNvalue[i]+= +1;*/
                  }
              }

              for(Colnb=0;Colnb<Nbcol_Basemat;Colnb++)
              {
                  if(Basemat[Laynb][Colnb]!=-1)
                  {
                      for(i=0,j=Basemat[Laynb][Colnb];i<Circulant;i++)
                      {
                          VNvalue1[j]= CNvalue[i];      //Temporary using VNvalue1 instead of declaring a new vector
                          j++;
                          if(j==Circulant)  j=0;
                      }
                      j= Colnb * Circulant;
                      for(i=0;i<Circulant;i++) 
						  // energy[j+i]+=VNvalue1[i] ;
						  if (VNvalue1[i] == 0) 
							  energy[j+i]+= -1;
						  else
							  energy[j+i]+= 1;
                  }
              }
              if(Syndrome!=1)   // this does not make sense
				  for(i=0;i<Circulant;i++)    
					  if(CNvalue[i]!=0)   { 
							Syndrome=1; 
							break;
						}
          }

          if(Syndrome==0)   {
			  IsCodeword=1; 
			  break;
		  }

          for(n=0,EnergyMax=-255;n<N;n++)  
			  if(energy[n]>EnergyMax) 
				  EnergyMax=energy[n];
          #if GDBF
          for(n=0;n<N;n++)  if(energy[n]==EnergyMax)    Decide[n]= 1- Decide[n];
          #endif // GDBF

          #if PGDBF
          for(n=0;n<N;n++)  if((energy[n]==EnergyMax)&&(((double)(rand()) / RAND_MAX)<PoPGDBF))    Decide[n]= 1- Decide[n];
          #endif // PGDBF

		 #if PGDBF1
         if (iter<=1){
        	 for(n=0;n<N;n++){
				 if(energy[n]==EnergyMax) {
					 Decide[n]= 1- Decide[n];//gdbf
				 }
			 }
         
		 }
		 if (iter >1)
         {
         int batchStart = -1;
         int p = 1; //one column
         int cnt = 0; //count the number of flipped columns
         //select batch
		 
         for (int batch = 0; batch <N; batch+=Circulant*p)
         {
        	 int isMax = 0;
			 //circulant nodes
        	 for (int i = batch; i<batch+Circulant*p; i++) //circulant*p decoded messages at a time
        	 {
        		 //search for max
        		 if (energy[i]==EnergyMax)
        		 {
        			 isMax = 1;
        			 break;
        		 }
        	 }
        	 if (isMax) //flip coin to decide which batch stays
        	 {
        		 if (batch != oldBatch)
        		 {
        			 if ((batchStart== -1) || (((double)(rand()) / RAND_MAX)<0.5))
        			 {
        				 cnt ++;
        				 batchStart = batch; //index of first column message from the batch
        				 //flip these column bits
        				 for(int n=batchStart;n<batchStart+Circulant*p;n++){
							 if((energy[n]==EnergyMax))    
								 Decide[n]= 1- Decide[n];
						 }
        				    oldBatch = batchStart;
        			 }//else skip this column from flipping altogether
        		 }
        	 }
        	 if (cnt == 12) break; //if half the column have been flipped skip the rest
         }
         //selection of batch has been done
         if (batchStart<0) //only one column left with global max
         {
        	 batchStart = oldBatch;
        	 for(int n=batchStart;n<batchStart+Circulant*p;n++)  if((energy[n]==EnergyMax))    Decide[n]= 1- Decide[n];
         }
         }
         #endif // PGDBF1
	  }
	//============================================================================
  	// Compute Statistics
	//============================================================================
      nbtestedframes++;
	  NbError=0;for (k=0;k<N;k++)  if (Decide[k]!=Codeword[k]) NbError++;
	  NbBitError=NbBitError+NbError;
	// Case Divergence
	  if (!IsCodeword)
	  {
		  NiterMoy=NiterMoy+NbIter;
		  NbTotalErrors++;
	  }
	// Case Convergence to Right Codeword
	  if ((IsCodeword)&&(NbError==0)) { NiterMax=max(NiterMax,iter+1); NiterMoy=NiterMoy+(iter+1); }
	// Case Convergence to Wrong Codeword
	  if ((IsCodeword)&&(NbError!=0))
	  {
		  NiterMax=max(NiterMax,iter+1); NiterMoy=NiterMoy+(iter+1);
		  NbTotalErrors++; NbUnDetectedErrors++;
		  Dmin=min(Dmin,NbError);
	  }
	// Stopping Criterion
	 if (NbTotalErrors==NBframes) break;
	 if (nbtestedframes % 1000000 ==0)
	 {
		 printf("%1.5f\t\t",alpha);
		 printf("%10d (%1.15f)\t\t",NbBitError,(float)NbBitError/N/nbtestedframes);
		 printf("%4d (%1.15f)\t\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
		 printf("%10d\t\t",nbtestedframes);
		 printf("%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
		 printf("%d(%d)\n",NbUnDetectedErrors,Dmin);
		 fflush (stdout);
		 
		 fprintf(fout, "%1.5f\t\t",alpha);
		 fprintf(fout, "%10d (%1.15f)\t\t",NbBitError,(float)NbBitError/N/nbtestedframes);
		 fprintf(fout, "%4d (%1.15f)\t\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
		 fprintf(fout, "%10d\t\t",nbtestedframes);
		 fprintf(fout, "%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
		 fprintf(fout, "%d(%d)\n",NbUnDetectedErrors,Dmin);

		 fflush (fout);

	 }
  }
    printf("%1.5f\t\t",alpha);
    printf("%10d (%1.15f)\t\t",NbBitError,(float)NbBitError/N/nbtestedframes);
    printf("%4d (%1.15f)\t\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
    printf("%10d\t\t",nbtestedframes);
    printf("%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
    printf("%d(%d)\n",NbUnDetectedErrors,Dmin);
    fflush (stdout);
	
	fprintf(fout, "%1.5f\t\t",alpha);
	fprintf(fout, "%10d (%1.15f)\t\t",NbBitError,(float)NbBitError/N/nbtestedframes);
	fprintf(fout, "%4d (%1.15f)\t\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
	fprintf(fout, "%10d\t\t",nbtestedframes);
	fprintf(fout, "%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
	fprintf(fout, "%d(%d)\n",NbUnDetectedErrors,Dmin);

	fflush (fout);
  }
  if (fout)
	  fclose(fout);
  if (Flag != NULL)
	  free(Flag);
  if (Codeword==NULL)
	  free(Codeword);
  if (Receivedword==NULL)
    free(Receivedword);
  if (Decide==NULL)
	  free(Decide);
  if (CNvalue==NULL)
  	  free(CNvalue);
  if (VNvalue1==NULL)
  	  free(VNvalue1);
  if (VNvalue2==NULL)
  	  free(VNvalue2);

  return(0);
}
