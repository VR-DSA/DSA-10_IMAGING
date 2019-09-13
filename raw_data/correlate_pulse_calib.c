// -*- c++ -*-
/* Makes dedispersed visibility data */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "fitsio.h"

void usage()
{
  fprintf (stdout,
	   "correlate_pulse [options].  \n"
	   " -m start mjd of dumps [default 57892.0]\n"
	   " -d DM [default 56.78]\n"
	   " -f locations of dsa1-5 raw data files [default dsa1.out...]\n"
	   " -g location of output file (default test.fits)\n"
	   " -h        print usage\n");
}
  
int main (int argc, char *argv[]) {

  // command line
  long long nbytes = 2415919104;
  double mjd = 57892.0;
  int samples_per_bin = 128;
  int number_of_bins = 16;
  double dm = 56.78;
  double RA = 60.0;
  double DEC = 22.01444444;
  char dsa1[100], dsa2[100], dsa3[100], dsa4[100], dsa5[100], dsaout[100];
  sprintf(dsa1,"dsa1.out");
  sprintf(dsa2,"dsa2.out");
  sprintf(dsa3,"dsa3.out");
  sprintf(dsa4,"dsa4.out");
  sprintf(dsa5,"dsa5.out");
  sprintf(dsaout,"test.fits");

  samples_per_bin = 95000;
  number_of_bins = 3;
  
  for (int i=1;i<argc;i++) {

    if (strcmp(argv[i],"-m")==0) 
      mjd = (double)(atof(argv[i+1]));

    if (strcmp(argv[i],"-d")==0)
      dm = (double)(atof(argv[i+1]));

    if (strcmp(argv[i],"-f")==0) {
      strcpy(dsa1,argv[i+1]);
      strcpy(dsa2,argv[i+2]);
      strcpy(dsa3,argv[i+3]);
      strcpy(dsa4,argv[i+4]);
      strcpy(dsa5,argv[i+5]);
    }

    if (strcmp(argv[i],"-g")==0)
      strcpy(dsaout,argv[i+1]);

    if (strcmp(argv[i],"-h")==0) {
      usage();
      exit(1);
    }
    
  }

  long long start_sample = 100;
  printf("MJD: %lf\n",mjd);
  printf("DM: %g\n",dm);
  
  // make output fits file
  fitsfile *fptr;
  int status=0;
  int rownum = 1;
  char *ttype[] = {"Data"};
  char *tform[] = {"E"};  // 45 real and 45 imaginary products, 10 autos per pol
  char *tunit[] = {"\0"};
  char extname[] = "pulse_data";
  fits_create_file(&fptr, dsaout, &status);
  fits_create_tbl(fptr, BINARY_TBL, 0, 1, ttype, tform, tunit, extname, &status);
  fits_write_key(fptr, TDOUBLE, "MJD", &mjd, "Start MJD", &status);
  fits_write_key(fptr, TDOUBLE, "RA", &RA, "DEGREES", &status);
  fits_write_key(fptr, TDOUBLE, "DEC", &DEC, "DEGREES", &status);
  fits_write_key(fptr, TDOUBLE, "DM", &dm, "PC.CM-3", &status);
  float mytsamp = 8.192e-6*samples_per_bin;
  fits_write_key(fptr, TFLOAT, "TSAMP", &mytsamp, "Sample time (s)", &status);
  char * antennas;
  antennas = (char *)malloc(100);
  sprintf(antennas,"3-7-2-10-1-4-5-8-6-9");
  fits_write_key(fptr, TSTRING, "ANTENNAS", &antennas[0], "Antennas", &status);
  free(antennas);
  fits_close_file(fptr, &status);

  /* RUN CORRELATOR */

  // calculate channel offsets and allocate memory for output
  int dm_offsets[2048];
  float fq;
  for (int i=0;i<2048;i++) {
    fq = 1530.-i*250./2048.;
    fq = 4.149e3*dm*(pow(fq,-2.)-pow(1530.,-2.));
    dm_offsets[i] = (int)(round(fq/8.192e-6));
  }
  float tofile[2][200], wtofile[200];

  printf("Reading data into memory \n");
  
  // read all data into memory
  FILE *fin;
  char * cdsa1 = (char *)malloc(sizeof(char)*nbytes);
  char * cdsa2 = (char *)malloc(sizeof(char)*nbytes);
  char * cdsa3 = (char *)malloc(sizeof(char)*nbytes);
  char * cdsa4 = (char *)malloc(sizeof(char)*nbytes);
  char * cdsa5 = (char *)malloc(sizeof(char)*nbytes);
  int *idsa1, *idsa2, *idsa3, *idsa4, *idsa5;

  fin=fopen(dsa1,"rb");
  fread(cdsa1,sizeof(char),nbytes,fin);
  idsa1 = (int *)cdsa1;
  fclose(fin);

  fin=fopen(dsa2,"rb");
  fread(cdsa2,sizeof(char),nbytes,fin);
  idsa2 = (int *)cdsa2;
  fclose(fin);

  fin=fopen(dsa3,"rb");
  fread(cdsa3,sizeof(char),nbytes,fin);
  idsa3 = (int *)cdsa3;
  fclose(fin);

  fin=fopen(dsa4,"rb");
  fread(cdsa4,sizeof(char),nbytes,fin);
  idsa4 = (int *)cdsa4;
  fclose(fin);

  fin=fopen(dsa5,"rb");
  fread(cdsa5,sizeof(char),nbytes,fin);
  idsa5 = (int *)cdsa5;
  fclose(fin);  

  // loop over time and frequency
  // fill vector of samples for each point, and apply bandpass correction
  char d[2][2][10]; // pol, r/i, ant
  int i1, ci;
  long long idx;
  // correlate. Order: [antenna pair, pol, r/i]
  // write to output file

  int nidx=0, siidx=0;   
  int fqi, dmi;
  for (int tidx=0;tidx<number_of_bins;tidx++) {
    printf("tidx %d of %d\n",tidx+1,number_of_bins);
    for (int fidx=0;fidx<1024;fidx++) {
      
      for (int sidx=0;sidx<2;sidx++) {

	fqi = 2*fidx+sidx;
	dmi = 2*fidx+1-sidx;
	
	for (int i=0;i<200;i++) wtofile[i] = 0.;
      
	// loop over samples per bin
	for (int bidx=0;bidx<samples_per_bin;bidx++) {
	  
	  idx = 2048*(start_sample+tidx*samples_per_bin+bidx+dm_offsets[dmi]) + fqi;
	  
	  // DSA1
	  i1 = idsa1[idx];
	  d[0][0][0] = (char)(((unsigned int)(i1) & (unsigned int)(15)) << 4);
	  d[0][1][0] = (char)((unsigned int)(i1) & (unsigned int)(240));
	  d[0][0][1] = (char)(((unsigned int)(i1) & (unsigned int)(983040)) >> 12);
	  d[0][1][1] = (char)(((unsigned int)(i1) & (unsigned int)(15728640)) >> 16);
	  d[1][0][0] = (char)(((unsigned int)(i1) & (unsigned int)(3840)) >> 4);
	  d[1][1][0] = (char)(((unsigned int)(i1) & (unsigned int)(61440)) >> 8);
	  d[1][0][1] = (char)(((unsigned int)(i1) & (unsigned int)(251658240)) >> 20);
	  d[1][1][1] = (char)(((unsigned int)(i1) & (unsigned int)(4026531840)) >> 24);	    
	  
	  // DSA2
	  i1 = idsa2[idx];
	  d[0][0][2] = (char)(((unsigned int)(i1) & (unsigned int)(15)) << 4);
	  d[0][1][2] = (char)((unsigned int)(i1) & (unsigned int)(240));
	  d[0][0][3] = (char)(((unsigned int)(i1) & (unsigned int)(983040)) >> 12);
	  d[0][1][3] = (char)(((unsigned int)(i1) & (unsigned int)(15728640)) >> 16);
	  d[1][0][2] = (char)(((unsigned int)(i1) & (unsigned int)(3840)) >> 4);
	  d[1][1][2] = (char)(((unsigned int)(i1) & (unsigned int)(61440)) >> 8);
	  d[1][0][3] = (char)(((unsigned int)(i1) & (unsigned int)(251658240)) >> 20);
	  d[1][1][3] = (char)(((unsigned int)(i1) & (unsigned int)(4026531840)) >> 24);
	  
	  // DSA3
	  i1 = idsa3[idx];
	  d[0][0][4] = (char)(((unsigned int)(i1) & (unsigned int)(15)) << 4);
	  d[0][1][4] = (char)((unsigned int)(i1) & (unsigned int)(240));
	  d[0][0][5] = (char)(((unsigned int)(i1) & (unsigned int)(983040)) >> 12);
	  d[0][1][5] = (char)(((unsigned int)(i1) & (unsigned int)(15728640)) >> 16);
	  d[1][0][4] = (char)(((unsigned int)(i1) & (unsigned int)(3840)) >> 4);
	  d[1][1][4] = (char)(((unsigned int)(i1) & (unsigned int)(61440)) >> 8);
	  d[1][0][5] = (char)(((unsigned int)(i1) & (unsigned int)(251658240)) >> 20);
	  d[1][1][5] = (char)(((unsigned int)(i1) & (unsigned int)(4026531840)) >> 24);
	  
	  // DSA4
	  i1 = idsa4[idx];
	  d[0][0][6] = (char)(((unsigned int)(i1) & (unsigned int)(15)) << 4);
	  d[0][1][6] = (char)((unsigned int)(i1) & (unsigned int)(240));
	  d[0][0][7] = (char)(((unsigned int)(i1) & (unsigned int)(983040)) >> 12);
	  d[0][1][7] = (char)(((unsigned int)(i1) & (unsigned int)(15728640)) >> 16);
	  d[1][0][6] = (char)(((unsigned int)(i1) & (unsigned int)(3840)) >> 4);
	  d[1][1][6] = (char)(((unsigned int)(i1) & (unsigned int)(61440)) >> 8);
	  d[1][0][7] = (char)(((unsigned int)(i1) & (unsigned int)(251658240)) >> 20);
	  d[1][1][7] = (char)(((unsigned int)(i1) & (unsigned int)(4026531840)) >> 24);
	  
	  // DSA5
	  i1 = idsa5[idx];
	  d[0][0][8] = (char)(((unsigned int)(i1) & (unsigned int)(15)) << 4);
	  d[0][1][8] = (char)((unsigned int)(i1) & (unsigned int)(240));
	  d[0][0][9] = (char)(((unsigned int)(i1) & (unsigned int)(983040)) >> 12);
	  d[0][1][9] = (char)(((unsigned int)(i1) & (unsigned int)(15728640)) >> 16);
	  d[1][0][8] = (char)(((unsigned int)(i1) & (unsigned int)(3840)) >> 4);
	  d[1][1][8] = (char)(((unsigned int)(i1) & (unsigned int)(61440)) >> 8);
	  d[1][0][9] = (char)(((unsigned int)(i1) & (unsigned int)(251658240)) >> 20);
	  d[1][1][9] = (char)(((unsigned int)(i1) & (unsigned int)(4026531840)) >> 24);
	  
	  // do correlation
	  ci = 0;
	  for (int i=0;i<9;i++) {
	    for (int j=i+1;j<10;j++) {
	      
	      wtofile[ci] += ((float)((((int)((d[0][0][i]*d[0][0][j]+d[0][1][i]*d[0][1][j])/256))*1.))); // A_R
	      ci++;
	      wtofile[ci] += ((float)((((int)((d[0][1][i]*d[0][0][j]-d[0][0][i]*d[0][1][j])/256))*1.))); // A_I
	      ci++;
	      wtofile[ci] += ((float)((((int)((d[1][0][i]*d[1][0][j]+d[1][1][i]*d[1][1][j])/256))*1.))); // B_R
	      ci++;
	      wtofile[ci] += ((float)((((int)((d[1][1][i]*d[1][0][j]-d[1][0][i]*d[1][1][j])/256))*1.))); // B_I
	      ci++;
	      
	    }
	  }
	  
	  // do autos
	  for (int i=0;i<10;i++) {

	    /*wtofile[ci] += bandpass[0][i][dmi]*((float)(((int)((d[0][0][i]*d[0][0][i]+d[0][1][i]*d[0][1][i])/256))*1.)); // AA
	    ci++;
	    wtofile[ci] += bandpass[1][i][dmi]*((float)(((int)((d[1][0][i]*d[1][0][i]+d[1][1][i]*d[1][1][i])/256))*1.)); // BB
	    ci++;*/
	    wtofile[ci] += ((float)(((int)((d[0][0][i]*d[0][0][i]+d[0][1][i]*d[0][1][i])/256))*1.)); // AA
	    ci++;
	    wtofile[ci] += ((float)(((int)((d[1][0][i]*d[1][0][i]+d[1][1][i]*d[1][1][i])/256))*1.)); // BB
	    ci++;
	    
	  }
	  
	}

	for (int i=0;i<200;i++)
	  tofile[sidx][i] = wtofile[i];
	
	
      }
      
      // write to file
      
      fits_open_table(&fptr, dsaout, READWRITE, &status);

      // implement channel swap
      for (int i=0;i<200;i++)
	wtofile[i] = tofile[1][i];
      fits_write_col(fptr, TFLOAT, 1, rownum, 1, 200, wtofile, &status);
      rownum+=200;
      for (int i=0;i<200;i++)
	wtofile[i] = tofile[0][i];
      fits_write_col(fptr, TFLOAT, 1, rownum, 1, 200, wtofile, &status);
      rownum+=200;
      fits_update_key(fptr, TINT, "NAXIS2", &rownum, "", &status);
      fits_close_file(fptr, &status);
      //printf("%d\n",status);
      
    }
  }
  
  
  
  free(cdsa1);
  free(cdsa2);
  free(cdsa3);
  free(cdsa4);
  free(cdsa5);

}
