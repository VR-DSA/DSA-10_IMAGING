#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <sigproc.h>
#include <header.h>

FILE *output;


void send_string(char *string) /* includefile */
{
  int len;
  len=strlen(string);
  fwrite(&len, sizeof(int), 1, output);
  fwrite(string, sizeof(char), len, output);
}

void send_float(char *name,float floating_point) /* includefile */
{
  send_string(name);
  fwrite(&floating_point,sizeof(float),1,output);
}

void send_double (char *name, double double_precision) /* includefile */
{
  send_string(name);
  fwrite(&double_precision,sizeof(double),1,output);
}

void send_int(char *name, int integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(int),1,output);
}

void send_char(char *name, char integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(char),1,output);
}


void send_long(char *name, long integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(long),1,output);
}

void send_coords(double raj, double dej, double az, double za) /*includefile*/
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za);
}


int main (int argc, char *argv[]) {

  // open output file
  if (!(output = fopen("/home/user/tmp/src_coh.fil","wb"))) {
    printf("Couldn't open output file\n");
    return 0;
  }

  send_string("HEADER_START");
  send_string("source_name");
  send_string("FRB190523");
  send_int("machine_id",1);
  send_int("telescope_id",82);
  send_int("data_type",1); // filterbank data
  send_double("fch1",1487.275390625);
  //send_double("fch1",1493.37890625);
  send_double("foff",-0.1220703125);
  send_int("nchans",1250);
  //send_int("nchans",1500);
  send_int("nbits",32);
  send_double("tstart",58626.2541144408);
  send_double("tsamp",0.000131072);
  send_int("nifs",1);
  send_string("HEADER_END");

  // open input file and allocate memory
  FILE *fin;
  fin=fopen(argv[1],"r");
  float * write_data = (float *)malloc(sizeof(float)*256*1250);

  // read
  float tt;
  for (int i=0;i<256*1250;i++)
    fscanf(fin,"%f %f\n",&write_data[i],&tt);
  
  // write
  fwrite(write_data,sizeof(float),1250*256,output);
       
    
  fclose(fin);
  fclose(output);
  free(write_data);

}
