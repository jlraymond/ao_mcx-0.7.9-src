#include "printf.h"
#include "stdlib.h"
#include "stdio.h"

typedef struct MCXAcoustics{
	float Px;  // x component of pressure (Mag(P)*ihat)
	float Py;
	float Pz;
	float USphase;
} Acoustics;

const int dim = 3;
const int myLength = 4;

int main(){
	int i,j,k,n;
	char * file_name = "myfile.bin";
	FILE * fp = fopen(file_name,"rb");
	int data_length=dim*dim*dim*myLength;
	int index;
	int current_position=0;

	float * my_data = (float *)malloc( data_length*sizeof(float) );
	Acoustics * sorted_data = (Acoustics*)malloc(dim*dim*dim*sizeof(Acoustics));
	//float * sorted_data = (float *)malloc(data_length*sizeof(float));
		
	fread( my_data, sizeof(float), data_length, fp);
	
	for( k =0; k < dim; ++k){
		for( j =0; j < dim; ++j){
			for( i =0; i < dim; ++i){
				for( n =0; n < myLength; ++n){
				
					index =  dim*dim*k + dim*j + i;
					sorted_data[current_position].Px = my_data[index];
	 				sorted_data[current_position].Py = my_data[dim*dim*dim + index];
					sorted_data[current_position].Pz = my_data[dim*dim*dim*2 + index];
					sorted_data[current_position].USphase = my_data[dim*dim*dim*3 + index];
					}
					printf("%f\t%f\t%f\t%f\n", sorted_data[current_position].Px, sorted_data[current_position].Py, sorted_data[current_position].Pz, sorted_data[current_position].USphase );
					current_position = current_position+1;
					
				}
			}
		}
		
	fclose(fp);
	return 0;	
}
