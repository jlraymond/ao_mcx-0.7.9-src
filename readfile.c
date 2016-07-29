#include "printf.h"
#include "stdlib.h"
#include "stdio.h"


const int Stride = 3;

int main(){
	int i,j,k,n;
	char * file_name = "myfile.bin";
	FILE * fp = fopen(file_name,"rb");
	int data_length=Stride*Stride*Stride*4;
	int index;

	int current_position=0;
	float * my_data = (float *)malloc( data_length*sizeof(float) );
	float * sorted_data = (float *)malloc(data_length*sizeof(float));
		
	fread( my_data, sizeof(float), data_length, fp);

	
	for( i =0; i < Stride; ++i){
		for( j =0; j < Stride; ++j){
			for( k =0; k < Stride; ++k){
				for( n =0; n < 4; ++n){
					index = Stride*Stride*Stride*n + Stride*Stride*k + Stride*j + i;
					sorted_data[current_position] = my_data[index];
					printf("%f\t", sorted_data[current_position] );
					current_position=current_position+1;
				}
				/*
				 * float4 * sorted_data = (float4*)malloc(data_length*sizeof(float4));
				 * Replace inner most loop with the following
				 * index =  Stride*Stride*k + Stride*j + i;
				 * sorted_data[current_position].x = my_data[index] // is a float4 *;
				 * sorted_data[current_position].y = my_data[Stride*Stride*Stride 	+index]
				 * sorted_data[current_position].z = my_data[Stride*Stride*Stride*2 +index]
				 * sorted_data[current_position].w = my_data[Stride*Stride*Stride*3 +index]
				 * 	*/
				printf("\n");
			}
		}
	}
	fclose(fp);
	return 0;	
}
