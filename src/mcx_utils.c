/*******************************************************************************
**
**  Acousto-Optic MCX (AO-MCX) - Matt Adams <adamsm2@bu.edu>
**
**	Written based on:
**  Monte Carlo eXtreme (MCX)  - GPU accelerated 3D Monte Carlo transport simulation
**  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
**
**  Reference (Fang2009):
**        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
**        Migration in 3D Turbid Media Accelerated by Graphics Processing 
**        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
**
**  mcx_utils.c: configuration and command line option processing unit
**
**  License: GNU General Public License v3, see LICENSE.txt for details
**
**	Changes from MCX are marked with //MTA
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mcx_utils.h"
#include "mcx_const.h"
#include "mcx_shapes.h"

#define FIND_JSON_KEY(id,idfull,parent,fallback,val) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? fallback : tmp->val) \
                     : tmp->val)

#define FIND_JSON_OBJ(id,idfull,parent) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? NULL : tmp) \
                     : tmp)


//MTA. These are the tags for the command line options.
// It may be good to add an option to perform an optical simulation only w/o acoustics
const char shortopt[]={'h','i','f','n','t','T','s','a','g','b','B','z','u','H','P',
                 'd','r','S','p','e','U','R','l','L','I','o','G','M','A','E','v','\0'};
const char *fullopt[]={"--help","--interactive","--input","--photon",
                 "--thread","--blocksize","--session","--array",
                 "--gategroup","--reflect","--reflectin","--srcfrom0",
                 "--unitinmm","--maxdetphoton","--shapes","--savedet",
                 "--repeat","--save2pt","--printlen","--minenergy",
                 "--normalize","--skipradius","--log","--listgpu",
                 "--printgpu","--root","--gpu","--dumpmask","--autopilot","--seed","--version",""};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//MTA. These are the default settings.
void mcx_initcfg(Config *cfg){
     cfg->medianum=0;
     cfg->detnum=0;
     cfg->dim.x=0;
     cfg->dim.y=0;
     cfg->dim.z=0;
     cfg->steps.x=1.f;
     cfg->steps.y=1.f;
     cfg->steps.z=1.f;
     cfg->nblocksize=64;
     cfg->nphoton=0;
     cfg->nthread=2048;
     cfg->isrowmajor=0; /* default is Matlab array*/
     cfg->maxgate=1;
     cfg->isreflect=1;
     cfg->isref3=1;
     cfg->isrefint=0;
     cfg->isnormalized=1;
     cfg->issavedet=1;
     cfg->respin=1;
     cfg->issave2pt=1;
     cfg->isgpuinfo=0;
	 cfg->Acon=NULL;		//MTA
	 cfg->Ocon=NULL;		//MTA
     cfg->prop=NULL;
	 cfg->detpos=NULL;
     cfg->vol=NULL;
     cfg->pressure=NULL;	//MTA
     cfg->session[0]='\0';
     cfg->printnum=0;
     cfg->minenergy=0.f;
     cfg->flog=stdout;
     cfg->sradius=0.f;
     cfg->rootpath[0]='\0';
     cfg->gpuid=0;
     cfg->issrcfrom0=0;
     cfg->unitinmm=1.f;
     cfg->isdumpmask=0;
     cfg->maxdetphoton=1000000;
     cfg->autopilot=0;
     cfg->seed=0;
     cfg->exportfield0=NULL;
     cfg->exportfield1=NULL;
     cfg->exportdetected=NULL;
     /*cfg->his=(History){{'M','C','X','H'},1,0,0,0,0,0,0,1.f,{0,0,0,0,0,0,0}};*/
     memset(&cfg->his,0,sizeof(History));
     memcpy(cfg->his.magic,"MCXH",4);
     cfg->his.version=1;
     cfg->his.unitinmm=1.f;
     cfg->shapedata=NULL;
}



//MTA. Clears the config file when called 
void mcx_clearcfg(Config *cfg){
     if(cfg->medianum)
		free(cfg->prop);
		free(cfg->Acon);	//MTA
		free(cfg->Ocon);	//MTA
	 if(cfg->detnum)
     	free(cfg->detpos);
     if(cfg->dim.x && cfg->dim.y && cfg->dim.z)
        free(cfg->vol);
		free(cfg->pressure);	//MTA
		
     mcx_initcfg(cfg);
}

//MTA. Identifies  data that needs to be save and saves it 
void mcx_savedata(float *dat, int len, int doappend, char *suffix, Config *cfg, char *fieldnum){
     FILE *fp;
     char name[MAX_PATH_LENGTH];
     if(strcmp(fieldnum,"none")==0){
     	sprintf(name,"%s.%s",cfg->session,suffix);
     }else{
	 	sprintf(name,"%s_%s.%s",cfg->session,fieldnum,suffix);
	 }	
	 	    
     if(doappend){
        fp=fopen(name,"ab");
     }else{
        fp=fopen(name,"wb");
     }
     if(fp==NULL){
	mcx_error(-2,"can not save data to disk",__FILE__,__LINE__);
     }
     if(strcmp(suffix,"mch")==0){
	fwrite(&(cfg->his),sizeof(History),1,fp);
     }
     fwrite(dat,sizeof(float),len,fp);  
     fclose(fp);
}

//MTA. This just prints the simulation log in the command window. 
void mcx_printlog(Config *cfg, char *str){
     if(cfg->flog>0){ /*stdout is 1*/
         fprintf(cfg->flog,"%s\n",str);
     }
}

//MTA. Normalizes a field by some scale
void mcx_normalize(float field[], float scale, int fieldlen){
     int i;
     for(i=0;i<fieldlen;i++){
         field[i]*=scale;
     }
}

//MTA. Prints out messages for certain errors 
void mcx_error(const int id,const char *msg,const char *file,const int linenum){
     fprintf(stdout,"\nMCX ERROR(%d):%s in unit %s:%d\n",id,msg,file,linenum);
     if(id==-CUDA_ERROR_LAUNCH_TIMEOUT){
         fprintf(stdout,"This error often happens when you are using a non-dedicated GPU.\n\
Please checkout FAQ #1 for more details:\n\
URL: http://mcx.sf.net/cgi-bin/index.cgi?Doc/FAQ\n");
     }
#ifdef MCX_CONTAINER
     mcx_throw_exception(id,msg,file,linenum);
#else
     exit(id);
#endif
}

//MTA. Erroring function.  Makes sure the proper number of optical properties, etc. have been entered by the user
void mcx_assert(int ret){
     if(!ret) mcx_error(ret,"assert error",__FILE__,__LINE__);
}

//MTA. Reads config files.
void mcx_readconfig(char *fname, Config *cfg){
     if(fname[0]==0){
     	mcx_loadconfig(stdin,cfg);
     }else{
        FILE *fp=fopen(fname,"rt");
        if(fp==NULL) mcx_error(-2,"can not load the specified config file",__FILE__,__LINE__);
        if(strstr(fname,".json")!=NULL){
            char *jbuf;
            int len;
            cJSON *jroot;

            fseek (fp, 0, SEEK_END);
            len=ftell(fp)+1;
            jbuf=(char *)malloc(len);
            rewind(fp);
            if(fread(jbuf,len-1,1,fp)!=1)
                mcx_error(-2,"reading input file is terminated",__FILE__,__LINE__);
            jbuf[len-1]='\0';
            jroot = cJSON_Parse(jbuf);
            if(jroot){
                mcx_loadjson(jroot,cfg);
                cJSON_Delete(jroot);
            }else{
                char *ptrold, *ptr=(char*)cJSON_GetErrorPtr();
                if(ptr) ptrold=strstr(jbuf,ptr);
                fclose(fp);
                if(ptr && ptrold){
                   char *offs=(ptrold-jbuf>=50) ? ptrold-50 : jbuf;
                   while(offs<ptrold){
                      fprintf(stderr,"%c",*offs);
                      offs++;
                   }
                   fprintf(stderr,"<error>%.50s\n",ptrold);
                }
                free(jbuf);
                mcx_error(-9,"invalid JSON input file",__FILE__,__LINE__);
            }
            free(jbuf);
        }else{
	    mcx_loadconfig(fp,cfg); 
        }
        fclose(fp);
	if(cfg->session[0]=='\0'){
	    strncpy(cfg->session,fname,MAX_SESSION_LENGTH);
	}
     }
}

//MTA. Writes to a config file.
void mcx_writeconfig(char *fname, Config *cfg){
     if(fname[0]==0)
     	mcx_saveconfig(stdout,cfg);
     else{
     	FILE *fp=fopen(fname,"wt");
	if(fp==NULL) mcx_error(-2,"can not write to the specified config file",__FILE__,__LINE__);
	mcx_saveconfig(fp,cfg);     
	fclose(fp);
     }
}


//MTA. Configures simulation domain based on volume bin file (eg. semi60x60x60.bin).
void mcx_prepdomain(char *op_filename, char *ac_filename, Config *cfg){
     int idx1d;
     if(op_filename[0] || cfg->vol){
        if(cfg->vol==NULL){
	    	mcx_loadvolume(op_filename,cfg);
	     	if(cfg->shapedata && strstr(cfg->shapedata,":")!=NULL){
                  int status;
     		  	  Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
        	  	  if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
		  			status=mcx_parse_shapestring(&grid,cfg->shapedata);
		  		  if(status){
		      		MCX_ERROR(status,mcx_last_shapeerror());
		  		  }
	     	 }
		}
	 //MTA		
	 if(ac_filename[0] || cfg->pressure){
        if(cfg->pressure==NULL){
	     mcx_loadacoustics(ac_filename,cfg);
		}
	 }
	
	if(cfg->isrowmajor){
		/*from here on, the array is always col-major*/
		mcx_convertrow2col(&(cfg->vol), &(cfg->dim));
		cfg->isrowmajor=0;
	}
	if(cfg->issavedet)
		mcx_maskdet(cfg);
	if(cfg->srcpos.x<0.f || cfg->srcpos.y<0.f || cfg->srcpos.z<0.f || 
		cfg->srcpos.x>=cfg->dim.x || cfg->srcpos.y>=cfg->dim.y || cfg->srcpos.z>=cfg->dim.z)
		mcx_error(-4,"source position is outside of the volume",__FILE__,__LINE__);
	idx1d=(int)(floor(cfg->srcpos.z)*cfg->dim.y*cfg->dim.x+floor(cfg->srcpos.y)*cfg->dim.x+floor(cfg->srcpos.x));

        /* if the specified source position is outside the domain, move the source
	   along the initial vector until it hit the domain */
	if(cfg->vol && cfg->vol[idx1d]==0){
                printf("source (%f %f %f) is located outside the domain, vol[%d]=%d\n",
		      cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,idx1d,cfg->vol[idx1d]);
		while(cfg->vol[idx1d]==0){
			cfg->srcpos.x+=cfg->srcdir.x;
			cfg->srcpos.y+=cfg->srcdir.y;
			cfg->srcpos.z+=cfg->srcdir.z;
                        if(cfg->srcpos.x<0.f || cfg->srcpos.y<0.f || cfg->srcpos.z<0.f ||
                               cfg->srcpos.x>=cfg->dim.x || cfg->srcpos.y>=cfg->dim.y || cfg->srcpos.z>=cfg->dim.z)
                               mcx_error(-4,"searching non-zero voxel failed along the incident vector",__FILE__,__LINE__);
			idx1d=(int)(floor(cfg->srcpos.z)*cfg->dim.y*cfg->dim.x+floor(cfg->srcpos.y)*cfg->dim.x+floor(cfg->srcpos.x));
		}
		printf("fixing source position to (%f %f %f)\n",cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z);
	}
     }else{
     	mcx_error(-4,"one must specify a binary volume file in order to run the simulation",__FILE__,__LINE__);
     }
}


//MTA. This sets the input parameters for the simulation.
void mcx_loadconfig(FILE *in, Config *cfg){
     uint i,gates,itmp;
     float dtmp;
     char op_filename[MAX_PATH_LENGTH]={0}, ac_filename[MAX_PATH_LENGTH]={0}, comment[MAX_PATH_LENGTH],*comm;
     
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of photons: [1000000]\n\t");
     mcx_assert(fscanf(in,"%d", &(i) )==1); 
     if(cfg->nphoton==0) cfg->nphoton=i;
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d\nPlease specify the random number generator seed: [1234567]\n\t",cfg->nphoton);
     if(cfg->seed==0)
        mcx_assert(fscanf(in,"%d", &(cfg->seed) )==1);
     else
        mcx_assert(fscanf(in,"%d", &itmp )==1);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d\nPlease specify the position of the source (in grid unit): [10 10 5]\n\t",cfg->seed);
     mcx_assert(fscanf(in,"%f %f %f", &(cfg->srcpos.x),&(cfg->srcpos.y),&(cfg->srcpos.z) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(cfg->issrcfrom0==0 && comm!=NULL && sscanf(comm,"%d",&itmp)==1)
         cfg->issrcfrom0=itmp;

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the normal direction of the source fiber: [0 0 1]\n\t",
                                   cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z);
     if(!cfg->issrcfrom0){
        cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
     }
     mcx_assert(fscanf(in,"%f %f %f", &(cfg->srcdir.x),&(cfg->srcdir.y),&(cfg->srcdir.z) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the time gates (format: start end step) in seconds [0.0 1e-9 1e-10]\n\t",
                                   cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z);
     mcx_assert(fscanf(in,"%f %f %f", &(cfg->tstart),&(cfg->tend),&(cfg->tstep) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the path to the volume binary file:\n\t",
                                   cfg->tstart,cfg->tend,cfg->tstep);
     if(cfg->tstart>cfg->tend || cfg->tstep==0.f){
         mcx_error(-9,"incorrect time gate settings",__FILE__,__LINE__);
     }
     gates=(uint)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
     if(cfg->maxgate>gates)
	 cfg->maxgate=gates;

     mcx_assert(fscanf(in,"%s", op_filename)==1);
     if(cfg->rootpath[0]){
#ifdef WIN32
         sprintf(comment,"%s\\%s",cfg->rootpath,op_filename);
#else
         sprintf(comment,"%s/%s",cfg->rootpath,op_filename);
#endif
         strncpy(op_filename,comment,MAX_PATH_LENGTH);
     }
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     
     if(in==stdin)
     	fprintf(stdout,"%s\nPlease specify the path to the Acoustics binary file:\n\t",
                                   op_filename);
     mcx_assert(fscanf(in,"%s", ac_filename)==1);
     if(cfg->rootpath[0]){
#ifdef WIN32
         sprintf(comment,"%s\\%s",cfg->rootpath,ac_filename);
#else
         sprintf(comment,"%s/%s",cfg->rootpath,ac_filename);
#endif
         strncpy(ac_filename,comment,MAX_PATH_LENGTH);
     }
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%s\nPlease specify the x voxel size (in mm), x dimension, min and max x-index [1.0 100 1 100]:\n\t",ac_filename);
     mcx_assert(fscanf(in,"%f %d %d %d", &(cfg->steps.x),&(cfg->dim.x),&(cfg->crop0.x),&(cfg->crop1.x))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(cfg->steps.x!=cfg->steps.y || cfg->steps.y!=cfg->steps.z)
        mcx_error(-9,"MCX currently does not support anisotropic voxels",__FILE__,__LINE__);
	 
	 //MTA		
     if(cfg->steps.x!=1.f && cfg->unitinmm==1.f){
        cfg->unitinmm=cfg->steps.x;
        cfg->steps.x=1.f;cfg->steps.y=1.f;cfg->steps.z=1.f;
     }

     /*if(cfg->unitinmm!=1.f){
        cfg->steps.x=cfg->unitinmm; cfg->steps.y=cfg->unitinmm; cfg->steps.z=cfg->unitinmm;
     }*/

     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the y voxel size (in mm), y dimension, min and max y-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.x,cfg->dim.x,cfg->crop0.x,cfg->crop1.x);
     mcx_assert(fscanf(in,"%f %d %d %d", &(cfg->steps.y),&(cfg->dim.y),&(cfg->crop0.y),&(cfg->crop1.y))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the z voxel size (in mm), z dimension, min and max z-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.y,cfg->dim.y,cfg->crop0.y,cfg->crop1.y);
     mcx_assert(fscanf(in,"%f %d %d %d", &(cfg->steps.z),&(cfg->dim.z),&(cfg->crop0.z),&(cfg->crop1.z))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(cfg->sradius>0.f){
     	cfg->crop0.x=MAX((uint)(cfg->srcpos.x-cfg->sradius),0);
     	cfg->crop0.y=MAX((uint)(cfg->srcpos.y-cfg->sradius),0);
     	cfg->crop0.z=MAX((uint)(cfg->srcpos.z-cfg->sradius),0);
     	cfg->crop1.x=MIN((uint)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
     	cfg->crop1.y=MIN((uint)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
     	cfg->crop1.z=MIN((uint)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
     }else if(cfg->sradius==0.f){
     	memset(&(cfg->crop0),0,sizeof(uint3));
     	memset(&(cfg->crop1),0,sizeof(uint3));
     }else{
        /*
           if I define -R with a negative radius, I will use crop0/crop1 to set the cachebox
           nothing need to change here.
        */
     }
     
     //MTA.
     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify mass density (kg/m^3), speed of sound (m/s), source frequency (hz): [1000.0 1500.0 1100000.0]:\n\t",
                                  cfg->steps.z,cfg->dim.z,cfg->crop0.z,cfg->crop1.z);
        cfg->Acon=(Aconstants*)malloc(sizeof(Aconstants));	//MTA
     	mcx_assert(fscanf(in,"%f %f %f", &(cfg->Acon[0].rho),&(cfg->Acon[0].va),&(cfg->Acon[0].f))==3); //MTA
     	comm=fgets(comment,MAX_PATH_LENGTH,in);//MTA
     	if(cfg->Acon[0].f < 1e3) //MTA Change frequency if entered in MHz
     		cfg->Acon[0].f *= 1e6;   //MTA  

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify optical wavelength in vacuum (nm) and elasto-optic coefficient: [1064.0 0.32]:\n\t",
                                  cfg->Acon[0].rho,cfg->Acon[0].va,cfg->Acon[0].f);	//MTA
        cfg->Ocon=(Oconstants*)malloc(sizeof(Oconstants));	//MTA
     	mcx_assert(fscanf(in,"%f %f", &(cfg->Ocon[0].lambda),&(cfg->Ocon[0].nu))==2);	//MTA
     	comm=fgets(comment,MAX_PATH_LENGTH,in);	//MTA
     	
     	if(cfg->Ocon[0].lambda > 1)	//MTA Adjust if entered in nm
     		cfg->Ocon[0].lambda /= 1e9;      //MTA
     
     ///////////////////////////////////////////////////////////////////////////////////////
     
     if(in==stdin)
     	fprintf(stdout,"%f %f\nPlease specify the total types of media:\n\t",
                                  cfg->Ocon[0].lambda,cfg->Ocon[0].nu);
     mcx_assert(fscanf(in,"%d", &(cfg->medianum))==1);
     cfg->medianum++;
     if(cfg->medianum>MAX_PROP)
         mcx_error(-4,"input media types exceed the maximum (255)",__FILE__,__LINE__); //MTA.
     comm=fgets(comment,MAX_PATH_LENGTH,in);

	 if(in==stdin)
	 	fprintf(stdout,"%d\n",cfg->medianum);
	 	
     cfg->prop=(Medium*)malloc(sizeof(Medium)*cfg->medianum);
		
     cfg->prop[0].mua=0.f; /*property 0 is already air*/
     cfg->prop[0].mus=0.f;
     cfg->prop[0].g=1.f;
     cfg->prop[0].n=1.f;

     for(i=1;i<cfg->medianum;i++){
        if(in==stdin)
			fprintf(stdout,"Please define medium #%d: mus(1/mm), anisotropy, mua(1/mm), and refractive index: [1.01 0.01 0.04 1.37]\n\t",i);
     	mcx_assert(fscanf(in, "%f %f %f %f", &(cfg->prop[i].mus),&(cfg->prop[i].g),&(cfg->prop[i].mua),&(cfg->prop[i].n))==4);
		comm=fgets(comment,MAX_PATH_LENGTH,in);
        if(in==stdin)
			fprintf(stdout,"Optical Properties %f %f %f %f \n",cfg->prop[i].mus,cfg->prop[i].g,cfg->prop[i].mua,cfg->prop[i].n);
     }
     
     if(cfg->unitinmm!=1.f){
         for(i=1;i<cfg->medianum;i++){
		cfg->prop[i].mus*=cfg->unitinmm;
		cfg->prop[i].mua*=cfg->unitinmm;
         }
     }
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of detectors and fiber diameter (in grid unit):\n\t");
     mcx_assert(fscanf(in,"%d %f", &(cfg->detnum), &(cfg->detradius))==2);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d %f\n",cfg->detnum,cfg->detradius);
     cfg->detpos=(float4*)malloc(sizeof(float4)*cfg->detnum);
     if(cfg->issavedet && cfg->detnum==0) 
      	cfg->issavedet=0;
     for(i=0;i<cfg->detnum;i++){
        if(in==stdin)
		fprintf(stdout,"Please define detector #%d: x,y,z (in grid unit): [5 5 5 1]\n\t",i);
     	mcx_assert(fscanf(in, "%f %f %f", &(cfg->detpos[i].x),&(cfg->detpos[i].y),&(cfg->detpos[i].z))==3);
	cfg->detpos[i].w=cfg->detradius;
        if(!cfg->issrcfrom0){
		cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
	}
        comm=fgets(comment,MAX_PATH_LENGTH,in);
        if(comm!=NULL && sscanf(comm,"%f",&dtmp)==1)
            cfg->detpos[i].w=dtmp;

        if(in==stdin)
		fprintf(stdout,"%f %f %f\n",cfg->detpos[i].x,cfg->detpos[i].y,cfg->detpos[i].z);
     }
     mcx_prepdomain(op_filename,ac_filename,cfg);
     cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.colcount=cfg->medianum+3;  //MTA
}

// JSON NOT UPDATED FOR AO-MCX!!!
int mcx_loadjson(cJSON *root, Config *cfg){
     int i;
     cJSON *Domain, *Optode, *Forward, *Session, *Shapes, *tmp, *subitem;
     char filename[MAX_PATH_LENGTH]={'\0'};
     Domain  = cJSON_GetObjectItem(root,"Domain");
     Optode  = cJSON_GetObjectItem(root,"Optode");
     Session = cJSON_GetObjectItem(root,"Session");
     Forward = cJSON_GetObjectItem(root,"Forward");
     Shapes  = cJSON_GetObjectItem(root,"Shapes");
     
     //char FOO;
     
     if(Domain){
        char volfile[MAX_PATH_LENGTH];
	cJSON *meds,*val;
	val=FIND_JSON_OBJ("VolumeFile","Domain.VolumeFile",Domain);
	if(val){
          strncpy(volfile, val->valuestring, MAX_PATH_LENGTH);
          if(cfg->rootpath[0]){
#ifdef WIN32
           sprintf(filename,"%s\\%s",cfg->rootpath,volfile);
#else
           sprintf(filename,"%s/%s",cfg->rootpath,volfile);
#endif
          }else{
	     strncpy(filename,volfile,MAX_PATH_LENGTH);
	  }
	}
        if(cfg->unitinmm==1.f)
	    cfg->unitinmm=FIND_JSON_KEY("LengthUnit","Domain.LengthUnit",Domain,1.f,valuedouble);
        meds=FIND_JSON_OBJ("Media","Domain.Media",Domain);
        if(meds){
           cJSON *med=meds->child;
           if(med){
             cfg->medianum=cJSON_GetArraySize(meds);
             if(cfg->medianum>MAX_PROP)
                 MCX_ERROR(-4,"input media types exceed the maximum (255)");
             cfg->prop=(Medium*)malloc(sizeof(Medium)*cfg->medianum);
             for(i=0;i<cfg->medianum;i++){
               cJSON *val=FIND_JSON_OBJ("mua",(MCX_ERROR(-1,"You must specify absorption coeff, default in 1/mm"),""),med);
               if(val) cfg->prop[i].mua=val->valuedouble;
	       val=FIND_JSON_OBJ("mus",(MCX_ERROR(-1,"You must specify scattering coeff, default in 1/mm"),""),med);
               if(val) cfg->prop[i].mus=val->valuedouble;
	       val=FIND_JSON_OBJ("g",(MCX_ERROR(-1,"You must specify anisotropy [0-1]"),""),med);
               if(val) cfg->prop[i].g=val->valuedouble;
	       val=FIND_JSON_OBJ("n",(MCX_ERROR(-1,"You must specify refractive index"),""),med);
	       if(val) cfg->prop[i].n=val->valuedouble;

               med=med->next;
               if(med==NULL) break;
             }
	     if(cfg->unitinmm!=1.f){
        	 for(i=0;i<cfg->medianum;i++){
			cfg->prop[i].mus*=cfg->unitinmm;
			cfg->prop[i].mua*=cfg->unitinmm;
        	 }
	     }
           }
        }
	val=FIND_JSON_OBJ("Dim","Domain.Dim",Domain);
	if(val && cJSON_GetArraySize(val)>=3){
	   cfg->dim.x=val->child->valueint;
           cfg->dim.y=val->child->next->valueint;
           cfg->dim.z=val->child->next->next->valueint;
	}else{
	   MCX_ERROR(-1,"You must specify the dimension of the volume");
	}
	val=FIND_JSON_OBJ("Step","Domain.Step",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->steps.x=val->child->valuedouble;
               cfg->steps.y=val->child->next->valuedouble;
               cfg->steps.z=val->child->next->next->valuedouble;
           }else{
	       MCX_ERROR(-1,"Domain::Step has incorrect element numbers");
           }
	}
	if(cfg->steps.x!=cfg->steps.y || cfg->steps.y!=cfg->steps.z)
           mcx_error(-9,"MCX currently does not support anisotropic voxels",__FILE__,__LINE__);

	//MTA
	if(cfg->steps.x!=1.f && cfg->unitinmm==1.f){
           cfg->unitinmm=cfg->steps.x;
           cfg->steps.x=1.f;cfg->steps.y=1.f;cfg->steps.z=1.f;
    }

/*	if(cfg->unitinmm!=1.f){
           cfg->steps.x=cfg->unitinmm; cfg->steps.y=cfg->unitinmm; cfg->steps.z=cfg->unitinmm;
	}*/
	val=FIND_JSON_OBJ("CacheBoxP0","Domain.CacheBoxP0",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->crop0.x=val->child->valueint;
               cfg->crop0.y=val->child->next->valueint;
               cfg->crop0.z=val->child->next->next->valueint;
           }else{
	       MCX_ERROR(-1,"Domain::CacheBoxP0 has incorrect element numbers");
           }
	}
	val=FIND_JSON_OBJ("CacheBoxP1","Domain.CacheBoxP1",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->crop1.x=val->child->valueint;
               cfg->crop1.y=val->child->next->valueint;
               cfg->crop1.z=val->child->next->next->valueint;
           }else{
	       MCX_ERROR(-1,"Domain::CacheBoxP1 has incorrect element numbers");
           }
	}
	val=FIND_JSON_OBJ("OriginType","Domain.OriginType",Domain);
	if(val && cfg->issrcfrom0==0) cfg->issrcfrom0=val->valueint;

	if(cfg->sradius>0.f){
     	   cfg->crop0.x=MAX((uint)(cfg->srcpos.x-cfg->sradius),0);
     	   cfg->crop0.y=MAX((uint)(cfg->srcpos.y-cfg->sradius),0);
     	   cfg->crop0.z=MAX((uint)(cfg->srcpos.z-cfg->sradius),0);
     	   cfg->crop1.x=MIN((uint)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
     	   cfg->crop1.y=MIN((uint)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
     	   cfg->crop1.z=MIN((uint)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
	}else if(cfg->sradius==0.f){
     	   memset(&(cfg->crop0),0,sizeof(uint3));
     	   memset(&(cfg->crop1),0,sizeof(uint3));
	}else{
           /*
              if I define -R with a negative radius, I will use crop0/crop1 to set the cachebox
              nothing need to change here.
           */
	}
     }
     if(Optode){
        cJSON *dets, *src=FIND_JSON_OBJ("Source","Optode.Source",Optode);
        if(src){
           subitem=FIND_JSON_OBJ("Pos","Optode.Source.Pos",src);
           if(subitem){
              cfg->srcpos.x=subitem->child->valuedouble;
              cfg->srcpos.y=subitem->child->next->valuedouble;
              cfg->srcpos.z=subitem->child->next->next->valuedouble;
           }
           subitem=FIND_JSON_OBJ("Dir","Optode.Source.Dir",src);
           if(subitem){
              cfg->srcdir.x=subitem->child->valuedouble;
              cfg->srcdir.y=subitem->child->next->valuedouble;
              cfg->srcdir.z=subitem->child->next->next->valuedouble;
           }
	   if(!cfg->issrcfrom0){
              cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
	   }
        }
        dets=FIND_JSON_OBJ("Detector","Optode.Detector",Optode);
        if(dets){
           cJSON *det=dets->child;
           if(det){
             cfg->detnum=cJSON_GetArraySize(dets);
             cfg->detpos=(float4*)malloc(sizeof(float4)*cfg->detnum);
	     if(cfg->issavedet && cfg->detnum==0) 
      		cfg->issavedet=0;
             for(i=0;i<cfg->detnum;i++){
               cJSON *pos=dets, *rad=NULL;
               rad=FIND_JSON_OBJ("R","Optode.Detector.R",det);
               if(cJSON_GetArraySize(det)==2){
                   pos=FIND_JSON_OBJ("Pos","Optode.Detector.Pos",det);
               }
               if(pos){
	           cfg->detpos[i].x=pos->child->valuedouble;
                   cfg->detpos[i].y=pos->child->next->valuedouble;
	           cfg->detpos[i].z=pos->child->next->next->valuedouble;
               }
               if(rad){
                   cfg->detpos[i].w=rad->valuedouble;
               }
               if(!cfg->issrcfrom0){
		   cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
	       }
               det=det->next;
               if(det==NULL) break;
             }
           }
        }
     }
     if(Session){
        if(cfg->seed==0)      cfg->seed=FIND_JSON_KEY("RNGSeed","Session.RNGSeed",Session,-1,valueint);
        if(cfg->nphoton==0)   cfg->nphoton=FIND_JSON_KEY("Photons","Session.Photons",Session,0,valuedouble);
        if(cfg->session[0]=='\0')  strncpy(cfg->session, FIND_JSON_KEY("ID","Session.ID",Session,"default",valuestring), MAX_SESSION_LENGTH);
        if(cfg->rootpath[0]=='\0') strncpy(cfg->rootpath, FIND_JSON_KEY("RootPath","Session.RootPath",Session,"",valuestring), MAX_PATH_LENGTH);

        if(!cfg->isreflect)   cfg->isreflect=FIND_JSON_KEY("DoMismatch","Session.DoMismatch",Session,cfg->isreflect,valueint);
        if(cfg->issave2pt)    cfg->issave2pt=FIND_JSON_KEY("DoSaveVolume","Session.DoSaveVolume",Session,cfg->issave2pt,valueint);
        if(cfg->isnormalized) cfg->isnormalized=FIND_JSON_KEY("DoNormalize","Session.DoNormalize",Session,cfg->isnormalized,valueint);
        if(!cfg->issavedet)   cfg->issavedet=FIND_JSON_KEY("DoPartialPath","Session.DoPartialPath",Session,cfg->issavedet,valueint);
     }
     if(Forward){
        uint gates;
        cfg->tstart=FIND_JSON_KEY("T0","Forward.T0",Forward,0.0,valuedouble);
        cfg->tend  =FIND_JSON_KEY("T1","Forward.T1",Forward,0.0,valuedouble);
        cfg->tstep =FIND_JSON_KEY("Dt","Forward.Dt",Forward,0.0,valuedouble);
	if(cfg->tstart>cfg->tend || cfg->tstep==0.f)
            mcx_error(-9,"incorrect time gate settings",__FILE__,__LINE__);

        gates=(uint)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
        if(cfg->maxgate>gates)
	    cfg->maxgate=gates;
     }
     if(filename[0]=='\0'){
         if(Shapes){
             int status;
             Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
             if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
	     status=mcx_parse_jsonshapes(root, &grid);
	     if(status){
	         MCX_ERROR(status,mcx_last_shapeerror());
	     }
	 }else{
	     MCX_ERROR(-1,"You must either define Domain.VolumeFile, or define a Shapes section");
	 }
     }else if(Shapes){
         MCX_ERROR(-1,"You can not specify both Domain.VolumeFile and Shapes sections");
     }
     //mcx_prepdomain(filename,FOO,cfg);
     cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.colcount=cfg->medianum+3; /*column count=maxmedia+2*/  //MTA
     return 0;
}

//MTA. Saves a file detailing the config. I'M NOT SURE IF THIS WORKS WITH AO-MCX
void mcx_saveconfig(FILE *out, Config *cfg){
     uint i;

     fprintf(out,"%d\n", (cfg->nphoton) );
     fprintf(out,"%d\n", (cfg->seed) );
     fprintf(out,"%f %f %f\n", (cfg->srcpos.x),(cfg->srcpos.y),(cfg->srcpos.z) );
     fprintf(out,"%f %f %f\n", (cfg->srcdir.x),(cfg->srcdir.y),(cfg->srcdir.z) );
     fprintf(out,"%e %e %e\n", (cfg->tstart),(cfg->tend),(cfg->tstep) );
     fprintf(out,"%f %d %d %d\n", (cfg->steps.x),(cfg->dim.x),(cfg->crop0.x),(cfg->crop1.x));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.y),(cfg->dim.y),(cfg->crop0.y),(cfg->crop1.y));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.z),(cfg->dim.z),(cfg->crop0.z),(cfg->crop1.z));
     fprintf(out,"%d\n", (cfg->medianum));
	// fprintf(out,"%f %f %f\n", (cfg->Acon.rho),(cfg->Acon.va),(cfg->Acon.f));	//MTA added 6/27/12
	 //fprintf(out,"%f %f\n", (cfg->.lambda),(cfg->.nu));	//MTA added 6/26/12, changed 6/27/12


     for(i=0;i<cfg->medianum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->prop[i].mus),(cfg->prop[i].g),(cfg->prop[i].mua),(cfg->prop[i].n));
     }
     fprintf(out,"%d", (cfg->detnum));
     for(i=0;i<cfg->detnum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->detpos[i].x),(cfg->detpos[i].y),(cfg->detpos[i].z),(cfg->detpos[i].w));
     }
}
 
//MTA. This sets up the simulation domain based on the volume binary file (eg. semi60x60x60.bin).
void mcx_loadvolume(char *filename,Config *cfg){
     unsigned int i,datalen,res;
     FILE *fp;
     
     if(strstr(filename,".json")!=NULL){
         int status;
         Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
	 if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
         status=mcx_load_jsonshapes(&grid,filename);
	 if(status){
	     MCX_ERROR(status,mcx_last_shapeerror());
	 }
	 return;
     }
     fp=fopen(filename,"rb");
     if(fp==NULL){
     	     mcx_error(-5,"the specified binary volume file does not exist",__FILE__,__LINE__);
     }
     if(cfg->vol){
     	     free(cfg->vol);
     	     cfg->vol=NULL;
     }
     datalen=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     cfg->vol=(unsigned char*)malloc(sizeof(unsigned char)*datalen);
     res=fread(cfg->vol,sizeof(unsigned char),datalen,fp);
     fclose(fp);
     if(res!=datalen){
     	 mcx_error(-6,"file size does not match specified dimensions",__FILE__,__LINE__);
     }
     for(i=0;i<datalen;i++){
         if(cfg->vol[i]>=cfg->medianum)
            mcx_error(-6,"medium index exceeds the specified medium types",__FILE__,__LINE__);
     }
}

// MTA This entire sub-function was written by me
void mcx_loadacoustics(char *filename,Config *cfg){
     unsigned int i,j,k,index,datalen,res;
     FILE *fp;
     
     unsigned int current_pos=0;
     
     fp=fopen(filename,"rb");
     if(fp==NULL){
     	     mcx_error(-5,"the specified binary acoustics file does not exist",__FILE__,__LINE__);
     }
     if(cfg->pressure){
     	     free(cfg->pressure);
     	     cfg->pressure=NULL;
     }
     
     datalen=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     float * my_data = (float *)malloc(datalen*4*sizeof(float));
     cfg->pressure = (Acoustics*)malloc(datalen*sizeof(Acoustics));
     
     res=fread( my_data, sizeof(float), datalen*4, fp);
     if(res!=datalen*4){
     	 mcx_error(-6,"file size does not match specified dimensions",__FILE__,__LINE__);
     }
     
     for( k =0; k < cfg->dim.z; ++k){
		for( j =0; j < cfg->dim.y; ++j){
			for( i =0; i < cfg->dim.x; ++i){
					index =  cfg->dim.x*cfg->dim.y*k + cfg->dim.x*j + i;
					cfg->pressure[current_pos].Px = my_data[index];
	 				cfg->pressure[current_pos].Py = my_data[datalen + index];
					cfg->pressure[current_pos].Pz = my_data[datalen*2 + index];
					cfg->pressure[current_pos].USphase = my_data[datalen*3 + index];
					
					/*if(i<10 && j==0 && k==0)
						printf("%f\t%f\t%f\t%f\n", cfg->pressure[current_pos].Px, cfg->pressure[current_pos].Py, cfg->pressure[current_pos].Pz, cfg->pressure[current_pos].USphase );*/
											
					current_pos = current_pos+1;
				}
			}
		} 
     
     fclose(fp);
     
     free(my_data);
     
     /*for(i=0;i<datalen;i++){
         if(sqrtf(cfg->pressure[i].Px*cfg->pressure[i].Px+cfg->pressure[i].Py*cfg->pressure[i].Py+cfg->pressure[i].Pz*cfg->pressure[i].Pz)<0.01f)
            mcx_error(-6,"cfg->pressure isn't set properly",__FILE__,__LINE__);
     }*/
}

void  mcx_convertrow2col(unsigned char **vol, uint3 *dim){
     uint x,y,z;
     unsigned int dimxy,dimyz;
     unsigned char *newvol=NULL;
          
     if(*vol==NULL || dim->x==0 || dim->y==0 || dim->z==0){
     	return;
     }     
     newvol=(unsigned char*)malloc(sizeof(unsigned char)*dim->x*dim->y*dim->z);
     dimxy=dim->x*dim->y;
     dimyz=dim->y*dim->z;
     for(x=0;x<dim->x;x++)
      for(y=0;y<dim->y;y++)
       for(z=0;z<dim->z;z++){
       		newvol[z*dimxy+y*dim->x+x]=*vol[x*dimyz+y*dim->z+z];
       }
     free(*vol);
     *vol=newvol;
}


//MTA. This function determines which boundary voxels are used as "detectors" (given a detector input)
void  mcx_maskdet(Config *cfg){
     uint d,dx,dy,dz,idx1d,zi,yi,c,count;
     float x,y,z,ix,iy,iz,rx,ry,rz,d2,mind2,d2max;
     unsigned char *padvol;
     const float corners[8][3]={{0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f},
                                {1.f,1.f,0.f},{1.f,0.f,1.f},{0.f,1.f,1.f},{1.f,1.f,1.f}};
     
     dx=cfg->dim.x+2;
     dy=cfg->dim.y+2;
     dz=cfg->dim.z+2;
     
     /*handling boundaries in a volume search is tedious, I first pad vol by a layer of zeros,
       then I don't need to worry about boundaries any more*/

     padvol=(unsigned char*)calloc(dx*dy,dz);

     for(zi=1;zi<=cfg->dim.z;zi++)
        for(yi=1;yi<=cfg->dim.y;yi++)
	        memcpy(padvol+zi*dy*dx+yi*dx+1,cfg->vol+(zi-1)*cfg->dim.y*cfg->dim.x+(yi-1)*cfg->dim.x,cfg->dim.x);

     /**
        The goal here is to find a set of voxels for each 
	detector so that the intersection between a sphere
	of R=cfg->detradius,c0=cfg->detpos[d] and the object 
	surface (or bounding box) is fully covered.
     */
     for(d=0;d<cfg->detnum;d++){                             /*loop over each detector*/
        count=0;
        d2max=(cfg->detpos[d].w+1.7321f)*(cfg->detpos[d].w+1.7321f);
        for(z=-cfg->detpos[d].w-1.f;z<=cfg->detpos[d].w+1.f;z+=0.5f){   /*search in a cube with edge length 2*R+3*/
           iz=z+cfg->detpos[d].z;
           for(y=-cfg->detpos[d].w-1.f;y<=cfg->detpos[d].w+1.f;y+=0.5f){
              iy=y+cfg->detpos[d].y;
              for(x=-cfg->detpos[d].w-1.f;x<=cfg->detpos[d].w+1.f;x+=0.5f){
	         ix=x+cfg->detpos[d].x;

		 if(iz<0||ix<0||iy<0||ix>=cfg->dim.x||iy>=cfg->dim.y||iz>=cfg->dim.z||
		    x*x+y*y+z*z > (cfg->detpos[d].w+1.f)*(cfg->detpos[d].w+1.f))
		     continue;
		 mind2=VERY_BIG;
                 for(c=0;c<8;c++){ /*test each corner of a voxel*/
			rx=(int)ix-cfg->detpos[d].x+corners[c][0];
			ry=(int)iy-cfg->detpos[d].y+corners[c][1];
			rz=(int)iz-cfg->detpos[d].z+corners[c][2];
			d2=rx*rx+ry*ry+rz*rz;
		 	if(d2>d2max){ /*R+sqrt(3) to make sure the circle is fully corvered*/
				mind2=VERY_BIG;
		     		break;
			}
			if(d2<mind2) mind2=d2;
		 }
		 if(mind2==VERY_BIG || mind2>=cfg->detpos[d].w*cfg->detpos[d].w) continue;
		 idx1d=((int)(iz+1.f)*dy*dx+(int)(iy+1.f)*dx+(int)(ix+1.f)); /*1.f comes from the padded layer*/

		 if(padvol[idx1d])  /*looking for a voxel on the interface or bounding box*/
                  if(!(padvol[idx1d+1]&&padvol[idx1d-1]&&padvol[idx1d+dx]&&padvol[idx1d-dx]&&padvol[idx1d+dy*dx]&&padvol[idx1d-dy*dx]&&
		     padvol[idx1d+dx+1]&&padvol[idx1d+dx-1]&&padvol[idx1d-dx+1]&&padvol[idx1d-dx-1]&&
		     padvol[idx1d+dy*dx+1]&&padvol[idx1d+dy*dx-1]&&padvol[idx1d-dy*dx+1]&&padvol[idx1d-dy*dx-1]&&
		     padvol[idx1d+dy*dx+dx]&&padvol[idx1d+dy*dx-dx]&&padvol[idx1d-dy*dx+dx]&&padvol[idx1d-dy*dx-dx]&&
		     padvol[idx1d+dy*dx+dx+1]&&padvol[idx1d+dy*dx+dx-1]&&padvol[idx1d+dy*dx-dx+1]&&padvol[idx1d+dy*dx-dx-1]&&
		     padvol[idx1d-dy*dx+dx+1]&&padvol[idx1d-dy*dx+dx-1]&&padvol[idx1d-dy*dx-dx+1]&&padvol[idx1d-dy*dx-dx-1])){
		          cfg->vol[((int)iz*cfg->dim.y*cfg->dim.x+(int)iy*cfg->dim.x+(int)ix)]|=(1<<7);/*set the highest bit to 1*/
                          count++;
	          }
	       }
	   }
        }
        if(cfg->issavedet && count==0)
              fprintf(stderr,"MCX WARNING: detector %d is not located on an interface, please check coordinates.\n",d+1);
     }
     /**
         To test the results, you should use -M to dump the det-mask, load 
	 it in matlab, and plot the interface containing the detector with
	 pcolor() (has the matching index), and then draw a circle with the
	 radius and center set in the input file. the pixels should completely
	 cover the circle.
     */
     if(cfg->isdumpmask){
     	 char fname[MAX_PATH_LENGTH];
	 FILE *fp;
	 sprintf(fname,"%s.mask",cfg->session);
	 if((fp=fopen(fname,"wb"))==NULL){
	 	mcx_error(-10,"can not save mask file",__FILE__,__LINE__);
	 }
	 if(fwrite(cfg->vol,cfg->dim.x*cfg->dim.y,cfg->dim.z,fp)!=cfg->dim.z){
	 	mcx_error(-10,"can not save mask file",__FILE__,__LINE__);
	 }
	 fclose(fp);
         free(padvol);
	 exit(0);
     }
     free(padvol);
}

//MTA. I'm really not sure what's going on here.
int mcx_readarg(int argc, char *argv[], int id, void *output,const char *type){
     /*
         when a binary option is given without a following number (0~1), 
         we assume it is 1
     */
     if(strcmp(type,"char")==0 && (id>=argc-1||(argv[id+1][0]<'0'||argv[id+1][0]>'9'))){
	*((char*)output)=1;
	return id;
     }
     if(id<argc-1){
         if(strcmp(type,"char")==0)
             *((char*)output)=atoi(argv[id+1]);
	 else if(strcmp(type,"int")==0)
             *((int*)output)=atoi(argv[id+1]);
	 else if(strcmp(type,"float")==0)
             *((float*)output)=atof(argv[id+1]);
	 else if(strcmp(type,"string")==0)
	     strcpy((char *)output,argv[id+1]);
     }else{
     	 mcx_error(-1,"incomplete input",__FILE__,__LINE__);
     }
     return id+1;
}

//MTA. Again, no idea. Some function that looks for errors in the config file.
int mcx_remap(char *opt){
    int i=0;
    while(shortopt[i]!='\0'){
	if(strcmp(opt,fullopt[i])==0){
		opt[1]=shortopt[i];
		opt[2]='\0';
		return 0;
	}
	i++;
    }
    return 1;
}

//MTA. Parses input config file or command line options.  Maybe add an option here for AO in the future.
void mcx_parsecmd(int argc, char* argv[], Config *cfg){
     int i=1,isinteractive=1,issavelog=0;
     char filename[MAX_PATH_LENGTH]={0};
     char logfile[MAX_PATH_LENGTH]={0};
     float np=0.f;
//MTA. Outputs text with instructions detailing how to use MCX
     if(argc<=1){
     	mcx_usage(argv[0]);
     	exit(0);
     }
//MTA. Error Message
     while(i<argc){
     	    if(argv[i][0]=='-'){
		if(argv[i][1]=='-'){
			if(mcx_remap(argv[i])){
				mcx_error(-2,"unknown verbose option",__FILE__,__LINE__);
			}
		}
	        switch(argv[i][1]){
		     case 'h': 
		                mcx_usage(argv[0]);
				exit(0);
		     case 'i':
				if(filename[0]){
					mcx_error(-2,"you can not specify both interactive mode and config file",__FILE__,__LINE__);
				}
		     		isinteractive=1;
				break;
		     case 'f': 
		     		isinteractive=0;
		     	        i=mcx_readarg(argc,argv,i,filename,"string");
				break;
		     case 'm':
                                /*from rev 191, we have enabled -n and disabled -m*/
				mcx_error(-2,"specifying photon move is not supported any more, please use -n",__FILE__,__LINE__);
		     	        i=mcx_readarg(argc,argv,i,&(cfg->nphoton),"int");
		     	        break;
		     case 'n':
		     	        i=mcx_readarg(argc,argv,i,&(np),"float");
				cfg->nphoton=(int)np;
		     	        break;
		     case 't':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->nthread),"int");
		     	        break;
                     case 'T':
                               	i=mcx_readarg(argc,argv,i,&(cfg->nblocksize),"int");
                               	break;
		     case 's':
		     	        i=mcx_readarg(argc,argv,i,cfg->session,"string");
		     	        break;
		     case 'a':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isrowmajor),"char");
		     	        break;
		     case 'g':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->maxgate),"int");
		     	        break;
		     case 'b':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isreflect),"char");
				cfg->isref3=cfg->isreflect;
		     	        break;
                     case 'B':
                                i=mcx_readarg(argc,argv,i,&(cfg->isrefint),"char");
                               	break;
		     case 'd':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->issavedet),"char");
		     	        break;
		     case 'r':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->respin),"int");
		     	        break;
		     case 'S':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->issave2pt),"char");
		     	        break;
		     case 'p':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->printnum),"int");
		     	        break;
                     case 'e':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->minenergy),"float");
                                break;
		     case 'U':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isnormalized),"char");
		     	        break;
                     case 'R':
                                i=mcx_readarg(argc,argv,i,&(cfg->sradius),"float");
                                break;
                     case 'u':
                                i=mcx_readarg(argc,argv,i,&(cfg->unitinmm),"float");
                                break;
                     case 'l':
                                issavelog=1;
                                break;
		     case 'L':
                                cfg->isgpuinfo=2;
		                break;
		     case 'I':
                                cfg->isgpuinfo=1;
		                break;
		     case 'o':
		     	        i=mcx_readarg(argc,argv,i,cfg->rootpath,"string");
		     	        break;
                     case 'G':
                                i=mcx_readarg(argc,argv,i,&(cfg->gpuid),"int");
                                break;
                     case 'z':
                                i=mcx_readarg(argc,argv,i,&(cfg->issrcfrom0),"char");
                                break;
		     case 'M':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isdumpmask),"char");
		     	        break;
		     case 'H':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->maxdetphoton),"int");
		     	        break;
                     case 'P':
                                cfg->shapedata=argv[++i];
                                break;
                     case 'A':
                                i=mcx_readarg(argc,argv,i,&(cfg->autopilot),"char");
                                break;
                     case 'E':
                                i=mcx_readarg(argc,argv,i,&(cfg->seed),"int");
                                break;
                     case 'v':
                                mcx_version(cfg);
				break;
		}
	    }
	    i++;
     }
     if(issavelog && cfg->session){
          sprintf(logfile,"%s.log",cfg->session);
          cfg->flog=fopen(logfile,"wt");
          if(cfg->flog==NULL){
		cfg->flog=stdout;
		fprintf(cfg->flog,"unable to save to log file, will print from stdout\n");
          }
     }
     if(cfg->isgpuinfo!=2){ /*print gpu info only*/
	  if(isinteractive){
             mcx_readconfig((char*)"",cfg);
	  }else{
     	     mcx_readconfig(filename,cfg);
	  }
     }
}

void mcx_version(Config *cfg){
    const char ver[]="$Rev:: 272  $";
    int v=0;
    sscanf(ver,"$Rev::%d",&v);
    fprintf(cfg->flog, "MCX Revision %d\n",v);
    exit(0);
}

//MTA. Just prints text with instructions on how to use MCX.
void mcx_usage(char *exename){
     printf("\
###############################################################################\n\
#               Acousto-Optic Monte Carlo eXtreme (AO-MCX) -- CUDA            #\n\
#   Orig. Copyright (c) 2009-2012 Qianqian Fang <fangq@nmr.mgh.harvard.edu>   #\n\
#    Martinos Center for Biomedical Imaging, Massachusetts General Hospital   #\n\
#																			  #\n\
#				  AO-MCX Created by Matt Adams <adamsm2@bu.edu>       		  #\n\
#								Boston University							  #\n\
#									   2013									  #\n\
###############################################################################\n\
$MCX-AOI $Rev:: 2 $ Last Commit $Date:: 2014-03-21 $ by $Author:: adamsm2$\n\
###############################################################################\n\
\n\
usage: %s <param1> <param2> ...\n\
where possible parameters include (the first item in [] is the default value)\n\
 -i 	       (--interactive) interactive mode\n\
 -s sessionid  (--session)     a string to label all output file names\n\
 -f config     (--input)       read config from a file\n\
 -n [0|int]    (--photon)      total photon number (exponential form accepted)\n\
 -t [2048|int] (--thread)      total thread number\n\
 -T [64|int]   (--blocksize)   thread number per block\n\
 -A [0|int]    (--autopilot)   auto thread config:1 dedicated GPU;2 non-dedic.\n\
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto\n\
 -r [1|int]    (--repeat)      number of repetitions\n\
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array\n\
 -z [0|1]      (--srcfrom0)    1 volume coord. origin [0 0 0]; 0 use [1 1 1]\n\
 -g [1|int]    (--gategroup)   number of time gates per run\n\
 -b [1|0]      (--reflect)     1 to reflect photons at ext. boundary;0 to exit\n\
 -B [0|1]      (--reflectin)   1 to reflect photons at int. boundary; 0 do not\n\
 -R [0.|float] (--skipradius)  cached zone radius from source to use atomics\n\
 -u [1.|float] (--unitinmm)    defines the length unit for the grid edge\n\
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw\n\
 -d [1|0]      (--savedet)     1 to save photon info at detectors; 0 not save\n\
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save\n\
 -H [1000000]  (--maxdetphoton)max number of detected photons\n\
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save\n\
 -E [0|int]    (--seed)        set random-number-generator seed, -1 to generate\n\
 -h            (--help)        print this message\n\
 -l            (--log)         print messages to a log file instead\n\
 -L            (--listgpu)     print GPU information only\n\
 -I            (--printgpu)    print GPU information and run program\n\
 -v            (--version)     print MCX-AOI revision number\n\
example:\n\
       %s -A -n 1e7 -f input.inp -G 1 \n\
or\n\
       %s -t 2048 -T 64 -n 1e7 -f input.inp -s test -r 2 -g 10 -U 0 -b 1 -G 1\n\
or\n\
       %s -f input.json -P '{\"Shapes\":[{\"ZLayers\":[[1,10,1],[11,30,2],[31,60,3]]}]}'\n",
              exename,exename,exename,exename);
}
