 
 
 
 
 
 
 
 
 
 






















 


#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "snic.h"


void push (HEAP *h, const unsigned int ind, const unsigned int klab, const double dist)
{
    if (h->len + 1 >= h->size)
    {
        h->size = h->size ? h->size * 2 : 4;  
        h->nodes = (NODE *)realloc(h->nodes, h->size * sizeof (NODE));
    }
    int i = h->len + 1;
    int j = i / 2;
    while (i > 1 && h->nodes[j].d > dist)
    {
        h->nodes[i] = h->nodes[j];
        i = j;
        j = j / 2;
    }
    h->nodes[i].i = ind;
    h->nodes[i].k = klab;
    h->nodes[i].d = dist;
    h->len++;
}
 
void pop(HEAP* h, unsigned int* ind, unsigned int* klab, double* dist)
{
    if(h->len > 1)
    {
        int i, j, k;
         
        *ind  = h->nodes[1].i;
        *klab = h->nodes[1].k;
        *dist = h->nodes[1].d;
     
        h->nodes[1] = h->nodes[h->len];
     
        h->len--;
     
        i = 1;
        while (i!=h->len+1)
        {
            k = h->len+1;
            j = 2 * i;
            if (j <= h->len && h->nodes[j].d < h->nodes[k].d)
            {
                k = j;
            }
            if (j + 1 <= h->len && h->nodes[j + 1].d < h->nodes[k].d)
            {
                k = j + 1;
            }
            h->nodes[i] = h->nodes[k];
            i = k;
        }
    }
}

void rgbtolab(double* rin, double* gin, double* bin, double sz, double* lvec, double* avec, double* bvec)
{
    int i;
    double sR, sG, sB;
    double R,G,B;
    double X,Y,Z;
    double r, g, b;
    const double epsilon = 0.008856;	 
    const double kappa   = 903.3;		 
    
    const double Xr = 0.950456;	 
    const double Yr = 1.0;		 
    const double Zr = 1.088754;	 
    double xr,yr,zr;
    double fx, fy, fz;
    double lval,aval,bval;
    
    for(i = 0; i < sz; i++)
    {
        sR = rin[i]; sG = gin[i]; sB = bin[i];
        R = sR/255.0;
        G = sG/255.0;
        B = sB/255.0;
        
        if(R <= 0.04045)	r = R/12.92;
        else				r = pow((R+0.055)/1.055,2.4);
        if(G <= 0.04045)	g = G/12.92;
        else				g = pow((G+0.055)/1.055,2.4);
        if(B <= 0.04045)	b = B/12.92;
        else				b = pow((B+0.055)/1.055,2.4);
        
        X = r*0.4124564 + g*0.3575761 + b*0.1804375;
        Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        
         
         
         
        xr = X/Xr;
        yr = Y/Yr;
        zr = Z/Zr;
        
        if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
        else				fx = (kappa*xr + 16.0)/116.0;
        if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
        else				fy = (kappa*yr + 16.0)/116.0;
        if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
        else				fz = (kappa*zr + 16.0)/116.0;
        
        lval = 116.0*fy-16.0;
        aval = 500.0*(fx-fy);
        bval = 200.0*(fy-fz);
        
        lvec[i] = lval; avec[i] = aval; bvec[i] = bval;
    }
}

 
 
 
 
 
 
 
void FindSeeds(const int width, const int height, const int numk, int* kx, int* ky, int* outnumk)
{
    const int sz = width*height;
    double gridstep = sqrt((double)(sz)/(double)(numk)) + 0.5;
     
     
    if(1)
    {
        int minerr = 999999;
        double minstep = gridstep-1.0; double maxstep = gridstep+1.0;
        for(double x = minstep; x <= maxstep; x += 0.1)
        {
            int err = abs( (int)(0.5 + width/x)*(int)(0.5 + height/x) - numk);
            if(err < minerr)
            {
                minerr = err; gridstep = x;
            }
        }
    }

    double halfstep = gridstep/2.0;

    int n = 0;
    for(double y = halfstep; y <= height; y += gridstep)
    {
        int yval = (int)(y+0.5);
        if(yval < height)
        {
            for(double x = halfstep; x <= width; x += gridstep)
            {
                int xval = (int)(x+0.5);
                if(xval < width)
                {
                    kx[n] = xval;
                    ky[n] = yval;
                    n++;
                }
            }
        }
    }
    *outnumk = n;
}

 
 
 
 
 
void runSNIC(
            double**                      chans,
            const int                   nchans,
             const int					width,
             const int					height,
             int*                       labels,
             int*						outnumk,
             const int                  innumk,
             const double               compactness)
{
    const int w = width;
    const int h = height;
    const int sz = w*h;
    const int dx8[8] = {-1,  0, 1, 0, -1,  1, 1, -1}; 
    const int dy8[8] = { 0, -1, 0, 1, -1, -1, 1,  1}; 
    const int dn8[8] = {-1, -w, 1, w, -1-w,1-w,1+w,-1+w};
    
    int* cx = (int*)malloc( sizeof(int)     * (int)(innumk * 1.1 + 10) );
    int* cy = (int*)malloc( sizeof(int)     * (int)(innumk * 1.1 + 10) );

    int numk = 0; 
    FindSeeds(width,height,innumk,cx,cy,&numk);
     
     
     
    HEAP* heap = (HEAP *)calloc(1, sizeof (HEAP));
    HEAP* heap2 = (HEAP *)calloc(1, sizeof (HEAP));
    heap->nodes = (NODE*)calloc(sz,sizeof(NODE)); heap->len = 0; heap->size = sz;  

     
    for(int i = 0; i < sz; i++) labels[i] = -1;

    for(int k = 0; k < numk; k++)
    {
        push(heap,(cx[k] << 16 | cy[k]),k,0);
    }
    
    double** kc = (double**)malloc(sizeof(double*)*nchans);
    for(int c = 0; c < nchans; c++)
    {
        kc[c] = (double*)calloc(numk,sizeof(double));
    }
    double* kx = (double*)calloc(numk,sizeof(double));
    double* ky = (double*)calloc(numk,sizeof(double));
    double* ksize = (double*)calloc(numk,sizeof(double));
    NODE* pnode = (NODE*)calloc(1,sizeof(NODE));

    const int CONNECTIVITY = 4; 
    const double M = compactness; 
    const double invwt = (M*M*numk)/(double)(sz);
    
    int pixelcount = 0;
    int xx = 0, yy = 0, ii = 0;
    double cdiff = 0,xdiff = 0,ydiff = 0,colordist = 0,xydist = 0,slicdist = 0;
     
     
     
    int loopcount = 0;
    unsigned int ind;
    unsigned int klab;
    double dist;
    while(pixelcount < sz)
    {

        pop(heap,&ind,&klab,&dist);
        const int k = klab;
        const int x = ind >> 16 & 0xffff;
        const int y = ind & 0xffff;
        const int i = y*width+x;
        
        if(labels[i] < 0)
        {
            labels[i] = k; pixelcount++;

            for(int c = 0; c < nchans; c++)
            {
                kc[c][k] += chans[c][i];
            }
            kx[k] += x;
            ky[k] += y;
            ksize[k] += 1.0;
            
            for(int p = 0; p < CONNECTIVITY; p++)
            {
                xx = x + dx8[p];
                yy = y + dy8[p];
                if(!(xx < 0 || xx >= w || yy < 0 || yy >= h))
                {
                    ii = i + dn8[p];
                    if(labels[ii] < 0) 
                    {
                        colordist = 0;
                        for(int c = 0; c < nchans; c++)
                        {
                            cdiff = kc[c][k]-(chans[c][ii]*ksize[k]);
                            colordist += (cdiff*cdiff);
                        }
                        xdiff = kx[k] - xx*ksize[k];
                        ydiff = ky[k] - yy*ksize[k];
                        xydist      = xdiff*xdiff + ydiff*ydiff;

                        slicdist    = (colordist + xydist*invwt)/(ksize[k]*ksize[k]); 

                        push(heap,(xx << 16 | yy),k,slicdist);
                    }
                }
            }
        }
        loopcount++;if(loopcount > sz*10){printf("heap1 len: %d heap2 len: %d\n", heap->len, heap2->len); break;}
    }
    *outnumk = numk;
     
     
     
    if(labels[0] < 0) labels[0] = 0;
    for(int y = 1; y < height; y++)
    {
        for(int x = 1; x < width; x++)
        {
            int i = y*width+x;
            if(labels[i] < 0) 
            {
                if(labels[i-1] >= 0) labels[i] = labels[i-1];
                else if(labels[i-width] >= 0) labels[i] = labels[i-width];
            } 
        }
    }

    free(cx);
    free(cy);
    for(int c = 0; c < nchans; c++)
    {
        free(kc[c]);
    }
    free(kc);    
    free(kx);
    free(ky);
    free(ksize);
    free(pnode);
    if(heap->nodes) free(heap->nodes);
    if(heap)free(heap);
    if(heap2->nodes) free(heap2->nodes);
    if(heap2)free(heap2);

}

 
 
 
 
 
void SNIC_main(double* img, const int width, const int height,
                const int nchannels, const int numSuperpixels, const double compactness,
                const int doRGBtoLAB, int* klabels, int* numlabels)
{
    int sz = width*height;
    double** channels = (double**)malloc(sizeof(double*)*nchannels);
    for(int c = 0; c < nchannels; c++)
    {
        channels[c] = img + c*sz;
    }
     
     
     
    if(doRGBtoLAB && nchannels==3)
    {
        rgbtolab(channels[0],channels[1],channels[2],sz,channels[0],channels[1],channels[2]);
    }
     
     
     
    int numklabels = 0;
    runSNIC(channels,nchannels,width,height,klabels,&numklabels,numSuperpixels,compactness);
    
    *numlabels = numklabels;

    free(channels);
}

