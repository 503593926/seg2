 
 
 
 
 
 
 
 
 
 






















 


typedef struct
{
    unsigned int i;  
    unsigned int k;  
    double d;        
} NODE;

typedef struct
{
    NODE *nodes;
    int len;  
    int size;  
} HEAP;

void push (HEAP *h, const unsigned int ind, const unsigned int klab, const double dist);
 
void pop(HEAP* h, unsigned int* ind, unsigned int* klab, double* dist);

void rgbtolab(double* rin, double* gin, double* bin, double sz, double* lvec, double* avec, double* bvec);
void FindSeeds(const int width, const int height, const int numk, int* kx, int* ky, int* outnumk);
void runSNIC(
            double**                      chans,
            const int                   nchans,
             const int                  width,
             const int                  height,
             int*                       labels,
             int*                       outnumk,
             const int                  innumk,
             const double               compactness);
void SNIC_main(double* img, const int width, const int height,
                const int nchannels, const int numSuperpixels, const double compactness,
                const int doRGBtoLAB, int* klabels, int* numlabels);

