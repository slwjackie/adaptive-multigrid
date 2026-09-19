// FP64 cached, spatially varying 9-point stencil. No fast-math, no hidden scaling.
// Coefficients are [basis,9,nx,ny]; output is [basis,nx,ny].
// OFFSETS_9: center, -x, +x, -y, +y, (-x,-y), (-x,+y), (+x,-y), (+x,+y).
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
extern "C" {
EXPORT int anmg_stencil_abi() { return 1; }
EXPORT int anmg_stencil_openmp() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
EXPORT int anmg_stencil9(const double* c, const double* r, const double* gate,
                        double* out, int nx, int ny, int q,
                        int threads, int parallel_min) {
    if (!c || !r || !out || nx < 1 || ny < 1 || q < 1 || threads < 1) return -1;
    const std::ptrdiff_t N = std::ptrdiff_t(nx)*ny;
    // Parallel launch is deliberately avoided on tiny coarse grids.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads) if(N >= parallel_min && threads > 1)
    for (int b=0; b<q; ++b) {
        for (int i=0; i<nx; ++i) {
            const double* cb = c + std::ptrdiff_t(b)*9*N;
            double* dst = out + std::ptrdiff_t(b)*N;
            // Interior contiguous loop has no indirect sparse-column indexing.
            if (i>0 && i<nx-1) {
                #pragma omp simd
                for (int j=1; j<ny-1; ++j) {
                    const std::ptrdiff_t k=std::ptrdiff_t(i)*ny+j;
                    double v=cb[k]*r[k];
                    v += cb[N+k]*r[k-ny]; v += cb[2*N+k]*r[k+ny];
                    v += cb[3*N+k]*r[k-1]; v += cb[4*N+k]*r[k+1];
                    v += cb[5*N+k]*r[k-ny-1]; v += cb[6*N+k]*r[k-ny+1];
                    v += cb[7*N+k]*r[k+ny-1]; v += cb[8*N+k]*r[k+ny+1];
                    dst[k] = gate ? v*gate[k] : v;
                }
            }
            // Explicit zero-exterior boundary, including nx/ny=1 or 2.
            for (int j=0; j<ny; ++j) {
                if (i>0 && i<nx-1 && j>0 && j<ny-1) continue;
                const std::ptrdiff_t k=std::ptrdiff_t(i)*ny+j;
                double v=cb[k]*r[k];
                if(i>0) v+=cb[N+k]*r[k-ny];
                if(i+1<nx) v+=cb[2*N+k]*r[k+ny];
                if(j>0) v+=cb[3*N+k]*r[k-1];
                if(j+1<ny) v+=cb[4*N+k]*r[k+1];
                if(i>0 && j>0) v+=cb[5*N+k]*r[k-ny-1];
                if(i>0 && j+1<ny) v+=cb[6*N+k]*r[k-ny+1];
                if(i+1<nx && j>0) v+=cb[7*N+k]*r[k+ny-1];
                if(i+1<nx && j+1<ny) v+=cb[8*N+k]*r[k+ny+1];
                dst[k] = gate ? v*gate[k] : v;
            }
        }
    }
    return 0;
}
}

// v6.7: selected-row application skips unselected work (unlike output masking).
#include <cstdint>
#include <algorithm>
#include <cmath>
extern "C" {
EXPORT int anmg_stencil9_rows(const double* c, const double* r,
        const std::int64_t* rows, double* out, std::int64_t count,
        int nx, int ny, int threads, int parallel_min) {
    if (!c || !r || !rows || !out || count<0 || nx<1 || ny<1) return -1;
    const std::int64_t N=std::int64_t(nx)*ny;
    const int di[9]={0,-1,1,0,0,-1,-1,1,1}, dj[9]={0,0,0,-1,1,-1,1,-1,1};
    for(std::int64_t z=0;z<count;++z) if(rows[z]<0 || rows[z]>=N) return -2;
    #pragma omp parallel for schedule(static) num_threads(threads) if(count>=parallel_min && threads>1)
    for(std::int64_t z=0;z<count;++z) {
        const auto k=rows[z]; const int i=int(k/ny),j=int(k%ny);
        double value=0.;
        for(int t=0;t<9;++t) if(i+di[t]>=0 && i+di[t]<nx && j+dj[t]>=0 && j+dj[t]<ny)
            value+=c[std::int64_t(t)*N+k]*r[k+std::int64_t(di[t])*ny+dj[t]];
        out[z]=value;
    }
    return 0;
}
// Fine-row P scatter equals R*r for R=P^T. Statistics and restriction share
// this fine-grid pass. Serial accumulation is intentional and deterministic.
EXPORT int anmg_restrict_features(const std::int64_t* ip,const std::int64_t* col,
        const double* val,const double* r,double* coarse,double* features,
        int nx,int ny,int nc,int bx,int by) {
    if(!r || !features || nx<1 || ny<1 || bx<1 || by<1) return -1;
    if(coarse) std::fill(coarse,coarse+nc,0.);
    std::fill(features,features+std::int64_t(bx)*by*6,0.);
    for(int i=0;i<nx;++i) for(int j=0;j<ny;++j) {
        const std::int64_t k=std::int64_t(i)*ny+j;
        const int block=(i*bx/nx)*by+j*by/ny;
        double* f=features+std::int64_t(block)*6;
        const double v=r[k];
        f[0]+=v*v; f[1]+=std::abs(v); f[2]=std::max(f[2],std::abs(v));
        if(i+1<nx) {const double d=v-r[k+ny];f[3]+=d*d;}
        if(j+1<ny) {const double d=v-r[k+1];f[4]+=d*d;}
        f[5]+=1.;
        if(coarse) for(std::int64_t t=ip[k];t<ip[k+1];++t) coarse[col[t]]+=val[t]*v;
    }
    return 0;
}
}
