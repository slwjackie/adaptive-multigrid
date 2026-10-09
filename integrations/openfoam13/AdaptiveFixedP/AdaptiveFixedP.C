// SPDX-License-Identifier: GPL-3.0-or-later
// Target: OpenFOAM Foundation 13 (not the OpenCFD version-number series).
#include "AdaptiveFixedP.H"
#include "fvMesh.H"
#include "volFields.H"
#include "Time.H"
#include "Wire.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>

namespace Foam {
defineTypeNameAndDebug(AdaptiveFixedP,0);
lduMatrix::solver::addsymMatrixConstructorToTable<AdaptiveFixedP>
    addAdaptiveFixedPSymMatrixConstructorToTable_;
// Register the asymmetric path solely to issue our explicit admission error.
lduMatrix::solver::addasymMatrixConstructorToTable<AdaptiveFixedP>
    addAdaptiveFixedPAsymMatrixConstructorToTable_;
}
namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
struct Session {
    std::unique_ptr<adaptiveFixedP::Connection> connection;
    std::uint64_t index=0;
    std::string mesh,boundary;
    std::vector<Foam::label> permutation;
};
std::map<std::string,Session>& sessions(){static std::map<std::string,Session> s;return s;}
std::string digest(const std::string& s){
    std::uint64_t h=14695981039346656037ULL;
    for(unsigned char c:s){h^=c;h*=1099511628211ULL;}
    std::ostringstream o;o<<"fnv1a64:"<<std::hex<<h;return o.str();
}
template<class Values> void array(std::ostream& out,const Values& v){
    out<<'[';for(std::size_t i=0;i<std::size_t(v.size());++i){if(i)out<<',';out<<v[i];}out<<']';
}
double norm(const Foam::scalarField& v){double n=0;for(Foam::label i=0;i<v.size();++i)n=std::hypot(n,double(v[i]));return n;}
bool nested(Foam::label n){return n>=3 && ((n+1)&n)==0;}
std::vector<double> coordinates(const Foam::vectorField& c,int dim,double eps){
    std::vector<double> a;for(Foam::label i=0;i<c.size();++i)a.push_back(c[i][dim]);
    std::sort(a.begin(),a.end());std::vector<double> u;
    for(double v:a)if(u.empty()||std::abs(v-u.back())>eps)u.push_back(v);
    return u;
}
std::vector<Foam::label> mapMesh(const Foam::fvMesh& mesh,Foam::label nx,Foam::label ny){
    const Foam::vectorField& c=mesh.C().primitiveField();
    if(!nested(nx)||!nested(ny)||nx*ny!=c.size()||mesh.dynamic())
        throw std::runtime_error("requires a static 2D grid with each dimension 2**L-1 >= 3");
    double scale=0;for(Foam::label i=0;i<c.size();++i)for(int d=0;d<3;++d)scale=std::max(scale,std::abs(double(c[i][d])));
    const double eps=std::max(1e-12,scale*1e-10);
    auto x=coordinates(c,0,eps),y=coordinates(c,1,eps),z=coordinates(c,2,eps);
    if(x.size()!=std::size_t(nx)||y.size()!=std::size_t(ny)||z.size()!=1)
        throw std::runtime_error("mesh centres are not one Cartesian x-y layer");
    for(const auto* a:{&x,&y})for(std::size_t i=2;i<a->size();++i)
        if(std::abs(((*a)[i]-(*a)[i-1])-((*a)[1]-(*a)[0]))>eps)
            throw std::runtime_error("this bridge requires uniform Cartesian cell spacing");
    std::vector<Foam::label> order(std::size_t(nx*ny),-1),inverse(order.size(),-1);
    for(Foam::label native=0;native<c.size();++native){
        long ix=std::lround((c[native].x()-x[0])/(x[1]-x[0]));
        long iy=std::lround((c[native].y()-y[0])/(y[1]-y[0]));
        if(ix<0||ix>=nx||iy<0||iy>=ny||order[ix*ny+iy]>=0
            ||std::abs(c[native].x()-x[ix])>eps||std::abs(c[native].y()-y[iy])>eps)
            throw std::runtime_error("ambiguous structured-to-native cell map");
        order[ix*ny+iy]=native;inverse[native]=ix*ny+iy;
    }
    const auto& lo=mesh.lduAddr().lowerAddr();const auto& up=mesh.lduAddr().upperAddr();
    if(lo.size()!=(nx-1)*ny+nx*(ny-1))throw std::runtime_error("non-Cartesian internal face count");
    for(Foam::label f=0;f<lo.size();++f){auto i=inverse[lo[f]],j=inverse[up[f]];
        if(std::abs(i/ny-j/ny)+std::abs(i%ny-j%ny)!=1)
            throw std::runtime_error("matrix addressing is inconsistent with Cartesian cell neighbours");}
    return order;
}
}

Foam::AdaptiveFixedP::AdaptiveFixedP(const word& fieldName,const lduMatrix& matrix,
    const FieldField<Field,scalar>& bou,const FieldField<Field,scalar>& in,
    const lduInterfaceFieldPtrsList& interfaces,const dictionary& controls)
 : lduMatrix::solver(fieldName,matrix,bou,in,interfaces,controls) {}

Foam::solverPerformance Foam::AdaptiveFixedP::solve(scalarField& psi,
    const scalarField& source,const direction cmpt) const {
    const auto started=Clock::now();
    try {
        if(sizeof(scalar)!=sizeof(double))throw std::runtime_error("double-precision OpenFOAM build required");
        if(Pstream::parRun()||cmpt!=0||fieldName_!="p")
            throw std::runtime_error("AdaptiveFixedP supports only serial scalar pressure p");
        forAll(interfaces_,i)if(interfaces_.set(i))
            throw std::runtime_error("processor/cyclic/coupled interfaces are unsupported");
        if(!matrix_.symmetric())throw std::runtime_error("pressure matrix must use symmetric LDU storage");
        const fvMesh& mesh=refCast<const fvMesh>(matrix_.mesh().thisDb());
        const label nx=controlDict_.lookup<label>("nx"),ny=controlDict_.lookup<label>("ny");
        const word mode=controlDict_.lookupOrDefault<word>("mode","classical");
        if(mode!="classical"&&mode!="hs"&&mode!="native")throw std::runtime_error("invalid AdaptiveFixedP mode");
        const fileName path(controlDict_.lookup("socketPath"));
        const scalar atol=controlDict_.lookupOrDefault<scalar>("adaptiveAtol",1e-12);
        const scalar rtol=controlDict_.lookupOrDefault<scalar>("adaptiveRtol",1e-8);
        const label timeout=controlDict_.lookupOrDefault<label>("socketTimeout",120);
        if(!std::isfinite(atol)||!std::isfinite(rtol)||atol<=0||rtol<=0||rtol>=1||maxIter_<=0)
            throw std::runtime_error("invalid raw L2 stopping configuration");
        auto order=mapMesh(mesh,nx,ny);
        std::ostringstream geometry,boundaries;geometry<<std::setprecision(17);boundaries<<std::setprecision(17);
        const auto& centres=mesh.C().primitiveField();
        forAll(centres,i)geometry<<centres[i].x()<<','<<centres[i].y()<<','<<centres[i].z()<<';';
        forAll(mesh.points(),i)geometry<<mesh.points()[i].x()<<','<<mesh.points()[i].y()<<','<<mesh.points()[i].z()<<';';
        array(geometry,matrix_.lduAddr().lowerAddr());array(geometry,matrix_.lduAddr().upperAddr());
        const volScalarField& p=mesh.lookupObject<volScalarField>(fieldName_);
        forAll(mesh.boundary(),i){
            if(mesh.boundary()[i].coupled())throw std::runtime_error("coupled boundary unsupported");
            boundaries<<mesh.boundary()[i].name()<<':'<<mesh.boundary()[i].type()<<':'<<p.boundaryField()[i].type()<<';';
            array(boundaries,mesh.boundary()[i].faceCells());
        }
        const auto meshId=digest(geometry.str()),boundaryId=digest(boundaries.str());
        Session& state=sessions()[path.c_str()];
        if(state.connection&&(state.mesh!=meshId||state.boundary!=boundaryId||state.permutation!=order))
            throw std::runtime_error("mesh/boundary changed in a fixed-P case; start a new service/case");
        if(!state.connection){state.connection.reset(new adaptiveFixedP::Connection(path.c_str(),timeout));
            state.mesh=meshId;state.boundary=boundaryId;state.permutation=order;}
        const scalarField x0(psi);scalarField ax(psi.size());
        matrix_.Amul(ax,x0,interfaceBouCoeffs_,interfaces_,cmpt);
        const double initial=norm(scalarField(source-ax));
        // Native edge accumulation and CSR summation use different orders.
        // Diagnostic comparison is cancellation-aware; acceptance below still
        // uses the exact native target and a fresh native Amul.
        const double residualRoundoff=128*std::numeric_limits<double>::epsilon()
            *(norm(source)+norm(ax));
        const double threshold=std::max(double(atol),double(rtol)*initial);
        scalarField v0(psi.size()),v1(psi.size()),av0(psi.size()),av1(psi.size());
        forAll(v0,i){v0[i]=std::sin(0.731*(i+1));v1[i]=std::cos(0.417*(i+1));}
        matrix_.Amul(av0,v0,interfaceBouCoeffs_,interfaces_,cmpt);
        matrix_.Amul(av1,v1,interfaceBouCoeffs_,interfaces_,cmpt);
        label nativeCycles=0;double nativeSeconds=0,nativeFinal=initial;
        if(mode=="native"){
            const auto nativeStart=Clock::now();
            if(initial>threshold){
                scalarField tmp(psi.size());const scalar normalization=normFactor(x0,source,ax,tmp);
                dictionary controls;controls.add("solver",word("PCG"));controls.add("preconditioner",word("DIC"));
                // PCG uses normalized L1. This conservative target implies the raw L2 target.
                controls.add("tolerance",scalar(threshold/std::max(double(normalization),1e-300)));
                controls.add("relTol",scalar(0));controls.add("maxIter",maxIter_);
                auto native=lduMatrix::solver::New(fieldName_,matrix_,interfaceBouCoeffs_,interfaceIntCoeffs_,interfaces_,controls);
                nativeCycles=native->solve(psi,source,cmpt).nIterations();
            }
            matrix_.Amul(ax,psi,interfaceBouCoeffs_,interfaces_,cmpt);nativeFinal=norm(scalarField(source-ax));
            nativeSeconds=seconds(nativeStart);
            if(!std::isfinite(nativeFinal)||nativeFinal>threshold)
                throw std::runtime_error("native PCG reference did not meet common raw L2 threshold");
        }
        std::ostringstream out;out<<std::setprecision(17)
            <<"{\"schema\":\"h2-fixed-p-live-v1\",\"op\":"<<adaptiveFixedP::quote(mode=="native"?"record":"solve")
            <<",\"mode\":"<<adaptiveFixedP::quote(mode.c_str())
            <<",\"producer\":\"OpenFOAM Foundation 13 AdaptiveFixedP\",\"source_kind\":\"external_cfd\""
            <<",\"boundary_finalized\":true,\"nullspace\":\"none\",\"coupled_interfaces\":0"
            <<",\"shape\":["<<nx<<','<<ny<<"],\"time\":"<<mesh.time().value()<<",\"index\":"<<state.index
            <<",\"mesh_id\":"<<adaptiveFixedP::quote(meshId)<<",\"boundary_id\":"<<adaptiveFixedP::quote(boundaryId)
            <<",\"atol\":"<<atol<<",\"rtol\":"<<rtol<<",\"max_cycles\":"<<maxIter_
            <<",\"native_initial_residual\":"<<initial<<",\"native_threshold\":"<<threshold;
        out<<",\"diag\":";array(out,matrix_.diag());out<<",\"lower\":";array(out,matrix_.lower());
        out<<",\"upper\":";array(out,matrix_.upper());out<<",\"lower_addr\":";array(out,matrix_.lduAddr().lowerAddr());
        out<<",\"upper_addr\":";array(out,matrix_.lduAddr().upperAddr());out<<",\"b\":";array(out,source);
        out<<",\"x0\":";array(out,x0);out<<",\"structured_to_native\":";array(out,order);
        out<<",\"probe_vectors\":[";forAll(v0,i){if(i)out<<',';out<<'['<<v0[i]<<','<<v1[i]<<']';}out<<']';
        out<<",\"probe_products\":[";forAll(v0,i){if(i)out<<',';out<<'['<<av0[i]<<','<<av1[i]<<']';}out<<']';
        out<<",\"context\":{\"native_time_index\":"<<mesh.time().timeIndex()<<'}';
        if(mode=="native"){
            out<<",\"x_native_reference\":";array(out,psi);
            out<<",\"reference_solver\":\"PCG/DIC\",\"reference_seconds\":"<<nativeSeconds
                <<",\"reference_cycles\":"<<nativeCycles<<",\"reference_initial_residual\":"<<initial
                <<",\"reference_final_residual\":"<<nativeFinal;
        }
        out<<'}';
        const auto response=adaptiveFixedP::Decoder(state.connection->request(out.str())).parse(psi.size());
        if(!response.success)throw std::runtime_error("Python pressure solve rejected: "+response.error);
        if(std::abs(response.threshold-threshold)>rtol*residualRoundoff+1e-12*std::max(threshold,1e-300)
            ||std::abs(response.initial-initial)>residualRoundoff+1e-10*std::max(initial,1e-300))
            throw std::runtime_error("Python/native stopping threshold or initial residual disagree");
        scalarField candidate(psi.size());forAll(candidate,i)candidate[i]=response.x[i];
        if(mode=="native")forAll(candidate,i)if(candidate[i]!=psi[i])
            throw std::runtime_error("record acknowledgement changed native pressure solution");
        matrix_.Amul(ax,candidate,interfaceBouCoeffs_,interfaces_,cmpt);
        const double final=norm(scalarField(source-ax));
        if(!std::isfinite(final)||final>threshold)
            throw std::runtime_error("returned pressure fails OpenFOAM native raw L2 residual check");
        psi=candidate;++state.index;
        solverPerformance performance(typeName,fieldName_,scalar(initial),scalar(final),
            label(response.cycles),true,false);
        Ostream& timingLog=Info();
        const int previousPrecision=timingLog.precision(17);
        timingLog<<"H2FixedPPressureSeconds "<<seconds(started)<<" index "<<(state.index-1)
            <<" time "<<mesh.time().value()<<" mode "<<mode<<" residual "<<final
            <<" threshold "<<threshold<<nl;
        timingLog.precision(previousPrecision);
        return performance;
    }catch(const std::exception& e){
        FatalErrorInFunction<<"AdaptiveFixedP: "<<e.what()<<exit(FatalError);
    }
    return solverPerformance(typeName,fieldName_);
}
