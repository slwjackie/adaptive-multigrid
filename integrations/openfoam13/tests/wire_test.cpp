#include "../AdaptiveFixedP/Wire.hpp"
#include <cassert>
#include <iostream>

bool rejects(const std::string& s){try{adaptiveFixedP::Decoder(s).parse(2);return false;}catch(const std::exception&){return true;}}
int main(int argc,char** argv){
    const std::string valid="{\"success\":true,\"x_native\":[1,-2.5e-1],\"cycles\":3,\"threshold\":1e-9,\"initial_residual\":1,\"final_residual\":1e-10,\"error\":null}";
    auto r=adaptiveFixedP::Decoder(valid).parse(2);assert(r.success&&r.x[1]==-0.25&&r.cycles==3);
    assert(rejects("{\"success\":true}"));
    assert(rejects(valid+"junk"));
    assert(rejects("{\"success\":false,\"success\":true}"));
    assert(rejects("{\"success\":false,\"cycles\":1.5}"));
    assert(rejects("{\"success\":false,\"threshold\":NaN}"));
    assert(rejects("{\"success\":false,\"x_native\":[1,2,3]}"));
    assert(rejects("{\"success\":false,\"x_native\":[1,Infinity]}"));
    assert(!adaptiveFixedP::Decoder("{\"success\":false,\"error\":\"not ready\\nretry\"}").parse(2).success);
    assert(adaptiveFixedP::quote("x\n\"y") == "\"x\\u000a\\\"y\"");
    if(argc>1){
        adaptiveFixedP::Connection c(argv[1],5);
        for(int i=0;i<2;++i){auto v=adaptiveFixedP::Decoder(c.request("{\"n\":2}")).parse(2);assert(v.success&&v.x[0]==1);}
    }
    std::cout<<"wire checks passed\n";
}
