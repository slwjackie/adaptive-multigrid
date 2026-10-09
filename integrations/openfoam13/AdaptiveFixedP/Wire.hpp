// Standalone C++11 framing and strict response decoder; no OpenFOAM dependency.
#ifndef ADAPTIVE_FIXED_P_WIRE_HPP
#define ADAPTIVE_FIXED_P_WIRE_HPP
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace adaptiveFixedP {
inline std::string quote(const std::string& s) {
    static const char* hex="0123456789abcdef";
    std::string o="\"";
    for(unsigned char c:s) {
        if(c=='"'||c=='\\'){o+='\\';o+=char(c);}
        else if(c<32){o+="\\u00";o+=hex[c>>4];o+=hex[c&15];}
        else o+=char(c);
    }
    return o+'"';
}
struct Response {
    bool success=false;
    long cycles=0;
    double threshold=0,initial=0,final=0;
    std::string error;
    std::vector<double> x;
};
class Decoder {
    const std::string& s; std::size_t p=0;
    void ws(){while(p<s.size()&&(s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t'))++p;}
    [[noreturn]] void bad(){throw std::runtime_error("invalid JSON solver response");}
    void take(char c){ws();if(p>=s.size()||s[p++]!=c)bad();}
    bool at(char c){ws();return p<s.size()&&s[p]==c;}
    std::string str(){
        take('"');std::string o;
        while(p<s.size()){
            unsigned char c=s[p++]; if(c=='"')return o; if(c<32)bad();
            if(c=='\\'){
                if(p>=s.size())bad();
                c=s[p++];
                if(c=='"'||c=='\\'||c=='/')o+=char(c);
                else if(c=='n')o+='\n';else if(c=='r')o+='\r';else if(c=='t')o+='\t';
                else if(c=='b')o+='\b';else if(c=='f')o+='\f';
                else if(c=='u'){
                    unsigned n=0;
                    for(int i=0;i<4;++i){if(p>=s.size())bad();char h=s[p++];n<<=4;
                        if(h>='0'&&h<='9')n+=h-'0';else if(h>='a'&&h<='f')n+=h-'a'+10;
                        else if(h>='A'&&h<='F')n+=h-'A'+10;else bad();}
                    if(n<128)o+=char(n);else o+='?';
                }else bad();
            }else o+=char(c);
        }bad();
    }
    double number(){
        ws();std::size_t start=p;
        if(p<s.size()&&s[p]=='-')++p;
        if(p>=s.size())bad();
        if(s[p]=='0')++p;
        else{if(s[p]<'1'||s[p]>'9')bad();while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;}
        if(p<s.size()&&s[p]=='.'){++p;std::size_t q=p;while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;if(q==p)bad();}
        if(p<s.size()&&(s[p]=='e'||s[p]=='E')){++p;if(p<s.size()&&(s[p]=='+'||s[p]=='-'))++p;
            std::size_t q=p;while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;if(q==p)bad();}
        std::string v=s.substr(start,p-start);char* end=nullptr;errno=0;double d=std::strtod(v.c_str(),&end);
        if(!std::isfinite(d)||errno==ERANGE||*end)bad();
        return d;
    }
    bool boolean(){ws();if(s.compare(p,4,"true")==0){p+=4;return true;}if(s.compare(p,5,"false")==0){p+=5;return false;}bad();}
    void skip(unsigned depth=0){
        if(depth>16)bad();
        ws();if(p>=s.size())bad();
        if(at('"')){str();return;}
        if(at('{')){take('{');if(!at('}'))while(true){str();take(':');skip(depth+1);if(!at(','))break;take(',');}take('}');return;}
        if(at('[')){take('[');if(!at(']'))while(true){skip(depth+1);if(!at(','))break;take(',');}take(']');return;}
        if(s.compare(p,4,"null")==0){p+=4;return;}
        if(s[p]=='t'||s[p]=='f'){boolean();return;}number();
    }
public:
    explicit Decoder(const std::string& input):s(input){}
    Response parse(std::size_t n){
        Response r;std::set<std::string> seen;take('{');
        if(!at('}'))while(true){
            std::string k=str();if(!seen.insert(k).second)bad();take(':');
            if(k=="success")r.success=boolean();
            else if(k=="cycles"){double v=number();if(v<0||v>2147483647||v!=std::floor(v))bad();r.cycles=long(v);}
            else if(k=="threshold")r.threshold=number();
            else if(k=="initial_residual")r.initial=number();
            else if(k=="final_residual")r.final=number();
            else if(k=="error"){if(at('"'))r.error=str();else{ws();if(s.compare(p,4,"null"))bad();p+=4;}}
            else if(k=="x_native"){
                take('[');if(!at(']'))while(true){if(r.x.size()>=n)bad();r.x.push_back(number());if(!at(','))break;take(',');}take(']');
            }else skip();
            if(!at(','))break;
            take(',');
        }
        take('}');ws();if(p!=s.size()||!seen.count("success"))bad();
        if(r.success && (r.x.size()!=n || !seen.count("cycles") || !seen.count("threshold")
            || !seen.count("initial_residual") || !seen.count("final_residual")
            || r.threshold<0||r.initial<0||r.final<0))bad();
        return r;
    }
};
class Connection {
    int fd=-1;
    static void transfer(int socket,void* buffer,std::size_t size,bool write){
        char* p=static_cast<char*>(buffer);
        while(size){ssize_t n=write ? ::send(socket,p,size,MSG_NOSIGNAL) : ::recv(socket,p,size,0);
            if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error(write?"solver socket write failed":"solver socket read failed");
            p+=n;size-=std::size_t(n);}
    }
public:
    explicit Connection(const std::string& path,int timeoutSeconds){
        sockaddr_un addr;std::memset(&addr,0,sizeof(addr));addr.sun_family=AF_UNIX;
        if(path.empty()||path.size()>=sizeof(addr.sun_path))throw std::runtime_error("invalid Unix socket path");
        std::memcpy(addr.sun_path,path.c_str(),path.size()+1);
        fd=::socket(AF_UNIX,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("cannot create solver socket");
        timeval timeout={timeoutSeconds,0};
        if(timeoutSeconds<=0||::setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout))
            ||::setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout))
            ||::connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))){::close(fd);fd=-1;throw std::runtime_error("cannot connect to solver service or set timeout");}
    }
    Connection(const Connection&)=delete;Connection& operator=(const Connection&)=delete;
    ~Connection(){if(fd>=0)::close(fd);}
    std::string request(const std::string& body){
        const std::uint64_t maxFrame=128ULL*1024*1024;
        if(body.empty()||body.size()>maxFrame)throw std::runtime_error("request frame too large");
        unsigned char head[8];std::uint64_t n=body.size();for(int i=7;i>=0;--i){head[i]=n&255;n>>=8;}
        transfer(fd,head,8,true);transfer(fd,const_cast<char*>(body.data()),body.size(),true);
        transfer(fd,head,8,false);n=0;for(int i=0;i<8;++i)n=(n<<8)|head[i];
        if(!n||n>maxFrame)throw std::runtime_error("invalid response frame length");
        std::string out(std::size_t(n),'\0');transfer(fd,&out[0],out.size(),false);return out;
    }
};
}
#endif
