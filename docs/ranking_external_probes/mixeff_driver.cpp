#include <cmath>
#include <iostream>
extern "C" double ranking_probe_mixeff();
int main(){double beta=ranking_probe_mixeff();std::cout<<"slope="<<beta<<'\n';return std::isfinite(beta)&&std::abs(beta-0.75)<0.02?0:1;}
