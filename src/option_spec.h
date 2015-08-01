#ifndef __OPTION_SPEC_H__
#define __OPTION_SPEC_H__
#include <iostream>
struct OptionSpec {
    /**
     * Type of option:
     *      1  -> Call option
     *      -1 -> Put option
     */ 
    int type;
    float stockPrice;
    float strikePrice;
    float yearsToMaturity;
    float volatility;
    float riskFreeRate;
    int numSteps;   
    //TODO(disiok): Implement for OpenCL pricers
    bool isAmerican;
};
std::ostream& operator<<(std::ostream& out, const OptionSpec& other);
#endif
