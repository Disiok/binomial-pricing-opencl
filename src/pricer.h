#ifndef __PRICER_H__
#define __PRICER_H__
#include "option_spec.h"

class OptionPricer {
public:
    virtual ~OptionPricer() {}
    virtual double price(OptionSpec& optionSpec) = 0;
};

class LatticePricer: public OptionPricer {
};

class SerialPricer: public LatticePricer {
public:
    virtual double price(OptionSpec& optionSpec);
};

class OpenCLPricer: public LatticePricer {
public:
    OpenCLPricer();
    virtual double price(OptionSpec& optionSpec);
};
#endif
