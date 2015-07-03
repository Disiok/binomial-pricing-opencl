#ifndef __PRICER_H__
#define __PRICER_H__
#include "option_spec.h"
#include <vector>
#include <OpenCL/cl.hpp>

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
    virtual ~OpenCLPricer();
    virtual double price(OptionSpec& optionSpec);
private:
    double priceImplSync(OptionSpec& optionSpec, int stepSize);
    double priceImplGroup(OptionSpec& optionSpec, int stepSize);

    std::vector<cl::Platform>* platforms;
    cl::Platform* defaultPlatform;
    std::vector<cl::Device>* devices;
    cl::Device* defaultDevice;
    cl::Context* context;
    std::string* kernelCode;
    cl::Program::Sources* sources;
    cl::Program* program;
};
#endif
