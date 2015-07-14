#ifndef __PRICER_H__
#define __PRICER_H__
// System Libraries
#include <vector>

// OpenCL C++ Binding
#include "cl.hpp"

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
    virtual ~OpenCLPricer();
    virtual double price(OptionSpec& optionSpec);
private:
    double priceImplGroup(OptionSpec& optionSpec, int groupSize);
    double priceImplTriangle(OptionSpec& optionSpec, int stepSize);

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
