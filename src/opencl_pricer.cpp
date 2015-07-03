#include <OpenCL/cl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include "pricer.h"
#include "option_spec.h"

OpenCLPricer::OpenCLPricer() {
    // Retrieve platforms
    platforms = new std::vector<cl::Platform>();
    cl::Platform::get(platforms);

    // Check number of platforms found
    if (platforms->size() == 0) {
        std::cerr   << "[ERROR] No platform found. Check OpenCL installation!" 
                    << std::endl;
        exit(1);
    } else {
        std::cout   << "[INFO] " << platforms->size() << " platforms found." 
                    << std::endl;
    }

    // TODO(disiok): Add parameters to choose platforms
    // Select default platform
    defaultPlatform = &(*platforms)[0];
    std::cout   << "[INFO] Using platform: " 
                << defaultPlatform->getInfo<CL_PLATFORM_NAME>() 
                << std::endl;

    // Retrieve devices
    devices = new std::vector<cl::Device>();
    defaultPlatform->getDevices(CL_DEVICE_TYPE_ALL, devices);

    // Check number of devices found
    if (devices->size() == 0) {
        std::cerr   << "[ERROR] No devices found. Check OpenCL installation!" 
                    << std::endl;
        exit(2);
    } else {
        std::cout   << "[INFO] " << devices->size() << " devices found." 
                    << std::endl;
    }

    // TODO(disiok): Add parameters to choose devices
    // Select default device
    defaultDevice = &(*devices)[1];
    std::cout   << "[INFO] Using device: " 
                << defaultDevice->getInfo<CL_DEVICE_NAME>() 
                << " (Max work item sizes: "
                << defaultDevice->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0]
                << ", Max work group size: "
                << defaultDevice->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                << ", Max computing units: "
                << defaultDevice->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                << ")"
                << std::endl;

    // Create context
    context = new cl::Context({*defaultDevice});

    // Define kernel code
    std::ifstream ifs("kernel.cl");
    kernelCode = new std::string(
            (std::istreambuf_iterator<char>(ifs)),
            (std::istreambuf_iterator<char>()));
    sources = new cl::Program::Sources();
    sources->push_back({kernelCode->c_str(), kernelCode->length()});

    // Build kernel code
    program = new cl::Program(*context, *sources);
    if (program->build({*defaultDevice}) != CL_SUCCESS) {
        std::cerr   << "[ERROR] Error building: " 
                    << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*defaultDevice) 
                    << std::endl;
        exit(4);
    } else {
        std::cout   << "[INFO] Successfully built kernel program" 
                    << std::endl;
    }

}

double OpenCLPricer::price(OptionSpec& optionSpec) {
   return priceImplReduce(optionSpec); 
}

double OpenCLPricer::priceImplSolo(OptionSpec& optionSpec) {
    // ------------------------Derived Parameters------------------------------
    float deltaT = optionSpec.yearsToMaturity / optionSpec.numSteps;

    float upFactor = exp(optionSpec.volatility * sqrt(deltaT));
    float downFactor = 1.0f / upFactor;

    float discountFactor = exp(optionSpec.riskFreeRate * deltaT);

    float upWeight = (discountFactor - downFactor) / (upFactor - downFactor);
    float downWeight = 1.0f - upWeight;
    
    // Create buffers on the devices
    cl::Buffer valueBufferA(*context, 
                           CL_MEM_READ_WRITE,
                           sizeof(float) * (optionSpec.numSteps + 1));

    cl::Buffer valueBufferB(*context, 
                           CL_MEM_READ_WRITE,
                           sizeof(float) * (optionSpec.numSteps + 1));

    // Create qeueue to push commands for the devices
    cl::CommandQueue queue(*context, *defaultDevice);
    
    // Build and run init kernel 
    cl::Kernel kernel(*program, "init");
    kernel.setArg(0, optionSpec.stockPrice);
    kernel.setArg(1, optionSpec.strikePrice);
    kernel.setArg(2, optionSpec.numSteps);
    kernel.setArg(3, optionSpec.type);
    kernel.setArg(4, deltaT);
    kernel.setArg(5, upFactor);
    kernel.setArg(6, downFactor);
    kernel.setArg(7, valueBufferA);
    queue.enqueueNDRangeKernel(kernel, 
                              cl::NullRange, 
                              cl::NDRange(optionSpec.numSteps + 1), 
                              cl::NullRange);
    std::cout << "[INFO] Executing init kernel with " << optionSpec.numSteps + 1
            << " work items" << std::endl;

    // Block until init kernel finishes execution
    queue.finish();

    // Note(disiok): After each iteration, the number of nodes is reduced by 1 
    cl::Kernel iterateKernel(*program, "solo");
    iterateKernel.setArg(0, upWeight);
    iterateKernel.setArg(1, downWeight);
    iterateKernel.setArg(2, discountFactor);

    for (int i = 1; i <= optionSpec.numSteps; i ++) {
        if (i % 2 == 1) {
            iterateKernel.setArg(3, valueBufferA);
            iterateKernel.setArg(4, valueBufferB);
        } else {
            iterateKernel.setArg(3, valueBufferB);
            iterateKernel.setArg(4, valueBufferA);
        }
        queue.enqueueNDRangeKernel(iterateKernel,
                            cl::NullRange,
                            cl::NDRange(optionSpec.numSteps + 1 - i),
                            cl::NullRange);
        queue.finish();
    }

    // Read results
    float* value = new float;
    queue.enqueueReadBuffer(optionSpec.numSteps % 2 == 1? 
                            valueBufferB : valueBufferA, 
                            CL_TRUE, 
                            0, 
                            sizeof(float), 
                            value);
    return *value; 
}

double OpenCLPricer::priceImplReduce(OptionSpec& optionSpec) {
    // ------------------------Derived Parameters------------------------------
    float deltaT = optionSpec.yearsToMaturity / optionSpec.numSteps;

    float upFactor = exp(optionSpec.volatility * sqrt(deltaT));
    float downFactor = 1.0f / upFactor;

    float discountFactor = exp(optionSpec.riskFreeRate * deltaT);

    float upWeight = (discountFactor - downFactor) / (upFactor - downFactor);
    float downWeight = 1.0f - upWeight;
    
    // Create buffers on the devices
    cl::Buffer valueBufferA(*context, 
                           CL_MEM_READ_WRITE,
                           sizeof(float) * (optionSpec.numSteps + 1));

    cl::Buffer valueBufferB(*context, 
                       CL_MEM_READ_WRITE,
                       sizeof(float) * (optionSpec.numSteps + 1));

    // Create qeueue to push commands for the devices
    cl::CommandQueue queue(*context, *defaultDevice);
    
    // Build and run init kernel 
    cl::Kernel kernel(*program, "init");
    kernel.setArg(0, optionSpec.stockPrice);
    kernel.setArg(1, optionSpec.strikePrice);
    kernel.setArg(2, optionSpec.numSteps);
    kernel.setArg(3, optionSpec.type);
    kernel.setArg(4, deltaT);
    kernel.setArg(5, upFactor);
    kernel.setArg(6, downFactor);
    kernel.setArg(7, valueBufferA);
    queue.enqueueNDRangeKernel(kernel, 
                              cl::NullRange, 
                              cl::NDRange(optionSpec.numSteps + 1), 
                              cl::NullRange);
    std::cout << "[INFO] Executing init kernel with " << optionSpec.numSteps + 1
            << " work items" << std::endl;

    // Block until init kernel finishes execution
    queue.finish();

    // Note(disiok), After each iteration, the number of nodes is reduced by 500
    cl::Kernel iterateKernel(*program, "reduce");
    iterateKernel.setArg(0, upWeight);
    iterateKernel.setArg(1, downWeight);
    iterateKernel.setArg(2, discountFactor);
    iterateKernel.setArg(5, cl::Local(sizeof(float) * 501));

    for (int i = 1; i <= optionSpec.numSteps / 500; i ++) {
        if (i % 2 == 1) {
            iterateKernel.setArg(3, valueBufferA);
            iterateKernel.setArg(4, valueBufferB);
        } else {
            iterateKernel.setArg(3, valueBufferB);
            iterateKernel.setArg(4, valueBufferA);
        }
        int numWorkItems = optionSpec.numSteps + 1 - 500 * i;

        queue.enqueueNDRangeKernel(iterateKernel,
                            cl::NullRange,
                            cl::NDRange(numWorkItems),
                            cl::NDRange(1));
        queue.finish();
    }

    // Read results
    float* value = new float;
    queue.enqueueReadBuffer((optionSpec.numSteps / 500) % 2 == 1? 
                            valueBufferB : valueBufferA, 
                            CL_TRUE, 
                            0, 
                            sizeof(float), 
                            value);
    return *value; 
}

double OpenCLPricer::priceImplSync(OptionSpec& optionSpec) {
    // ------------------------Derived Parameters------------------------------
    float deltaT = optionSpec.yearsToMaturity / optionSpec.numSteps;

    float upFactor = exp(optionSpec.volatility * sqrt(deltaT));
    float downFactor = 1.0f / upFactor;

    float discountFactor = exp(optionSpec.riskFreeRate * deltaT);

    float upWeight = (discountFactor - downFactor) / (upFactor - downFactor);
    float downWeight = 1.0f - upWeight;
    
    // Create buffers on the devices
    cl::Buffer valueBufferA(*context, 
                           CL_MEM_READ_WRITE,
                           sizeof(float) * (optionSpec.numSteps + 1));

    cl::Buffer valueBufferB(*context, 
                           CL_MEM_READ_WRITE,
                           sizeof(float) * (optionSpec.numSteps + 1));

    // Create qeueue to push commands for the devices
    cl::CommandQueue queue(*context, *defaultDevice);
    
    // Build and run init kernel 
    cl::Kernel kernel(*program, "init");
    kernel.setArg(0, optionSpec.stockPrice);
    kernel.setArg(1, optionSpec.strikePrice);
    kernel.setArg(2, optionSpec.numSteps);
    kernel.setArg(3, optionSpec.type);
    kernel.setArg(4, deltaT);
    kernel.setArg(5, upFactor);
    kernel.setArg(6, downFactor);
    kernel.setArg(7, valueBufferA);
    queue.enqueueNDRangeKernel(kernel, 
                              cl::NullRange, 
                              cl::NDRange(optionSpec.numSteps + 1), 
                              cl::NullRange);
    std::cout << "[INFO] Executing init kernel with " << optionSpec.numSteps + 1
            << " work items" << std::endl;

    // Block until init kernel finishes execution
    queue.finish();

    // Note(disiok): Here we use work groups of size 501 so that after each
    // iteration, the number of nodes is reduced by 500
    cl::Kernel iterateKernel(*program, "iterate");
    iterateKernel.setArg(0, upWeight);
    iterateKernel.setArg(1, downWeight);
    iterateKernel.setArg(2, discountFactor);
    iterateKernel.setArg(5, cl::Local(sizeof(float) * 501));

    for (int i = 1; i <= optionSpec.numSteps / 500; i ++) {
        if (i % 2 == 1) {
            iterateKernel.setArg(3, valueBufferA);
            iterateKernel.setArg(4, valueBufferB);
        } else {
            iterateKernel.setArg(3, valueBufferB);
            iterateKernel.setArg(4, valueBufferA);
        }
        int numWorkGroups = optionSpec.numSteps + 1 - 500 * i;
        int groupSize = 501;
        int numWorkItems = numWorkGroups * groupSize;

        queue.enqueueNDRangeKernel(iterateKernel,
                            cl::NullRange,
                            cl::NDRange(numWorkItems)),
                            cl::NDRange(groupSize);
        std::cout << "[INFO] Executing iterate kernel with " << numWorkGroups
                << " work groups and " << groupSize << " work items per group"
                << std::endl; 

        queue.finish();
    }

    // Read results
    float* value = new float;
    queue.enqueueReadBuffer((optionSpec.numSteps / 500) % 2 == 1? 
                            valueBufferB : valueBufferA, 
                            CL_TRUE, 
                            0, 
                            sizeof(float), 
                            value);
    return *value; 
}

OpenCLPricer::~OpenCLPricer() {

}
