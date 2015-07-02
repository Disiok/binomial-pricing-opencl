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
    // ------------------------Derived Parameters------------------------------
    float deltaT = optionSpec.yearsToMaturity / optionSpec.numSteps;

    float upFactor = exp(optionSpec.volatility * sqrt(deltaT));
    float downFactor = 1.0f / upFactor;

    float discountFactor = exp(optionSpec.riskFreeRate * deltaT);

    float upWeight = (discountFactor - downFactor) / (upFactor - downFactor);
    float downWeight = 1.0f - upWeight;
    
    // Create buffers on the devices
    cl::Buffer valueAtExpiryBuffer(*context, 
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
    kernel.setArg(7, valueAtExpiryBuffer);
    queue.enqueueNDRangeKernel(kernel, 
                              cl::NullRange, 
                              cl::NDRange(optionSpec.numSteps + 1), 
                              cl::NullRange);
    
    // Block until init kernel finishes execution
    queue.finish();

    cl::Kernel iterateKernel(*program, "iterate");
    iterateKernel.setArg(0, upWeight);
    iterateKernel.setArg(1, downWeight);
    iterateKernel.setArg(2, discountFactor);
    iterateKernel.setArg(3, valueAtExpiryBuffer);
    iterateKernel.setArg(4, cl::Local(sizeof(float) * 512));

    for (int i = 0; i < (optionSpec.numSteps + 1) / 512; i ++) {
        queue.enqueueNDRangeKernel(iterateKernel,
                            cl::NullRange,
                            cl::NDRange(optionSpec.numSteps + 1 - i * 512),
                            cl::NDRange(512));
        queue.finish();
    }

    // Read results
    float* values = new float[optionSpec.numSteps + 1];
    queue.enqueueReadBuffer(valueAtExpiryBuffer, 
                            CL_TRUE, 
                            0, 
                            sizeof(float) * (optionSpec.numSteps + 1), 
                            values);

    // Print results
    std::cout   << "The result is: "
                << std::endl;

    for (int i = 0; i < optionSpec.numSteps + 1; i ++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    return values[0]; 
}

OpenCLPricer::~OpenCLPricer() {

}
