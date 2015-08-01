// System Libraries
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

// OpenCL C++ Binding
#include "cl.hpp"

#include "pricer.h"
#include "option_spec.h"

// ---------------------------Constructor--------------------------------------
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
    // NOTE(disiok): Default to improved triangle algorithm
    return priceImplTriangle(optionSpec, 500); 
    // return priceImplGroup(optionSpec, 5); 
}

/**
 * Algorithm:
 *  init kernel:
 *      Use (optionSpec.numSteps + 1) work-items to compute the option values
 *      at expiry
 *      Only executed once
 *
 *  group kernel:
 *      Each work-item calculate the previous option values of (groupSize)
 *      lattice points
 *      Kernel executed (optionSpec.numSteps) times
 *      Each execution reduces the number of lattice points by 1
 */
double OpenCLPricer::priceImplGroup(OptionSpec& optionSpec, int groupSize) {
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
    cl::Kernel initKernel(*program, "init");
    initKernel.setArg(0, optionSpec.stockPrice);
    initKernel.setArg(1, optionSpec.strikePrice);
    initKernel.setArg(2, optionSpec.numSteps);
    initKernel.setArg(3, optionSpec.type);
    initKernel.setArg(4, deltaT);
    initKernel.setArg(5, upFactor);
    initKernel.setArg(6, downFactor);
    initKernel.setArg(7, valueBufferA);
    queue.enqueueNDRangeKernel(initKernel, 
                              cl::NullRange, 
                              cl::NDRange(optionSpec.numSteps + 1), 
                              cl::NullRange);
    // std::cout << "[INFO] Executing init kernel with " << optionSpec.numSteps + 1
    //        << " work items" << std::endl;

    // Block until init kernel finishes execution
    queue.enqueueBarrierWithWaitList();

    // Build and run group kernel 
    cl::Kernel groupKernel(*program, "group");
    groupKernel.setArg(0, upWeight);
    groupKernel.setArg(1, downWeight);
    groupKernel.setArg(2, discountFactor);
    for (int i = 1; i <= optionSpec.numSteps; i ++) {
        int numLatticePoints = optionSpec.numSteps + 1 - i;
        int numWorkItems = ceil((float) numLatticePoints / groupSize);
        groupKernel.setArg(3, i % 2 == 1 ? valueBufferA : valueBufferB);
        groupKernel.setArg(4, i % 2 == 1 ? valueBufferB: valueBufferA);
        groupKernel.setArg(5, numLatticePoints);
        groupKernel.setArg(6, groupSize);
        queue.enqueueNDRangeKernel(groupKernel,
                            cl::NullRange,
                            cl::NDRange(numWorkItems),
                            cl::NullRange);

        // std::cout << "[INFO] Executing group kernel with " << numWorkItems
        //         << " work items" << std::endl;
        queue.enqueueBarrierWithWaitList();
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


double OpenCLPricer::priceImplTriangle(OptionSpec& optionSpec, int stepSize) {
    if (stepSize >= 512) {
        std::cerr << "[Error] Step size not valid."
            << "Cannot have more than 512 work items per work group" 
            << std::endl;
        exit(5);
    }

    // ------------------------Derived Parameters------------------------------
    float deltaT = optionSpec.yearsToMaturity / optionSpec.numSteps;

    float upFactor = exp(optionSpec.volatility * sqrt(deltaT));
    float downFactor = 1.0f / upFactor;

    float discountFactor = exp(optionSpec.riskFreeRate * deltaT);

    float upWeight = (discountFactor - downFactor) / (upFactor - downFactor);
    float downWeight = 1.0f - upWeight;
    
    // Create buffers on the devices
    cl::Buffer valueBuffer(*context, 
                           CL_MEM_READ_WRITE,
                           sizeof(float) * (optionSpec.numSteps + 1));

    cl::Buffer triangleBuffer(*context, 
                           CL_MEM_READ_WRITE,
                           sizeof(float) * (optionSpec.numSteps + 1));

    // Create qeueue to push commands for the devices
    cl::CommandQueue queue(*context, *defaultDevice);
    
    // Build and run init kernel 
    cl::Kernel initKernel(*program, "init");
    initKernel.setArg(0, optionSpec.stockPrice);
    initKernel.setArg(1, optionSpec.strikePrice);
    initKernel.setArg(2, optionSpec.numSteps);
    initKernel.setArg(3, optionSpec.type);
    initKernel.setArg(4, deltaT);
    initKernel.setArg(5, upFactor);
    initKernel.setArg(6, downFactor);
    initKernel.setArg(7, valueBuffer);
    queue.enqueueNDRangeKernel(initKernel, 
                              cl::NullRange, 
                              cl::NDRange(optionSpec.numSteps + 1), 
                              cl::NullRange);
    // std::cout << "[INFO] Executing init kernel with " << optionSpec.numSteps + 1
    //         << " work items" << std::endl;

    // Block until init kernel finishes execution
    queue.enqueueBarrierWithWaitList();

    // Note(disiok): Here we use work groups of size stepSize + 1 
    // so that after each iteration, the number of nodes is reduced by stepSize
    int groupSize = stepSize + 1;

    cl::Kernel upKernel(*program, "upTriangle");
    upKernel.setArg(0, upWeight);
    upKernel.setArg(1, downWeight);
    upKernel.setArg(2, discountFactor);
    upKernel.setArg(3, valueBuffer);
    upKernel.setArg(4, cl::Local(sizeof(float) * groupSize));
    upKernel.setArg(5, triangleBuffer);

    cl::Kernel downKernel(*program, "downTriangle");
    downKernel.setArg(0, upWeight);
    downKernel.setArg(1, downWeight);
    downKernel.setArg(2, discountFactor);
    downKernel.setArg(3, valueBuffer);
    downKernel.setArg(4, cl::Local(sizeof(float) * groupSize));
    downKernel.setArg(5, triangleBuffer);
    for (int i = 0; i < optionSpec.numSteps / stepSize; i ++) {
        int numWorkGroupsUp = optionSpec.numSteps / stepSize - i;
        int numWorkGroupsDown = numWorkGroupsUp - 1;
        int numWorkItemsUp = numWorkGroupsUp * groupSize;
        int numWorkItemsDown = numWorkGroupsDown * groupSize;

        queue.enqueueNDRangeKernel(upKernel,
                            cl::NullRange,
                            cl::NDRange(numWorkItemsUp)),
                            cl::NDRange(groupSize);
        // std::cout << "[INFO] Executing up kernel with " << numWorkGroupsUp
        //         << " work groups and " << groupSize << " work items per group"
        //         << std::endl; 

        queue.enqueueBarrierWithWaitList();

        if (numWorkGroupsDown > 0) {
            queue.enqueueNDRangeKernel(downKernel,
                    cl::NullRange,
                    cl::NDRange(numWorkItemsDown)),
                    cl::NDRange(groupSize);
            // std::cout << "[INFO] Executing down kernel with " << numWorkGroupsDown
            //     << " work groups and " << groupSize << " work items per group"
            //     << std::endl; 
            queue.enqueueBarrierWithWaitList();
        }
    }

    // Read results
    float* value = new float;
    queue.enqueueReadBuffer(valueBuffer, 
                            CL_TRUE, 
                            0, 
                            sizeof(float), 
                            value);
    return *value; 
}

// -------------------------Destructor-----------------------------------------
OpenCLPricer::~OpenCLPricer() {
    delete platforms;
    delete devices;
    delete context;
    delete kernelCode;
    delete sources;
    delete program;
}
