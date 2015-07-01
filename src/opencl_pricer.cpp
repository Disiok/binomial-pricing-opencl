#include <OpenCL/cl.hpp>
#include <vector>
#include <iostream>
#include <fstream>

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
   return 0; 
}

OpenCLPricer::~OpenCLPricer() {

}
