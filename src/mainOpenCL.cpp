// NOTE(disiok): Supposedly use cl::vector instead of std::vector
// It is, however, causing an error right now
// #define __NO_STD_VECTOR 
// #define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "option_spec.h"

int main() {
    std::cout << "[INFO] Starting main function" << std::endl;
    //try {
        // Retrieve platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Check number of platforms found
        if (platforms.size() == 0) {
            std::cerr   << "[ERROR] No platform found. Check OpenCL installation!" 
                        << std::endl;
            exit(1);
        } else {
            std::cout   << "[INFO] " << platforms.size() << " platforms found." 
                        << std::endl;
        }

        // TODO(disiok): Add parameters to choose platforms
        // Select default platform
        cl::Platform defaultPlatform = platforms[0];
        std::cout   << "[INFO] Using platform: " 
                    << defaultPlatform.getInfo<CL_PLATFORM_NAME>() 
                    << std::endl;

        // Retrieve devices
        std::vector<cl::Device> devices;
        defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        // Check number of devices found
        if (devices.size() == 0) {
            std::cerr   << "[ERROR] No devices found. Check OpenCL installation!" 
                        << std::endl;
            exit(2);
        } else {
            std::cout   << "[INFO] " << devices.size() << " devices found." 
                        << std::endl;
        }

        // TODO(disiok): Add parameters to choose devices
        // Select default device
        cl::Device defaultDevice = devices[1];
        std::cout   << "[INFO] Using device: " 
                    << defaultDevice.getInfo<CL_DEVICE_NAME>() 
                    << std::endl;

        // Create context
        cl::Context context({defaultDevice});

        // Define kernel code
        std::ifstream ifs("kernel.cl");
        std::string kernelCode(
                (std::istreambuf_iterator<char>(ifs)),
                (std::istreambuf_iterator<char>()));
        cl::Program::Sources sources;
        sources.push_back({kernelCode.c_str(), kernelCode.length()});

        // Build kernel code
        cl::Program program(context, sources);
        if (program.build({defaultDevice}) != CL_SUCCESS) {
            std::cerr   << "[ERROR] Error building: " 
                        << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) 
                        << std::endl;
            exit(4);
        } else {
            std::cout   << "[INFO] Successfully built kernel program" 
                        << std::endl;
        }
        
        // Hardcode option spec for testing
        OptionSpec optionSpec = {0, 50, 50, 1, 0.5, 0.05, 100}; 
        std::vector<int> v;
        for (int i = 0; i < optionSpec.numSteps; i++) {
            v.push_back(i);
        }

        std::cout << "[INFO] " << optionSpec;

        // Create buffers on the devices
        cl::Buffer indexBuffer(context, 
                               CL_MEM_READ_WRITE, 
                               sizeof(int) * optionSpec.numSteps,
                               v.data());

        cl::Buffer valueBuffer(context, 
                               CL_MEM_READ_WRITE,
                               sizeof(float) * optionSpec.numSteps);

        // Create qeueue to push commands for the devices
        cl::CommandQueue queue(context, defaultDevice);
        
        // Run kernel with functor
        cl::Kernel kernel = cl::Kernel(program, "init");
        kernel.setArg(0, optionSpec.stockPrice);
        kernel.setArg(1, optionSpec.strikePrice);
        kernel.setArg(2, optionSpec.yearsToMaturity);
        kernel.setArg(3, optionSpec.volatility);
        kernel.setArg(4, optionSpec.riskFreeRate);
        kernel.setArg(5, optionSpec.numSteps);
        kernel.setArg(6, indexBuffer);
        kernel.setArg(7, valueBuffer);
        queue.enqueueNDRangeKernel(kernel, 
                                  cl::NullRange, 
                                  cl::NDRange(optionSpec.numSteps), 
                                  cl::NullRange);
        queue.finish();
    
        // Read results
        float* values = new float[optionSpec.numSteps];
        queue.enqueueReadBuffer(valueBuffer, 
                                CL_TRUE, 
                                0, 
                                sizeof(float) * optionSpec.numSteps, 
                                values);

        // Print results
        std::cout   << "The result is: "
                    << std::endl;

        for (int i = 0; i < 10; i ++) {
            std::cout   << values[i] << " ";
        }
        std::cout   << std::endl;
    //} catch (cl::Error error) {
    //    std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
    //}
}
