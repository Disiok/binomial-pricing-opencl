// NOTE(disiok): Supposedly use cl::vector instead of std::vector
// It is, however, causing an error right now
// #define __NO_STD_VECTOR 
#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::cout << "[INFO] Starting main function" << std::endl;
    try {
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
        cl::Program::Sources sources;
        std::ifstream ifs("addvec.cl");
        std::string kernelCode(
                (std::istreambuf_iterator<char>(ifs)),
                (std::istreambuf_iterator<char>()));
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

        // Create buffers on the devices
        cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
        cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
        cl::Buffer bufferC(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

        // Hardcode some arrays for testing
        int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int b[] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

        // Create qeueue to push commands for the devices
        cl::CommandQueue queue(context, defaultDevice);
        
        // Write data to the device
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(int) * 10, a);
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(int) * 10, b);
        
        //  NOTE(disiok): KernelFunctor is removed from the API in v1.2
        //  cl::KernelFunctor simple_add(
        //         cl::Kernel(program, "simple_add"),
        //         queue,
        //         cl::NullRrange,
        //         cl::NDRange(10),
        //         cl::NullRange);
        
        // Run kernel with functor
        cl::Kernel kernel = cl::Kernel(program, "simple_add");
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(10), cl::NullRange);
        queue.finish();
    
        // Read results
        int c[10];
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * 10, c);

        // Print results
        std::cout   << "The result is: "
                    << std::endl;

        for (int i = 0; i < 10; i ++) {
            std::cout   << c[i] << " ";
        }
        std::cout   << std::endl;
    } catch (cl::Error error) {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
    }
}
