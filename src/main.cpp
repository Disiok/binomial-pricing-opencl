#include <iostream>
#include <iomanip>
#include <chrono>
#include "option_spec.h"
#include "pricer.h"

void iterativeBenchmark(int initialNumSteps, int growthRate, int seconds) {
    auto start = std::chrono::steady_clock::now();
    std::chrono::seconds maxDuration(seconds);

    OptionPricer* serialPricer = new SerialPricer(); 
    OptionPricer* openclPricer = new OpenCLPricer();
    while (std::chrono::steady_clock::now() - start < maxDuration) {
        std::cout << "-------------------------------------" << std::endl;

        // Construct test option specification
        OptionSpec optionSpec = {1, 100, 100, 1.0, 0.3, 0.02, initialNumSteps, false};
        std::cout << "Number of steps: " << initialNumSteps << std::endl;
        
        // Price with serial pricer
        auto start = std::chrono::steady_clock::now();
        double benchmarkPrice = serialPricer->price(optionSpec);
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        std::cout << "[Benchmark] Value: " << std::setprecision(10) << benchmarkPrice << std::endl;
        std::cout << "[Benchmark] Time: " << std::chrono::duration<double, std::milli> (diff).count()
            << " ms" << std::endl;

        // Price with opencl pricer
        start = std::chrono::steady_clock::now();
        double openclPrice = openclPricer->price(optionSpec); 
        end = std::chrono::steady_clock::now();
        diff = end - start;
        std::cout << "[OpenCL] Value: " << std::setprecision(10) << openclPrice << std::endl; 
        std::cout << "[OpenCL] Time: "  << std::chrono::duration<double, std::milli> (diff).count()
            << " ms" << std::endl;

        initialNumSteps *= growthRate;
    }
}

int main() {
    std::cout << "[INFO] Starting tester main function." << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    
    iterativeBenchmark(500, 2, 5);

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "[INFO] Terminating tester main function." << std::endl;
}
