#include <iostream>
#include <chrono>
#include "option_spec.h"
#include "pricer.h"

void iterativeBenchmark(int initialNumSteps, int growthRate, int seconds) {
    auto start = std::chrono::steady_clock::now();
    std::chrono::seconds maxDuration(seconds);

    OptionPricer* serialPricer = new SerialPricer(); 
    OptionPricer* openclPricer = new OpenCLPricer();
    while (std::chrono::steady_clock::now() - start < maxDuration) {
        // Construct test option specification
        OptionSpec optionSpec = {1, 100, 100, 1.0, 0.3, 0.02, initialNumSteps, false};
        
        // Price with serial pricer
        auto start = std::chrono::steady_clock::now();
        double benchmarkPrice = serialPricer->price(optionSpec);
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        std::cout << "The benchmark value: " << benchmarkPrice << std::endl;
        std::cout << initialNumSteps << " steps priced in " << std::chrono::duration<double, std::milli> (diff).count()
            << " ms" << std::endl;

        // Price with opencl pricer
        start = std::chrono::steady_clock::now();
        double openclPrice = openclPricer->price(optionSpec); 
        end = std::chrono::steady_clock::now();
        diff = end - start;
        std::cout << "The opencl value: " << openclPrice << std::endl; 
        std::cout << initialNumSteps << " steps priced in " << std::chrono::duration<double, std::milli> (diff).count()
            << " ms" << std::endl;

        initialNumSteps *= growthRate;
    }
}

int main() {
    std::cout << "[INFO] Starting tester main function." << std::endl;
    
    iterativeBenchmark(500, 2, 10);
    std::cout << "[INFO] Terminating tester main function." << std::endl;
}
