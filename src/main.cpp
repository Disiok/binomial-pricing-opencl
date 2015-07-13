#include <iostream>
#include <chrono>
#include "option_spec.h"
#include "pricer.h"

int main() {
    std::cout << "[INFO] Starting tester main function." << std::endl;
    // Construct test option specification
    OptionSpec optionSpec = {1, 100, 100, 1.0, 0.3, 0.02, 10, false};
    std::cout << optionSpec;

    // Price with serial pricer
    OptionPricer* serialPricer = new SerialPricer(); 
    auto start = std::chrono::steady_clock::now();
    double benchmarkPrice;
    for (int i = 0; i < 1; i ++){
        benchmarkPrice = serialPricer->price(optionSpec);
    }
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "The benchmark value of the option is: " << benchmarkPrice << std::endl;
    std::cout << "Priced in " << std::chrono::duration<double, std::milli> (diff).count()
            << " ms" << std::endl;

    // Price with opencl pricer
    OptionPricer* openclPricer = new OpenCLPricer();
    start = std::chrono::steady_clock::now();
    double openclPrice;
    for (int i = 0; i < 1; i ++) {
        openclPrice = openclPricer->price(optionSpec); 
    }
    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "The opencl value of the option is: " << openclPrice << std::endl; 
    std::cout << "Priced in " << std::chrono::duration<double, std::milli> (diff).count()
            << " ms" << std::endl;

    std::cout << "[INFO] Terminating tester main function." << std::endl;
}
