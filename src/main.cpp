#include <iostream>
#include "option_spec.h"
#include "pricer.h"

int main() {
    std::cout << "[INFO] Starting tester main function." << std::endl;
    // Construct test option specification
    OptionSpec optionSpec = {1, 100, 100, 1.0, 0.3, 0.02, 500, false};
    std::cout << optionSpec;
    OptionPricer* openclPricer = new OpenCLPricer();
    OptionPricer* serialPricer = new SerialPricer(); 
    double benchmarkPrice = serialPricer->price(optionSpec);
    double openclPrice = openclPricer->price(optionSpec); 
    std::cout << "The benchmark value of the option is: " << benchmarkPrice << std::endl;
    std::cout << "The opencl value of the option is: " << openclPrice << std::endl; 
    std::cout << "[INFO] Terminating tester main function." << std::endl;
}
