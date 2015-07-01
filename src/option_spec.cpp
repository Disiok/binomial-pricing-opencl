#include <iostream>
#include "option_spec.h"

std::ostream& operator<<(std::ostream& os, const OptionSpec& other) {
    os << "Option Pricing Specification" << std::endl;
    os << "------------------------------" << std::endl;
    os << "Stock Price: " << other.stockPrice << std::endl;
    os << "Strike Price: " << other.strikePrice << std::endl;
    os << "Years to Maturity: " << other.yearsToMaturity << std::endl;
    os << "Volatility: " << other.volatility << std::endl;
    os << "Risk Free Rate: " << other.riskFreeRate << std::endl;
    os << "Number of Steps: " << other.numSteps << std::endl; 
    os << "------------------------------" << std::endl;
    return os;
}

