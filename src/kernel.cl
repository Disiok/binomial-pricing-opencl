__kernel void
init(
     const float stockPrice,
     const float strikePrice,
     const int numSteps,
     const int type,
     const float deltaT,
     const float upFactor,
     const float downFactor,
     __global float* valueAtExpiry
     )
{
    // ---------------------Calculate option value at expiry-------------------
    size_t id = get_global_id(0);
    float stockPriceAtExpiry = stockPrice * pow(upFactor, id) *
                                            pow(downFactor, numSteps - id);
    valueAtExpiry[id] = max(type * (stockPriceAtExpiry - strikePrice), 0.0f); 
    // printf("[TRACE] valueAtExpiry[%d] = %f\n", id, valueAtExpiry[id]);
}

__kernel void
group(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut,
        const int currentNumLattice,
        const int groupSize
        )
{
    int startIndex = get_global_id(0) * groupSize;
    int endIndex = min(startIndex + groupSize, currentNumLattice);

    for (int i = startIndex; i < endIndex; i++) {
        optionValueOut[i] = (downWeight * optionValueIn[i] + 
                            upWeight * optionValueIn[i + 1])
                            / discountFactor;
    }
}

__kernel  
void upTriangle(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValue,
        __local float* tempOptionValue,
        __global float* triangle
        )
{
    int localId = get_local_id(0);
    int groupSize = get_local_size(0);
    int groupId = get_group_id(0);

    int stepSize = groupSize - 1;
    int offset = stepSize * groupId;
    int globalId = offset + localId;

    tempOptionValue[localId] = optionValue[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1 ; i <= stepSize; i ++) {
        float value;
        if (localId <= stepSize - i) {
            value = (downWeight * tempOptionValue[localId] +
                    upWeight * tempOptionValue[localId + 1])
                    / discountFactor;
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId <= stepSize - i) {
            tempOptionValue[localId] = value;
        }

        if (localId == 0) {
            triangle[offset + i] = value; 
        } else if (localId == stepSize - i) {
           optionValue[globalId] = value;
        }
    }
}
__kernel  
void downTriangle(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValue,
        __local float* tempOptionValue,
        __global float* triangle
        )
{
    int localId = get_local_id(0);
    int groupSize = get_local_size(0);
    int groupId = get_group_id(0);

    int stepSize = groupSize - 1;
    int offset = stepSize * groupId;
    int globalId = offset + localId;

    for (int i = 1 ; i <= stepSize - 1; i ++) {
        float value;
        float upValue;
        float downValue;

        if (localId == stepSize - 1) {
            upValue = triangle[offset + stepSize + i];
        } else {
            upValue = tempOptionValue[localId + 1];
        }

        if (localId == stepSize - i) {
            downValue = optionValue[globalId];
        } else {
            downValue = tempOptionValue[localId];
        }

        if (localId >= stepSize - i && localId < stepSize) {
            value = (downWeight * downValue+
                    upWeight * upValue)
                    / discountFactor;
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId >= stepSize - i && localId < stepSize) {
            tempOptionValue[localId] = value;
        }
    }
    if (localId >= 0 && localId < stepSize) {
        optionValue[globalId] = tempOptionValue[localId]; 
    }
}
