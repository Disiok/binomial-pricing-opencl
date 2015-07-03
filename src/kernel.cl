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
}

__kernel void
iterate(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut,
        __local float* tempOptionValue
        )
{
    size_t localId = get_local_id(0);
    size_t groupId = get_group_id(0);

    tempOptionValue[localId] = optionValueIn[groupId + localId];
    // printf("Group %d, local %d, value %f \n", groupId, localId, optionValueIn[groupId + localId]);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 499 ; i >= 0; i --) {
        float value = tempOptionValue[localId];
        if (localId <= i) {
            value = (downWeight * tempOptionValue[localId] +
                    upWeight * tempOptionValue[localId + 1])
                    / discountFactor;
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
        tempOptionValue[localId] = value;
    }
    if (localId == 0) {
        optionValueOut[groupId] = tempOptionValue[localId];
        // printf("Last iteration with local %d, value %f \n", localId,optionValue[groupId]);
    }
}

__kernel void
reduce(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut,
        __local float* tempOptionValue
        )
{
    size_t id = get_global_id(0);

    for (int i = 0; i < 501; i ++) {
        tempOptionValue[i] = optionValueIn[id + i];         
    }

    for (int i = 499 ; i >= 0; i --) {
        for (int j = 0; j <= i; j++) {
            tempOptionValue[j] = (downWeight * tempOptionValue[j] +
                    upWeight * tempOptionValue[j + 1])
                    / discountFactor;
        }
    }
    optionValueOut[id] = tempOptionValue[0];
}

__kernel void
solo(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut
        )
{
    size_t id = get_global_id(0);

    optionValueOut[id] = (downWeight * optionValueIn[id] + 
                         upWeight * optionValueIn[id + 1])
                         / discountFactor;
}
