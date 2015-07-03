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
    size_t groupSize = get_local_size(0);
    size_t groupId = get_group_id(0);
    size_t stepSize = groupSize - 1;

    tempOptionValue[localId] = optionValueIn[groupId + localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = stepSize - 1 ; i >= 0; i --) {
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
        optionValueOut[groupId] = tempOptionValue[0];
        // printf("[TRACE] groupId = %d, stepSize = %d, optionValueOut[%d] = %f\n", groupId, stepSize, groupId, optionValueOut[groupId]);
    }
}

__kernel void
group(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut,
        const int currentNumLattice,
        const int stepSize
        )
{
    size_t id = get_global_id(0) * stepSize;

    for (int i = 0; i < stepSize; i++) {
        if (id + i < currentNumLattice) {
            optionValueOut[id + i] = (downWeight * optionValueIn[id + i] + 
                                   upWeight * optionValueIn[id + i + 1])
                                   / discountFactor;
        }
    }
}
