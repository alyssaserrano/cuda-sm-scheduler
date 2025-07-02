// Get the ID of the current SM
#define __SMC_getSMid                     \
    uint __SMC_smid;                     \
    asm("mov.u32 %0, %smid;" : "=r"(__SMC_smid) )

// Initialize the SM-control framework
#define __SMC_init                                      \
    unsigned int __SMC_workersNeeded = __SMC_numNeeded(); \
    unsigned int* __SMC_newChunkSeq = __SMC_buildChunkSeq(); \
    unsigned int* __SMC_workerCount = __SMC_initiateArray();

// Begin macro for SM partition logic
#define __SMC_Begin                                         \
    __shared__ int __SMC_workingCTAs;                       \
    __SMC_getSMid;                                          \
    if (offsetInCTA == 0)                                   \
        __SMC_workingCTAs =                                 \
            atomicInc(&__SMC_workerCount[__SMC_smid], INT_MAX); \
    __syncthreads();                                        \
    if (__SMC_workingCTAs >= __SMC_workersNeeded) return;   \
    int __SMC_chunksPerCTA =                                \
        __SMC_chunksPerSM / __SMC_workersNeeded;            \
    int __SMC_startChunkIDidx = __SMC_smid * __SMC_chunksPerSM + \
        __SMC_workingCTAs * __SMC_chunksPerCTA;             \
    for (int __SMC_chunkIDidx = __SMC_startChunkIDidx;      \
         __SMC_chunkIDidx < __SMC_startChunkIDidx + __SMC_chunksPerCTA; \
         __SMC_chunkIDidx++) {                              \
        __SMC_chunkID = __SMC_newChunkSeq[__SMC_chunkIDidx];

// End macro for loop closure
#define __SMC_End }

