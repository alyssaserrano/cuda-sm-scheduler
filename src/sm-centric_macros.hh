// sm_centric_macros.hh
#pragma once

/**** MACROS ****/
#define SMC_init(K) \
        unsigned int SMC_workersNeeded = SMC_numNeeded(); \
        unsigned int* SMC_newChunkSeq = SMC_buildChunkSeq((K), SMC_workersNeeded); \
        unsigned int* SMC_workerCount = SMC_initiateArray(SMC_workersNeeded);

#define SMC_getSMid \
    uint SMC_smid; \
    asm("mov.u32 %0, %smid;" : "=r"(SMC_smid) );

#define SMC_Begin \
    __shared__ int SMC_workingCTAs; \
    SMC_getSMid; \
    if (offsetInCTA == 0) \
        SMC_workingCTAs = atomicInc(&SMC_workerCount[SMC_smid], INT_MAX); \
    __syncthreads(); \
    if (SMC_workingCTAs >= SMC_workersNeeded) { \
        if (threadIdx.x == 0) \
    	    printf("Block %d rejected: SMC_workingCTAs=%d >= %d\n", blockIdx.x, SMC_workingCTAs, SMC_workersNeeded); \
        return; \
    } \
    int SMC_chunksPerCTA = max(1, SMC_chunksPerSM / SMC_workersNeeded); \
    int SMC_startChunkIDidx = SMC_smid * SMC_chunksPerSM + SMC_workingCTAs * SMC_chunksPerCTA; \
    for (int SMC_chunkIDidx = SMC_startChunkIDidx; \
         SMC_chunkIDidx < SMC_startChunkIDidx + SMC_chunksPerCTA; \
         SMC_chunkIDidx++) { \
        SMC_chunkID = SMC_newChunkSeq[SMC_chunkIDidx]; \
	/*if (threadIdx.x == 0) { \
            printf("Block %d is running chunk %d on SM %d\n", blockIdx.x, SMC_chunkID, SMC_smid); \
        }*/

#define SMC_End }
