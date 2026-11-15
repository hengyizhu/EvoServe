#include <nvtx3/nvToolsExt.h>
int trace_begin(const char *name) { return nvtxRangePushA(name); }
int trace_end(void) { return nvtxRangePop(); }
