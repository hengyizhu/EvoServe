#include <Python.h>
#include <time.h>
static double ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
/* ctypes.CDLL enters without the GIL; measure a real, explicit GIL acquisition. */
double sample_gil_wait_ms(void) {
 double start=ns(); PyGILState_STATE state=PyGILState_Ensure(); double end=ns(); PyGILState_Release(state); return (end-start)/1e6;
}
