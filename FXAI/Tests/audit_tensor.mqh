#ifndef __FXAI_AUDIT_TENSOR_MQH__
#define __FXAI_AUDIT_TENSOR_MQH__

#include "TensorCore\tensorcore_suite.mqh"

bool FXAI_AuditTensorKernelSelfTest(string &reason)
{
   FXAITestSuiteResult suite;
   FXAI_TensorCoreRunSuite(suite);
   reason = FXAI_TestSuiteLegacyReason(suite);
   return FXAI_TestSuitePassed(suite);
}

#endif // __FXAI_AUDIT_TENSOR_MQH__
