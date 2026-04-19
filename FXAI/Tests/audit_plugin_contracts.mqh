#ifndef __FXAI_AUDIT_PLUGIN_CONTRACTS_MQH__
#define __FXAI_AUDIT_PLUGIN_CONTRACTS_MQH__

#include "PluginContracts\plugin_contract_suite.mqh"

bool FXAI_AuditPluginContractSelfTest(string &reason)
{
   FXAITestSuiteResult suite;
   FXAI_PluginContractRunSuite(suite);
   reason = FXAI_TestSuiteLegacyReason(suite);
   return FXAI_TestSuitePassed(suite);
}

#endif // __FXAI_AUDIT_PLUGIN_CONTRACTS_MQH__
