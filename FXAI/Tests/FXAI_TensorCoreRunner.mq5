#property strict

input int    PredictionTargetMinutes = 5;
input int    TradeKiller = 0;

#include "audit_core.mqh"

input int    Test_Seed = 42;
input bool   Test_RunTensorCore = true;
input bool   Test_RunPluginContracts = true;
input bool   Test_FailOnError = true;
input string Test_ReportFile = "FXAI\\Reports\\tensorcore_contract_report.json";

int FXAI_GetM1SyncBars(void)
{
   return 3;
}

void FXAI_TestEnsureCommonDirectories(const string file_name)
{
   string normalized = file_name;
   StringReplace(normalized, "/", "\\");
   int last_sep = StringFind(normalized, "\\", 0);
   while(last_sep >= 0)
   {
      string folder = StringSubstr(normalized, 0, last_sep);
      if(StringLen(folder) > 0)
         FolderCreate(folder, FILE_COMMON);
      last_sep = StringFind(normalized, "\\", last_sep + 1);
   }
}

bool FXAI_TestWriteCombinedReport(const string file_name,
                                  const int seed,
                                  const FXAITestSuiteResult &tensor_suite,
                                  const FXAITestSuiteResult &plugin_suite)
{
   FXAI_TestEnsureCommonDirectories(file_name);
   int handle = FileOpen(file_name, FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   string json = "{";
   json += "\"seed\":" + IntegerToString(seed) + ",";
   json += "\"generated_at\":" + IntegerToString((int)TimeCurrent()) + ",";
   json += "\"ok\":" + string(FXAI_TestSuitePassed(tensor_suite) && FXAI_TestSuitePassed(plugin_suite) ? "true" : "false") + ",";
   json += "\"suites\":[";
   FXAI_TestSuiteAppendJson(tensor_suite, json);
   json += ",";
   FXAI_TestSuiteAppendJson(plugin_suite, json);
   json += "]";
   json += "}";

   FileWriteString(handle, json);
   FileClose(handle);
   return true;
}

int OnInit()
{
   MathSrand(Test_Seed);

   FXAITestSuiteResult tensor_suite;
   FXAI_TestSuiteReset(tensor_suite, "tensorcore");
   if(Test_RunTensorCore)
      FXAI_TensorCoreRunSuite(tensor_suite);
   else
      FXAI_TestSuiteAddCase(tensor_suite, "tensorcore_skipped", true, "");

   FXAITestSuiteResult plugin_suite;
   FXAI_TestSuiteReset(plugin_suite, "plugin_contracts");
   if(Test_RunPluginContracts)
      FXAI_PluginContractRunSuite(plugin_suite);
   else
      FXAI_TestSuiteAddCase(plugin_suite, "plugin_contracts_skipped", true, "");

   FXAI_TestWriteCombinedReport(Test_ReportFile, Test_Seed, tensor_suite, plugin_suite);

   Print("FXAI TensorCore runner: tensor_failed=", tensor_suite.failed,
         " plugin_failed=", plugin_suite.failed,
         " report=", Test_ReportFile);

   if(Test_FailOnError &&
      (!FXAI_TestSuitePassed(tensor_suite) || !FXAI_TestSuitePassed(plugin_suite)))
   {
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

void OnTick() {}
