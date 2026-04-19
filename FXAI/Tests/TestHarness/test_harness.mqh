#ifndef __FXAI_TEST_HARNESS_MQH__
#define __FXAI_TEST_HARNESS_MQH__

struct FXAITestSuiteResult
{
   string suite_name;
   int total;
   int failed;
   string case_names[];
   bool case_ok[];
   string case_reasons[];
};

void FXAI_TestSuiteReset(FXAITestSuiteResult &suite,
                         const string suite_name)
{
   suite.suite_name = suite_name;
   suite.total = 0;
   suite.failed = 0;
   ArrayResize(suite.case_names, 0);
   ArrayResize(suite.case_ok, 0);
   ArrayResize(suite.case_reasons, 0);
}

void FXAI_TestSuiteAddCase(FXAITestSuiteResult &suite,
                           const string case_name,
                           const bool passed,
                           const string reason = "")
{
   int idx = suite.total;
   ArrayResize(suite.case_names, idx + 1);
   ArrayResize(suite.case_ok, idx + 1);
   ArrayResize(suite.case_reasons, idx + 1);
   suite.case_names[idx] = case_name;
   suite.case_ok[idx] = passed;
   suite.case_reasons[idx] = reason;
   suite.total = idx + 1;
   if(!passed)
      suite.failed++;
}

bool FXAI_TestSuitePassed(const FXAITestSuiteResult &suite)
{
   return (suite.total > 0 && suite.failed == 0);
}

string FXAI_TestSuiteLegacyReason(const FXAITestSuiteResult &suite)
{
   for(int i=0; i<suite.total; i++)
   {
      if(suite.case_ok[i])
         continue;
      if(StringLen(suite.case_reasons[i]) > 0)
         return suite.case_names[i] + ":" + suite.case_reasons[i];
      return suite.case_names[i];
   }
   return "";
}

string FXAI_TestJsonEscape(const string value)
{
   string escaped = value;
   StringReplace(escaped, "\\", "\\\\");
   StringReplace(escaped, "\"", "\\\"");
   StringReplace(escaped, "\r", "\\r");
   StringReplace(escaped, "\n", "\\n");
   StringReplace(escaped, "\t", "\\t");
   return escaped;
}

void FXAI_TestSuiteAppendJson(const FXAITestSuiteResult &suite,
                              string &json)
{
   json += "{";
   json += "\"suite_name\":\"" + FXAI_TestJsonEscape(suite.suite_name) + "\",";
   json += "\"total\":" + IntegerToString(suite.total) + ",";
   json += "\"failed\":" + IntegerToString(suite.failed) + ",";
   json += "\"passed\":" + string(FXAI_TestSuitePassed(suite) ? "true" : "false") + ",";
   json += "\"cases\":[";
   for(int i=0; i<suite.total; i++)
   {
      if(i > 0)
         json += ",";
      json += "{";
      json += "\"name\":\"" + FXAI_TestJsonEscape(suite.case_names[i]) + "\",";
      json += "\"passed\":" + string(suite.case_ok[i] ? "true" : "false") + ",";
      json += "\"reason\":\"" + FXAI_TestJsonEscape(suite.case_reasons[i]) + "\"";
      json += "}";
   }
   json += "]";
   json += "}";
}

#endif // __FXAI_TEST_HARNESS_MQH__
