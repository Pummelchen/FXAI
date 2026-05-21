#property strict

#include "..\Engine\core.mqh"
#include "audit_core.mqh"

input bool   Audit_AllPlugins = true;
input ENUM_AI_TYPE Audit_Plugin = AI_M1SYNC;
input string Audit_PluginList = "{all}";
input string Audit_ScenarioList = "{random_walk, drift_up, drift_down, mean_revert, vol_cluster, monotonic_up, monotonic_down, regime_shift, market_recent, market_trend, market_chop, market_session_edges, market_spread_shock, market_walkforward, market_macro_event, market_adversarial}";
input int    Audit_Bars = 20000;
input int    PredictionTargetMinutes = 5;
input int    Audit_M1SyncBars = 3;
input ENUM_FXAI_FEATURE_NORMALIZATION Audit_Normalization = FXAI_NORM_EXISTING;
input int    Audit_SequenceBarsOverride = 0;
input int    Audit_SchemaOverride = 0;
input ulong  Audit_FeatureGroupsMaskOverride = 0;
input double Audit_CommissionPerLotSide = 0.0;
input double Audit_CostBufferPoints = 2.0;
input double Audit_SlippagePoints = 0.0;
input double Audit_FillPenaltyPoints = 0.0;
input int    Audit_WalkForwardTrainBars = 256;
input int    Audit_WalkForwardTestBars = 64;
input int    Audit_WalkForwardPurgeBars = 32;
input int    Audit_WalkForwardEmbargoBars = 24;
input int    Audit_WalkForwardFolds = 6;
input long   Audit_WindowStartUnix = 0;
input long   Audit_WindowEndUnix = 0;
input int    Audit_Seed = 42;
input bool   Audit_RunTensorKernelSanity = true;
input bool   Audit_RunPluginContractSanity = true;
input bool   Audit_ResetOutput = true;
input bool   Audit_StopOnFailure = false;
input int    TradeKiller = 0;

int FXAI_GetM1SyncBars(void)
{
   int v = Audit_M1SyncBars;
   if(v < 2) v = 2;
   if(v > 12) v = 12;
   return v;
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_GetFeatureNormalizationMethod()
{
   return Audit_Normalization;
}

int FXAI_AuditGetSequenceBarsOverride(void)
{
   return Audit_SequenceBarsOverride;
}

int FXAI_AuditGetSchemaOverride(void)
{
   return Audit_SchemaOverride;
}

double FXAI_AuditGetCommissionPerLotSide(void)
{
   return Audit_CommissionPerLotSide;
}

double FXAI_AuditGetCostBufferPoints(void)
{
   return Audit_CostBufferPoints;
}

double FXAI_AuditGetSlippagePoints(void)
{
   return Audit_SlippagePoints;
}

double FXAI_AuditGetFillPenaltyPoints(void)
{
   return Audit_FillPenaltyPoints;
}

void FXAI_ResolveExecutionProfile(FXAIExecutionProfile &profile)
{
   FXAI_ClearExecutionProfile(profile);
   profile.profile_id = (int)FXAI_EXEC_DEFAULT;
   profile.commission_per_lot_side = MathMax(Audit_CommissionPerLotSide, 0.0);
   profile.cost_buffer_points = 0.0;
   profile.slippage_points = MathMax(Audit_SlippagePoints, 0.0);
   profile.fill_penalty_points = MathMax(Audit_FillPenaltyPoints, 0.0);
   profile.allowed_deviation_points = 2.0 +
                                      4.0 * FXAI_Clamp(profile.slippage_points +
                                                       profile.fill_penalty_points,
                                                       0.0,
                                                       2.0);
}

int FXAI_AuditGetWalkForwardTrainBars(void)
{
   return Audit_WalkForwardTrainBars;
}

int FXAI_AuditGetWalkForwardTestBars(void)
{
   return Audit_WalkForwardTestBars;
}

int FXAI_AuditGetWalkForwardPurgeBars(void)
{
   int v = Audit_WalkForwardPurgeBars;
   if(v < 0) v = 0;
   if(v > 512) v = 512;
   return v;
}

int FXAI_AuditGetWalkForwardEmbargoBars(void)
{
   int v = Audit_WalkForwardEmbargoBars;
   if(v < 0) v = 0;
   if(v > 512) v = 512;
   return v;
}

int FXAI_AuditGetWalkForwardFolds(void)
{
   int v = Audit_WalkForwardFolds;
   if(v < 2) v = 2;
   if(v > 16) v = 16;
   return v;
}

datetime FXAI_AuditGetWindowStartTime(void)
{
   if(Audit_WindowStartUnix <= 0)
      return 0;
   return (datetime)Audit_WindowStartUnix;
}

datetime FXAI_AuditGetWindowEndTime(void)
{
   if(Audit_WindowEndUnix <= 0)
      return 0;
   return (datetime)Audit_WindowEndUnix;
}

ulong FXAI_AuditGetFeatureGroupsMaskOverride(void)
{
   return Audit_FeatureGroupsMaskOverride;
}

bool FXAI_AuditScenarioIdFromName(const string raw_name,
                                  int &out_id)
{
   string name = raw_name;
   StringTrimLeft(name);
   StringTrimRight(name);
   StringToLower(name);

   if(name == "random_walk") { out_id = 0; return true; }
   if(name == "drift_up") { out_id = 1; return true; }
   if(name == "drift_down") { out_id = 2; return true; }
   if(name == "mean_revert") { out_id = 3; return true; }
   if(name == "vol_cluster") { out_id = 4; return true; }
   if(name == "monotonic_up") { out_id = 5; return true; }
   if(name == "monotonic_down") { out_id = 6; return true; }
   if(name == "regime_shift") { out_id = 7; return true; }
   if(name == "market_recent") { out_id = 8; return true; }
   if(name == "market_trend") { out_id = 9; return true; }
   if(name == "market_chop") { out_id = 10; return true; }
   if(name == "market_session_edges") { out_id = 11; return true; }
   if(name == "market_spread_shock") { out_id = 12; return true; }
   if(name == "market_walkforward") { out_id = 13; return true; }
   if(name == "market_macro_event") { out_id = 14; return true; }
   if(name == "market_adversarial") { out_id = 15; return true; }
   return false;
}

void FXAI_AuditParseScenarioList(const string raw,
                                 int &scenario_ids[])
{
   ArrayResize(scenario_ids, 0);
   string clean = raw;
   StringReplace(clean, "{", "");
   StringReplace(clean, "}", "");
   StringReplace(clean, ";", ",");
   StringReplace(clean, "|", ",");

   string parts[];
   int n = StringSplit(clean, ',', parts);
   for(int i=0; i<n; i++)
   {
      int id = -1;
      if(!FXAI_AuditScenarioIdFromName(parts[i], id)) continue;
      bool exists = false;
      for(int j=0; j<ArraySize(scenario_ids); j++)
      {
         if(scenario_ids[j] == id) { exists = true; break; }
      }
      if(exists) continue;
      int sz = ArraySize(scenario_ids);
      ArrayResize(scenario_ids, sz + 1);
      scenario_ids[sz] = id;
   }

   if(ArraySize(scenario_ids) <= 0)
   {
      ArrayResize(scenario_ids, 4);
      scenario_ids[0] = 0;
      scenario_ids[1] = 1;
      scenario_ids[2] = 2;
      scenario_ids[3] = 4;
   }
}

bool FXAI_AuditPluginMatchesToken(CFXAIAIRegistry &registry,
                                  const int ai_id,
                                  const string token)
{
   string t = token;
   StringTrimLeft(t);
   StringTrimRight(t);
   StringToLower(t);
   if(StringLen(t) <= 0) return false;
   if(t == "all") return true;

   int num = (int)StringToInteger(t);
   if(IntegerToString(num) == t && num == ai_id)
      return true;

   CFXAIAIPlugin *plugin = registry.CreateInstance(ai_id);
   if(plugin == NULL) return false;
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(*plugin, manifest);
   string name = manifest.ai_name;
   delete plugin;
   StringToLower(name);
   return (name == t);
}

void FXAI_AuditResolvePlugins(CFXAIAIRegistry &registry,
                              int &plugin_ids[])
{
   ArrayResize(plugin_ids, 0);
   if(Audit_AllPlugins)
   {
      ArrayResize(plugin_ids, FXAI_AI_COUNT);
      for(int i=0; i<FXAI_AI_COUNT; i++) plugin_ids[i] = i;
      return;
   }

   string clean = Audit_PluginList;
   StringReplace(clean, "{", "");
   StringReplace(clean, "}", "");
   StringReplace(clean, ";", ",");
   StringReplace(clean, "|", ",");

   string parts[];
   int n = StringSplit(clean, ',', parts);
   for(int ai_id=0; ai_id<FXAI_AI_COUNT; ai_id++)
   {
      bool picked = false;
      for(int i=0; i<n; i++)
      {
         if(FXAI_AuditPluginMatchesToken(registry, ai_id, parts[i]))
         {
            picked = true;
            break;
         }
      }
      if(picked)
      {
         int sz = ArraySize(plugin_ids);
         ArrayResize(plugin_ids, sz + 1);
         plugin_ids[sz] = ai_id;
      }
   }

   if(ArraySize(plugin_ids) <= 0)
   {
      ArrayResize(plugin_ids, 1);
      plugin_ids[0] = (int)Audit_Plugin;
   }
}

bool FXAI_AuditOpenReport(int &handle)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_AUDIT_REPORT_DIR, FILE_COMMON);
   if(Audit_ResetOutput)
      FileDelete(FXAI_AUDIT_REPORT_FILE, FILE_COMMON);

   handle = FileOpen(FXAI_AUDIT_REPORT_FILE,
                     FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_WRITE | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   int file_size = (int)FileSize(handle);
   bool write_header = (Audit_ResetOutput || file_size <= 0);
   if(file_size > 0 && !Audit_ResetOutput)
      FileSeek(handle, 0, SEEK_END);
   else
      FileSeek(handle, 0, SEEK_SET);

   if(write_header && !FXAI_AuditWriteHeader(handle))
   {
      FileClose(handle);
      handle = INVALID_HANDLE;
      return false;
   }
   return true;
}

int OnInit()
{
   if(Audit_RunTensorKernelSanity)
   {
      string sanity_reason = "";
      if(!FXAI_AuditTensorKernelSelfTest(sanity_reason))
      {
         Print("FXAI audit tensor sanity failed: ", sanity_reason);
         return INIT_FAILED;
      }
   }

   if(Audit_RunPluginContractSanity)
   {
      string contract_reason = "";
      if(!FXAI_AuditPluginContractSelfTest(contract_reason))
      {
         Print("FXAI audit plugin contract sanity failed: ", contract_reason);
         return INIT_FAILED;
      }
   }

   int bars = Audit_Bars;
   if(bars < 2048) bars = 2048;
   if(bars > 100000) bars = 100000;
   int horizon = PredictionTargetMinutes;
   if(horizon < 1) horizon = 1;
   if(horizon > 720) horizon = 720;

   CFXAIAIRegistry registry;
   int plugin_ids[];
   int scenario_ids[];
   FXAI_AuditResolvePlugins(registry, plugin_ids);
   FXAI_AuditParseScenarioList(Audit_ScenarioList, scenario_ids);

   int handle = INVALID_HANDLE;
   if(!FXAI_AuditOpenReport(handle))
   {
      Print("FXAI audit: failed to open report file ", FXAI_AUDIT_REPORT_FILE);
      return INIT_FAILED;
   }

   Print("FXAI audit start: plugins=", ArraySize(plugin_ids), ", scenarios=", ArraySize(scenario_ids), ", bars=", bars, ", horizon=", horizon);

   int failures = 0;
   for(int p=0; p<ArraySize(plugin_ids); p++)
   {
      int ai_id = plugin_ids[p];
      for(int s=0; s<ArraySize(scenario_ids); s++)
      {
         FXAIAuditScenarioSpec spec;
         FXAI_AuditFillScenarioSpec(scenario_ids[s], spec);

         FXAIAuditScenarioMetrics metrics;
         bool ok = FXAI_AuditRunScenario(registry,
                                         ai_id,
                                         spec,
                                         bars,
                                         horizon,
                                         (ulong)Audit_Seed,
                                         Audit_Normalization,
                                         metrics);
         if(!ok)
         {
            failures++;
            Print("FXAI audit failed: ai_id=", ai_id, ", scenario=", spec.name);
            if(Audit_StopOnFailure)
            {
               FileClose(handle);
               return INIT_FAILED;
            }
            continue;
         }

         FXAI_AuditWriteMetrics(handle, metrics);
         Print("FXAI audit scenario: ", metrics.ai_name,
               " / ", metrics.scenario,
               " score=", DoubleToString(metrics.score, 2),
               " invalid=", metrics.invalid_preds,
               " skip=", DoubleToString(metrics.skip_ratio, 3),
               " active=", DoubleToString(metrics.active_ratio, 3),
               " drift=", DoubleToString(metrics.conf_drift, 3),
               " reset=", DoubleToString(metrics.reset_delta, 3),
               " seq=", DoubleToString(metrics.sequence_delta, 3),
               " wf_pbo=", DoubleToString(metrics.wf_pbo, 3),
               " wf_dsr=", DoubleToString(metrics.wf_dsr, 3),
               " wf_pass=", DoubleToString(metrics.wf_pass_rate, 3),
               " flags=", metrics.issue_flags);
         if(metrics.issue_flags != 0) failures++;
      }
   }

   FileClose(handle);
   Print("FXAI audit done: failures=", failures,
         ", report=FILE_COMMON/", FXAI_AUDIT_REPORT_FILE);

   if(MQLInfoInteger(MQL_TESTER)) TesterStop();
   else ExpertRemove();
   return INIT_SUCCEEDED;
}

void OnTick()
{
}
