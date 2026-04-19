#ifndef __FXAI_PLUGIN_CONTRACT_SUITE_MQH__
#define __FXAI_PLUGIN_CONTRACT_SUITE_MQH__

#include "..\TestHarness\test_harness.mqh"
#include "..\..\API\api.mqh"

void FXAI_PluginContractDefaultHyperParams(FXAIAIHyperParams &hp)
{
   hp.lr = 0.01;
   hp.l2 = 0.0001;
   hp.ftrl_alpha = 0.05;
   hp.ftrl_beta = 1.0;
   hp.ftrl_l1 = 0.0;
   hp.ftrl_l2 = 0.0001;
   hp.pa_c = 0.5;
   hp.pa_margin = 1.0;
   hp.xgb_lr = 0.05;
   hp.xgb_l2 = 0.0001;
   hp.xgb_split = 0.5;
   hp.mlp_lr = 0.01;
   hp.mlp_l2 = 0.0001;
   hp.mlp_init = 0.05;
   hp.quantile_lr = 0.01;
   hp.quantile_l2 = 0.0001;
   hp.enhash_lr = 0.01;
   hp.enhash_l1 = 0.0;
   hp.enhash_l2 = 0.0001;
   hp.tcn_layers = 2.0;
   hp.tcn_kernel = 3.0;
   hp.tcn_dilation_base = 2.0;
}

void FXAI_PluginContractBuildPredictRequest(const FXAIAIManifestV4 &manifest,
                                            FXAIAIPredictRequestV4 &req)
{
   FXAI_ClearPredictRequest(req);
   req.valid = true;
   req.ctx.api_version = FXAI_API_VERSION_V4;
   req.ctx.regime_id = 0;
   req.ctx.session_bucket = 0;
   req.ctx.horizon_minutes = MathMax(manifest.min_horizon_minutes, 1);
   req.ctx.feature_schema_id = manifest.feature_schema_id;
   req.ctx.normalization_method_id = FXAI_NORM_EXISTING;
   req.ctx.sequence_bars = MathMax(manifest.min_sequence_bars, 1);
   req.ctx.cost_points = 0.0;
   req.ctx.min_move_points = 0.0;
   req.ctx.point_value = 0.0001;
   req.ctx.domain_hash = 0.5;
   req.ctx.sample_time = D'2024.01.02 00:00';
   req.window_size = MathMax(req.ctx.sequence_bars - 1, 0);

   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = (k == 0 ? 1.0 : 0.01 * (double)(k + 1));
   for(int b=0; b<req.window_size; b++)
   {
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = (k == 0 ? 1.0 : 0.005 * (double)(b + 1) * (double)(k + 1));
   }
}

void FXAI_PluginContractBuildSyntheticSeries(datetime &time_arr[],
                                             double &open_arr[],
                                             double &high_arr[],
                                             double &low_arr[],
                                             double &close_arr[])
{
   int n = 32;
   ArrayResize(time_arr, n);
   ArrayResize(open_arr, n);
   ArrayResize(high_arr, n);
   ArrayResize(low_arr, n);
   ArrayResize(close_arr, n);
   datetime base_time = D'2024.01.02 00:00';
   for(int i=0; i<n; i++)
   {
      double base = 1.1000 + 0.0002 * (double)i;
      time_arr[i] = base_time + (datetime)(60 * i);
      open_arr[i] = base;
      close_arr[i] = base + (i % 2 == 0 ? 0.0001 : -0.00005);
      high_arr[i] = MathMax(open_arr[i], close_arr[i]) + 0.00015;
      low_arr[i] = MathMin(open_arr[i], close_arr[i]) - 0.00015;
   }
}

bool FXAI_PluginContractCheckFinitePrediction(const FXAIAIPredictionV4 &prediction,
                                              string &reason)
{
   double prob_sum = 0.0;
   for(int c=0; c<3; c++)
   {
      if(!MathIsValidNumber(prediction.class_probs[c]) ||
         prediction.class_probs[c] < 0.0 ||
         prediction.class_probs[c] > 1.0)
      {
         reason = "class_probs";
         return false;
      }
      prob_sum += prediction.class_probs[c];
   }
   if(MathAbs(prob_sum - 1.0) > 1e-6)
   {
      reason = "probability_sum";
      return false;
   }
   if(!MathIsValidNumber(prediction.move_mean_points) ||
      !MathIsValidNumber(prediction.move_q25_points) ||
      !MathIsValidNumber(prediction.move_q50_points) ||
      !MathIsValidNumber(prediction.move_q75_points) ||
      !MathIsValidNumber(prediction.mfe_mean_points) ||
      !MathIsValidNumber(prediction.mae_mean_points) ||
      !MathIsValidNumber(prediction.hit_time_frac) ||
      !MathIsValidNumber(prediction.path_risk) ||
      !MathIsValidNumber(prediction.fill_risk) ||
      !MathIsValidNumber(prediction.confidence) ||
      !MathIsValidNumber(prediction.reliability))
   {
      reason = "prediction_fields";
      return false;
   }
   reason = "";
   return true;
}

bool FXAI_PluginContractTestRegistryLifecycle(string &reason)
{
   CFXAIAIRegistry registry;
   if(!registry.Initialize())
   {
      reason = "registry_initialize";
      return false;
   }
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      if(registry.Get(ai) == NULL)
      {
         reason = "registry_missing_" + IntegerToString(ai);
         return false;
      }
   }
   registry.ResetAll();
   registry.Release();
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      if(registry.Get(ai) != NULL)
      {
         reason = "registry_release_" + IntegerToString(ai);
         return false;
      }
   }
   reason = "";
   return true;
}

bool FXAI_PluginContractTestManifestAndSelfTest(string &reason)
{
   CFXAIAIRegistry registry;
   if(!registry.Initialize())
   {
      reason = "registry_initialize";
      return false;
   }

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      CFXAIAIPlugin *plugin = registry.Get(ai);
      if(plugin == NULL)
      {
         reason = "plugin_null_" + IntegerToString(ai);
         return false;
      }
      FXAIAIManifestV4 manifest;
      plugin.DescribeResolved(manifest);
      if(plugin.AIId() != ai ||
         manifest.api_version != FXAI_API_VERSION_V4 ||
         manifest.ai_id != ai ||
         StringLen(plugin.AIName()) <= 0 ||
         StringLen(manifest.ai_name) <= 0 ||
         manifest.feature_schema_id <= 0 ||
         manifest.feature_groups_mask == 0 ||
         manifest.min_horizon_minutes <= 0 ||
         manifest.max_horizon_minutes < manifest.min_horizon_minutes ||
         manifest.min_sequence_bars <= 0 ||
         manifest.max_sequence_bars < manifest.min_sequence_bars ||
         StringLen(plugin.PersistentStateDepthTag()) <= 0 ||
         StringLen(plugin.PersistentStateCoverageTag()) <= 0 ||
         !plugin.SelfTest())
      {
         reason = "manifest_selftest_" + IntegerToString(ai);
         return false;
      }
   }

   reason = "";
   return true;
}

bool FXAI_PluginContractTestPredictContract(string &reason)
{
   CFXAIAIRegistry registry;
   if(!registry.Initialize())
   {
      reason = "registry_initialize";
      return false;
   }

   FXAIAIHyperParams hp;
   FXAI_PluginContractDefaultHyperParams(hp);

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      CFXAIAIPlugin *plugin = registry.Get(ai);
      if(plugin == NULL)
      {
         reason = "plugin_null_" + IntegerToString(ai);
         return false;
      }

      FXAIAIManifestV4 manifest;
      plugin.DescribeResolved(manifest);
      FXAIAIPredictRequestV4 req;
      FXAI_PluginContractBuildPredictRequest(manifest, req);
      FXAIAIPredictionV4 prediction;
      for(int c=0; c<3; c++)
         prediction.class_probs[c] = 0.0;
      prediction.move_mean_points = 0.0;
      prediction.move_q25_points = 0.0;
      prediction.move_q50_points = 0.0;
      prediction.move_q75_points = 0.0;
      prediction.mfe_mean_points = 0.0;
      prediction.mae_mean_points = 0.0;
      prediction.hit_time_frac = 1.0;
      prediction.path_risk = 0.0;
      prediction.fill_risk = 0.0;
      prediction.confidence = 0.0;
      prediction.reliability = 0.0;

      plugin.Reset();
      plugin.Predict(req, hp, prediction);

      string pred_reason = "";
      if(!FXAI_PluginContractCheckFinitePrediction(prediction, pred_reason))
      {
         reason = "predict_" + IntegerToString(ai) + "_" + pred_reason;
         return false;
      }
   }

   reason = "";
   return true;
}

bool FXAI_PluginContractTestPersistenceRoundTrip(string &reason)
{
   CFXAIAIRegistry registry;
   if(!registry.Initialize())
   {
      reason = "registry_initialize";
      return false;
   }

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      CFXAIAIPlugin *plugin = registry.Get(ai);
      if(plugin == NULL)
      {
         reason = "plugin_null_" + IntegerToString(ai);
         return false;
      }
      if(!plugin.SupportsPersistentState())
         continue;

      string file_name = plugin.PersistentStateFile("UNITTEST");
      FileDelete(file_name, FILE_COMMON);
      if(!plugin.SaveStateFile(file_name))
      {
         reason = "save_state_" + IntegerToString(ai);
         return false;
      }
      plugin.ResetState(0, 0);
      if(!plugin.LoadStateFile(file_name))
      {
         FileDelete(file_name, FILE_COMMON);
         reason = "load_state_" + IntegerToString(ai);
         return false;
      }
      FileDelete(file_name, FILE_COMMON);
   }

   reason = "";
   return true;
}

bool FXAI_PluginContractTestSyntheticSeriesSupport(string &reason)
{
   CFXAIAIRegistry registry;
   if(!registry.Initialize())
   {
      reason = "registry_initialize";
      return false;
   }

   datetime time_arr[];
   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   FXAI_PluginContractBuildSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      CFXAIAIPlugin *plugin = registry.Get(ai);
      if(plugin == NULL)
      {
         reason = "plugin_null_" + IntegerToString(ai);
         return false;
      }
      if(!plugin.SupportsSyntheticSeries())
         continue;
      if(!plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr))
      {
         reason = "synthetic_series_" + IntegerToString(ai);
         return false;
      }
      plugin.ClearSyntheticSeries();
   }

   reason = "";
   return true;
}

void FXAI_PluginContractRunSuite(FXAITestSuiteResult &suite)
{
   FXAI_TestSuiteReset(suite, "plugin_contracts");

   string reason = "";
   bool passed = FXAI_PluginContractTestRegistryLifecycle(reason);
   FXAI_TestSuiteAddCase(suite, "registry_lifecycle", passed, reason);

   reason = "";
   passed = FXAI_PluginContractTestManifestAndSelfTest(reason);
   FXAI_TestSuiteAddCase(suite, "manifest_and_selftest", passed, reason);

   reason = "";
   passed = FXAI_PluginContractTestPredictContract(reason);
   FXAI_TestSuiteAddCase(suite, "predict_request_contract", passed, reason);

   reason = "";
   passed = FXAI_PluginContractTestPersistenceRoundTrip(reason);
   FXAI_TestSuiteAddCase(suite, "persistent_state_roundtrip", passed, reason);

   reason = "";
   passed = FXAI_PluginContractTestSyntheticSeriesSupport(reason);
   FXAI_TestSuiteAddCase(suite, "synthetic_series_contract", passed, reason);
}

#endif // __FXAI_PLUGIN_CONTRACT_SUITE_MQH__
