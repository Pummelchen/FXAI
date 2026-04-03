bool FXAI_ValidateNativePluginAPI()
{
   if(!FXAI_FeatureRegistrySelfTest())
   {
      Print("FXAI error: feature registry self-test failed.");
      return false;
   }

   double x_dummy[FXAI_AI_WEIGHTS];
   for(int k=0; k<FXAI_AI_WEIGHTS; k++) x_dummy[k] = 0.0;
   x_dummy[0] = 1.0;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
      {
         Print("FXAI error: API v4 plugin missing at id=", ai_idx);
         return false;
      }

      FXAIAIHyperParams hp;
      FXAI_GetModelHyperParams(ai_idx, hp);
      plugin.EnsureInitialized(hp);

      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);
      string reason = "";
      if(!FXAI_ValidateManifestV4(manifest, reason))
      {
         Print("FXAI error: API v4 manifest invalid. model=", plugin.AIName(),
               " id=", ai_idx,
               " reason=", reason);
         return false;
      }

      FXAIAIPredictRequestV4 req_v4;
      FXAI_ClearPredictRequest(req_v4);
      req_v4.valid = true;
      req_v4.ctx.api_version = FXAI_API_VERSION_V4;
      req_v4.ctx.regime_id = 0;
      req_v4.ctx.session_bucket = FXAI_DeriveSessionBucket(TimeCurrent());
      req_v4.ctx.horizon_minutes = 5;
      req_v4.ctx.feature_schema_id = manifest.feature_schema_id;
      req_v4.ctx.normalization_method_id = (int)AI_FeatureNormalization;
      req_v4.ctx.sequence_bars = FXAI_GetPluginSequenceBars(*plugin, req_v4.ctx.horizon_minutes);
      req_v4.ctx.cost_points = 0.5;
      req_v4.ctx.min_move_points = 0.8;
      req_v4.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
      req_v4.ctx.domain_hash = FXAI_SymbolHash01(_Symbol);
      req_v4.ctx.sample_time = TimeCurrent();
      for(int kk=0; kk<FXAI_AI_WEIGHTS; kk++)
         req_v4.x[kk] = x_dummy[kk];
      FXAI_FillComplianceWindow(req_v4.ctx, req_v4.x, req_v4.x_window, req_v4.window_size);
      FXAI_ApplyFeatureSchemaToPayloadEx(manifest.feature_schema_id,
                                       manifest.feature_groups_mask,
                                       req_v4.ctx.sequence_bars,
                                       req_v4.x_window,
                                       req_v4.window_size,
                                       req_v4.x);

      FXAIAIPredictionV4 pred_v4;
      if(!plugin.Predict(req_v4, hp, pred_v4))
      {
         Print("FXAI error: API v4 predict failed. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }

      if(!FXAI_ValidatePredictionV4(pred_v4, reason))
      {
         Print("FXAI error: API v4 prediction invalid. model=", plugin.AIName(),
               " id=", ai_idx,
               " reason=", reason);
         return false;
      }

      if(!plugin.SelfTest())
      {
         Print("FXAI error: API v4 self-test failed. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }
   }

   return true;
}

