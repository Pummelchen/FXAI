void FXAI_ClearContextV4(FXAIAIContextV4 &ctx)
{
   ctx.api_version = FXAI_API_VERSION_V4;
   ctx.regime_id = 0;
   ctx.session_bucket = 0;
   ctx.horizon_minutes = 1;
   ctx.feature_schema_id = FXAI_SCHEMA_FULL;
   ctx.normalization_method_id = FXAI_NORM_EXISTING;
   ctx.sequence_bars = 1;
   ctx.cost_points = 0.0;
   ctx.min_move_points = 0.0;
   ctx.point_value = 0.0;
   ctx.domain_hash = FXAI_SymbolHash01(_Symbol);
   ctx.sample_time = 0;
}

bool FXAI_ValidateContextV4(const FXAIAIContextV4 &ctx,
                            string &reason)
{
   if(ctx.api_version != FXAI_API_VERSION_V4)
   {
      reason = "ctx.api_version";
      return false;
   }
   if(ctx.regime_id < 0 || ctx.regime_id >= FXAI_PLUGIN_REGIME_BUCKETS)
   {
      reason = "ctx.regime_id";
      return false;
   }
   if(ctx.session_bucket < 0 || ctx.session_bucket >= FXAI_PLUGIN_SESSION_BUCKETS)
   {
      reason = "ctx.session_bucket";
      return false;
   }
   if(ctx.horizon_minutes <= 0)
   {
      reason = "ctx.horizon_minutes";
      return false;
   }
   if(ctx.feature_schema_id < FXAI_SCHEMA_FULL || ctx.feature_schema_id > FXAI_SCHEMA_CONTEXTUAL)
   {
      reason = "ctx.feature_schema_id";
      return false;
   }
   if(ctx.normalization_method_id < 0 || ctx.normalization_method_id >= FXAI_NORM_METHOD_COUNT)
   {
      reason = "ctx.normalization_method_id";
      return false;
   }
   if(ctx.sequence_bars <= 0 || ctx.sequence_bars > FXAI_MAX_SEQUENCE_BARS)
   {
      reason = "ctx.sequence_bars";
      return false;
   }
   if(!MathIsValidNumber(ctx.cost_points) || ctx.cost_points < 0.0)
   {
      reason = "ctx.cost_points";
      return false;
   }
   if(!MathIsValidNumber(ctx.min_move_points) || ctx.min_move_points < 0.0)
   {
      reason = "ctx.min_move_points";
      return false;
   }
   if(!MathIsValidNumber(ctx.domain_hash) || ctx.domain_hash < 0.0 || ctx.domain_hash > 1.0)
   {
      reason = "ctx.domain_hash";
      return false;
   }
   if(!MathIsValidNumber(ctx.point_value) || ctx.point_value <= 0.0)
   {
      reason = "ctx.point_value";
      return false;
   }
   reason = "";
   return true;
}

bool FXAI_ValidateWindowPayloadV4(const double &x_window[][FXAI_AI_WEIGHTS],
                                  const int window_size,
                                  string &reason)
{
   if(window_size < 0 || window_size > FXAI_MAX_SEQUENCE_BARS)
   {
      reason = "req.window_size";
      return false;
   }

   for(int b=0; b<window_size; b++)
   {
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         if(!MathIsValidNumber(x_window[b][k]))
         {
            reason = "req.x_window";
            return false;
         }
      }
   }

   reason = "";
   return true;
}

bool FXAI_ValidateManifestContextCompatibilityV4(const FXAIAIManifestV4 &manifest,
                                                 const FXAIAIContextV4 &ctx,
                                                 string &reason)
{
   if(ctx.horizon_minutes < manifest.min_horizon_minutes ||
      ctx.horizon_minutes > manifest.max_horizon_minutes)
   {
      reason = "ctx.horizon_manifest";
      return false;
   }
   if(ctx.sequence_bars < manifest.min_sequence_bars ||
      ctx.sequence_bars > manifest.max_sequence_bars)
   {
      reason = "ctx.sequence_manifest";
      return false;
   }

   bool expects_window = (FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT) ||
                          FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL));
   if(!expects_window && ctx.sequence_bars != 1)
   {
      reason = "ctx.sequence_unexpected";
      return false;
   }

   reason = "";
   return true;
}

void FXAI_ClearPredictRequest(FXAIAIPredictRequestV4 &req)
{
   req.valid = false;
   FXAI_ClearContextV4(req.ctx);
   req.window_size = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = 0.0;
}

void FXAI_ClearTrainRequest(FXAIAITrainRequestV4 &req)
{
   req.valid = false;
   FXAI_ClearContextV4(req.ctx);
   req.label_class = (int)FXAI_LABEL_SKIP;
   req.move_points = 0.0;
   req.sample_weight = 0.0;
   req.mfe_points = 0.0;
   req.mae_points = 0.0;
   req.time_to_hit_frac = 1.0;
   req.path_flags = 0;
   req.path_risk = 0.0;
   req.fill_risk = 0.0;
   req.masked_step_target = 0.0;
   req.next_vol_target = 0.0;
   req.regime_shift_target = 0.0;
   req.context_lead_target = 0.5;
   req.window_size = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = 0.0;
}

bool FXAI_ValidatePredictRequestV4(const FXAIAIPredictRequestV4 &req,
                                   string &reason)
{
   if(!req.valid)
   {
      reason = "req.valid";
      return false;
   }
   if(!FXAI_ValidateContextV4(req.ctx, reason))
      return false;
   if(!FXAI_ValidateWindowPayloadV4(req.x_window, req.window_size, reason))
      return false;
   if(req.window_size > MathMax(req.ctx.sequence_bars - 1, 0))
   {
      reason = "req.window_size_ctx";
      return false;
   }
   if(req.ctx.sequence_bars > 1 && req.window_size <= 0)
   {
      reason = "req.window_payload";
      return false;
   }
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      if(!MathIsValidNumber(req.x[k]))
      {
         reason = "req.x";
         return false;
      }
   }
   reason = "";
   return true;
}

bool FXAI_ValidateTrainRequestV4(const FXAIAITrainRequestV4 &req,
                                 string &reason)
{
   if(!req.valid)
   {
      reason = "req.valid";
      return false;
   }
   if(!FXAI_ValidateContextV4(req.ctx, reason))
      return false;
   if(!FXAI_ValidateWindowPayloadV4(req.x_window, req.window_size, reason))
      return false;
   if(req.window_size > MathMax(req.ctx.sequence_bars - 1, 0))
   {
      reason = "req.window_size_ctx";
      return false;
   }
   if(req.ctx.sequence_bars > 1 && req.window_size <= 0)
   {
      reason = "req.window_payload";
      return false;
   }
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      if(!MathIsValidNumber(req.x[k]))
      {
         reason = "req.x";
         return false;
      }
   }
   if(req.label_class < (int)FXAI_LABEL_SELL || req.label_class > (int)FXAI_LABEL_SKIP)
   {
      reason = "req.label_class";
      return false;
   }
   if(!MathIsValidNumber(req.move_points))
   {
      reason = "req.move_points";
      return false;
   }
   if(!MathIsValidNumber(req.sample_weight) || req.sample_weight < 0.0)
   {
      reason = "req.sample_weight";
      return false;
   }
   if(!MathIsValidNumber(req.mfe_points) || req.mfe_points < 0.0 ||
      !MathIsValidNumber(req.mae_points) || req.mae_points < 0.0)
   {
      reason = "req.path_excursions";
      return false;
   }
   if(!MathIsValidNumber(req.time_to_hit_frac) || req.time_to_hit_frac < 0.0 || req.time_to_hit_frac > 1.0)
   {
      reason = "req.time_to_hit_frac";
      return false;
   }
   if(!MathIsValidNumber(req.path_risk) || req.path_risk < 0.0 || req.path_risk > 1.0)
   {
      reason = "req.path_risk";
      return false;
   }
   if(!MathIsValidNumber(req.fill_risk) || req.fill_risk < 0.0 || req.fill_risk > 1.0)
   {
      reason = "req.fill_risk";
      return false;
   }
   if(!MathIsValidNumber(req.masked_step_target))
   {
      reason = "req.masked_step_target";
      return false;
   }
   if(!MathIsValidNumber(req.next_vol_target) || req.next_vol_target < 0.0)
   {
      reason = "req.next_vol_target";
      return false;
   }
   if(!MathIsValidNumber(req.regime_shift_target) || req.regime_shift_target < 0.0 || req.regime_shift_target > 1.0)
   {
      reason = "req.regime_shift_target";
      return false;
   }
   if(!MathIsValidNumber(req.context_lead_target) || req.context_lead_target < 0.0 || req.context_lead_target > 1.0)
   {
      reason = "req.context_lead_target";
      return false;
   }
   reason = "";
   return true;
}

void FXAI_SetTrainRequestPathTargets(FXAIAITrainRequestV4 &req,
                                     const double mfe_points,
                                     const double mae_points,
                                     const double time_to_hit_frac,
                                     const int path_flags,
                                     const double spread_stress_points)
{
   req.mfe_points = MathMax(MathAbs(mfe_points), 0.0);
   req.mae_points = MathMax(MathAbs(mae_points), 0.0);
   req.time_to_hit_frac = FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
   req.path_flags = path_flags;
   req.path_risk = FXAI_PathRiskFromTargets(req.mfe_points,
                                            req.mae_points,
                                            req.ctx.min_move_points,
                                            req.time_to_hit_frac,
                                            req.path_flags);
   req.fill_risk = FXAI_FillRiskFromTargets(spread_stress_points,
                                            req.ctx.min_move_points,
                                            req.ctx.cost_points);
}

void FXAI_SetTrainRequestAuxTargets(FXAIAITrainRequestV4 &req,
                                    const double masked_step_target,
                                    const double next_vol_target,
                                    const double regime_shift_target,
                                    const double context_lead_target)
{
   req.masked_step_target = masked_step_target;
   req.next_vol_target = MathMax(next_vol_target, 0.0);
   req.regime_shift_target = FXAI_Clamp(regime_shift_target, 0.0, 1.0);
   req.context_lead_target = FXAI_Clamp(context_lead_target, 0.0, 1.0);
}

void FXAI_CopyWindowPayload(const double &src[][FXAI_AI_WEIGHTS], const int src_size, double &dst[][FXAI_AI_WEIGHTS], int &dst_size)
{
   dst_size = src_size;
   if(dst_size < 0) dst_size = 0;
   if(dst_size > FXAI_MAX_SEQUENCE_BARS) dst_size = FXAI_MAX_SEQUENCE_BARS;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         dst[b][k] = (b < dst_size ? src[b][k] : 0.0);
}
