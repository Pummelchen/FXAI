void FXAI_FillComplianceContext(CFXAIAIPlugin &plugin,
                                FXAIAIContextV4 &ctx,
                                const double cost_points,
                                const datetime sample_time,
                                const int regime_id,
                                const int horizon_minutes)
{
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   ctx.api_version = FXAI_API_VERSION_V4;
   ctx.regime_id = regime_id;
   ctx.session_bucket = FXAI_DeriveSessionBucket(sample_time);
   ctx.horizon_minutes = horizon_minutes;
   ctx.feature_schema_id = manifest.feature_schema_id;
   ctx.normalization_method_id = (int)AI_FeatureNormalization;
   ctx.sequence_bars = FXAI_GetPluginSequenceBars(plugin, horizon_minutes);
   ctx.min_move_points = MathMax(cost_points + 0.30, 0.50);
   ctx.cost_points = MathMax(cost_points, 0.0);
   ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
   ctx.domain_hash = FXAI_SymbolHash01(_Symbol);
   ctx.sample_time = sample_time;
}

void FXAI_FillComplianceWindow(const FXAIAIContextV4 &ctx,
                               const double &x[],
                               double &x_window[][FXAI_AI_WEIGHTS],
                               int &window_size)
{
   FXAI_ClearInputWindow(x_window, window_size);
   int seq = ctx.sequence_bars;
   if(seq < 1) seq = 1;
   if(seq > FXAI_MAX_SEQUENCE_BARS) seq = FXAI_MAX_SEQUENCE_BARS;
   for(int b=0; b<seq-1; b++)
   {
      double decay = 1.0 - 0.08 * (double)(b + 1);
      if(decay < 0.30) decay = 0.30;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_window[b][k] = x[k] * decay;
      window_size++;
   }
}

void FXAI_FillComplianceTrainRequest(CFXAIAIPlugin &plugin,
                                     FXAIAITrainRequestV4 &req,
                                     const int label_class,
                                     const double move_points,
                                     const double cost_points,
                                     const double v1,
                                     const double v2,
                                     const double v3,
                                     const datetime sample_time,
                                     const int regime_id,
                                     const int horizon_minutes)
{
   FXAI_ClearTrainRequest(req);
   req.valid = true;
   FXAI_FillComplianceContext(plugin, req.ctx, cost_points, sample_time, regime_id, horizon_minutes);
   req.label_class = label_class;
   req.move_points = move_points;
   req.sample_weight = 1.0;
   double mfe_points = MathMax(MathAbs(move_points), MathMax(req.ctx.min_move_points, 0.10));
   double mae_points = (label_class == (int)FXAI_LABEL_SKIP ? mfe_points : 0.35 * mfe_points);
   double hit_time_frac = (label_class == (int)FXAI_LABEL_SKIP ? 0.75 : 0.40);
   int path_flags = (label_class == (int)FXAI_LABEL_SKIP ? 1 : 0);
   FXAI_SetTrainRequestPathTargets(req,
                                   mfe_points,
                                   mae_points,
                                   hit_time_frac,
                                   path_flags,
                                   0.0);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   req.x[0] = 1.0;
   req.x[1] = v1;
   req.x[2] = v2;
   req.x[3] = v3;
   req.x[4] = 0.35 * v2;
   req.x[5] = 0.25 * v3;
   req.x[6] = 0.15 * v1;
   req.x[7] = req.ctx.cost_points;
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   FXAI_FillComplianceWindow(req.ctx, req.x, req.x_window, req.window_size);
   FXAI_ApplyPayloadTransformPipelineEx(manifest.feature_schema_id,
                                        manifest.feature_groups_mask,
                                        req.ctx.normalization_method_id,
                                        req.ctx.horizon_minutes,
                                        req.ctx.sequence_bars,
                                        req.x_window,
                                        req.window_size,
                                        req.x);
}

void FXAI_FillCompliancePredictRequest(CFXAIAIPlugin &plugin,
                                       FXAIAIPredictRequestV4 &req,
                                       const double cost_points,
                                       const double v1,
                                       const double v2,
                                       const double v3,
                                       const datetime sample_time,
                                       const int regime_id,
                                       const int horizon_minutes)
{
   FXAI_ClearPredictRequest(req);
   req.valid = true;
   FXAI_FillComplianceContext(plugin, req.ctx, cost_points, sample_time, regime_id, horizon_minutes);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   req.x[0] = 1.0;
   req.x[1] = v1;
   req.x[2] = v2;
   req.x[3] = v3;
   req.x[4] = 0.35 * v2;
   req.x[5] = 0.25 * v3;
   req.x[6] = 0.15 * v1;
   req.x[7] = req.ctx.cost_points;
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   FXAI_FillComplianceWindow(req.ctx, req.x, req.x_window, req.window_size);
   FXAI_ApplyPayloadTransformPipelineEx(manifest.feature_schema_id,
                                        manifest.feature_groups_mask,
                                        req.ctx.normalization_method_id,
                                        req.ctx.horizon_minutes,
                                        req.ctx.sequence_bars,
                                        req.x_window,
                                        req.window_size,
                                        req.x);
}

bool FXAI_ValidatePredictionOutput(const CFXAIAIPlugin &plugin,
                                   const FXAIAIPredictionV4 &pred,
                                   const string tag)
{
   string reason = "";
   if(!FXAI_ValidatePredictionV4(pred, reason))
   {
      Print("FXAI compliance error: invalid prediction. model=", plugin.AIName(),
            " tag=", tag, " reason=", reason);
      return false;
   }

   double s = pred.class_probs[(int)FXAI_LABEL_SELL]
            + pred.class_probs[(int)FXAI_LABEL_BUY]
            + pred.class_probs[(int)FXAI_LABEL_SKIP];
   for(int c=0; c<3; c++)
   {
      if(!MathIsValidNumber(pred.class_probs[c]) || pred.class_probs[c] < 0.0 || pred.class_probs[c] > 1.0)
      {
         Print("FXAI compliance error: probability range invalid. model=", plugin.AIName(),
               " tag=", tag, " class=", c, " value=", DoubleToString(pred.class_probs[c], 6));
         return false;
      }
   }

   if(!MathIsValidNumber(pred.move_mean_points) || pred.move_mean_points < 0.0)
   {
      Print("FXAI compliance error: expected move invalid. model=", plugin.AIName(),
            " tag=", tag, " ev=", DoubleToString(pred.move_mean_points, 6));
      return false;
   }

   return true;
}

int FXAI_ComplianceSequenceBars(const FXAIAIManifestV4 &manifest)
{
   if(manifest.max_sequence_bars <= 1)
      return 1;
   int seq = manifest.max_sequence_bars;
   if(seq < manifest.min_sequence_bars)
      seq = manifest.min_sequence_bars;
   if(seq < 2) seq = 2;
   if(seq > 16) seq = 16;
   return seq;
}

int FXAI_ComplianceHorizon(const FXAIAIManifestV4 &manifest,
                           const int desired_horizon)
{
   int h = desired_horizon;
   if(h < manifest.min_horizon_minutes) h = manifest.min_horizon_minutes;
   if(h > manifest.max_horizon_minutes) h = manifest.max_horizon_minutes;
   if(h < 1) h = 1;
   return h;
}

double FXAI_PredictionDistance(const FXAIAIPredictionV4 &a,
                               const FXAIAIPredictionV4 &b)
{
   double d = 0.0;
   for(int c=0; c<3; c++)
      d += MathAbs(a.class_probs[c] - b.class_probs[c]);
   d += 0.05 * MathAbs(a.move_mean_points - b.move_mean_points);
   d += 0.02 * MathAbs(a.move_q25_points - b.move_q25_points);
   d += 0.02 * MathAbs(a.move_q50_points - b.move_q50_points);
   d += 0.02 * MathAbs(a.move_q75_points - b.move_q75_points);
   d += 0.02 * MathAbs(a.mfe_mean_points - b.mfe_mean_points);
   d += 0.02 * MathAbs(a.mae_mean_points - b.mae_mean_points);
   d += 0.05 * MathAbs(a.hit_time_frac - b.hit_time_frac);
   d += 0.05 * MathAbs(a.path_risk - b.path_risk);
   d += 0.05 * MathAbs(a.fill_risk - b.fill_risk);
   d += 0.10 * MathAbs(a.confidence - b.confidence);
   d += 0.10 * MathAbs(a.reliability - b.reliability);
   return d;
}

bool FXAI_RunStateResetCompliance(CFXAIAIPlugin &plugin,
                                  const FXAIAIManifestV4 &manifest,
                                  const FXAIAIHyperParams &hp,
                                  const datetime now_t)
{
   if(!FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL))
      return true;

   int seq = FXAI_ComplianceSequenceBars(manifest);
   int horizon = FXAI_ComplianceHorizon(manifest, 5);

   FXAIAITrainRequestV4 train_req;
   FXAI_FillComplianceTrainRequest(plugin, train_req, (int)FXAI_LABEL_BUY, 5.5, 0.8,
                                   0.80, 0.45, 0.20, now_t - 90, 2, horizon);
   train_req.ctx.sequence_bars = seq;

   FXAIAIPredictRequestV4 pred_req;
   FXAI_FillCompliancePredictRequest(plugin, pred_req, 0.8, 0.80, 0.45, 0.20, now_t - 30, 2, horizon);
   pred_req.ctx.sequence_bars = seq;

   for(int i=0; i<6; i++)
      FXAI_TrainViaV4(plugin, train_req, hp);

   FXAIAIPredictionV4 pred_before, pred_after_reset_a, pred_after_reset_b;
   FXAI_PredictViaV4(plugin, pred_req, hp, pred_before);

   plugin.ResetState((int)FXAI_RESET_SESSION_CHANGE, now_t);
   if(!FXAI_PredictViaV4(plugin, pred_req, hp, pred_after_reset_a) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_after_reset_a, "reset_a"))
      return false;

   plugin.ResetState((int)FXAI_RESET_SESSION_CHANGE, now_t + 60);
   if(!FXAI_PredictViaV4(plugin, pred_req, hp, pred_after_reset_b) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_after_reset_b, "reset_b"))
      return false;

   double drift = FXAI_PredictionDistance(pred_after_reset_a, pred_after_reset_b);
   if(drift > 1e-4)
   {
      Print("FXAI compliance error: reset not idempotent. model=", plugin.AIName(),
            " drift=", DoubleToString(drift, 8));
      return false;
   }

   return true;
}

bool FXAI_RunSequenceWindowCompliance(CFXAIAIPlugin &plugin,
                                      const FXAIAIManifestV4 &manifest,
                                      const FXAIAIHyperParams &hp,
                                      const datetime now_t)
{
   bool wants_window = FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT);
   if(!wants_window)
      return (manifest.min_sequence_bars == 1 && manifest.max_sequence_bars == 1);

   int seq = FXAI_ComplianceSequenceBars(manifest);
   int horizon = FXAI_ComplianceHorizon(manifest, 13);

   FXAIAITrainRequestV4 train_req;
   FXAI_FillComplianceTrainRequest(plugin, train_req, (int)FXAI_LABEL_SELL, -5.0, 0.9,
                                   -0.75, -0.42, -0.18, now_t - 150, 3, horizon);
   train_req.ctx.sequence_bars = seq;
   FXAI_FillComplianceWindow(train_req.ctx, train_req.x, train_req.x_window, train_req.window_size);
   FXAI_ApplyPayloadTransformPipelineEx(manifest.feature_schema_id,
                                        manifest.feature_groups_mask,
                                        train_req.ctx.normalization_method_id,
                                        train_req.ctx.horizon_minutes,
                                        train_req.ctx.sequence_bars,
                                        train_req.x_window,
                                        train_req.window_size,
                                        train_req.x);

   FXAIAIPredictRequestV4 pred_req_seq, pred_req_one;
   FXAI_FillCompliancePredictRequest(plugin, pred_req_seq, 0.9, -0.75, -0.42, -0.18, now_t - 10, 3, horizon);
   FXAI_FillCompliancePredictRequest(plugin, pred_req_one, 0.9, -0.75, -0.42, -0.18, now_t - 10, 3, horizon);
   pred_req_seq.ctx.sequence_bars = seq;
   FXAI_FillComplianceWindow(pred_req_seq.ctx, pred_req_seq.x, pred_req_seq.x_window, pred_req_seq.window_size);
   FXAI_ApplyPayloadTransformPipelineEx(manifest.feature_schema_id,
                                        manifest.feature_groups_mask,
                                        pred_req_seq.ctx.normalization_method_id,
                                        pred_req_seq.ctx.horizon_minutes,
                                        pred_req_seq.ctx.sequence_bars,
                                        pred_req_seq.x_window,
                                        pred_req_seq.window_size,
                                        pred_req_seq.x);
   pred_req_one.ctx.sequence_bars = 1;
   FXAI_FillComplianceWindow(pred_req_one.ctx, pred_req_one.x, pred_req_one.x_window, pred_req_one.window_size);
   FXAI_ApplyPayloadTransformPipelineEx(manifest.feature_schema_id,
                                        manifest.feature_groups_mask,
                                        pred_req_one.ctx.normalization_method_id,
                                        pred_req_one.ctx.horizon_minutes,
                                        pred_req_one.ctx.sequence_bars,
                                        pred_req_one.x_window,
                                        pred_req_one.window_size,
                                        pred_req_one.x);

   for(int i=0; i<4; i++)
      FXAI_TrainViaV4(plugin, train_req, hp);

   FXAIAIPredictionV4 pred_seq, pred_one;
   if(!FXAI_PredictViaV4(plugin, pred_req_seq, hp, pred_seq) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_seq, "sequence"))
      return false;

   if(!FXAI_PredictViaV4(plugin, pred_req_one, hp, pred_one) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_one, "sequence_one"))
      return false;

   plugin.ResetState((int)FXAI_RESET_REGIME_CHANGE, now_t + 120);
   if(!FXAI_PredictViaV4(plugin, pred_req_seq, hp, pred_seq) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_seq, "sequence_reset"))
      return false;

   return true;
}

bool FXAI_RunCalibrationDriftCompliance(CFXAIAIPlugin &plugin,
                                        const FXAIAIManifestV4 &manifest,
                                        const FXAIAIHyperParams &hp,
                                        const datetime now_t)
{
   int seq = FXAI_ComplianceSequenceBars(manifest);
   int base_h = FXAI_ComplianceHorizon(manifest, 5);
   int alt_h = FXAI_ComplianceHorizon(manifest, 13);
   int long_h = FXAI_ComplianceHorizon(manifest, 34);

   for(int step=0; step<180; step++)
   {
      int cls = (step % 3 == 0 ? (int)FXAI_LABEL_BUY :
                 (step % 3 == 1 ? (int)FXAI_LABEL_SELL : (int)FXAI_LABEL_SKIP));
      int regime = step % 4;
      int horizon = (step % 3 == 0 ? base_h : (step % 3 == 1 ? alt_h : long_h));
      double cost = (step % 5 == 0 ? 1.4 : 0.7);
      double v1 = (cls == (int)FXAI_LABEL_BUY ? 0.85 : (cls == (int)FXAI_LABEL_SELL ? -0.85 : 0.03));
      double v2 = 0.55 * v1;
      double v3 = 0.30 * v1;
      double move = (cls == (int)FXAI_LABEL_BUY ? 5.0 + 0.04 * step :
                    (cls == (int)FXAI_LABEL_SELL ? -(5.0 + 0.04 * step) : 0.15 + 0.002 * step));

      FXAIAITrainRequestV4 train_req;
      FXAI_FillComplianceTrainRequest(plugin, train_req, cls, move, cost,
                                      v1, v2, v3,
                                      now_t - (step * 60), regime, horizon);
      train_req.ctx.sequence_bars = seq;
      FXAI_TrainViaV4(plugin, train_req, hp);

      if((step % 24) != 23)
         continue;

      FXAIAIPredictRequestV4 pred_buy, pred_sell, pred_skip;
      FXAI_FillCompliancePredictRequest(plugin, pred_buy, 0.7, 0.82, 0.46, 0.21, now_t + step, regime, base_h);
      FXAI_FillCompliancePredictRequest(plugin, pred_sell, 0.7, -0.82, -0.46, -0.21, now_t + step, regime, alt_h);
      FXAI_FillCompliancePredictRequest(plugin, pred_skip, 1.2, 0.02, 0.01, 0.00, now_t + step, regime, base_h);
      pred_buy.ctx.sequence_bars = seq;
      pred_sell.ctx.sequence_bars = seq;
      pred_skip.ctx.sequence_bars = seq;

      FXAIAIPredictionV4 out_buy, out_sell, out_skip;
      if(!FXAI_PredictViaV4(plugin, pred_buy, hp, out_buy) ||
         !FXAI_PredictViaV4(plugin, pred_sell, hp, out_sell) ||
         !FXAI_PredictViaV4(plugin, pred_skip, hp, out_skip))
         return false;

      if(!FXAI_ValidatePredictionOutput(plugin, out_buy, "drift_buy") ||
         !FXAI_ValidatePredictionOutput(plugin, out_sell, "drift_sell") ||
         !FXAI_ValidatePredictionOutput(plugin, out_skip, "drift_skip"))
         return false;

      if(out_buy.class_probs[(int)FXAI_LABEL_BUY] + 0.03 < out_buy.class_probs[(int)FXAI_LABEL_SELL])
      {
         Print("FXAI compliance error: calibration drift buy ordering failed. model=", plugin.AIName());
         return false;
      }
      if(out_sell.class_probs[(int)FXAI_LABEL_SELL] + 0.03 < out_sell.class_probs[(int)FXAI_LABEL_BUY])
      {
         Print("FXAI compliance error: calibration drift sell ordering failed. model=", plugin.AIName());
         return false;
      }
      if(out_skip.class_probs[(int)FXAI_LABEL_SKIP] < 0.15)
      {
         Print("FXAI compliance error: calibration drift skip weak. model=", plugin.AIName());
         return false;
      }
   }

   return true;
}

double FXAI_ComplianceRandSymmetric(ulong &state)
{
   state = state * (ulong)1664525 + (ulong)1013904223;
   ulong bucket = state % (ulong)20001;
   return ((double)bucket / 10000.0) - 1.0;
}

bool FXAI_RunDeterminismCompliance(CFXAIAIPlugin &plugin,
                                   const FXAIAIManifestV4 &manifest,
                                   const FXAIAIHyperParams &hp,
                                   const datetime now_t)
{
   int horizon = FXAI_ComplianceHorizon(manifest, 13);
   FXAIAIPredictRequestV4 req;
   FXAI_FillCompliancePredictRequest(plugin, req, 0.8, 0.72, 0.38, 0.18, now_t, 2, horizon);

   FXAIAIPredictionV4 pred_a, pred_b;
   if(!FXAI_PredictViaV4(plugin, req, hp, pred_a) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_a, "determinism_a"))
      return false;
   if(!FXAI_PredictViaV4(plugin, req, hp, pred_b) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_b, "determinism_b"))
      return false;

   double drift = FXAI_PredictionDistance(pred_a, pred_b);
   if(drift > 1e-8)
   {
      Print("FXAI compliance error: deterministic replay failed. model=", plugin.AIName(),
            " drift=", DoubleToString(drift, 10));
      return false;
   }
   return true;
}

bool FXAI_RunPersistenceRoundTripCompliance(CFXAIAIPlugin &plugin,
                                            const FXAIAIManifestV4 &manifest,
                                            const FXAIAIHyperParams &hp,
                                            const datetime now_t)
{
   if(!plugin.SupportsPersistentState())
      return true;

   int seq = FXAI_ComplianceSequenceBars(manifest);
   int horizon = FXAI_ComplianceHorizon(manifest, 5);
   FXAIAITrainRequestV4 buy_s;
   FXAIAITrainRequestV4 sell_s;
   FXAIAITrainRequestV4 skip_s;
   FXAI_FillComplianceTrainRequest(plugin, buy_s, (int)FXAI_LABEL_BUY, 4.5, 0.8, 0.75, 0.40, 0.20, now_t - 180, 1, horizon);
   FXAI_FillComplianceTrainRequest(plugin, sell_s, (int)FXAI_LABEL_SELL, -4.5, 0.8, -0.75, -0.40, -0.20, now_t - 120, 1, horizon);
   FXAI_FillComplianceTrainRequest(plugin, skip_s, (int)FXAI_LABEL_SKIP, 0.2, 0.8, 0.02, 0.01, 0.00, now_t - 60, 1, horizon);
   buy_s.ctx.sequence_bars = seq;
   sell_s.ctx.sequence_bars = seq;
   skip_s.ctx.sequence_bars = seq;
   FXAI_TrainViaV4(plugin, buy_s, hp);
   FXAI_TrainViaV4(plugin, sell_s, hp);
   FXAI_TrainViaV4(plugin, skip_s, hp);

   string file_name = plugin.PersistentStateFile("__compliance__");
   FileDelete(file_name, FILE_COMMON);
   if(!plugin.SaveStateFile(file_name))
   {
      Print("FXAI compliance error: failed to save persistent state. model=", plugin.AIName());
      return false;
   }

   CFXAIAIPlugin *twin = g_plugins.CreateInstance(plugin.AIId());
   if(twin == NULL)
   {
      FileDelete(file_name, FILE_COMMON);
      Print("FXAI compliance error: failed to create persistence twin. model=", plugin.AIName());
      return false;
   }

   twin.Reset();
   twin.EnsureInitialized(hp);
   if(!twin.LoadStateFile(file_name))
   {
      delete twin;
      FileDelete(file_name, FILE_COMMON);
      Print("FXAI compliance error: failed to reload persistent state. model=", plugin.AIName());
      return false;
   }

   FXAIAIPredictRequestV4 req;
   FXAI_FillCompliancePredictRequest(plugin, req, 0.9, 0.84, 0.44, 0.22, now_t, 1, FXAI_ComplianceHorizon(manifest, 21));

   FXAIAIPredictionV4 pred_live, pred_restored;
   bool live_ok = FXAI_PredictViaV4(plugin, req, hp, pred_live) &&
                  FXAI_ValidatePredictionOutput(plugin, pred_live, "persist_live");
   bool twin_ok = FXAI_PredictViaV4(*twin, req, hp, pred_restored) &&
                  FXAI_ValidatePredictionOutput(*twin, pred_restored, "persist_twin");

   delete twin;
   FileDelete(file_name, FILE_COMMON);
   if(!live_ok || !twin_ok)
      return false;

   double drift = FXAI_PredictionDistance(pred_live, pred_restored);
   if(drift > 1e-6)
   {
      Print("FXAI compliance error: persistence round-trip drift too high. model=", plugin.AIName(),
            " drift=", DoubleToString(drift, 8));
      return false;
   }
   return true;
}

bool FXAI_RunPersistenceCoverageCompliance(CFXAIAIPlugin &plugin,
                                           const FXAIAIManifestV4 &manifest)
{
   if(!plugin.SupportsPersistentState())
      return true;

   string file_name = plugin.PersistentStateFile("__coverage__");
   FileDelete(file_name, FILE_COMMON);
   if(!plugin.SaveStateFile(file_name))
   {
      Print("FXAI compliance error: failed to save persistence coverage artifact. model=", plugin.AIName());
      return false;
   }

   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      FileDelete(file_name, FILE_COMMON);
      Print("FXAI compliance error: failed to reopen persistence coverage artifact. model=", plugin.AIName());
      return false;
   }
   long file_size = (long)FileSize(handle);
   FileClose(handle);
   FileDelete(file_name, FILE_COMMON);

   string coverage_tag = plugin.PersistentStateCoverageTag();
   string expected_tier = FXAI_ReferenceTierName(manifest.reference_tier);
   if((coverage_tag == "native_model" || coverage_tag == "native_replay") && file_size < 256)
   {
      Print("FXAI compliance error: persistence artifact unexpectedly small. model=", plugin.AIName(),
            " bytes=", (int)file_size);
      return false;
   }

   bool stateful_checkpoint =
      (FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_ONLINE_LEARNING) ||
       FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY) ||
       FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL));

   if(stateful_checkpoint && coverage_tag != "native_model")
   {
      Print("FXAI compliance error: stateful model lacks native checkpoint coverage required for live promotion. model=",
            plugin.AIName(),
            " coverage=",
            coverage_tag);
      return false;
   }

   if(manifest.reference_tier == (int)FXAI_REFERENCE_FULL_NATIVE &&
      coverage_tag != "native_model")
   {
      Print("FXAI persistence note: full-native plugin is not running on native checkpoint coverage. model=", plugin.AIName(),
            " coverage=", coverage_tag);
   }
   else if(!stateful_checkpoint && coverage_tag != expected_tier)
   {
      Print("FXAI persistence note: checkpoint coverage tag differs from declared reference tier. model=", plugin.AIName(),
            " coverage=", coverage_tag,
            " tier=", expected_tier);
   }

   return true;
}

bool FXAI_RunFuzzStressCompliance(CFXAIAIPlugin &plugin,
                                  const FXAIAIManifestV4 &manifest,
                                  const FXAIAIHyperParams &hp,
                                  const datetime now_t)
{
   ulong state = (ulong)(plugin.AIId() + 1) * (ulong)2654435761;
   int seq = FXAI_ComplianceSequenceBars(manifest);
   int h_min = FXAI_ComplianceHorizon(manifest, manifest.min_horizon_minutes);
   int h_mid = FXAI_ComplianceHorizon(manifest, 13);
   int h_max = FXAI_ComplianceHorizon(manifest, manifest.max_horizon_minutes);

   for(int step=0; step<48; step++)
   {
      double v1 = FXAI_ComplianceRandSymmetric(state);
      double v2 = 0.55 * FXAI_ComplianceRandSymmetric(state);
      double v3 = 0.30 * FXAI_ComplianceRandSymmetric(state);
      double cost = 0.4 + 3.4 * FXAI_Clamp(0.5 + 0.5 * FXAI_ComplianceRandSymmetric(state), 0.0, 1.0);
      int regime = step % 4;
      int horizon = (step % 3 == 0 ? h_min : (step % 3 == 1 ? h_mid : h_max));

      if((step % 4) == 0)
      {
         int cls = (v1 > 0.12 ? (int)FXAI_LABEL_BUY :
                   (v1 < -0.12 ? (int)FXAI_LABEL_SELL : (int)FXAI_LABEL_SKIP));
         double move = (cls == (int)FXAI_LABEL_SKIP ? 0.15 + 0.20 * MathAbs(v1) :
                       (cls == (int)FXAI_LABEL_BUY ? 4.0 + 4.0 * MathAbs(v1) : -(4.0 + 4.0 * MathAbs(v1))));
         FXAIAITrainRequestV4 train_req;
         FXAI_FillComplianceTrainRequest(plugin, train_req, cls, move, cost,
                                         v1, v2, v3,
                                         now_t - (step * 60), regime, horizon);
         train_req.ctx.sequence_bars = seq;
         FXAI_TrainViaV4(plugin, train_req, hp);
      }

      FXAIAIPredictRequestV4 req;
      FXAI_FillCompliancePredictRequest(plugin, req, cost, v1, v2, v3,
                                        now_t + (step * 60), regime, horizon);
      req.ctx.sequence_bars = seq;
      FXAIAIPredictionV4 pred;
      if(!FXAI_PredictViaV4(plugin, req, hp, pred) ||
         !FXAI_ValidatePredictionOutput(plugin, pred, "fuzz"))
         return false;

      if(pred.move_q25_points > pred.move_q50_points + 1e-6 ||
         pred.move_q50_points > pred.move_q75_points + 1e-6)
      {
         Print("FXAI compliance error: move quantiles out of order. model=", plugin.AIName());
         return false;
      }
      if(pred.path_risk < 0.0 || pred.path_risk > 1.0 ||
         pred.fill_risk < 0.0 || pred.fill_risk > 1.0 ||
         pred.confidence < 0.0 || pred.confidence > 1.0 ||
         pred.reliability < 0.0 || pred.reliability > 1.0)
      {
         Print("FXAI compliance error: bounded risk/confidence outputs failed. model=", plugin.AIName());
         return false;
      }

      if((step % 12) == 11)
         plugin.ResetState((int)FXAI_RESET_SESSION_CHANGE, now_t + (step * 60));
   }

   return true;
}

bool FXAI_RunPluginComplianceHarness()
{
   datetime now_t = TimeCurrent();

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *plugin = g_plugins.CreateInstance(ai_idx);
      if(plugin == NULL)
      {
         Print("FXAI compliance error: could not create plugin id=", ai_idx);
         return false;
      }

      FXAIAIHyperParams hp;
      FXAI_GetModelHyperParams(ai_idx, hp);
      plugin.Reset();
      plugin.EnsureInitialized(hp);

      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);
      string reason = "";
      if(!FXAI_ValidateManifestV4(manifest, reason))
      {
         Print("FXAI compliance error: manifest invalid. model=", plugin.AIName(),
               " reason=", reason);
         delete plugin;
         return false;
      }

      FXAIAITrainRequestV4 buy_s, sell_s, skip_s, buy_big_s;
      FXAI_FillComplianceTrainRequest(*plugin, buy_s, (int)FXAI_LABEL_BUY, 4.5, 0.8, 0.75, 0.40, 0.20, now_t - 180, 1, 5);
      FXAI_FillComplianceTrainRequest(*plugin, sell_s, (int)FXAI_LABEL_SELL, -4.5, 0.8, -0.75, -0.40, -0.20, now_t - 120, 1, 5);
      FXAI_FillComplianceTrainRequest(*plugin, skip_s, (int)FXAI_LABEL_SKIP, 0.2, 0.8, 0.02, 0.01, 0.00, now_t - 60, 1, 5);
      FXAI_FillComplianceTrainRequest(*plugin, buy_big_s, (int)FXAI_LABEL_BUY, 8.0, 0.8, 1.20, 0.65, 0.35, now_t - 30, 1, 13);

      for(int rep=0; rep<10; rep++)
      {
         FXAI_TrainViaV4(*plugin, buy_s, hp);
         FXAI_TrainViaV4(*plugin, sell_s, hp);
         FXAI_TrainViaV4(*plugin, skip_s, hp);
         FXAI_TrainViaV4(*plugin, buy_big_s, hp);
      }

      FXAIAIPredictRequestV4 req_buy_lo, req_buy_hi, req_sell_lo, req_skip_lo, req_buy_big;
      FXAI_FillCompliancePredictRequest(*plugin, req_buy_lo, 0.8, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_buy_hi, 3.5, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_sell_lo, 0.8, -0.75, -0.40, -0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_skip_lo, 0.8, 0.02, 0.01, 0.00, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_buy_big, 0.8, 1.20, 0.65, 0.35, now_t, 1, 13);

      FXAIAIPredictionV4 pred_buy_lo, pred_buy_hi, pred_sell_lo, pred_skip_lo, pred_buy_big;
      FXAI_PredictViaV4(*plugin, req_buy_lo, hp, pred_buy_lo);
      FXAI_PredictViaV4(*plugin, req_buy_hi, hp, pred_buy_hi);
      FXAI_PredictViaV4(*plugin, req_sell_lo, hp, pred_sell_lo);
      FXAI_PredictViaV4(*plugin, req_skip_lo, hp, pred_skip_lo);
      FXAI_PredictViaV4(*plugin, req_buy_big, hp, pred_buy_big);

      bool ok = FXAI_ValidatePredictionOutput(*plugin, pred_buy_lo, "buy_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_buy_hi, "buy_hi")
             && FXAI_ValidatePredictionOutput(*plugin, pred_sell_lo, "sell_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_skip_lo, "skip_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_buy_big, "buy_big");
      if(!ok)
      {
         delete plugin;
         return false;
      }

      if(pred_buy_lo.class_probs[(int)FXAI_LABEL_BUY] + 0.05 < pred_buy_lo.class_probs[(int)FXAI_LABEL_SELL])
      {
         Print("FXAI compliance error: buy ordering failed. model=", plugin.AIName());
         delete plugin;
         return false;
      }
      if(pred_sell_lo.class_probs[(int)FXAI_LABEL_SELL] + 0.05 < pred_sell_lo.class_probs[(int)FXAI_LABEL_BUY])
      {
         Print("FXAI compliance error: sell ordering failed. model=", plugin.AIName());
         delete plugin;
         return false;
      }
      if(pred_skip_lo.class_probs[(int)FXAI_LABEL_SKIP] < 0.20)
      {
         Print("FXAI compliance error: skip response too weak. model=", plugin.AIName());
         delete plugin;
         return false;
      }

      double actionable_lo = pred_buy_lo.class_probs[(int)FXAI_LABEL_BUY] + pred_buy_lo.class_probs[(int)FXAI_LABEL_SELL];
      double actionable_hi = pred_buy_hi.class_probs[(int)FXAI_LABEL_BUY] + pred_buy_hi.class_probs[(int)FXAI_LABEL_SELL];
      if(actionable_hi > actionable_lo + 0.20)
      {
         Print("FXAI compliance error: cost awareness failed. model=", plugin.AIName(),
               " low=", DoubleToString(actionable_lo, 4),
               " high=", DoubleToString(actionable_hi, 4));
         delete plugin;
         return false;
      }

      if(pred_buy_big.move_mean_points + 0.25 < pred_buy_lo.move_mean_points)
      {
         Print("FXAI compliance error: EV monotonicity failed. model=", plugin.AIName(),
               " big=", DoubleToString(pred_buy_big.move_mean_points, 4),
               " base=", DoubleToString(pred_buy_lo.move_mean_points, 4));
         delete plugin;
         return false;
      }

      if(plugin.CorePredictFailures() > 0)
      {
         Print("FXAI compliance error: core predict failures detected. model=", plugin.AIName(),
               " failures=", plugin.CorePredictFailures());
         delete plugin;
         return false;
      }

      if(!FXAI_RunDeterminismCompliance(*plugin, manifest, hp, now_t + 120) ||
         !FXAI_RunPersistenceRoundTripCompliance(*plugin, manifest, hp, now_t + 150) ||
         !FXAI_RunPersistenceCoverageCompliance(*plugin, manifest) ||
         !FXAI_RunFuzzStressCompliance(*plugin, manifest, hp, now_t + 180))
      {
         delete plugin;
         return false;
      }

      plugin.ResetState((int)FXAI_RESET_FULL, now_t + 180);
      if(!FXAI_RunStateResetCompliance(*plugin, manifest, hp, now_t + 240) ||
         !FXAI_RunSequenceWindowCompliance(*plugin, manifest, hp, now_t + 300) ||
         !FXAI_RunCalibrationDriftCompliance(*plugin, manifest, hp, now_t + 360))
      {
         delete plugin;
         return false;
      }

      delete plugin;
   }

   return true;
}
