#ifndef __FXAI_AUDIT_SAMPLES_MQH__
#define __FXAI_AUDIT_SAMPLES_MQH__

bool FXAI_AuditBuildSample(const int i,
                           const int horizon_minutes,
                           const double point,
                           const double ev_threshold_points,
                           const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                           const datetime &time_arr[],
                           const double &open_arr[],
                           const double &high_arr[],
                           const double &low_arr[],
                           const double &close_arr[],
                           const int &spread_arr[],
                           const datetime &time_m5[],
                           const double &close_m5[],
                           const int &map_m5[],
                           const datetime &time_m15[],
                           const double &close_m15[],
                           const int &map_m15[],
                           const datetime &time_m30[],
                           const double &close_m30[],
                           const int &map_m30[],
                           const datetime &time_h1[],
                           const double &close_h1[],
                           const int &map_h1[],
                           const double &ctx_mean_arr[],
                           const double &ctx_std_arr[],
                           const double &ctx_up_arr[],
                           const double &ctx_extra_arr[],
                           FXAIAIContextV4 &ctx,
                           int &label_class,
                           double &move_points,
                           double &mfe_points,
                           double &mae_points,
                           double &time_to_hit_frac,
                           int &path_flags,
                           double &spread_stress,
                           FXAIExecutionTraceStats &trace_stats,
                           double &sample_weight,
                           double &x[])
{
   int n = ArraySize(close_arr);
   if(i < 0 || i >= n) return false;
   if(i - horizon_minutes < 0) return false;

   FXAIDataSnapshot snapshot;
   snapshot.symbol = _Symbol;
   snapshot.bar_time = time_arr[i];
   snapshot.point = point;
   snapshot.spread_points = FXAI_GetSpreadAtIndex(i, spread_arr, 1.0);
   snapshot.commission_points = FXAI_GetCommissionPointsRoundTripPerLot(snapshot.symbol, FXAI_AuditGetCommissionPerLotSide());
   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   snapshot.min_move_points = FXAI_ExecutionEntryCostPoints(snapshot.spread_points,
                                                            snapshot.commission_points,
                                                            FXAI_AuditGetCostBufferPoints(),
                                                            exec_profile);
   if(snapshot.min_move_points < 0.0) snapshot.min_move_points = 0.0;

   FXAIDataCoreBundle bundle;
   int align_upto = n - 1;
   if(align_upto < 0) align_upto = 0;
   FXAI_DataCoreBindArrayBundle(snapshot,
                                n,
                                align_upto,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                time_arr,
                                spread_arr,
                                close_m5,
                                time_m5,
                                map_m5,
                                close_m15,
                                time_m15,
                                map_m15,
                                close_m30,
                                time_m30,
                                map_m30,
                                close_h1,
                                time_h1,
                                map_h1,
                                ctx_mean_arr,
                                ctx_std_arr,
                                ctx_up_arr,
                                ctx_extra_arr,
                                bundle);

   FXAIFeatureCoreFrame feature_frame;
   if(!FXAI_FeatureCoreBuildFrameFromBundle(bundle,
                                            i,
                                            horizon_minutes,
                                            norm_method,
                                            feature_frame))
      return false;
   FXAINormalizationCoreFrame norm_frame;
   if(!FXAI_NormalizationCoreBuildInputFrameFromFeatureFrame(feature_frame, norm_frame))
      return false;
   ArrayResize(x, FXAI_AI_WEIGHTS);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      x[k] = norm_frame.model_input[k];

   double cost_points = snapshot.min_move_points;
   move_points = 0.0;
   mfe_points = 0.0;
   mae_points = 0.0;
   time_to_hit_frac = 1.0;
   path_flags = 0;
   label_class = FXAI_AuditBuildTripleBarrierLabelEx(i,
                                                     horizon_minutes,
                                                     cost_points,
                                                     ev_threshold_points,
                                                     snapshot,
                                                     high_arr,
                                                     low_arr,
                                                     close_arr,
                                                     move_points,
                                                     mfe_points,
                                                     mae_points,
                                                     time_to_hit_frac,
                                                     path_flags);
   spread_stress = FXAI_Clamp(snapshot.spread_points / MathMax(snapshot.min_move_points, 0.10), 0.0, 4.0);
   FXAI_BuildExecutionTraceStats(i,
                                 horizon_minutes,
                                 point,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
                                 spread_arr,
                                 trace_stats);
   sample_weight = FXAI_Clamp(FXAI_MoveEdgeWeight(move_points, cost_points), 0.25, 4.0);

   double spread_ref = FXAI_AuditGetIntArrayMean(spread_arr, i, 64, snapshot.spread_points);
   double vol_ref = FXAI_RollingAbsReturn(close_arr, i, 64);
    double vol_proxy = MathAbs(feature_frame.raw[5]);
   ctx.api_version = FXAI_API_VERSION_V4;
   ctx.regime_id = FXAI_AuditGetStaticRegimeId(snapshot.bar_time, snapshot.spread_points, spread_ref, vol_proxy, vol_ref);
   ctx.session_bucket = FXAI_DeriveSessionBucket(snapshot.bar_time);
   ctx.horizon_minutes = horizon_minutes;
   int schema_default = FXAI_AuditGetSchemaOverride();
   if(schema_default <= 0) schema_default = (int)FXAI_SCHEMA_FULL;
   ctx.feature_schema_id = schema_default;
   ctx.normalization_method_id = (int)norm_method;
   int seq_default = FXAI_AuditGetSequenceBarsOverride();
   if(seq_default <= 0) seq_default = 1;
   ctx.sequence_bars = seq_default;
   ctx.cost_points = cost_points;
   ctx.min_move_points = cost_points;
   ctx.point_value = point;
   ctx.domain_hash = FXAI_SymbolHash01(snapshot.symbol);
   ctx.sample_time = snapshot.bar_time;
   return true;
}

void FXAI_AuditBuildWindow(const int i,
                           const int requested_bars,
                           const int horizon_minutes,
                           const double point,
                           const double ev_threshold_points,
                           const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                           const datetime &time_arr[],
                           const double &open_arr[],
                           const double &high_arr[],
                           const double &low_arr[],
                           const double &close_arr[],
                           const int &spread_arr[],
                           const datetime &time_m5[],
                           const double &close_m5[],
                           const int &map_m5[],
                           const datetime &time_m15[],
                           const double &close_m15[],
                           const int &map_m15[],
                           const datetime &time_m30[],
                           const double &close_m30[],
                           const int &map_m30[],
                           const datetime &time_h1[],
                           const double &close_h1[],
                           const int &map_h1[],
                           const double &ctx_mean_arr[],
                           const double &ctx_std_arr[],
                           const double &ctx_up_arr[],
                           const double &ctx_extra_arr[],
                           double &x_window[][FXAI_AI_WEIGHTS],
                           int &window_size)
{
   FXAI_ClearInputWindow(x_window, window_size);
   int seq = requested_bars;
   if(seq < 1) seq = 1;
   if(seq > FXAI_MAX_SEQUENCE_BARS) seq = FXAI_MAX_SEQUENCE_BARS;
   for(int b=0; b<seq; b++)
   {
      int wi = i + 1 + b;
      FXAIAIContextV4 wctx;
      int wlabel = (int)FXAI_LABEL_SKIP;
      double wmove = 0.0;
      double wmfe = 0.0;
      double wmae = 0.0;
      double whit = 1.0;
      int wflags = 0;
      double wspread_stress = 0.0;
      FXAIExecutionTraceStats wtrace;
      double wweight = 1.0;
      double wx[];
      if(!FXAI_AuditBuildSample(wi,
                                horizon_minutes,
                                point,
                                ev_threshold_points,
                                norm_method,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                spread_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_m30,
                                close_m30,
                                map_m30,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_arr,
                                ctx_std_arr,
                                ctx_up_arr,
                                ctx_extra_arr,
                                wctx,
                                wlabel,
                                wmove,
                                wmfe,
                                wmae,
                                whit,
                                wflags,
                                wspread_stress,
                                wtrace,
                                wweight,
                                wx))
         break;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_window[b][k] = wx[k];
      window_size++;
   }
}


#endif // __FXAI_AUDIT_SAMPLES_MQH__
