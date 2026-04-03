   ulong DefaultFeatureGroupsMask(void) const
   {
      ulong mask = 0;
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_PRICE);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_VOLATILITY);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_TIME);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_CONTEXT);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_COST);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_FILTERS);
      return mask;
   }

   void FillManifest(FXAIAIManifestV4 &out,
                     const int family,
                     const ulong capability_mask,
                     const int min_sequence_bars,
                     const int max_sequence_bars,
                     const int min_horizon_minutes = 1,
                     const int max_horizon_minutes = 720,
                     const ulong feature_groups_mask = 0,
                     const int feature_schema_id = 0) const
   {
      int min_seq = min_sequence_bars;
      int max_seq = max_sequence_bars;
      if(min_seq < 1) min_seq = 1;
      if(min_seq > FXAI_MAX_SEQUENCE_BARS) min_seq = FXAI_MAX_SEQUENCE_BARS;
      if(max_seq < min_seq) max_seq = min_seq;
      if(max_seq > FXAI_MAX_SEQUENCE_BARS) max_seq = FXAI_MAX_SEQUENCE_BARS;

      out.api_version = FXAI_API_VERSION_V4;
      out.ai_id = AIId();
      out.ai_name = AIName();
      out.family = family;
      out.reference_tier = FXAI_DefaultReferenceTierForAI(out.ai_id);
      out.capability_mask = capability_mask;
      out.feature_schema_id = (feature_schema_id > 0 ? feature_schema_id : FXAI_DefaultFeatureSchemaForFamily(family));
      out.feature_groups_mask = (feature_groups_mask != 0 ? feature_groups_mask : FXAI_DefaultFeatureGroupsForFamily(family));
      out.min_horizon_minutes = min_horizon_minutes;
      out.max_horizon_minutes = max_horizon_minutes;
      out.min_sequence_bars = min_seq;
      out.max_sequence_bars = max_seq;
   }

   void ResetModelOutput(FXAIAIModelOutputV4 &out) const
   {
      out.class_probs[0] = 0.10;
      out.class_probs[1] = 0.10;
      out.class_probs[2] = 0.80;
      out.move_mean_points = 0.0;
      out.move_q25_points = 0.0;
      out.move_q50_points = 0.0;
      out.move_q75_points = 0.0;
      out.mfe_mean_points = 0.0;
      out.mae_mean_points = 0.0;
      out.hit_time_frac = 1.0;
      out.path_risk = 0.0;
      out.fill_risk = 0.0;
      out.confidence = 0.0;
      out.reliability = 0.0;
      out.has_quantiles = false;
      out.has_confidence = false;
      out.has_path_quality = false;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double move_mean_points = out.move_mean_points;
      if(!PredictModelCore(x, hp, out.class_probs, move_mean_points))
         return false;
      NormalizeClassDistribution(out.class_probs);
      out.move_mean_points = (MathIsValidNumber(move_mean_points) && move_mean_points > 0.0 ? move_mean_points : 0.0);

      double buy_p = out.class_probs[(int)FXAI_LABEL_BUY];
      double sell_p = out.class_probs[(int)FXAI_LABEL_SELL];
      double skip_p = out.class_probs[(int)FXAI_LABEL_SKIP];
      double directional_conf = MathMax(buy_p, sell_p);
      double entropy = 0.0;
      for(int c=0; c<3; c++)
      {
         double p = MathMax(out.class_probs[c], 1e-9);
         entropy -= p * MathLog(p);
      }
      entropy /= MathLog(3.0);

      double move_scale = MathMax(ResolveMinMovePoints(), 0.10);
      if(m_move_ready && m_move_ema_abs > 0.0)
         move_scale = MathMax(move_scale, 0.60 * m_move_ema_abs);
      if(CurrentWindowSize() > 1)
      {
         double mean1 = CurrentWindowFeatureMean(0);
         double var1 = 0.0;
         for(int b=0; b<CurrentWindowSize(); b++)
         {
            double d = CurrentWindowValue(b, 1) - mean1;
            var1 += d * d;
         }
         var1 = MathSqrt(var1 / (double)CurrentWindowSize());
         move_scale = MathMax(move_scale, 0.35 * var1);
      }

      double sigma = MathMax(0.10, 0.35 * out.move_mean_points + 0.35 * move_scale + 0.20 * skip_p + 0.15 * entropy);
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      double mfe_scale = 1.20 + 0.35 * directional_conf + 0.20 * (1.0 - skip_p);
      double mae_scale = 0.35 + 0.25 * skip_p + 0.20 * entropy;
      if(m_quality_head_ready)
      {
         double quality_base = MathMax(m_move_ema_abs, MathMax(move_scale, 0.10));
         mfe_scale = FXAI_Clamp(m_quality_mfe_ema / MathMax(quality_base, 0.10), 0.80, 3.00);
         mae_scale = FXAI_Clamp(m_quality_mae_ema / MathMax(MathMax(m_quality_mfe_ema, quality_base), 0.10), 0.05, 1.50);
      }
      out.mfe_mean_points = MathMax(out.move_q75_points, out.move_mean_points * mfe_scale);
      out.mae_mean_points = MathMax(0.0, out.move_mean_points * mae_scale);
      double hit_frac = 0.60 - 0.25 * directional_conf + 0.20 * skip_p + 0.15 * entropy;
      if(m_quality_head_ready)
         hit_frac = 0.55 * m_quality_hit_ema + 0.45 * hit_frac;
      out.hit_time_frac = FXAI_Clamp(hit_frac, 0.0, 1.0);
      out.confidence = FXAI_Clamp(0.60 * directional_conf + 0.20 * (1.0 - skip_p) + 0.20 * (1.0 - entropy), 0.0, 1.0);
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();
      double bank_mass = m_bank_total[r][s][h];
      double bank_rel = FXAI_Clamp(bank_mass / 120.0, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * bank_rel + 0.20 * (1.0 - entropy) + 0.10 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      double path_risk = 0.40 * FXAI_Clamp(out.mae_mean_points / MathMax(out.mfe_mean_points, move_scale), 0.0, 1.0) +
                         0.35 * out.hit_time_frac +
                         0.25 * skip_p;
      if(m_quality_head_ready)
         path_risk = 0.55 * path_risk + 0.45 * m_quality_path_risk_ema;
      out.path_risk = FXAI_Clamp(path_risk, 0.0, 1.0);
      double fill_risk = FXAI_Clamp((ResolveCostPoints(x) + 0.50 * ResolveMinMovePoints()) / MathMax(out.move_mean_points + move_scale, 0.25), 0.0, 1.0);
      if(m_quality_head_ready)
         fill_risk = 0.50 * fill_risk + 0.50 * m_quality_fill_risk_ema;
      out.fill_risk = FXAI_Clamp(fill_risk, 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      out.has_path_quality = true;
      return true;
   }

   void FillPredictionV4(const FXAIAIModelOutputV4 &model_out,
                         const double calibrated_move_mean_points,
                         FXAIAIPredictionV4 &dst) const
   {
      for(int c=0; c<3; c++)
         dst.class_probs[c] = model_out.class_probs[c];

      double buy_p = dst.class_probs[(int)FXAI_LABEL_BUY];
      double sell_p = dst.class_probs[(int)FXAI_LABEL_SELL];
      double skip_p = dst.class_probs[(int)FXAI_LABEL_SKIP];
      double directional_conf = MathMax(buy_p, sell_p);
      double uncertainty = FXAI_Clamp(1.0 - directional_conf + 0.50 * skip_p, 0.10, 1.50);
      double mean_move = (MathIsValidNumber(calibrated_move_mean_points) && calibrated_move_mean_points > 0.0 ? calibrated_move_mean_points : 0.0);

      dst.move_mean_points = mean_move;
      double raw_mean = (MathIsValidNumber(model_out.move_mean_points) && model_out.move_mean_points > 0.0 ? model_out.move_mean_points : 0.0);
      double scale = (raw_mean > 1e-9 ? mean_move / raw_mean : 1.0);

      if(model_out.has_quantiles && mean_move > 0.0)
      {
         dst.move_q25_points = MathMax(0.0, model_out.move_q25_points * scale);
         dst.move_q50_points = MathMax(dst.move_q25_points, model_out.move_q50_points * scale);
         dst.move_q75_points = MathMax(dst.move_q50_points, model_out.move_q75_points * scale);
      }
      else if(mean_move > 0.0)
      {
         dst.move_q25_points = MathMax(0.0, mean_move * MathMax(0.25, 1.0 - 0.45 * uncertainty));
         dst.move_q50_points = mean_move;
         dst.move_q75_points = MathMax(dst.move_q50_points, mean_move * (1.0 + 0.45 * uncertainty));
      }
      else
      {
         dst.move_q25_points = 0.0;
         dst.move_q50_points = 0.0;
         dst.move_q75_points = 0.0;
      }

      if(model_out.has_path_quality)
      {
         dst.mfe_mean_points = MathMax(0.0, model_out.mfe_mean_points * scale);
         dst.mae_mean_points = MathMax(0.0, model_out.mae_mean_points * scale);
         dst.hit_time_frac = FXAI_Clamp(model_out.hit_time_frac, 0.0, 1.0);
         dst.path_risk = FXAI_Clamp(model_out.path_risk, 0.0, 1.0);
         dst.fill_risk = FXAI_Clamp(model_out.fill_risk, 0.0, 1.0);
      }
      else
      {
         dst.mfe_mean_points = MathMax(dst.move_q75_points, dst.move_mean_points);
         dst.mae_mean_points = MathMax(0.0, 0.35 * dst.move_mean_points);
         dst.hit_time_frac = FXAI_Clamp(0.60 - 0.20 * directional_conf + 0.20 * skip_p, 0.0, 1.0);
         dst.path_risk = FXAI_Clamp(0.40 * skip_p + 0.35 * dst.hit_time_frac, 0.0, 1.0);
         dst.fill_risk = FXAI_Clamp((m_ctx_cost_points + 0.25 * ResolveMinMovePoints()) / MathMax(dst.move_mean_points + ResolveMinMovePoints(), 0.25), 0.0, 1.0);
      }

      dst.confidence = FXAI_Clamp(model_out.has_confidence ? model_out.confidence : directional_conf, 0.0, 1.0);
      dst.reliability = FXAI_Clamp(model_out.has_confidence ? model_out.reliability : (1.0 - 0.50 * skip_p), 0.0, 1.0);
   }

   double CurrentWindowSliceMean(const int input_idx,
                                 const int start_bar,
                                 const int count) const
   {
      if(input_idx < 0 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0 || count <= 0)
         return 0.0;

      int first = start_bar;
      if(first < 0) first = 0;
      if(first >= m_ctx_window_size) return 0.0;
      int last = first + count;
      if(last > m_ctx_window_size) last = m_ctx_window_size;
      if(last <= first) return 0.0;

      double sum = 0.0;
      int n = 0;
      for(int b=first; b<last; b++)
      {
         sum += m_ctx_window[b][input_idx];
         n++;
      }
      if(n <= 0) return 0.0;
      return sum / (double)n;
   }

   int CurrentWindowSize(void) const
   {
      return m_ctx_window_size;
   }

   double CurrentWindowValue(const int bar_idx, const int input_idx) const
   {
      if(bar_idx < 0 || bar_idx >= m_ctx_window_size) return 0.0;
      if(input_idx < 0 || input_idx >= FXAI_AI_WEIGHTS) return 0.0;
      return m_ctx_window[bar_idx][input_idx];
   }

   double CurrentWindowFeatureMean(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0) return 0.0;
      double full = CurrentWindowSliceMean(input_idx, 0, m_ctx_window_size);
      int half_n = MathMax(m_ctx_window_size / 2, 1);
      int quarter_n = MathMax(m_ctx_window_size / 4, 1);
      double half = CurrentWindowSliceMean(input_idx, m_ctx_window_size - half_n, half_n);
      double quarter = CurrentWindowSliceMean(input_idx, m_ctx_window_size - quarter_n, quarter_n);
      return 0.40 * full + 0.35 * half + 0.25 * quarter;
   }

   double CurrentWindowFeatureRecentMean(const int feature_idx,
                                         const int recent_bars) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 0) n = 1;
      if(n > m_ctx_window_size) n = m_ctx_window_size;
      return CurrentWindowSliceMean(input_idx, 0, n);
   }

   double CurrentWindowFeatureStd(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 1) return 0.0;
      double mean = CurrentWindowSliceMean(input_idx, 0, m_ctx_window_size);
      double acc = 0.0;
      for(int b=0; b<m_ctx_window_size; b++)
      {
         double d = m_ctx_window[b][input_idx] - mean;
         acc += d * d;
      }
      return MathSqrt(acc / (double)MathMax(m_ctx_window_size, 1));
   }

   double CurrentWindowFeatureRange(const int feature_idx,
                                    const int recent_bars = 0) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 0 || n > m_ctx_window_size) n = m_ctx_window_size;
      double lo = CurrentWindowValue(0, input_idx);
      double hi = lo;
      for(int b=0; b<n; b++)
      {
         double v = CurrentWindowValue(b, input_idx);
         if(v < lo) lo = v;
         if(v > hi) hi = v;
      }
      return hi - lo;
   }

   double CurrentWindowFeatureSlope(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 1) return 0.0;
      double first = m_ctx_window[0][input_idx];
      double last = m_ctx_window[m_ctx_window_size - 1][input_idx];
      return (first - last) / (double)MathMax(m_ctx_window_size - 1, 1);
   }

   double CurrentWindowFeatureRecentDelta(const int feature_idx,
                                          const int recent_bars) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 1) n = MathMax(m_ctx_window_size / 4, 2);
      if(n > m_ctx_window_size) n = m_ctx_window_size;
      int last_idx = n - 1;
      if(last_idx < 0) last_idx = 0;
      return CurrentWindowValue(0, input_idx) - CurrentWindowValue(last_idx, input_idx);
   }

   double CurrentWindowFeatureEMAMean(const int feature_idx,
                                      const double decay = 0.72) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      double a = FXAI_Clamp(decay, 0.05, 0.98);
      double w = 1.0;
      double sw = 0.0;
      double sum = 0.0;
      for(int b=0; b<m_ctx_window_size; b++)
      {
         double v = CurrentWindowValue(b, input_idx);
         sum += w * v;
         sw += w;
         w *= a;
      }
      if(sw <= 0.0) return 0.0;
      return sum / sw;
   }

   void BuildChronologicalSequenceTensor(const double &x[],
                                         double &seq[][FXAI_AI_WEIGHTS],
                                         int &seq_len) const
   {
      FXAISequenceRuntimeConfig cfg = FXAI_SequenceRuntimeMakeConfig(FXAI_MAX_SEQUENCE_BARS, 1, 1, false, true, 0.06);
      BuildChronologicalSequenceTensorConfigured(x, cfg, seq, seq_len);
   }

   int ContextSequenceCap(const int max_cap,
                          const int base_min = 8) const
   {
      return FXAI_ContextSequenceSpan(max_cap,
                                      (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1),
                                      _Symbol,
                                      base_min);
   }

   int ContextBatchCap(const int max_cap,
                       const int base_min = 4) const
   {
      return FXAI_ContextBatchSpan(max_cap,
                                   (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1),
                                   _Symbol,
                                   base_min);
   }
