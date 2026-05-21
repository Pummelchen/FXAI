   void GetQualityBankPriors(double &mfe_out,
                             double &mae_out,
                             double &hit_out,
                             double &path_out,
                             double &fill_out,
                             double &trust_out) const
   {
      mfe_out = m_quality_mfe_ema;
      mae_out = m_quality_mae_ema;
      hit_out = m_quality_hit_ema;
      path_out = m_quality_path_risk_ema;
      fill_out = m_quality_fill_risk_ema;
      trust_out = (m_quality_head_ready ? 0.35 : 0.0);

      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();
      if(!m_quality_bank_ready[r][s][h])
         return;

      double bank_trust = FXAI_Clamp(m_quality_bank_obs[r][s][h] / 120.0, 0.10, 0.85);
      mfe_out = (1.0 - bank_trust) * mfe_out + bank_trust * m_quality_bank_mfe[r][s][h];
      mae_out = (1.0 - bank_trust) * mae_out + bank_trust * m_quality_bank_mae[r][s][h];
      hit_out = (1.0 - bank_trust) * hit_out + bank_trust * m_quality_bank_hit[r][s][h];
      path_out = (1.0 - bank_trust) * path_out + bank_trust * m_quality_bank_path[r][s][h];
      fill_out = (1.0 - bank_trust) * fill_out + bank_trust * m_quality_bank_fill[r][s][h];
      trust_out = FXAI_Clamp(trust_out + 0.65 * bank_trust, 0.0, 1.0);
   }

   void PopulatePathQualityHeads(FXAIAIModelOutputV4 &out,
                                 const double &x[],
                                 const double activity_gate,
                                 const double structural_quality,
                                 const double execution_quality = -1.0) const
   {
      FXAIAIManifestV4 manifest;
      DescribeResolved(manifest);
      double bank_mfe = 0.0;
      double bank_mae = 0.0;
      double bank_hit = 1.0;
      double bank_path = 0.5;
      double bank_fill = 0.5;
      double bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);

      double active = FXAI_Clamp(activity_gate, 0.0, 1.0);
      double structure = FXAI_Clamp(structural_quality, 0.0, 1.0);
      double exec_q = (execution_quality >= 0.0 ? FXAI_Clamp(execution_quality, 0.0, 1.0) : structure);
      double move_scale = MathMax(out.move_mean_points,
                          MathMax(out.move_q50_points,
                          MathMax(ResolveMinMovePoints(), 0.10)));
      double qspan = MathMax(0.0, out.move_q75_points - out.move_q25_points);
      double sigma = MathMax(0.10, 0.30 * move_scale + 0.45 * qspan);
      double directional = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY],
                                              out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      double skip = FXAI_Clamp(out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      double cost_ratio = FXAI_Clamp(ResolveCostPoints(x) / MathMax(move_scale + 0.40 * sigma, 0.25), 0.0, 1.0);
      double trend = 0.0;
      double trend_fast = 0.0;
      double noise = 0.0;
      double ctx_shape = 0.0;
      if(CurrentWindowSize() > 1)
      {
         double slope = MathAbs(CurrentWindowFeatureSlope(0));
         double slope_fast = MathAbs(CurrentWindowFeatureRecentDelta(0, MathMax(CurrentWindowSize() / 4, 2)));
         double stdv = CurrentWindowFeatureStd(0);
         double level = MathAbs(CurrentWindowFeatureEMAMean(0));
         double local_range = CurrentWindowFeatureRange(0, MathMax(CurrentWindowSize() / 2, 2));
         double ctx_recent = MathAbs(CurrentWindowFeatureRecentMean(10, MathMax(CurrentWindowSize() / 4, 1)));
         double ctx_slow = MathAbs(CurrentWindowFeatureMean(10));
         trend = FXAI_Clamp(slope / MathMax(stdv + 0.20 * MathAbs(level), 0.10), 0.0, 1.25);
         trend_fast = FXAI_Clamp(slope_fast / MathMax(local_range + 0.10, 0.10), 0.0, 1.25);
         noise = FXAI_Clamp((0.65 * stdv + 0.35 * local_range) / MathMax(MathAbs(level) + 0.10, 0.10), 0.0, 1.25);
         ctx_shape = FXAI_Clamp((ctx_recent + 0.50 * ctx_slow) / MathMax(local_range + 0.10, 0.10), 0.0, 1.25);
      }

      double fam_trend = 1.0;
      double fam_ctx = 1.0;
      double fam_exec = 1.0;
      switch(manifest.family)
      {
         case FXAI_FAMILY_RECURRENT:
         case FXAI_FAMILY_CONVOLUTIONAL:
         case FXAI_FAMILY_TRANSFORMER:
         case FXAI_FAMILY_STATE_SPACE:
            fam_trend = 1.12;
            fam_ctx = 1.08;
            break;
         case FXAI_FAMILY_WORLD_MODEL:
         case FXAI_FAMILY_RETRIEVAL:
         case FXAI_FAMILY_MIXTURE:
            fam_ctx = 1.15;
            fam_exec = 0.92;
            break;
         case FXAI_FAMILY_TREE:
         case FXAI_FAMILY_LINEAR:
            fam_trend = 0.92;
            fam_exec = 1.06;
            break;
         case FXAI_FAMILY_RULE_BASED:
            fam_trend = 0.80;
            fam_ctx = 0.85;
            fam_exec = 1.18;
            break;
         default:
            break;
      }

      double mfe_scale = 1.05 + 0.30 * directional + 0.18 * active + 0.16 * structure +
                         0.16 * trend * fam_trend + 0.10 * trend_fast + 0.08 * ctx_shape * fam_ctx;
      double mae_scale = 0.14 + 0.24 * (1.0 - active) + 0.18 * (1.0 - structure) +
                         0.16 * cost_ratio * fam_exec + 0.12 * noise + 0.08 * skip;
      if(m_quality_head_ready || bank_trust > 0.0)
      {
         double quality_base = MathMax(move_scale, 0.10);
         double bank_mfe_scale = FXAI_Clamp(bank_mfe / quality_base, 0.80, 3.40);
         double bank_mae_scale = FXAI_Clamp(bank_mae / MathMax(MathMax(bank_mfe, quality_base), 0.10), 0.05, 1.70);
         mfe_scale = (1.0 - 0.55 * bank_trust) * mfe_scale + 0.55 * bank_trust * bank_mfe_scale;
         mae_scale = (1.0 - 0.55 * bank_trust) * mae_scale + 0.55 * bank_trust * bank_mae_scale;
      }
      out.mfe_mean_points = MathMax(out.move_q75_points, move_scale * FXAI_Clamp(mfe_scale, 0.80, 3.50));
      out.mae_mean_points = MathMax(0.0, move_scale * FXAI_Clamp(mae_scale, 0.05, 1.80));

      double hit_frac = 0.70 - 0.20 * active - 0.12 * structure - 0.08 * trend_fast -
                        0.06 * ctx_shape + 0.18 * noise + 0.16 * cost_ratio + 0.10 * skip;
      if(m_quality_head_ready || bank_trust > 0.0)
         hit_frac = (1.0 - 0.60 * bank_trust) * hit_frac + 0.60 * bank_trust * bank_hit;
      out.hit_time_frac = FXAI_Clamp(hit_frac, 0.0, 1.0);

      double path_risk = 0.34 * FXAI_Clamp(out.mae_mean_points / MathMax(out.mfe_mean_points, move_scale), 0.0, 1.0) +
                         0.22 * out.hit_time_frac +
                         0.18 * cost_ratio +
                         0.14 * (1.0 - structure) +
                         0.12 * noise +
                         0.08 * (1.0 - exec_q);
      if(m_quality_head_ready || bank_trust > 0.0)
         path_risk = (1.0 - 0.60 * bank_trust) * path_risk + 0.60 * bank_trust * bank_path;
      out.path_risk = FXAI_Clamp(path_risk, 0.0, 1.0);

      double fill_risk = 0.46 * cost_ratio +
                         0.26 * (1.0 - exec_q) +
                         0.16 * skip +
                         0.12 * noise;
      if(m_quality_head_ready || bank_trust > 0.0)
         fill_risk = (1.0 - 0.60 * bank_trust) * fill_risk + 0.60 * bank_trust * bank_fill;
      out.fill_risk = FXAI_Clamp(fill_risk, 0.0, 1.0);
      out.has_path_quality = true;
   }

   void PredictNativeQualityHeads(const double &x[],
                                  const double activity_gate,
                                  const double structural_quality,
                                  const double execution_quality,
                                  FXAIAIModelOutputV4 &out) const
   {
      double bank_mfe = 0.0;
      double bank_mae = 0.0;
      double bank_hit = 1.0;
      double bank_path = 0.5;
      double bank_fill = 0.5;
      double bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_native_quality_heads.Predict(x,
                                     out.move_mean_points,
                                     activity_gate,
                                     structural_quality,
                                     execution_quality,
                                     bank_mfe,
                                     bank_mae,
                                     bank_hit,
                                     bank_path,
                                     bank_fill,
                                     bank_trust,
                                     out);
   }

   void UpdateNativeQualityHeads(const double &x[],
                                 const double sample_w,
                                 const double lr,
                                 const double l2)
   {
      m_native_quality_heads.Update(x,
                                    sample_w,
                                    TargetMFEPoints(),
                                    FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                                    TargetHitTimeFrac(),
                                    TargetPathRisk(),
                                    TargetFillRisk(),
                                    TargetMaskedStep(),
                                    TargetNextVol(),
                                    TargetRegimeShift(),
                                    TargetContextLead(),
                                    lr,
                                    l2);
   }

   double TargetMFEPoints(void) const { return m_target_quality_ready ? m_target_mfe_points : 0.0; }
   double TargetMAEPoints(void) const { return m_target_quality_ready ? m_target_mae_points : 0.0; }
   double TargetHitTimeFrac(void) const { return m_target_quality_ready ? m_target_hit_time_frac : 1.0; }
   int TargetPathFlags(void) const { return m_target_quality_ready ? m_target_path_flags : 0; }
   double TargetPathRisk(void) const { return m_target_quality_ready ? m_target_path_risk : 0.0; }
   double TargetFillRisk(void) const { return m_target_quality_ready ? m_target_fill_risk : 0.0; }
   double TargetMaskedStep(void) const { return m_target_quality_ready ? m_target_masked_step : 0.0; }
   double TargetNextVol(void) const { return m_target_quality_ready ? m_target_next_vol : 0.0; }
   double TargetRegimeShift(void) const { return m_target_quality_ready ? m_target_regime_shift : 0.0; }
   double TargetContextLead(void) const { return m_target_quality_ready ? m_target_context_lead : 0.5; }

public:
