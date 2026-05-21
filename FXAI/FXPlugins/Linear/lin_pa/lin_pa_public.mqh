   CFXAIAIPA(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_PA_LINEAR; }
   virtual string AIName(void) const { return "lin_pa"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_LINEAR, caps, 1, 1);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      bool averaged = (m_guard_ready ? m_guard_use_avg : (m_steps > 24));
      double scores[FXAI_PA_CLASS_COUNT];
      ComputeScores(x, averaged, scores);

      double den = (m_margin_ready ? m_margin_ema : FXAI_Clamp(hp.pa_margin, 0.25, 4.0));
      if(den < 0.25) den = 0.25;

      double logits[FXAI_PA_CLASS_COUNT];
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         logits[c] = FXAI_ClipSym(scores[c] / den, 20.0);

      double p_raw[FXAI_PA_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, class_probs);

      expected_move_points = PredictExpectedMovePoints(x, hp);
      if(expected_move_points < 0.0)
         expected_move_points = 0.0;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      bool averaged = (m_guard_ready ? m_guard_use_avg : (m_steps > 24));
      double scores[FXAI_PA_CLASS_COUNT];
      ComputeScores(x, averaged, scores);
      double den = (m_margin_ready ? m_margin_ema : FXAI_Clamp(hp.pa_margin, 0.25, 4.0));
      if(den < 0.25) den = 0.25;
      double logits[FXAI_PA_CLASS_COUNT];
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         logits[c] = FXAI_ClipSym(scores[c] / den, 20.0);
      double p_raw[FXAI_PA_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double mu = 0.0, logv = 0.0;
      PredictMoveDistRaw(x, mu, logv);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double head = PredictMoveRaw(x);
      if(head <= 0.0 && m_move_ready) head = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, head);
      out.move_q25_points = MathMax(0.0, MathMax(0.0, mu - 0.60 * sigma));
      out.move_q50_points = MathMax(out.move_q25_points, MathMax(0.0, mu));
      out.move_q75_points = MathMax(out.move_q50_points, MathMax(0.0, mu + 0.60 * sigma));
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * (m_move_ready ? 1.0 : 0.0) + 0.30 * MathMin((double)m_mv_steps / 64.0, 1.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(x,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_quality_heads.Reset();

      m_steps = 0;
      m_mv_steps = 0;
      m_margin_ready = false;
      m_margin_ema = 0.0;
      m_use_hash = true;
      m_use_hash2 = true;

      m_loss_ready = false;
      m_loss_fast = 0.0;
      m_loss_slow = 0.0;
      m_drift_cooldown = 0;
      m_pa_mode = 0;
      m_conf_r = 1.0;

      m_guard_ready = false;
      m_guard_use_avg = false;
      m_guard_live_fast = 0.0;
      m_guard_live_slow = 0.0;
      m_guard_avg_fast = 0.0;
      m_guard_avg_slow = 0.0;

      m_hash_occ_ready = false;
      m_hash2_scale = 1.0;
      m_rep_head = 0;
      m_rep_size = 0;

      m_mv_lv_b = 0.0;
      m_cal3_steps = 0;

      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         m_cls_ema[c] = 1.0;
         m_cal3_b[c] = 0.0;
         for(int j=0; j<FXAI_PA_CLASS_COUNT; j++)
            m_cal3_w[c][j] = (c == j ? 1.0 : 0.0);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_w[c][i] = 0.0;
            m_w_avg[c][i] = 0.0;
            m_sigma[c][i] = 1.0;
         }

         for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
         {
            m_hw1[c][i] = 0.0;
            m_hw1_avg[c][i] = 0.0;
            m_hsigma1[c][i] = 1.0;
         }

         for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
         {
            m_hw2[c][i] = 0.0;
            m_hw2_avg[c][i] = 0.0;
            m_hsigma2[c][i] = 1.0;
         }

         for(int b=0; b<FXAI_PA_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_mv_mu_w[i] = 0.0;
         m_mv_lv_w[i] = 0.0;
      }
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         m_mv_mu_hw1[i] = 0.0;
         m_mv_lv_hw1[i] = 0.0;
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         m_mv_mu_hw2[i] = 0.0;
         m_mv_lv_hw2[i] = 0.0;
      }

      for(int i=0; i<FXAI_PA_REPLAY; i++)
      {
         m_rep_cls[i] = (int)FXAI_LABEL_SKIP;
         m_rep_move[i] = 0.0;
         m_rep_w[i] = 1.0;
         m_rep_hard[i] = 0.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_rep_x[i][k] = 0.0;
      }

      BuildCollisionProfile();
   }

   virtual bool SupportsPersistentState(void) const { return true; }
   virtual bool SupportsNativeParameterSnapshot(void) const { return true; }
   virtual string PersistentStateCoverageTag(void) const { return "native_model"; }

   virtual bool SaveModelState(const int handle) const
   {
      FileWriteInteger(handle, m_steps);
      FileWriteInteger(handle, m_mv_steps);
      FileWriteInteger(handle, (m_margin_ready ? 1 : 0));
      FileWriteDouble(handle, m_margin_ema);
      FileWriteInteger(handle, (m_use_hash ? 1 : 0));
      FileWriteInteger(handle, (m_use_hash2 ? 1 : 0));
      FileWriteInteger(handle, m_pa_mode);
      FileWriteDouble(handle, m_conf_r);
      FileWriteInteger(handle, (m_loss_ready ? 1 : 0));
      FileWriteDouble(handle, m_loss_fast);
      FileWriteDouble(handle, m_loss_slow);
      FileWriteInteger(handle, m_drift_cooldown);
      FileWriteInteger(handle, (m_guard_ready ? 1 : 0));
      FileWriteInteger(handle, (m_guard_use_avg ? 1 : 0));
      FileWriteDouble(handle, m_guard_live_fast);
      FileWriteDouble(handle, m_guard_live_slow);
      FileWriteDouble(handle, m_guard_avg_fast);
      FileWriteDouble(handle, m_guard_avg_slow);
      FileWriteInteger(handle, (m_hash_occ_ready ? 1 : 0));
      FileWriteDouble(handle, m_hash2_scale);
      FileWriteInteger(handle, m_rep_head);
      FileWriteInteger(handle, m_rep_size);
      FileWriteDouble(handle, m_mv_lv_b);
      FileWriteInteger(handle, m_cal3_steps);
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         FileWriteDouble(handle, m_cls_ema[c]);
         FileWriteDouble(handle, m_cal3_b[c]);
         for(int j=0; j<FXAI_PA_CLASS_COUNT; j++)
            FileWriteDouble(handle, m_cal3_w[c][j]);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            FileWriteDouble(handle, m_w[c][i]);
            FileWriteDouble(handle, m_w_avg[c][i]);
            FileWriteDouble(handle, m_sigma[c][i]);
         }
         for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
         {
            FileWriteDouble(handle, m_hw1[c][i]);
            FileWriteDouble(handle, m_hw1_avg[c][i]);
            FileWriteDouble(handle, m_hsigma1[c][i]);
         }
         for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
         {
            FileWriteDouble(handle, m_hw2[c][i]);
            FileWriteDouble(handle, m_hw2_avg[c][i]);
            FileWriteDouble(handle, m_hsigma2[c][i]);
         }
         for(int b=0; b<FXAI_PA_CAL_BINS; b++)
         {
            FileWriteDouble(handle, m_cal3_iso_pos[c][b]);
            FileWriteDouble(handle, m_cal3_iso_cnt[c][b]);
         }
      }
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         FileWriteDouble(handle, m_mv_mu_w[i]);
         FileWriteDouble(handle, m_mv_lv_w[i]);
      }
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         FileWriteInteger(handle, m_hash_occ1[i]);
         FileWriteDouble(handle, m_hash_bw1[i]);
         FileWriteDouble(handle, m_mv_mu_hw1[i]);
         FileWriteDouble(handle, m_mv_lv_hw1[i]);
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         FileWriteInteger(handle, m_hash_occ2[i]);
         FileWriteDouble(handle, m_hash_bw2[i]);
         FileWriteDouble(handle, m_mv_mu_hw2[i]);
         FileWriteDouble(handle, m_mv_lv_hw2[i]);
      }
      for(int i=0; i<FXAI_PA_REPLAY; i++)
      {
         FileWriteInteger(handle, m_rep_cls[i]);
         FileWriteDouble(handle, m_rep_move[i]);
         FileWriteDouble(handle, m_rep_w[i]);
         FileWriteDouble(handle, m_rep_hard[i]);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            FileWriteDouble(handle, m_rep_x[i][k]);
      }
      return true;
   }

   virtual bool LoadModelState(const int handle, const int version)
   {
      m_steps = FileReadInteger(handle);
      m_mv_steps = FileReadInteger(handle);
      m_margin_ready = (FileReadInteger(handle) != 0);
      m_margin_ema = FileReadDouble(handle);
      m_use_hash = (FileReadInteger(handle) != 0);
      m_use_hash2 = (FileReadInteger(handle) != 0);
      m_pa_mode = FileReadInteger(handle);
      m_conf_r = FileReadDouble(handle);
      m_loss_ready = (FileReadInteger(handle) != 0);
      m_loss_fast = FileReadDouble(handle);
      m_loss_slow = FileReadDouble(handle);
      m_drift_cooldown = FileReadInteger(handle);
      m_guard_ready = (FileReadInteger(handle) != 0);
      m_guard_use_avg = (FileReadInteger(handle) != 0);
      m_guard_live_fast = FileReadDouble(handle);
      m_guard_live_slow = FileReadDouble(handle);
      m_guard_avg_fast = FileReadDouble(handle);
      m_guard_avg_slow = FileReadDouble(handle);
      m_hash_occ_ready = (FileReadInteger(handle) != 0);
      m_hash2_scale = FileReadDouble(handle);
      m_rep_head = FileReadInteger(handle);
      m_rep_size = FileReadInteger(handle);
      m_mv_lv_b = FileReadDouble(handle);
      m_cal3_steps = FileReadInteger(handle);
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         m_cls_ema[c] = FileReadDouble(handle);
         m_cal3_b[c] = FileReadDouble(handle);
         for(int j=0; j<FXAI_PA_CLASS_COUNT; j++)
            m_cal3_w[c][j] = FileReadDouble(handle);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_w[c][i] = FileReadDouble(handle);
            m_w_avg[c][i] = FileReadDouble(handle);
            m_sigma[c][i] = FileReadDouble(handle);
         }
         for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
         {
            m_hw1[c][i] = FileReadDouble(handle);
            m_hw1_avg[c][i] = FileReadDouble(handle);
            m_hsigma1[c][i] = FileReadDouble(handle);
         }
         for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
         {
            m_hw2[c][i] = FileReadDouble(handle);
            m_hw2_avg[c][i] = FileReadDouble(handle);
            m_hsigma2[c][i] = FileReadDouble(handle);
         }
         for(int b=0; b<FXAI_PA_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = FileReadDouble(handle);
            m_cal3_iso_cnt[c][b] = FileReadDouble(handle);
         }
      }
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_mv_mu_w[i] = FileReadDouble(handle);
         m_mv_lv_w[i] = FileReadDouble(handle);
      }
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         m_hash_occ1[i] = FileReadInteger(handle);
         m_hash_bw1[i] = FileReadDouble(handle);
         m_mv_mu_hw1[i] = FileReadDouble(handle);
         m_mv_lv_hw1[i] = FileReadDouble(handle);
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         m_hash_occ2[i] = FileReadInteger(handle);
         m_hash_bw2[i] = FileReadDouble(handle);
         m_mv_mu_hw2[i] = FileReadDouble(handle);
         m_mv_lv_hw2[i] = FileReadDouble(handle);
      }
      for(int i=0; i<FXAI_PA_REPLAY; i++)
      {
         m_rep_cls[i] = FileReadInteger(handle);
         m_rep_move[i] = FileReadDouble(handle);
         m_rep_w[i] = FileReadDouble(handle);
         m_rep_hard[i] = FileReadDouble(handle);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_rep_x[i][k] = FileReadDouble(handle);
      }
      return true;
   }

   // Legacy compatibility path.
   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      UpdateWeighted(cls, x, hp, 1.0, 1.0, pseudo_move, false);
   }

