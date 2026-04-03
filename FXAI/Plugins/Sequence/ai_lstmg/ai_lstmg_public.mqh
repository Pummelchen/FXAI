   CFXAIAILSTMG(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_LSTMG; }
   virtual string AIName(void) const { return "ai_lstmg"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_RECURRENT, caps, 16, 128);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_seen_updates = 0;
      m_adam_t = 0;

      m_last_update_time = 0;
      m_last_session_bucket = -1;
      m_vol_ready = false;
      m_vol_ema = 0.0;

      m_ev_ready = false;
      m_ev_steps = 0;
      m_ev_a = 1.0;
      m_ev_b = 0.0;

      m_selftest_done = false;
      m_selftest_ok = false;
      m_quality_heads.Reset();

      for(int g=0; g<FXAI_LSTMG_OPT_GROUPS; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      for(int s=0; s<4; s++)
      {
         m_sess_total[s] = 0.0;
         for(int c=0; c<FXAI_LSTMG_CLASS_COUNT; c++)
            m_cls_count[s][c] = 0.0;
      }

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FXAI_LSTMG_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FXAI_LSTMG_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      m_ema_ready = false;
      m_ema_steps = 0;

      ResetState();
      ResetBatch();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights();

      if(!m_selftest_done)
         RunSelfTests();
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(80, 56);
   }

   void ForwardSequenceContext(const double &x[],
                               const bool use_ema,
                               double &h_out[],
                               double &c_out[]) const
   {
      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_RECURRENT, SequenceContextSpan());
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildChronologicalSequenceTensorConfigured(x, seq_cfg, seq, seq_len);

      double h_state[FXAI_AI_MLP_HIDDEN];
      double c_state[FXAI_AI_MLP_HIDDEN];
      BuildStateWithPending(h_state, c_state, use_ema);

      for(int t=0; t<seq_len; t++)
      {
         double x_step[FXAI_AI_WEIGHTS];
         FXAI_SequenceCopyRow(seq, t, x_step);
         double drop_mask[FXAI_AI_MLP_HIDDEN];
         double zone_mask[FXAI_AI_MLP_HIDDEN];
         double zi_hat[FXAI_AI_MLP_HIDDEN];
         double zf_hat[FXAI_AI_MLP_HIDDEN];
         double zo_hat[FXAI_AI_MLP_HIDDEN];
         double zg_hat[FXAI_AI_MLP_HIDDEN];
         double inv_i, inv_f, inv_o, inv_g;
         double ig[FXAI_AI_MLP_HIDDEN];
         double fg[FXAI_AI_MLP_HIDDEN];
         double og[FXAI_AI_MLP_HIDDEN];
         double gg[FXAI_AI_MLP_HIDDEN];
         double c_new[FXAI_AI_MLP_HIDDEN];
         double h_new[FXAI_AI_MLP_HIDDEN];

         ForwardOne(x_step, h_state, c_state, false, use_ema, 0,
                    drop_mask, zone_mask,
                    zi_hat, zf_hat, zo_hat, zg_hat,
                    inv_i, inv_f, inv_o, inv_g,
                    ig, fg, og, gg, c_new, h_new);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            h_state[h] = h_new[h];
            c_state[h] = c_new[h];
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         h_out[h] = h_state[h];
         c_out[h] = c_state[h];
      }
   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      if(ArraySize(class_probs) < FXAI_LSTMG_CLASS_COUNT)
         ArrayResize(class_probs, FXAI_LSTMG_CLASS_COUNT);
      bool use_ema = UseEMAInference();

      double h_state[FXAI_AI_MLP_HIDDEN];
      double c_state[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, use_ema, h_state, c_state);

      double logits[FXAI_LSTMG_CLASS_COUNT];
      double probs_raw[FXAI_LSTMG_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_state, use_ema, logits, probs_raw, mu, logv, q25, q75);

      Calibrate3(probs_raw, class_probs);

      double ev_raw = ExpectedMoveFromHeads(mu, logv, q25, q75, class_probs[(int)FXAI_LABEL_SKIP]);
      double ev_cal = CalibrateEV(ev_raw);
      if(ev_cal > 0.0 && m_move_ready && m_move_ema_abs > 0.0) expected_move_points = 0.70 * ev_cal + 0.30 * m_move_ema_abs;
      else if(ev_cal > 0.0) expected_move_points = ev_cal;
      else expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      return true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = 0.0;
      if(y == (int)FXAI_LABEL_BUY) pseudo_move = 1.0;
      else if(y == (int)FXAI_LABEL_SELL) pseudo_move = -1.0;
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      bool use_ema = UseEMAInference();
      double h_state[FXAI_AI_MLP_HIDDEN], c_state[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, use_ema, h_state, c_state);
      double logits[FXAI_LSTMG_CLASS_COUNT], probs_raw[FXAI_LSTMG_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_state, use_ema, logits, probs_raw, mu, logv, q25, q75);
      Calibrate3(probs_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double ev = CalibrateEV(ExpectedMoveFromHeads(mu, logv, q25, q75, out.class_probs[(int)FXAI_LABEL_SKIP]));
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, MathAbs(q25) * FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0));
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, MathAbs(q75) * FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0));
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * (1.0 - out.class_probs[(int)FXAI_LABEL_SKIP]) + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double bank_mfe = 0.0, bank_mae = 0.0, bank_hit = 1.0, bank_path = 0.5, bank_fill = 0.5, bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_quality_heads.Predict(xa,
                              out.move_mean_points,
                              FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                              out.reliability,
                              out.confidence,
                              bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust,
                              out);
      return true;
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_seen_updates++;
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double cost_points = ResolveCostPoints(xa);
      if(cost_points < 0.0) cost_points = 0.0;

      double min_move_points = ResolveMinMovePoints();
      if(min_move_points <= 0.0)
         min_move_points = MathMax(0.10, cost_points + 0.10);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);

      int sess = 0;
      int boundary = CheckRegimeBoundary(xa, sess);
      if(boundary > 0 && m_batch_size > 0)
         TrainBatch(h);

      if(boundary == 2)
      {
         ResetState();
         ResetBatch();
      }
      else if(boundary == 1)
      {
         DecayState(0.65);
         ResetBatch();
      }

      double reset_flag = (boundary > 0 ? 1.0 : 0.0);

      int cls = DeriveClassLabel(y, xa, move_points, cost_points, min_move_points);
      double w = MoveSampleWeight(xa, move_points) * ClassBalanceWeight(sess, cls);
      if(cls == (int)FXAI_LABEL_SKIP) w *= 0.80;
      w = FXAI_Clamp(w, 0.10, 6.00);
      m_quality_heads.Update(xa,
                             w,
                             TargetMFEPoints(),
                             FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                             TargetHitTimeFrac(),
                             TargetPathRisk(),
                             TargetFillRisk(),
                             TargetMaskedStep(),
                             TargetNextVol(),
                             TargetRegimeShift(),
                             TargetContextLead(),
                             h.lr,
                             h.l2);

      AppendBatch(cls, xa, move_points, cost_points, w, reset_flag);
      UpdateClassStats(sess, cls);

      if(m_batch_size >= FXAI_LSTMG_TBPTT ||
         (m_batch_size >= 8 && (m_seen_updates % 4) == 0))
      {
         TrainBatch(h);
      }
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return 0.5;

      double den = probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double ev = -1.0;
      if(PredictModelCore(x, hp, probs, ev) && ev > 0.0)
         return ev;

      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
