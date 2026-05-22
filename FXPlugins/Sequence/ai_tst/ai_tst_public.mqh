   CFXAIAITST(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TST; }
   virtual string AIName(void) const { return "ai_tst"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 256);

   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(FXAI_TST_SEQ, 72);
   }

   void ForwardSequenceContext(const double &x[],
                               double &state_embed[],
                               double &state_local[],
                               double &state_attn[],
                               double &state_final[],
                               double &group_gate[],
                               double &head_ctx[][FXAI_TST_D_HEAD])
   {
      int saved_ptr = m_seq_ptr;
      int saved_len = m_seq_len;
      double saved_seq[FXAI_TST_SEQ][FXAI_AI_MLP_HIDDEN];
      for(int t=0; t<FXAI_TST_SEQ; t++)
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            saved_seq[t][h] = m_seq_h[t][h];

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      int seq_mask[];
      double seq_pos_bias[];
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_TRANSFORMER, SequenceContextSpan());
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      double k_fast[3] = {1.00, 0.00, -1.00};
      double k_slow[5] = {0.10, 0.20, 0.40, 0.20, 0.10};
      BuildSequenceBlockSequence(x, dims, seq_cfg, k_fast, 3, k_slow, 5, seq, seq_len, seq_mask, seq_pos_bias);

      ResetSequence();
      for(int t=0; t<seq_len; t++)
      {
         double x_step[FXAI_AI_WEIGHTS];
         FXAI_SequenceCopyRow(seq, t, x_step);
         ForwardStep(x_step, true, false, -1, state_embed, state_local, state_attn, state_final, group_gate, head_ctx);
      }

      m_seq_ptr = saved_ptr;
      m_seq_len = saved_len;
      for(int t=0; t<FXAI_TST_SEQ; t++)
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            m_seq_h[t][h] = saved_seq[t][h];
   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double emb[FXAI_AI_MLP_HIDDEN];
      double loc[FXAI_AI_MLP_HIDDEN];
      double att[FXAI_AI_MLP_HIDDEN];
      double fin[FXAI_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FXAI_TST_HEADS][FXAI_TST_D_HEAD];
      ForwardSequenceContext(x, emb, loc, att, fin, grp, ctx);

      double logits[FXAI_TST_CLASS_COUNT];
      double probs[FXAI_TST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FXAI_TST_HORIZONS];
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75, mu_h[0], mu_h[1], mu_h[2]);

      double den = probs[FXAI_TST_BUY] + probs[FXAI_TST_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_TST_BUY] / den;
      double p_dir_cal = CalibrateSessionProb(SessionBucketNow(), p_dir_raw);
      double active = FXAI_Clamp(1.0 - probs[FXAI_TST_SKIP], 0.0, 1.0);
      class_probs[(int)FXAI_LABEL_BUY] = p_dir_cal * active;
      class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_dir_cal) * active;
      class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;

      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double ev_h = 0.50 * MathAbs(mu_h[0]) + 0.30 * MathAbs(mu_h[1]) + 0.20 * MathAbs(mu_h[2]);
      expected_move_points = (0.55 * MathAbs(mu) + 0.45 * ev_h + 0.22 * sigma + 0.10 * iqr) * active;
      if(expected_move_points <= 0.0)
         expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double emb[FXAI_AI_MLP_HIDDEN], loc[FXAI_AI_MLP_HIDDEN], att[FXAI_AI_MLP_HIDDEN], fin[FXAI_AI_MLP_HIDDEN], grp[3];
      double ctx[FXAI_TST_HEADS][FXAI_TST_D_HEAD];
      ForwardSequenceContext(x, emb, loc, att, fin, grp, ctx);
      double logits[FXAI_TST_CLASS_COUNT], probs[FXAI_TST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0, mu_h[FXAI_TST_HORIZONS];
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75, mu_h[0], mu_h[1], mu_h[2]);
      double den = probs[FXAI_TST_BUY] + probs[FXAI_TST_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_TST_BUY] / den;
      double p_dir_cal = CalibrateSessionProb(SessionBucketNow(), p_dir_raw);
      double active = FXAI_Clamp(1.0 - probs[FXAI_TST_SKIP], 0.0, 1.0);
      out.class_probs[(int)FXAI_LABEL_BUY] = p_dir_cal * active;
      out.class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_dir_cal) * active;
      out.class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;
      NormalizeClassDistribution(out.class_probs);
      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 30.0);
      double ev_h = 0.50 * MathAbs(mu_h[0]) + 0.30 * MathAbs(mu_h[1]) + 0.20 * MathAbs(mu_h[2]);
      double ev = (0.55 * MathAbs(mu) + 0.45 * ev_h + 0.22 * sigma + 0.10 * MathAbs(q75 - q25)) * active;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, MathAbs(q25) * active);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, MathAbs(q75) * active);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * active + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
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


   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_cfg_ready = false;
      m_step = 0;
      m_adam_t = 0;
      m_quality_heads.Reset();
      ResetSequence();
      ResetTrainBuffer();
      ResetInputNorm();
      ResetReplay();
      ResetSessionCal();
      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
      if(!m_cfg_ready) ConfigureFromHP(hp);
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      ConfigureFromHP(hp);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      // Controlled reset policy to avoid state bleed in non-stationary jumps.
      if((m_step % 512) == 0)
         ResetSequence();
      if(MathAbs(xa[1]) > 8.0 || MathAbs(xa[2]) > 8.0)
         ResetSequence();

      UpdateInputStats(xa);

      int cls = MapClass(y, xa, move_points);
      double sw = MoveSampleWeight(xa, move_points);
      sw = FXAI_Clamp(sw, 0.25, 4.00);
      m_quality_heads.Update(xa,
                             sw,
                             TargetMFEPoints(),
                             FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                             TargetHitTimeFrac(),
                             TargetPathRisk(),
                             TargetFillRisk(),
                             TargetMaskedStep(),
                             TargetNextVol(),
                             TargetRegimeShift(),
                             TargetContextLead(),
                             hp.lr,
                             hp.l2);
      double cost = InputCostProxyPoints(xa);

      // Hard-example replay and class-balance sampler.
      double p_before = PredictProb(xa, h);
      double y01 = (cls == FXAI_TST_BUY ? 1.0 : (cls == FXAI_TST_SELL ? 0.0 : 0.5));
      double hardness = MathAbs(y01 - p_before) + 0.20 * (cls == FXAI_TST_SKIP ? 1.0 : 0.0);
      hardness += 0.03 * MathAbs(MathAbs(move_points) - cost);
      PushReplay(cls, xa, move_points, cost, sw, hardness);

      AppendTrainSample(cls, xa, move_points, cost, sw);

      int local_cnt[FXAI_TST_CLASS_COUNT];
      for(int c=0; c<FXAI_TST_CLASS_COUNT; c++) local_cnt[c] = 0;
      for(int i=0; i<m_train_len; i++)
      {
         int cc = m_train_cls[i];
         if(cc < 0 || cc >= FXAI_TST_CLASS_COUNT) cc = FXAI_TST_SKIP;
         local_cnt[cc]++;
      }
      int minority = 0;
      for(int c=1; c<FXAI_TST_CLASS_COUNT; c++)
      {
         if(local_cnt[c] < local_cnt[minority]) minority = c;
      }

      int ridx = PickReplayIndex(minority);
      if(ridx >= 0)
         AppendReplayToTrain(ridx);
      if((m_step % 3) == 0)
      {
         int ridx2 = PickReplayIndex(-1);
         if(ridx2 >= 0)
            AppendReplayToTrain(ridx2);
      }

      TrainTBPTT(h);

      // Update live state with most recent bar after isolated TBPTT step.
      double emb[FXAI_AI_MLP_HIDDEN];
      double loc[FXAI_AI_MLP_HIDDEN];
      double att[FXAI_AI_MLP_HIDDEN];
      double fin[FXAI_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FXAI_TST_HEADS][FXAI_TST_D_HEAD];
      ForwardStep(xa, true, false, -1, emb, loc, att, fin, grp, ctx);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double emb[FXAI_AI_MLP_HIDDEN];
      double loc[FXAI_AI_MLP_HIDDEN];
      double att[FXAI_AI_MLP_HIDDEN];
      double fin[FXAI_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FXAI_TST_HEADS][FXAI_TST_D_HEAD];
      ForwardSequenceContext(x, emb, loc, att, fin, grp, ctx);

      double logits[FXAI_TST_CLASS_COUNT];
      double probs[FXAI_TST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FXAI_TST_HORIZONS];
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75, mu_h[0], mu_h[1], mu_h[2]);

      double den = probs[FXAI_TST_BUY] + probs[FXAI_TST_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_TST_BUY] / den;
      double p_dir_cal = CalibrateSessionProb(SessionBucketNow(), p_dir_raw);

      double p_up = p_dir_cal * FXAI_Clamp(1.0 - probs[FXAI_TST_SKIP], 0.0, 1.0);
      return FXAI_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double emb[FXAI_AI_MLP_HIDDEN];
      double loc[FXAI_AI_MLP_HIDDEN];
      double att[FXAI_AI_MLP_HIDDEN];
      double fin[FXAI_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FXAI_TST_HEADS][FXAI_TST_D_HEAD];
      ForwardSequenceContext(x, emb, loc, att, fin, grp, ctx);

      double logits[FXAI_TST_CLASS_COUNT];
      double probs[FXAI_TST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FXAI_TST_HORIZONS];
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75, mu_h[0], mu_h[1], mu_h[2]);

      double sigma = MathExp(0.5 * logv);
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double ev_h = 0.50 * MathAbs(mu_h[0]) + 0.30 * MathAbs(mu_h[1]) + 0.20 * MathAbs(mu_h[2]);
      double ev = (0.55 * MathAbs(mu) + 0.45 * ev_h + 0.22 * sigma + 0.10 * iqr) * FXAI_Clamp(1.0 - probs[FXAI_TST_SKIP], 0.0, 1.0);
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.65 * ev + 0.35 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
