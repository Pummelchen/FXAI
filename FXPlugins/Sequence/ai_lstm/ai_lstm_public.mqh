   CFXAIAILSTM(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_LSTM; }
   virtual string AIName(void) const { return "ai_lstm"; }


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
      m_last_sample_time = 0;
      m_pending_reset_flag = 0.0;
      m_vol_ready = false;
      m_vol_ema = 0.0;

      m_ema_ready = false;
      m_ema_steps = 0;
      m_lstm_replay_head = 0;
      m_lstm_replay_size = 0;
      m_val_ready = false;
      m_val_steps = 0;
      m_quality_degraded = false;
      m_quality_heads.Reset();

      for(int g=0; g<6; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FXAI_LSTM_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      ResetState();
      ResetBatch();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(64, 48);
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
         double ig[FXAI_AI_MLP_HIDDEN];
         double fg[FXAI_AI_MLP_HIDDEN];
         double og[FXAI_AI_MLP_HIDDEN];
         double gg[FXAI_AI_MLP_HIDDEN];
         double zi_hat[FXAI_AI_MLP_HIDDEN];
         double zf_hat[FXAI_AI_MLP_HIDDEN];
         double zo_hat[FXAI_AI_MLP_HIDDEN];
         double zg_hat[FXAI_AI_MLP_HIDDEN];
         double inv_i, inv_f, inv_o, inv_g;
         double c_new[FXAI_AI_MLP_HIDDEN];
         double h_new[FXAI_AI_MLP_HIDDEN];
         ForwardOne(x_step, h_state, c_state, false, use_ema, 0,
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
      if(ArraySize(class_probs) < FXAI_LSTM_CLASS_COUNT)
         ArrayResize(class_probs, FXAI_LSTM_CLASS_COUNT);
      bool use_ema = UseEMAInference();

      double h_state[FXAI_AI_MLP_HIDDEN];
      double c_state[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, use_ema, h_state, c_state);

      double logits[FXAI_LSTM_CLASS_COUNT];
      double probs_raw[FXAI_LSTM_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_state, use_ema, logits, probs_raw, mu, logv, q25, q75);
      Calibrate3(probs_raw, class_probs);

      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, class_probs[(int)FXAI_LABEL_SKIP]);
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0) expected_move_points = 0.70 * ev + 0.30 * m_move_ema_abs;
      else if(ev > 0.0) expected_move_points = ev;
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

      double h_state[FXAI_AI_MLP_HIDDEN];
      double c_state[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, use_ema, h_state, c_state);
      double logits[FXAI_LSTM_CLASS_COUNT], probs_raw[FXAI_LSTM_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_state, use_ema, logits, probs_raw, mu, logv, q25, q75);
      Calibrate3(probs_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, out.class_probs[(int)FXAI_LABEL_SKIP]);
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

      MaybeResetForRegime(xa);

      int cls = NormalizeClassLabel(y, xa, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (int)FXAI_LABEL_SKIP;

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = MoveSampleWeight(x, move_points);
      if(cls == (int)FXAI_LABEL_SKIP) w *= 0.90;
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

      datetime t_now = ResolveContextTime();
      if(t_now <= 0) t_now = TimeCurrent();
      double cost_points = MathMax(0.0, ResolveCostPoints(xa));
      int sess = SessionBucket(t_now);

      AppendBatch(cls, xa, move_points, w, m_pending_reset_flag);
      PushReplay(cls, xa, move_points, cost_points, w, t_now, sess);

      if(m_lstm_replay_size >= 64 && m_batch_size < FXAI_LSTM_TBPTT && (m_seen_updates % 6) == 0)
      {
         int add_n = (m_batch_size <= 6 ? 2 : 1);
         for(int a=0; a<add_n; a++)
         {
            int pick = -1;
            double best_score = -1.0;
            for(int tries=0; tries<12; tries++)
            {
               int li = PluginRandIndex(m_lstm_replay_size);
               int rp = ReplayPos(li);
               double rw = ReplayAgeWeight(m_lstm_replay_time[rp], t_now);
               if(m_replay_session[rp] == sess) rw *= 1.20;
               if(m_replay_y[rp] == (int)FXAI_LABEL_SKIP) rw *= 0.95;
               double edge_r = MathAbs(m_lstm_replay_move[rp]) - MathMax(0.0, m_lstm_replay_cost[rp]);
               if(edge_r > 0.0) rw *= (1.0 + 0.04 * MathMin(edge_r, 20.0));
               else rw *= 0.80;
               if(rw > best_score)
               {
                  best_score = rw;
                  pick = rp;
               }
            }
            if(pick >= 0)
            {
               double xr[FXAI_AI_WEIGHTS];
               for(int i=0; i<FXAI_AI_WEIGHTS; i++) xr[i] = m_lstm_replay_x[pick][i];
               double edge_r = MathAbs(m_lstm_replay_move[pick]) - MathMax(0.0, m_lstm_replay_cost[pick]);
               double rw = FXAI_Clamp(m_replay_w[pick] * ReplayAgeWeight(m_lstm_replay_time[pick], t_now), 0.10, 6.00);
               if(edge_r > 0.0) rw *= (1.0 + 0.03 * MathMin(edge_r, 20.0));
               AppendBatch(m_replay_y[pick], xr, m_lstm_replay_move[pick], rw, 1.0);
               if(m_batch_size >= FXAI_LSTM_TBPTT) break;
            }
         }
      }

      if(m_batch_size >= FXAI_LSTM_TBPTT ||
         (m_batch_size >= 4 && (m_seen_updates % (m_quality_degraded ? 2 : 4)) == 0))
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
