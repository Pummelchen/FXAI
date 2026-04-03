   CFXAIAISTMN(void)
   {
      m_initialized = false;
      InitWeights();
   }

   virtual int AIId(void) const { return (int)AI_STMN; }
   virtual string AIName(void) const { return "ai_stmn"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 16, 128);

   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(xa, probs, mu, logv, q25, q75);

      double den = probs[FXAI_STMN_BUY] + probs[FXAI_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_STMN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double active = FXAI_Clamp(1.0 - probs[FXAI_STMN_SKIP], 0.0, 1.0);
      class_probs[(int)FXAI_LABEL_BUY] = p_dir_cal * active;
      class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_dir_cal) * active;
      class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;

      double spread_amp = MathMax(0.0, q75 - q25);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * spread_amp + 0.20 * sigma);
      expected_move_points = amp * active;
      if(expected_move_points <= 0.0)
         expected_move_points = MathMax(PredictMoveHeadRaw(xa), m_move_ema_abs);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      EnsureInitialized(hp);

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(xa, probs, mu, logv, q25, q75);

      double den = probs[FXAI_STMN_BUY] + probs[FXAI_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_STMN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double active = FXAI_Clamp(1.0 - probs[FXAI_STMN_SKIP], 0.0, 1.0);
      out.class_probs[(int)FXAI_LABEL_BUY] = p_dir_cal * active;
      out.class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_dir_cal) * active;
      out.class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;
      NormalizeClassDistribution(out.class_probs);

      double spread_amp = MathMax(0.0, q75 - q25);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * spread_amp + 0.20 * sigma);
      out.move_mean_points = MathMax(0.0, amp * active);
      out.move_q25_points = MathMax(0.0, q25 * active);
      out.move_q50_points = MathMax(out.move_q25_points, MathAbs(mu) * active);
      out.move_q75_points = MathMax(out.move_q50_points, q75 * active);
      double dir_conf = MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]);
      out.confidence = FXAI_Clamp(0.75 * dir_conf + 0.25 * active, 0.0, 1.0);
      out.reliability = FXAI_Clamp(1.0 / (1.0 + sigma), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      if(out.move_mean_points <= 0.0)
         out.move_mean_points = MathMax(PredictMoveHeadRaw(xa), m_move_ema_abs);
      PredictNativeQualityHeads(xa,
                                active,
                                FXAI_Clamp(1.0 / (1.0 + spread_amp + sigma), 0.0, 1.0),
                                out.confidence,
                                out);
      return true;
   }


   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      InitWeights();
      m_initialized = true;
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_initialized) return;
      InitWeights();
      m_initialized = true;
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

      FXAIAIHyperParams hs = ScaleHyperParamsForMove(hp, move_points);
      double sw = FXAI_Clamp(MoveSampleWeight(x, move_points), 0.25, 8.00);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      UpdateInputStats(xa);

      int cls = ResolveClass(y, x, move_points);
      double cost = InputCostProxyPoints(x);

      double h_prev[FXAI_AI_MLP_HIDDEN];
      GetLastHidden(h_prev);

      PushTrainSample(cls, xa, move_points, cost, sw, h_prev);
      if(m_train_len >= 4)
         TrainTBPTT(hs);

      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;

      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, xn);

      double node_z[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double attn[FXAI_STMN_NODES][FXAI_STMN_NODES];
      double msg[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double node_o[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double pool[FXAI_STMN_NODES];
      double graph[FXAI_AI_MLP_HIDDEN];

      double z_gate[FXAI_AI_MLP_HIDDEN];
      double r_gate[FXAI_AI_MLP_HIDDEN];
      double h_cand[FXAI_AI_MLP_HIDDEN];
      double h_new[FXAI_AI_MLP_HIDDEN];
      double logits[FXAI_STMN_CLASS_COUNT];

      SpatialForward(xn, node_z, attn, msg, node_o, pool, graph);
      TemporalForward(graph, h_prev, z_gate, r_gate, h_cand, h_new);
      HeadForward(h_new, logits, probs, mu, logv, q25, q75);

      PushSequenceState(graph, h_new);

      double den = probs[FXAI_STMN_BUY] + probs[FXAI_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir = probs[FXAI_STMN_BUY] / den;

      double cw = ClassWeight(cls, move_points, cost, sw);
      if(cls == FXAI_STMN_BUY)
         UpdateCalibration(p_dir, 1, cw);
      else if(cls == FXAI_STMN_SELL)
         UpdateCalibration(p_dir, 0, cw);
      else
         UpdateCalibration(p_dir, (move_points >= 0.0 ? 1 : 0), 0.25 * cw);

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(xa, move_points, hs, sw);
      UpdateNativeQualityHeads(xa, sw, hs.lr, hs.l2);

      m_step++;
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(x, probs, mu, logv, q25, q75);

      double den = probs[FXAI_STMN_BUY] + probs[FXAI_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_STMN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double p_up = p_dir_cal * FXAI_Clamp(1.0 - probs[FXAI_STMN_SKIP], 0.0, 1.0);
      return FXAI_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(x, probs, mu, logv, q25, q75);

      double spread_amp = MathMax(0.0, q75 - q25);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * spread_amp + 0.20 * sigma);

      double active = FXAI_Clamp(1.0 - probs[FXAI_STMN_SKIP], 0.0, 1.0);
      double ev = amp * active;

      if(ev > 0.0) return ev;
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      return MathMax(PredictMoveHeadRaw(xa), m_move_ema_abs);
   }
