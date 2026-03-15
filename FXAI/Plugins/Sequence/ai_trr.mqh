#ifndef __FXAI_AI_TRR_MQH__
#define __FXAI_AI_TRR_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_TRR_DIMS      8
#define FXAI_TRR_BUF       32
#define FXAI_TRR_TOPO      12
#define FXAI_TRR_RES       24
#define FXAI_TRR_ZF        (1 + FXAI_TRR_TOPO + 4)
#define FXAI_TRR_READOUT   (1 + (2 * FXAI_TRR_RES))

class CFXAIAITRR : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_steps;
   int    m_hist_count;
   int    m_head;
   double m_hist[FXAI_TRR_DIMS][FXAI_TRR_BUF];
   double m_last_feat[FXAI_TRR_DIMS];
   double m_reservoir[FXAI_TRR_RES];
   double m_reservoir_slow[FXAI_TRR_RES];
   double m_in_w[FXAI_TRR_RES][FXAI_TRR_ZF];
   double m_rec_w[FXAI_TRR_RES][FXAI_TRR_RES];
   double m_cls_w[3][FXAI_TRR_READOUT];
   double m_cls_g2[3][FXAI_TRR_READOUT];
   double m_mv_w[FXAI_TRR_READOUT];
   double m_mv_g2[FXAI_TRR_READOUT];
   double m_transition_ema;
   double m_dispersion_ema;
   double m_recurrence_ema;
   double m_move_edge_ema;
   double m_session_bias[4][3];
   double m_session_move_scale[4];
   CFXAITernaryCalibrator m_cal3;

   double ClampProb(const double p) const
   {
      return FXAI_Clamp(p, 0.0005, 0.9995);
   }

   int SessionBucket(void) const
   {
      MqlDateTime dt;
      TimeToStruct(ResolveContextTime(), dt);
      if(dt.hour < 6) return 0;
      if(dt.hour < 12) return 1;
      if(dt.hour < 17) return 2;
      return 3;
   }

   void ResetModel(void)
   {
      m_initialized = false;
      m_steps = 0;
      m_hist_count = 0;
      m_head = 0;
      for(int d=0; d<FXAI_TRR_DIMS; d++)
      {
         m_last_feat[d] = 0.0;
         for(int i=0; i<FXAI_TRR_BUF; i++) m_hist[d][i] = 0.0;
      }
      for(int i=0; i<FXAI_TRR_RES; i++)
      {
         m_reservoir[i] = 0.0;
         m_reservoir_slow[i] = 0.0;
         for(int k=0; k<FXAI_TRR_ZF; k++) m_in_w[i][k] = 0.0;
         for(int j=0; j<FXAI_TRR_RES; j++) m_rec_w[i][j] = 0.0;
      }
      for(int c=0; c<3; c++)
      {
         for(int k=0; k<FXAI_TRR_READOUT; k++)
         {
            m_cls_w[c][k] = 0.0;
            m_cls_g2[c][k] = 0.0;
         }
      }
      for(int k=0; k<FXAI_TRR_READOUT; k++)
      {
         m_mv_w[k] = 0.0;
         m_mv_g2[k] = 0.0;
      }
      m_transition_ema = 0.0;
      m_dispersion_ema = 0.0;
      m_recurrence_ema = 0.0;
      m_move_edge_ema = 0.0;
      for(int b=0; b<4; b++)
      {
         for(int c=0; c<3; c++) m_session_bias[b][c] = 0.0;
         m_session_move_scale[b] = 1.0;
      }
   }

   double HashToWeight(const int a, const int b, const double scale) const
   {
      int h = (a + 1) * 1103 + (b + 7) * 313 + (a + 11) * (b + 17);
      h = h ^ (h >> 3);
      double u = (double)(h % 2001 - 1000) / 1000.0;
      return scale * u;
   }

   void SeedFixedWeights(void)
   {
      for(int i=0; i<FXAI_TRR_RES; i++)
      {
         for(int k=0; k<FXAI_TRR_ZF; k++)
            m_in_w[i][k] = HashToWeight(i, k, 0.22);

         for(int j=0; j<FXAI_TRR_RES; j++)
         {
            double w = 0.0;
            if(j == i) w = 0.15;
            else if(((i * 7 + j * 11) % 9) == 0) w = HashToWeight(i, j, 0.16);
            else if(((i * 13 + j * 5) % 17) == 0) w = HashToWeight(i + 17, j + 19, 0.10);
            m_rec_w[i][j] = w;
         }
      }

      // Conservative readout priors.
      m_cls_w[(int)FXAI_LABEL_BUY][0]  =  0.05;
      m_cls_w[(int)FXAI_LABEL_SELL][0] =  0.05;
      m_cls_w[(int)FXAI_LABEL_SKIP][0] =  0.20;
      m_mv_w[0] = 0.10;

      m_initialized = true;
   }

   void EnsureBootstrapped(void)
   {
      if(m_initialized) return;
      ResetModel();
      SeedFixedWeights();
   }

   double SafeX(const double &x[], const int idx) const
   {
      if(idx < 0 || idx >= ArraySize(x)) return 0.0;
      double v = x[idx];
      if(!MathIsValidNumber(v)) return 0.0;
      return FXAI_ClipSym(v, 8.0);
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         xa[i] = (i < xn ? SafeX(x, i) : 0.0);

      int win_n = CurrentWindowSize();
      if(win_n <= 1) return;

      double mean1 = CurrentWindowFeatureMean(0);
      double mean2 = CurrentWindowFeatureMean(1);
      double mean6 = CurrentWindowFeatureMean(5);
      double first1 = CurrentWindowValue(0, 1);
      double last1  = CurrentWindowValue(win_n - 1, 1);
      double first2 = CurrentWindowValue(0, 2);
      double last2  = CurrentWindowValue(win_n - 1, 2);
      double vol1 = 0.0;
      for(int i=0; i<win_n; i++)
      {
         double d = CurrentWindowValue(i, 1) - mean1;
         vol1 += d * d;
      }
      vol1 = MathSqrt(vol1 / (double)win_n);
      double attn[];
      double conv_fast[];
      double conv_slow[];
      double block[];
      double k_fast[3] = {0.62, 0.24, 0.14};
      double k_slow[5] = {0.32, 0.24, 0.18, 0.16, 0.10};
      int seq_span = MathMax(MathMin(win_n, FXAI_MAX_SEQUENCE_BARS), 10);
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_STATE_SPACE, seq_span);
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildSequenceBlockSummaries(x, dims, seq_cfg, k_fast, 3, k_slow, 5, attn, conv_fast, conv_slow, block);

      xa[1] = FXAI_ClipSym(0.40 * xa[1] + 0.16 * mean1 + 0.12 * (first1 - last1) + 0.12 * attn[1] + 0.10 * conv_fast[1] + 0.10 * block[1], 8.0);
      xa[2] = FXAI_ClipSym(0.40 * xa[2] + 0.16 * mean2 + 0.12 * (first2 - last2) + 0.12 * attn[2] + 0.10 * conv_slow[2] + 0.10 * block[2], 8.0);
      xa[6] = FXAI_ClipSym(0.42 * xa[6] + 0.18 * vol1 + 0.12 * MathAbs(attn[6]) + 0.10 * MathAbs(conv_fast[6]) + 0.18 * MathAbs(block[6]), 8.0);
      xa[7] = FXAI_ClipSym(0.42 * xa[7] + 0.18 * mean6 + 0.12 * attn[7] + 0.12 * conv_slow[7] + 0.16 * block[7], 8.0);
      xa[10] = FXAI_ClipSym(0.72 * xa[10] + 0.10 * attn[10] + 0.08 * conv_fast[10] + 0.10 * block[10], 8.0);
      xa[11] = FXAI_ClipSym(0.72 * xa[11] + 0.10 * attn[11] + 0.08 * conv_slow[11] + 0.10 * block[11], 8.0);
   }

   void BuildDerived(const double &x[], double &f[]) const
   {
      double a1 = SafeX(x, 1), a2 = SafeX(x, 2), a3 = SafeX(x, 3), a4 = SafeX(x, 4);
      double a5 = SafeX(x, 5), a6 = SafeX(x, 6), a7 = SafeX(x, 7), a8 = SafeX(x, 8);
      double a9 = SafeX(x, 9), a10 = SafeX(x, 10), a11 = SafeX(x, 11), a12 = SafeX(x, 12);

      f[0] = FXAI_ClipSym(0.55*a1 + 0.28*a2 + 0.10*a3, 6.0);                           // impulse
      f[1] = FXAI_ClipSym(0.75*MathAbs(a6) + 0.30*MathAbs(a7), 6.0);                   // vol
      f[2] = FXAI_ClipSym(-0.42*a1 + 0.34*a5 - 0.14*a10, 6.0);                         // reversion
      f[3] = FXAI_ClipSym(0.50*MathAbs(a2 - a5) + 0.25*MathAbs(a3 - a4), 6.0);         // compression/breakout
      f[4] = FXAI_ClipSym(0.32*a9 + 0.22*a10 - 0.18*a11 + 0.12*a12, 6.0);              // asymmetry
      f[5] = FXAI_ClipSym(0.85*MathAbs(a7) + 0.15*MathAbs(a8), 6.0);                    // friction/noise
      f[6] = FXAI_ClipSym(0.30*a3 + 0.25*a4 + 0.18*a8, 6.0);                            // curvature hint
      f[7] = FXAI_ClipSym(0.35*a2 - 0.20*a4 + 0.15*a12, 6.0);                           // loop hint
   }

   double HistVal(const int dim, const int lag) const
   {
      if(m_hist_count <= 0) return 0.0;
      int use_lag = lag;
      if(use_lag < 0) use_lag = 0;
      if(use_lag >= m_hist_count) use_lag = m_hist_count - 1;
      int idx = m_head - 1 - use_lag;
      while(idx < 0) idx += FXAI_TRR_BUF;
      return m_hist[dim][idx];
   }

   void PushFeatures(const double &f[])
   {
      for(int d=0; d<FXAI_TRR_DIMS; d++) m_hist[d][m_head] = f[d];
      m_head++;
      if(m_head >= FXAI_TRR_BUF) m_head = 0;
      if(m_hist_count < FXAI_TRR_BUF) m_hist_count++;
      for(int d=0; d<FXAI_TRR_DIMS; d++) m_last_feat[d] = f[d];
   }

   void WindowStats(const int dim, const int win, double &mean_abs, double &sign_changes, double &curv, double &disp) const
   {
      int w = win;
      if(w > m_hist_count) w = m_hist_count;
      if(w <= 1)
      {
         mean_abs = MathAbs(HistVal(dim, 0));
         sign_changes = 0.0;
         curv = 0.0;
         disp = 0.0;
         return;
      }

      double sum_abs = 0.0;
      double prev = HistVal(dim, 0);
      double prev_diff = 0.0;
      double sc = 0.0;
      double cv = 0.0;
      double mean = 0.0;
      for(int i=0; i<w; i++)
      {
         double v = HistVal(dim, i);
         sum_abs += MathAbs(v);
         mean += v;
         if(i > 0)
         {
            if((v > 0.0 && prev < 0.0) || (v < 0.0 && prev > 0.0)) sc += 1.0;
            double diff = v - prev;
            if(i > 1) cv += MathAbs(diff - prev_diff);
            prev_diff = diff;
         }
         prev = v;
      }
      mean /= (double)w;
      double var = 0.0;
      for(int i=0; i<w; i++)
      {
         double dv = HistVal(dim, i) - mean;
         var += dv * dv;
      }
      var /= (double)w;

      mean_abs = sum_abs / (double)w;
      sign_changes = sc / (double)(w - 1);
      curv = cv / (double)MathMax(1, w - 2);
      disp = MathSqrt(var);
   }

   void BuildTopologyLiteFeatures(const double &current_f[], double &topo[]) const
   {
      double ma8 = 0.0, sc8 = 0.0, cv8 = 0.0, dp8 = 0.0;
      double ma16 = 0.0, sc16 = 0.0, cv16 = 0.0, dp16 = 0.0;
      double ma32 = 0.0, sc32 = 0.0, cv32 = 0.0, dp32 = 0.0;

      WindowStats(0, 8, ma8, sc8, cv8, dp8);
      WindowStats(0, 16, ma16, sc16, cv16, dp16);
      WindowStats(0, 32, ma32, sc32, cv32, dp32);

      double vol8 = 0.0, scv8 = 0.0, ccv8 = 0.0, dvol8 = 0.0;
      double vol16 = 0.0, scv16 = 0.0, ccv16 = 0.0, dvol16 = 0.0;
      WindowStats(1, 8, vol8, scv8, ccv8, dvol8);
      WindowStats(1, 16, vol16, scv16, ccv16, dvol16);

      double loop_ma = 0.0, loop_sc = 0.0, loop_cv = 0.0, loop_dp = 0.0;
      WindowStats(7, 16, loop_ma, loop_sc, loop_cv, loop_dp);

      double path_len = 0.0;
      double net_disp = 0.0;
      int w = MathMin(m_hist_count, 16);
      if(w >= 2)
      {
         double first = HistVal(0, w - 1);
         double last  = HistVal(0, 0);
         net_disp = MathAbs(last - first);
         for(int i=0; i<w - 1; i++)
            path_len += MathAbs(HistVal(0, i) - HistVal(0, i + 1));
      }
      double tort = (net_disp > 1e-6 ? path_len / net_disp : path_len);

      double revisit = 0.0;
      if(w >= 4)
      {
         double now = HistVal(0, 0);
         int close_n = 0;
         for(int i=2; i<w; i++)
            if(MathAbs(HistVal(0, i) - now) < 0.25) close_n++;
         revisit = (double)close_n / (double)(w - 2);
      }

      double compress_expand = FXAI_Clamp((dp8 + 0.10) / (dp16 + 0.10), 0.0, 6.0);
      double sign_density = 0.60 * sc8 + 0.40 * sc16;
      double curvature = 0.60 * cv8 + 0.40 * cv16;
      double recurrence = revisit;
      double excursion_asym = FXAI_ClipSym(current_f[4] / (0.25 + ma8), 6.0);
      double local_dispersion = 0.55 * dp8 + 0.45 * dvol8;
      double tortuosity = FXAI_Clamp(tort, 0.0, 12.0);
      double loop_proxy = FXAI_ClipSym(loop_ma - 0.50 * loop_dp + 0.20 * loop_sc, 6.0);
      double resonance = FXAI_ClipSym((ma8 - ma16) + 0.40 * (vol8 - vol16), 6.0);
      double tear = FXAI_ClipSym(compress_expand * (0.5 + curvature) - recurrence, 8.0);
      double transition = FXAI_ClipSym(0.45 * tear + 0.25 * MathAbs(excursion_asym) + 0.20 * tortuosity - 0.15 * recurrence, 8.0);
      double noise_gate = FXAI_ClipSym(current_f[5] + sign_density + 0.30 * dvol16, 8.0);

      topo[0]  = compress_expand;
      topo[1]  = sign_density;
      topo[2]  = curvature;
      topo[3]  = recurrence;
      topo[4]  = excursion_asym;
      topo[5]  = local_dispersion;
      topo[6]  = tortuosity;
      topo[7]  = loop_proxy;
      topo[8]  = resonance;
      topo[9]  = tear;
      topo[10] = transition;
      topo[11] = noise_gate;
   }

   void BuildReservoirInput(const double &topo[], const double &f[], double &z[]) const
   {
      z[0] = 1.0;
      for(int i=0; i<FXAI_TRR_TOPO; i++) z[1 + i] = FXAI_ClipSym(topo[i], 8.0);
      z[1 + FXAI_TRR_TOPO + 0] = f[0];
      z[1 + FXAI_TRR_TOPO + 1] = f[1];
      z[1 + FXAI_TRR_TOPO + 2] = f[2];
      z[1 + FXAI_TRR_TOPO + 3] = f[5];
   }

   void PreviewReservoir(const double &z[], double &fast_out[], double &slow_out[]) const
   {
      for(int i=0; i<FXAI_TRR_RES; i++)
      {
         double in_sum = 0.0;
         for(int k=0; k<FXAI_TRR_ZF; k++) in_sum += m_in_w[i][k] * z[k];
         double rec_sum = 0.0;
         for(int j=0; j<FXAI_TRR_RES; j++) rec_sum += m_rec_w[i][j] * m_reservoir[j];
         double drive = FXAI_Tanh(FXAI_ClipSym(in_sum + rec_sum, 12.0));
         double leak_fast = 0.22 + 0.01 * (double)(i % 5);
         double leak_slow = 0.08 + 0.005 * (double)(i % 5);
         fast_out[i] = (1.0 - leak_fast) * m_reservoir[i] + leak_fast * drive;
         slow_out[i] = (1.0 - leak_slow) * m_reservoir_slow[i] + leak_slow * drive;
      }
   }

   void UpdateReservoir(const double &z[])
   {
      double fast_out[FXAI_TRR_RES];
      double slow_out[FXAI_TRR_RES];
      PreviewReservoir(z, fast_out, slow_out);
      for(int i=0; i<FXAI_TRR_RES; i++)
      {
         m_reservoir[i] = fast_out[i];
         m_reservoir_slow[i] = slow_out[i];
      }
   }

   void BuildReadoutInput(const double &fast_state[], const double &slow_state[], double &r[]) const
   {
      r[0] = 1.0;
      for(int i=0; i<FXAI_TRR_RES; i++)
      {
         r[1 + i] = fast_state[i];
         r[1 + FXAI_TRR_RES + i] = slow_state[i];
      }
   }

   void SoftmaxReadout(const double &r[], double &probs[]) const
   {
      double logits[3];
      for(int c=0; c<3; c++)
      {
         double z = 0.0;
         for(int k=0; k<FXAI_TRR_READOUT; k++) z += m_cls_w[c][k] * r[k];
         logits[c] = FXAI_ClipSym(z, 20.0);
      }
      Softmax3(logits, probs);
   }

   double PredictMoveHeadInternal(const double &r[]) const
   {
      double z = 0.0;
      for(int k=0; k<FXAI_TRR_READOUT; k++) z += m_mv_w[k] * r[k];
      return MathMax(0.0, z);
   }

public:
   CFXAIAITRR(void) : CFXAIAIPlugin()
   {
      Reset();
   }

   virtual int AIId(void) const { return AI_TRR; }
   virtual string AIName(void) const { return "ai_trr"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_RECURRENT, caps, 16, 128);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      ResetModel();
      m_cal3.Reset();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      EnsureBootstrapped();
   }


   virtual bool PredictModelCore(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      EnsureBootstrapped();

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[FXAI_TRR_DIMS];
      double topo[FXAI_TRR_TOPO];
      double z[FXAI_TRR_ZF];
      BuildDerived(xa, f);
      BuildTopologyLiteFeatures(f, topo);
      BuildReservoirInput(topo, f, z);
      double fast_state[FXAI_TRR_RES];
      double slow_state[FXAI_TRR_RES];
      PreviewReservoir(z, fast_state, slow_state);

      double r[FXAI_TRR_READOUT];
      BuildReadoutInput(fast_state, slow_state, r);
      SoftmaxReadout(r, class_probs);

      double move_hat = PredictMoveHeadInternal(r);
      double cost = ResolveCostPoints(x);
      if(cost < 0.0) cost = 0.0;
      double mm = ResolveMinMovePoints();
      if(mm <= 0.0) mm = MathMax(0.10, cost);
      int sb = SessionBucket();
      double edge_hat = MathMax(0.0, move_hat * m_session_move_scale[sb]);

      double transition = FXAI_Clamp(topo[10] / 6.0, 0.0, 1.5);
      double recurrence = FXAI_Clamp(topo[3], 0.0, 1.0);
      double noise = FXAI_Clamp(topo[11] / 6.0, 0.0, 1.5);
      double active = FXAI_Clamp(0.55 * transition + 0.35 * FXAI_Sigmoid(edge_hat / MathMax(mm, 0.10)) - 0.25 * recurrence - 0.20 * noise, 0.0, 1.0);

      double dir_mass = class_probs[(int)FXAI_LABEL_BUY] + class_probs[(int)FXAI_LABEL_SELL];
      double p_buy = 0.5;
      double p_sell = 0.5;
      if(dir_mass > 1e-9)
      {
         p_buy = class_probs[(int)FXAI_LABEL_BUY] / dir_mass;
         p_sell = class_probs[(int)FXAI_LABEL_SELL] / dir_mass;
      }

      double p_raw[3];
      p_raw[(int)FXAI_LABEL_BUY]  = ClampProb(active * p_buy);
      p_raw[(int)FXAI_LABEL_SELL] = ClampProb(active * p_sell);
      p_raw[(int)FXAI_LABEL_SKIP] = ClampProb(MathMax(1.0 - active, recurrence * 0.40 + noise * 0.25));
      double logits_cal[3];
      for(int c=0; c<3; c++) logits_cal[c] = MathLog(MathMax(p_raw[c], 1e-6)) + m_session_bias[sb][c];
      Softmax3(logits_cal, p_raw);
      m_cal3.Calibrate(p_raw, class_probs);
      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int c=0; c<3; c++) class_probs[c] /= s;

      // Publish raw move amplitude; the shared framework handles cost-aware EV gating.
      expected_move_points = MathMax(0.0, edge_hat + cost);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double probs[3];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return false;

      for(int c=0; c<3; c++)
         out.class_probs[c] = probs[c];

      double dir_conf = MathMax(probs[(int)FXAI_LABEL_BUY], probs[(int)FXAI_LABEL_SELL]);
      double skip = probs[(int)FXAI_LABEL_SKIP];
      double sigma = MathMax(0.10, 0.35 * ev + 0.60 * m_recurrence_ema + 0.40 * m_dispersion_ema);
      out.move_mean_points = MathMax(ev, (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = out.move_mean_points;
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      out.confidence = FXAI_Clamp(0.55 * dir_conf + 0.25 * (1.0 - skip) + 0.20 * (1.0 - FXAI_Clamp(m_recurrence_ema, 0.0, 1.0)), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.55 + 0.20 * (1.0 - FXAI_Clamp(m_dispersion_ema, 0.0, 1.0)) + 0.25 * FXAI_Clamp(m_session_move_scale[SessionBucket()] - 0.85, 0.0, 0.5), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(xa,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

protected:
   virtual void TrainModelCore(const int y, const double &x[], const FXAIAIHyperParams &hp, const double move_points)
   {
      EnsureBootstrapped();

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[FXAI_TRR_DIMS];
      double topo[FXAI_TRR_TOPO];
      double z[FXAI_TRR_ZF];
      BuildDerived(xa, f);
      BuildTopologyLiteFeatures(f, topo);
      BuildReservoirInput(topo, f, z);
      UpdateReservoir(z);

      double r[FXAI_TRR_READOUT];
      BuildReadoutInput(m_reservoir, m_reservoir_slow, r);

      double probs[3];
      SoftmaxReadout(r, probs);
      double pred_move = PredictMoveHeadInternal(r);

      int label = y;
      if(label < (int)FXAI_LABEL_SELL || label > (int)FXAI_LABEL_SKIP)
      {
         if(move_points > 0.0) label = (int)FXAI_LABEL_BUY;
         else if(move_points < 0.0) label = (int)FXAI_LABEL_SELL;
         else label = (int)FXAI_LABEL_SKIP;
      }

      double sw = FXAI_Clamp(MoveSampleWeight(x, move_points), 0.25, 4.0);
      double lr = FXAI_Clamp(0.18 * hp.lr * sw, 0.0002, 0.04);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.05);

      double raw_probs[3];
      {
         double cost_now = ResolveCostPoints(xa);
         if(cost_now < 0.0) cost_now = 0.0;
         double mm = ResolveMinMovePoints();
         if(mm <= 0.0) mm = MathMax(0.10, cost_now);
         double transition = FXAI_Clamp(topo[10] / 6.0, 0.0, 1.5);
         double recurrence = FXAI_Clamp(topo[3], 0.0, 1.0);
         double noise = FXAI_Clamp(topo[11] / 6.0, 0.0, 1.5);
         double edge_hat = MathMax(0.0, pred_move * m_session_move_scale[SessionBucket()]);
         double active = FXAI_Clamp(0.55 * transition + 0.35 * FXAI_Sigmoid(edge_hat / MathMax(mm, 0.10)) - 0.25 * recurrence - 0.20 * noise, 0.0, 1.0);
         double dir_mass = probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL];
         double p_buy = 0.5;
         double p_sell = 0.5;
         if(dir_mass > 1e-9)
         {
            p_buy = probs[(int)FXAI_LABEL_BUY] / dir_mass;
            p_sell = probs[(int)FXAI_LABEL_SELL] / dir_mass;
         }
         raw_probs[(int)FXAI_LABEL_BUY] = ClampProb(active * p_buy);
         raw_probs[(int)FXAI_LABEL_SELL] = ClampProb(active * p_sell);
         raw_probs[(int)FXAI_LABEL_SKIP] = ClampProb(MathMax(1.0 - active, recurrence * 0.40 + noise * 0.25));
         double logits_cal[3];
         int sb = SessionBucket();
         for(int c=0; c<3; c++)
            logits_cal[c] = MathLog(MathMax(raw_probs[c], 1e-6)) + m_session_bias[sb][c];
         Softmax3(logits_cal, raw_probs);
      }
      m_cal3.Update(raw_probs, label, sw, lr);

      for(int c=0; c<3; c++)
      {
         double tgt = (c == label ? 1.0 : 0.0);
         double err = tgt - probs[c];
         for(int k=0; k<FXAI_TRR_READOUT; k++)
         {
            double grad = err * r[k] - l2 * m_cls_w[c][k];
            m_cls_g2[c][k] += grad * grad;
            double step = lr / MathSqrt(1.0 + m_cls_g2[c][k]);
            m_cls_w[c][k] = FXAI_ClipSym(m_cls_w[c][k] + step * grad, 4.0);
         }
      }

      double cost = ResolveCostPoints(xa);
      if(cost < 0.0) cost = 0.0;
      double target_move = MathMax(0.0, MathAbs(move_points) - cost);
      double mv_err = target_move - pred_move;
      for(int k=0; k<FXAI_TRR_READOUT; k++)
      {
         double grad = mv_err * r[k] - 0.01 * m_mv_w[k];
         m_mv_g2[k] += grad * grad;
         double step = FXAI_Clamp(0.15 * hp.lr * sw, 0.0002, 0.03) / MathSqrt(1.0 + m_mv_g2[k]);
         m_mv_w[k] = FXAI_ClipSym(m_mv_w[k] + step * grad, 8.0);
      }

      m_transition_ema = 0.96 * m_transition_ema + 0.04 * topo[10];
      m_dispersion_ema = 0.96 * m_dispersion_ema + 0.04 * topo[5];
      m_recurrence_ema = 0.96 * m_recurrence_ema + 0.04 * topo[3];
      FXAI_UpdateMoveEMA(m_move_edge_ema, m_move_ready, target_move, 0.02);
      int sb = SessionBucket();
      for(int c=0; c<3; c++)
      {
         double tgt = (c == label ? 1.0 : 0.0);
         m_session_bias[sb][c] = FXAI_ClipSym(0.995 * m_session_bias[sb][c] + 0.010 * (tgt - probs[c]), 1.5);
      }
      m_session_move_scale[sb] = FXAI_Clamp(0.995 * m_session_move_scale[sb] + 0.005 * (target_move / MathMax(pred_move, 0.10)), 0.70, 1.50);

      PushFeatures(f);
      UpdateNativeQualityHeads(xa, sw, hp.lr, hp.l2);
      m_steps++;
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3];
      double exp_move = 0.0;
      if(PredictModelCore(x, hp, probs, exp_move))
         return probs[(int)FXAI_LABEL_BUY] / MathMax(probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL], 1e-6);
      return 0.50;
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      FXAIAIModelOutputV4 out;
      if(PredictDistributionCore(x, hp, out) && out.move_mean_points > 0.0)
         return out.move_mean_points;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return 0.0;
   }
};

#endif // __FXAI_AI_TRR_MQH__
