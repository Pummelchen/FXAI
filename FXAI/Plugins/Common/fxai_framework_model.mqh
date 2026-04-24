#ifndef __FXAI_FRAMEWORK_MODEL_MQH__
#define __FXAI_FRAMEWORK_MODEL_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_FW_STATE_DIM 16
#define FXAI_FW_REGIMES 3
#define FXAI_FW_RF_TREES 13
#define FXAI_FW_RF_LEAVES 8
#define FXAI_FW_SEQ_HIDDEN 8

#define FXAI_FW_KIND_MSGARCH 1
#define FXAI_FW_KIND_ARIMAX_GARCH 2
#define FXAI_FW_KIND_RANDOM_FOREST 3
#define FXAI_FW_KIND_COINT_VECM 4
#define FXAI_FW_KIND_OU_SPREAD 5
#define FXAI_FW_KIND_PPO 6
#define FXAI_FW_KIND_MICROFLOW 7
#define FXAI_FW_KIND_HMM_REGIME 8
#define FXAI_FW_KIND_ELASTIC_LOGIT 9
#define FXAI_FW_KIND_PROFIT_LOGIT 10
#define FXAI_FW_KIND_CNN_LSTM 11
#define FXAI_FW_KIND_ATTN_CNN_BILSTM 12
#define FXAI_FW_KIND_EMD_HHT 13
#define FXAI_FW_KIND_VMD 14
#define FXAI_FW_KIND_TVP_KALMAN 15
#define FXAI_FW_KIND_PCA_PANEL 16
#define FXAI_FW_KIND_PPP_VALUE 17
#define FXAI_FW_KIND_CARRY 18
#define FXAI_FW_KIND_CMV_PANEL 19
#define FXAI_FW_KIND_TSMOM_VOL 20
#define FXAI_FW_KIND_XSMOM_RANK 21
#define FXAI_FW_KIND_VOL_BREAKOUT 22
#define FXAI_FW_KIND_XRATE_CONSISTENCY 23
#define FXAI_FW_KIND_GRU 24
#define FXAI_FW_KIND_BILSTM 25
#define FXAI_FW_KIND_LSTM_TCN 26

class CFXAIFrameworkModelPlugin : public CFXAIAIPlugin
{
protected:
   int    m_step;
   double m_w[3][FXAI_FW_STATE_DIM];
   double m_v[3][FXAI_FW_STATE_DIM];
   double m_move_ema;
   bool   m_move_ready_local;
   double m_class_mass[3];

   double m_hmm_pi[FXAI_FW_REGIMES];
   double m_hmm_A[FXAI_FW_REGIMES][FXAI_FW_REGIMES];
   double m_regime_mu[FXAI_FW_REGIMES];
   double m_regime_var[FXAI_FW_REGIMES];
   double m_garch_h[FXAI_FW_REGIMES];
   double m_garch_omega[FXAI_FW_REGIMES];
   double m_garch_alpha[FXAI_FW_REGIMES];
   double m_garch_beta[FXAI_FW_REGIMES];

   double m_rf_leaf_mass[FXAI_FW_RF_TREES][FXAI_FW_RF_LEAVES][3];
   double m_rf_leaf_move[FXAI_FW_RF_TREES][FXAI_FW_RF_LEAVES];
   int    m_rf_feature[FXAI_FW_RF_TREES][3];
   double m_rf_threshold[FXAI_FW_RF_TREES][3];

   double m_kalman_state[FXAI_FW_STATE_DIM];
   double m_kalman_cov[FXAI_FW_STATE_DIM];
   double m_pca_mean[FXAI_FW_STATE_DIM];
   double m_pca_loading[3][FXAI_FW_STATE_DIM];
   double m_pca_var[3];

   double m_ou_mean;
   double m_ou_speed;
   double m_ou_var;
   double m_prev_spread;
   bool   m_prev_spread_ready;

   double m_policy_old[3];
   double m_value_w[FXAI_FW_STATE_DIM];

   virtual int FrameworkKind(void) const = 0;
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_OTHER; }
   virtual int FrameworkMinSequenceBars(void) const { return (UsesSequenceWindow() ? 12 : 1); }
   virtual int FrameworkMaxSequenceBars(void) const { return (UsesSequenceWindow() ? FXAI_MAX_SEQUENCE_BARS : 1); }

   bool UsesSequenceWindow(void) const
   {
      int k = FrameworkKind();
      return (k == FXAI_FW_KIND_CNN_LSTM ||
              k == FXAI_FW_KIND_ATTN_CNN_BILSTM ||
              k == FXAI_FW_KIND_GRU ||
              k == FXAI_FW_KIND_BILSTM ||
              k == FXAI_FW_KIND_LSTM_TCN ||
              k == FXAI_FW_KIND_EMD_HHT ||
              k == FXAI_FW_KIND_VMD ||
              k == FXAI_FW_KIND_TSMOM_VOL ||
              k == FXAI_FW_KIND_VOL_BREAKOUT);
   }

   double Logistic(const double x) const
   {
      return 1.0 / (1.0 + MathExp(-FXAI_Clamp(x, -40.0, 40.0)));
   }

   double StableTanh(const double x) const
   {
      double z = FXAI_Clamp(x, -20.0, 20.0);
      double e2 = MathExp(2.0 * z);
      return (e2 - 1.0) / (e2 + 1.0);
   }

   double SeedWeight(const int a,
                     const int b) const
   {
      double x = MathSin((double)((AIId() + 17) * (a + 3) * 37 + (b + 11) * 101));
      return 0.035 * x;
   }

   double SafeX(const double &x[],
                const int idx) const
   {
      if(idx < 0 || idx >= ArraySize(x))
         return 0.0;
      double v = x[idx];
      if(!MathIsValidNumber(v))
         return 0.0;
      return FXAI_Clamp(v, -50.0, 50.0);
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];
      double e0 = MathExp(FXAI_Clamp(logits[0] - m, -35.0, 35.0));
      double e1 = MathExp(FXAI_Clamp(logits[1] - m, -35.0, 35.0));
      double e2 = MathExp(FXAI_Clamp(logits[2] - m, -35.0, 35.0));
      double s = e0 + e1 + e2;
      if(s <= 0.0)
      {
         probs[0] = 0.10;
         probs[1] = 0.10;
         probs[2] = 0.80;
         return;
      }
      probs[0] = e0 / s;
      probs[1] = e1 / s;
      probs[2] = e2 / s;
   }

   void BuildState(const double &x[],
                   double &z[]) const
   {
      ArrayResize(z, FXAI_FW_STATE_DIM);
      z[0] = 1.0;
      z[1] = SafeX(x, 1);
      z[2] = SafeX(x, 2);
      z[3] = SafeX(x, 3);
      z[4] = SafeX(x, 4);
      z[5] = SafeX(x, 7);
      z[6] = SafeX(x, 12);
      z[7] = SafeX(x, 40);
      z[8] = CurrentWindowFeatureSlope(0);
      z[9] = CurrentWindowFeatureStd(0);
      z[10] = CurrentWindowFeatureRange(0, 16);
      z[11] = CurrentWindowFeatureRecentDelta(0, 8);
      z[12] = CurrentWindowFeatureEMAMean(1, 0.70);
      z[13] = SafeX(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 14);
      z[14] = SafeX(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 19);
      z[15] = FXAI_Clamp((double)m_ctx_horizon_minutes / 60.0, 0.0, 2.0);
      for(int i=1; i<FXAI_FW_STATE_DIM; i++)
         z[i] = FXAI_Clamp(z[i], -8.0, 8.0);
   }

   double LinearMargin(const double &z[]) const
   {
      double mb = 0.0;
      double ms = 0.0;
      for(int i=0; i<FXAI_FW_STATE_DIM; i++)
      {
         mb += m_w[(int)FXAI_LABEL_BUY][i] * z[i];
         ms += m_w[(int)FXAI_LABEL_SELL][i] * z[i];
      }
      return FXAI_ClipSym(mb - ms, 12.0);
   }

   double WindowVolatility(void) const
   {
      double v = MathMax(CurrentWindowFeatureStd(0), 0.0);
      if(v <= 1e-6)
         v = MathMax(MathAbs(CurrentWindowFeatureRecentDelta(0, 8)), 0.0);
      if(v <= 1e-6)
         v = MathMax(m_move_ready_local ? 0.01 * m_move_ema : 0.05, 0.01);
      return v;
   }

   void HMMForward(const double obs,
                   const double vol,
                   double &pi_out[]) const
   {
      ArrayResize(pi_out, FXAI_FW_REGIMES);
      double un[FXAI_FW_REGIMES];
      double total = 0.0;
      for(int j=0; j<FXAI_FW_REGIMES; j++)
      {
         double pred = 0.0;
         for(int i=0; i<FXAI_FW_REGIMES; i++)
            pred += m_hmm_pi[i] * m_hmm_A[i][j];
         double var = MathMax(m_regime_var[j] + vol * vol, 1e-6);
         double d = obs - m_regime_mu[j];
         double emission = MathExp(FXAI_Clamp(-0.5 * d * d / var, -40.0, 0.0)) / MathSqrt(var);
         un[j] = MathMax(pred * emission, 1e-12);
         total += un[j];
      }
      if(total <= 0.0)
         total = 1.0;
      for(int j=0; j<FXAI_FW_REGIMES; j++)
         pi_out[j] = un[j] / total;
   }

   double HMMRegimeMargin(const double &z[],
                          double &confidence) const
   {
      double pi[];
      HMMForward(z[8] + 0.25 * z[11], WindowVolatility(), pi);
      double m = 0.0;
      double best = 0.0;
      for(int r=0; r<FXAI_FW_REGIMES; r++)
      {
         m += pi[r] * m_regime_mu[r] / MathSqrt(MathMax(m_regime_var[r], 1e-6));
         if(pi[r] > best) best = pi[r];
      }
      confidence = FXAI_Clamp(best, 0.0, 1.0);
      return FXAI_ClipSym(m, 8.0);
   }

   double MSGARCHMargin(const double &z[],
                        double &forecast_vol,
                        double &confidence) const
   {
      double pi[];
      HMMForward(z[8] + 0.15 * z[1], WindowVolatility(), pi);
      double r2 = MathPow(z[8], 2.0);
      forecast_vol = 0.0;
      double drift = 0.0;
      confidence = 0.0;
      for(int k=0; k<FXAI_FW_REGIMES; k++)
      {
         double h = m_garch_omega[k] + m_garch_alpha[k] * r2 + m_garch_beta[k] * MathMax(m_garch_h[k], 1e-6);
         h = MathMax(h, 1e-8);
         forecast_vol += pi[k] * MathSqrt(h);
         drift += pi[k] * m_regime_mu[k] / MathSqrt(MathMax(h, 1e-8));
         if(pi[k] > confidence) confidence = pi[k];
      }
      return FXAI_ClipSym(drift, 8.0);
   }

   int RFFeatureValueIndex(const int tree,
                           const int depth) const
   {
      int f = m_rf_feature[tree][depth];
      if(f < 0) f = 0;
      if(f >= FXAI_FW_STATE_DIM) f = FXAI_FW_STATE_DIM - 1;
      return f;
   }

   int RFLeaf(const int tree,
              const double &z[]) const
   {
      int leaf = 0;
      for(int d=0; d<3; d++)
      {
         int f = RFFeatureValueIndex(tree, d);
         if(z[f] > m_rf_threshold[tree][d])
            leaf |= (1 << d);
      }
      if(leaf < 0) leaf = 0;
      if(leaf >= FXAI_FW_RF_LEAVES) leaf = FXAI_FW_RF_LEAVES - 1;
      return leaf;
   }

   double RandomForestMargin(const double &z[],
                             double &confidence) const
   {
      double vote[3] = {1e-3, 1e-3, 1e-3};
      for(int t=0; t<FXAI_FW_RF_TREES; t++)
      {
         int leaf = RFLeaf(t, z);
         double total = 0.0;
         for(int c=0; c<3; c++) total += m_rf_leaf_mass[t][leaf][c];
         if(total <= 0.0) total = 1.0;
         for(int c=0; c<3; c++)
            vote[c] += m_rf_leaf_mass[t][leaf][c] / total;
      }
      double den = vote[0] + vote[1] + vote[2];
      if(den <= 0.0) den = 1.0;
      confidence = FXAI_Clamp(MathMax(vote[0], vote[1]) / den, 0.0, 1.0);
      return FXAI_ClipSym((vote[(int)FXAI_LABEL_BUY] - vote[(int)FXAI_LABEL_SELL]) / den * 5.0, 8.0);
   }

   double KalmanMargin(const double &z[],
                       double &confidence) const
   {
      double yhat = 0.0;
      double var = 0.0;
      for(int i=0; i<FXAI_FW_STATE_DIM; i++)
      {
         yhat += m_kalman_state[i] * z[i];
         var += m_kalman_cov[i] * z[i] * z[i];
      }
      confidence = FXAI_Clamp(1.0 / (1.0 + MathSqrt(MathMax(var, 1e-9))), 0.0, 1.0);
      return FXAI_ClipSym(yhat / MathMax(MathSqrt(MathMax(var, 1e-6)), 0.05), 8.0);
   }

   double PCAMargin(const double &z[],
                    double &confidence) const
   {
      double score[3] = {0.0, 0.0, 0.0};
      double total_var = 1e-6;
      for(int pc=0; pc<3; pc++)
      {
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
            score[pc] += (z[i] - m_pca_mean[i]) * m_pca_loading[pc][i];
         total_var += MathMax(m_pca_var[pc], 0.0);
      }
      confidence = FXAI_Clamp((m_pca_var[0] + m_pca_var[1]) / total_var, 0.0, 1.0);
      return FXAI_ClipSym(0.55 * score[0] + 0.30 * score[1] - 0.15 * MathAbs(score[2]), 8.0);
   }

   double SequenceMargin(const double &z[],
                         double &confidence) const
   {
      int n = CurrentWindowSize();
      if(n <= 0)
      {
         confidence = 0.20;
         return LinearMargin(z);
      }

      double h[FXAI_FW_SEQ_HIDDEN];
      double hb[FXAI_FW_SEQ_HIDDEN];
      for(int j=0; j<FXAI_FW_SEQ_HIDDEN; j++)
      {
         h[j] = 0.0;
         hb[j] = 0.0;
      }

      int kind = FrameworkKind();
      for(int b=n - 1; b>=0; b--)
      {
         double v0 = CurrentWindowValue(b, 1);
         double v1 = CurrentWindowValue(b, 2);
         double conv = 0.55 * v0 + 0.30 * v1 + 0.15 * CurrentWindowValue(b, 4);
         if(kind == FXAI_FW_KIND_LSTM_TCN || kind == FXAI_FW_KIND_CNN_LSTM || kind == FXAI_FW_KIND_ATTN_CNN_BILSTM)
            conv += 0.20 * (CurrentWindowValue(MathMax(b - 1, 0), 1) - CurrentWindowValue(MathMin(b + 1, n - 1), 1));
         for(int j=0; j<FXAI_FW_SEQ_HIDDEN; j++)
         {
            double u = conv * SeedWeight(j + 1, kind + 3) + h[j] * SeedWeight(j + 5, kind + 7);
            if(kind == FXAI_FW_KIND_GRU)
            {
               double zg = Logistic(u + SeedWeight(j + 8, kind));
               double rg = Logistic(0.60 * u + SeedWeight(j + 9, kind));
               double cand = StableTanh(conv * SeedWeight(j + 11, kind) + rg * h[j]);
               h[j] = (1.0 - zg) * h[j] + zg * cand;
            }
            else
            {
               double ing = Logistic(u);
               double forg = Logistic(0.50 + 0.25 * h[j]);
               double cand = StableTanh(u + 0.10 * conv);
               h[j] = forg * h[j] + ing * cand;
            }
         }
      }

      if(kind == FXAI_FW_KIND_BILSTM || kind == FXAI_FW_KIND_ATTN_CNN_BILSTM)
      {
         for(int b=0; b<n; b++)
         {
            double v0 = CurrentWindowValue(b, 1);
            double v1 = CurrentWindowValue(b, 2);
            double conv = 0.55 * v0 + 0.30 * v1 + 0.15 * CurrentWindowValue(b, 4);
            for(int j=0; j<FXAI_FW_SEQ_HIDDEN; j++)
            {
               double u = conv * SeedWeight(j + 13, kind + 5) + hb[j] * SeedWeight(j + 17, kind + 11);
               double ing = Logistic(u);
               double forg = Logistic(0.50 + 0.25 * hb[j]);
               hb[j] = forg * hb[j] + ing * StableTanh(u);
            }
         }
      }

      double margin = 0.0;
      double energy = 0.0;
      for(int j=0; j<FXAI_FW_SEQ_HIDDEN; j++)
      {
         margin += h[j] * SeedWeight(j + 23, kind);
         energy += h[j] * h[j];
         if(kind == FXAI_FW_KIND_BILSTM || kind == FXAI_FW_KIND_ATTN_CNN_BILSTM)
         {
            margin += hb[j] * SeedWeight(j + 29, kind);
            energy += hb[j] * hb[j];
         }
      }
      margin += 0.40 * LinearMargin(z);
      confidence = FXAI_Clamp(MathSqrt(energy / (double)FXAI_FW_SEQ_HIDDEN), 0.0, 1.0);
      return FXAI_ClipSym(margin, 8.0);
   }

   double FrameworkMargin(const double &z[],
                          const double cost_points,
                          double &forecast_vol,
                          double &confidence) const
   {
      forecast_vol = WindowVolatility();
      confidence = 0.55;
      int kind = FrameworkKind();

      if(kind == FXAI_FW_KIND_MSGARCH)
         return MSGARCHMargin(z, forecast_vol, confidence);
      if(kind == FXAI_FW_KIND_HMM_REGIME)
         return HMMRegimeMargin(z, confidence);
      if(kind == FXAI_FW_KIND_RANDOM_FOREST)
         return RandomForestMargin(z, confidence);
      if(kind == FXAI_FW_KIND_TVP_KALMAN)
         return KalmanMargin(z, confidence);
      if(kind == FXAI_FW_KIND_PCA_PANEL)
         return PCAMargin(z, confidence);
      if(kind == FXAI_FW_KIND_GRU || kind == FXAI_FW_KIND_BILSTM ||
         kind == FXAI_FW_KIND_LSTM_TCN || kind == FXAI_FW_KIND_CNN_LSTM ||
         kind == FXAI_FW_KIND_ATTN_CNN_BILSTM)
         return SequenceMargin(z, confidence);

      double base = LinearMargin(z);
      switch(kind)
      {
         case FXAI_FW_KIND_ARIMAX_GARCH:
            return FXAI_ClipSym(0.55 * base + 0.45 * (z[8] + 0.35 * z[13]) / MathMax(forecast_vol, 0.05), 8.0);
         case FXAI_FW_KIND_COINT_VECM:
         case FXAI_FW_KIND_OU_SPREAD:
         {
            double spread = z[1] - 0.65 * z[6] - 0.35 * z[12];
            double zdev = (spread - m_ou_mean) / MathSqrt(MathMax(m_ou_var, 1e-6));
            confidence = FXAI_Clamp(MathAbs(zdev) / 2.5, 0.0, 1.0);
            return FXAI_ClipSym(-m_ou_speed * zdev, 8.0);
         }
         case FXAI_FW_KIND_MICROFLOW:
            confidence = FXAI_Clamp(0.40 + 0.25 * MathAbs(z[6]) + 0.20 * MathAbs(z[10]), 0.0, 1.0);
            return FXAI_ClipSym(0.50 * z[6] + 0.35 * z[8] - 0.25 * MathAbs(z[5]), 8.0);
         case FXAI_FW_KIND_ELASTIC_LOGIT:
            return FXAI_ClipSym(base, 8.0);
         case FXAI_FW_KIND_PROFIT_LOGIT:
            return FXAI_ClipSym(base * (1.0 + 0.25 * FXAI_Clamp(m_move_ema / MathMax(cost_points, 0.10), 0.0, 2.0)), 8.0);
         case FXAI_FW_KIND_EMD_HHT:
            return FXAI_ClipSym(0.45 * base + 0.35 * CurrentWindowFeatureRecentDelta(0, 8) + 0.20 * (CurrentWindowFeatureRecentMean(0, 8) - CurrentWindowFeatureMean(0)), 8.0);
         case FXAI_FW_KIND_VMD:
            return FXAI_ClipSym(0.40 * base + 0.30 * CurrentWindowFeatureEMAMean(0, 0.55) + 0.30 * CurrentWindowFeatureEMAMean(1, 0.85), 8.0);
         case FXAI_FW_KIND_PPP_VALUE:
            confidence = FXAI_Clamp(0.20 + 0.10 * (double)MathMin(m_ctx_horizon_minutes, 240) / 60.0, 0.0, 0.75);
            return FXAI_ClipSym(-0.55 * z[4] - 0.35 * z[13], 8.0);
         case FXAI_FW_KIND_CARRY:
            return FXAI_ClipSym(0.75 * z[13] - 0.30 * forecast_vol + 0.25 * base, 8.0);
         case FXAI_FW_KIND_CMV_PANEL:
            return FXAI_ClipSym(0.40 * z[13] + 0.35 * z[8] - 0.25 * z[4] + 0.25 * base, 8.0);
         case FXAI_FW_KIND_TSMOM_VOL:
            return FXAI_ClipSym((0.45 * CurrentWindowFeatureRecentDelta(0, 16) + 0.35 * z[8] + 0.20 * base) / MathMax(forecast_vol, 0.05), 8.0);
         case FXAI_FW_KIND_XSMOM_RANK:
            return FXAI_ClipSym(0.55 * z[6] + 0.35 * z[8] + 0.10 * base, 8.0);
         case FXAI_FW_KIND_VOL_BREAKOUT:
         {
            double expansion = CurrentWindowFeatureRange(0, 8) - CurrentWindowFeatureRange(0, 32);
            confidence = FXAI_Clamp(MathAbs(expansion) / MathMax(forecast_vol, 0.05), 0.0, 1.0);
            return FXAI_ClipSym((expansion > 0.0 ? 1.0 : -1.0) * z[8] + 0.25 * base, 8.0);
         }
         case FXAI_FW_KIND_XRATE_CONSISTENCY:
            confidence = FXAI_Clamp(MathAbs(z[6] - z[13]) / 2.0, 0.0, 1.0);
            return FXAI_ClipSym(-0.60 * (z[6] - z[13]) + 0.25 * base, 8.0);
         case FXAI_FW_KIND_PPO:
            return FXAI_ClipSym(base + 0.35 * z[8] - 0.25 * z[5], 8.0);
      }
      return base;
   }

   void UpdateRegimeModels(const int cls,
                           const double &z[],
                           const double sample_w)
   {
      double obs = z[8] + 0.25 * z[11];
      double pi[];
      HMMForward(obs, WindowVolatility(), pi);
      int best = 0;
      for(int r=1; r<FXAI_FW_REGIMES; r++)
         if(pi[r] > pi[best]) best = r;

      for(int r=0; r<FXAI_FW_REGIMES; r++)
      {
         double a = FXAI_Clamp(0.015 * sample_w * (0.35 + pi[r]), 0.001, 0.08);
         double d = obs - m_regime_mu[r];
         m_regime_mu[r] += a * d;
         m_regime_var[r] = MathMax((1.0 - a) * m_regime_var[r] + a * d * d, 1e-6);
         m_garch_h[r] = MathMax(m_garch_omega[r] + m_garch_alpha[r] * d * d + m_garch_beta[r] * m_garch_h[r], 1e-8);
         m_hmm_pi[r] = 0.80 * m_hmm_pi[r] + 0.20 * pi[r];
      }
      for(int r=0; r<FXAI_FW_REGIMES; r++)
      {
         for(int c=0; c<FXAI_FW_REGIMES; c++)
            m_hmm_A[r][c] = 0.995 * m_hmm_A[r][c] + 0.005 * (c == best ? 1.0 : 0.0);
         double rowsum = 0.0;
         for(int c=0; c<FXAI_FW_REGIMES; c++)
            rowsum += m_hmm_A[r][c];
         if(rowsum <= 0.0) rowsum = 1.0;
         for(int c=0; c<FXAI_FW_REGIMES; c++)
            m_hmm_A[r][c] /= rowsum;
      }
   }

   void UpdateRF(const int cls,
                 const double &z[],
                 const double move_points,
                 const double sample_w)
   {
      for(int t=0; t<FXAI_FW_RF_TREES; t++)
      {
         uint h = (uint)(m_step * 1103515245U + t * 2654435761U + AIId() * 97U);
         if((h % 5U) == 0U)
            continue;
         int leaf = RFLeaf(t, z);
         double w = MathMax(0.05, sample_w);
         m_rf_leaf_mass[t][leaf][cls] += w;
         m_rf_leaf_move[t][leaf] = 0.96 * m_rf_leaf_move[t][leaf] + 0.04 * MathAbs(move_points);
         for(int d=0; d<3; d++)
         {
            int f = RFFeatureValueIndex(t, d);
            m_rf_threshold[t][d] = 0.995 * m_rf_threshold[t][d] + 0.005 * z[f];
         }
      }
   }

   void UpdateKalmanPCAOU(const int cls,
                          const double &z[],
                          const double signed_move,
                          const double sample_w)
   {
      double pred = 0.0;
      double pred_var = 1e-6;
      for(int i=0; i<FXAI_FW_STATE_DIM; i++)
      {
         pred += m_kalman_state[i] * z[i];
         pred_var += m_kalman_cov[i] * z[i] * z[i];
      }
      double err = signed_move - pred;
      double obs_var = MathMax(m_ou_var, 1e-4);
      for(int i=0; i<FXAI_FW_STATE_DIM; i++)
      {
         double gain = m_kalman_cov[i] * z[i] / (pred_var + obs_var);
         m_kalman_state[i] += FXAI_Clamp(gain * err, -0.25, 0.25);
         m_kalman_cov[i] = FXAI_Clamp((1.0 - gain * z[i]) * m_kalman_cov[i] + 0.0002, 0.0001, 4.0);
         double a = FXAI_Clamp(0.01 * sample_w, 0.001, 0.05);
         double centered = z[i] - m_pca_mean[i];
         m_pca_mean[i] += a * centered;
         for(int pc=0; pc<3; pc++)
         {
            double proj = centered * m_pca_loading[pc][i];
            m_pca_loading[pc][i] += a * proj * centered;
            m_pca_var[pc] = 0.995 * m_pca_var[pc] + 0.005 * proj * proj;
         }
      }
      double spread = z[1] - 0.65 * z[6] - 0.35 * z[12];
      if(m_prev_spread_ready)
      {
         double delta = spread - m_prev_spread;
         double dev = m_prev_spread - m_ou_mean;
         double a = FXAI_Clamp(0.02 * sample_w, 0.001, 0.08);
         if(MathAbs(dev) > 1e-6)
            m_ou_speed = FXAI_Clamp((1.0 - a) * m_ou_speed + a * FXAI_Clamp(-delta / dev, 0.001, 1.0), 0.001, 1.0);
         m_ou_mean += a * (spread - m_ou_mean);
         double e = spread - m_ou_mean;
         m_ou_var = MathMax((1.0 - a) * m_ou_var + a * e * e, 1e-6);
      }
      m_prev_spread = spread;
      m_prev_spread_ready = true;
   }

   void UpdateLinearPolicy(const int cls,
                           const double &z[],
                           const double signed_move,
                           const FXAIAIHyperParams &hp,
                           const double cost_points,
                           const double sample_w)
   {
      double logits[3];
      for(int c=0; c<3; c++)
      {
         logits[c] = 0.0;
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
            logits[c] += m_w[c][i] * z[i];
      }
      double p[3];
      Softmax3(logits, p);
      double lr = FXAI_Clamp(hp.lr, 0.0002, 0.08) * FXAI_Clamp(sample_w, 0.2, 5.0);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.20);
      double l1 = (FrameworkKind() == FXAI_FW_KIND_ELASTIC_LOGIT ? 0.0008 : 0.0);
      if(FrameworkKind() == FXAI_FW_KIND_PROFIT_LOGIT)
         lr *= FXAI_Clamp(MathAbs(signed_move) / MathMax(cost_points, 0.10), 0.5, 4.0);
      if(FrameworkKind() == FXAI_FW_KIND_PPO)
      {
         double value = 0.0;
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
            value += m_value_w[i] * z[i];
         double reward = signed_move - cost_points - 0.10 * MathAbs(signed_move);
         double adv = FXAI_Clamp(reward - value, -8.0, 8.0);
         double ratio = p[cls] / MathMax(m_policy_old[cls], 1e-4);
         double clipped = FXAI_Clamp(ratio, 0.80, 1.20);
         lr *= (MathAbs(ratio) <= MathAbs(clipped) ? ratio : clipped) * FXAI_Clamp(MathAbs(adv), 0.2, 3.0);
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
            m_value_w[i] += 0.20 * FXAI_Clamp(hp.lr, 0.0002, 0.05) * (reward - value) * z[i];
      }
      for(int c=0; c<3; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double g = target - p[c];
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
         {
            double shrink = l2 * m_w[c][i] + l1 * (m_w[c][i] > 0.0 ? 1.0 : (m_w[c][i] < 0.0 ? -1.0 : 0.0));
            m_w[c][i] += lr * (g * z[i] - shrink);
            m_w[c][i] = FXAI_Clamp(m_w[c][i], -8.0, 8.0);
         }
      }
      for(int c=0; c<3; c++)
         m_policy_old[c] = p[c];
   }

public:
   CFXAIFrameworkModelPlugin(void) {}

   virtual void Describe(FXAIAIManifestV4 &out) const
   {
      ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING | FXAI_CAP_REPLAY | FXAI_CAP_MULTI_HORIZON | FXAI_CAP_SELF_TEST);
      if(UsesSequenceWindow())
         caps |= (ulong)(FXAI_CAP_WINDOW_CONTEXT | FXAI_CAP_STATEFUL);
      FillManifest(out,
                   FrameworkFamily(),
                   caps,
                   FrameworkMinSequenceBars(),
                   FrameworkMaxSequenceBars());
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_step = 0;
      m_move_ema = 0.0;
      m_move_ready_local = false;
      m_prev_spread = 0.0;
      m_prev_spread_ready = false;
      m_ou_mean = 0.0;
      m_ou_speed = 0.08;
      m_ou_var = 0.05;
      for(int c=0; c<3; c++)
      {
         m_class_mass[c] = 1.0;
         m_policy_old[c] = 0.3333333333;
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
         {
            m_w[c][i] = SeedWeight(c + 1, i + 1);
            m_v[c][i] = 0.0;
         }
      }
      for(int r=0; r<FXAI_FW_REGIMES; r++)
      {
         m_hmm_pi[r] = 1.0 / (double)FXAI_FW_REGIMES;
         m_regime_mu[r] = (double)(r - 1) * 0.05;
         m_regime_var[r] = 0.05 + 0.04 * (double)r;
         m_garch_h[r] = 0.02 + 0.02 * (double)r;
         m_garch_omega[r] = 0.0005 + 0.0002 * (double)r;
         m_garch_alpha[r] = 0.07 + 0.02 * (double)r;
         m_garch_beta[r] = 0.86 - 0.03 * (double)r;
         for(int q=0; q<FXAI_FW_REGIMES; q++)
            m_hmm_A[r][q] = (r == q ? 0.92 : 0.04);
      }
      for(int t=0; t<FXAI_FW_RF_TREES; t++)
      {
         for(int d=0; d<3; d++)
         {
            m_rf_feature[t][d] = 1 + (int)((uint)(t * 7 + d * 11 + AIId()) % (uint)(FXAI_FW_STATE_DIM - 1));
            m_rf_threshold[t][d] = SeedWeight(t + 3, d + 5) * 4.0;
         }
         for(int l=0; l<FXAI_FW_RF_LEAVES; l++)
         {
            m_rf_leaf_move[t][l] = 0.0;
            for(int c=0; c<3; c++)
               m_rf_leaf_mass[t][l][c] = 1.0;
         }
      }
      for(int i=0; i<FXAI_FW_STATE_DIM; i++)
      {
         m_kalman_state[i] = SeedWeight(7, i + 1);
         m_kalman_cov[i] = 1.0;
         m_pca_mean[i] = 0.0;
         m_value_w[i] = 0.0;
         for(int pc=0; pc<3; pc++)
            m_pca_loading[pc][i] = (pc == (i % 3) ? 1.0 : 0.0) * (i == 0 ? 0.0 : 1.0);
      }
      for(int pc=0; pc<3; pc++)
         m_pca_var[pc] = 1.0;
   }

   virtual bool SupportsNativeParameterSnapshot(void) const { return true; }
   virtual string PersistentStateCoverageTag(void) const { return "native_model"; }

   virtual bool PredictModelCore(const double &x[],
                                 const FXAIAIHyperParams &hp,
                                 double &class_probs[],
                                 double &expected_move_points)
   {
      FXAIAIModelOutputV4 out;
      if(!PredictDistributionCore(x, hp, out))
         return false;
      for(int c=0; c<3; c++)
         class_probs[c] = out.class_probs[c];
      expected_move_points = out.move_mean_points;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double z[];
      BuildState(x, z);
      double forecast_vol = 0.0;
      double model_conf = 0.0;
      double cost = ResolveCostPoints(x);
      double margin = FrameworkMargin(z, cost, forecast_vol, model_conf);
      double min_move = MathMax(ResolveMinMovePoints(), 0.10);
      double scale = MathMax(MathMax(forecast_vol, min_move * 0.20), 0.05);
      double p_dir = Logistic(margin / scale);
      double edge = MathAbs(margin) * MathMax(m_move_ready_local ? m_move_ema : min_move, min_move);
      double active = Logistic((edge - cost) / MathMax(min_move, 0.10));
      double weak = FXAI_Clamp(1.0 - model_conf, 0.0, 1.0);
      double skip = FXAI_Clamp(0.12 + 0.58 * (1.0 - active) + 0.25 * weak, 0.05, 0.92);
      out.class_probs[(int)FXAI_LABEL_BUY] = (1.0 - skip) * p_dir;
      out.class_probs[(int)FXAI_LABEL_SELL] = (1.0 - skip) * (1.0 - p_dir);
      out.class_probs[(int)FXAI_LABEL_SKIP] = skip;
      NormalizeClassDistribution(out.class_probs);
      out.move_mean_points = MathMax(0.0, MathMax(edge, m_move_ready_local ? m_move_ema : 0.0));
      double sigma = MathMax(0.10, forecast_vol + 0.25 * out.move_mean_points + 0.25 * min_move);
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      out.confidence = FXAI_Clamp(0.50 * MathMax(out.class_probs[0], out.class_probs[1]) + 0.35 * model_conf + 0.15 * active, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.30 + 0.30 * MathMin((double)m_step / 128.0, 1.0) + 0.25 * model_conf + 0.15 * (m_move_ready_local ? 1.0 : 0.0), 0.0, 1.0);
      out.mfe_mean_points = MathMax(out.move_q75_points, out.move_mean_points * (1.05 + 0.35 * out.confidence));
      out.mae_mean_points = MathMax(0.0, out.move_mean_points * (0.30 + 0.35 * skip + 0.15 * weak));
      out.hit_time_frac = FXAI_Clamp(0.70 - 0.35 * active + 0.25 * skip, 0.0, 1.0);
      out.path_risk = FXAI_Clamp(0.35 * skip + 0.30 * weak + 0.35 * out.mae_mean_points / MathMax(out.mfe_mean_points, 0.10), 0.0, 1.0);
      out.fill_risk = FXAI_Clamp(cost / MathMax(out.move_mean_points + min_move, 0.10), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      out.has_path_quality = true;
      PredictNativeQualityHeads(x,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

protected:
   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < 0 || cls > 2)
         cls = (move_points > 0.0 ? (int)FXAI_LABEL_BUY : (move_points < 0.0 ? (int)FXAI_LABEL_SELL : (int)FXAI_LABEL_SKIP));
      double z[];
      BuildState(x, z);
      double signed_move = (cls == (int)FXAI_LABEL_BUY ? MathAbs(move_points) :
                            (cls == (int)FXAI_LABEL_SELL ? -MathAbs(move_points) : 0.0));
      double cost = ResolveCostPoints(x);
      double sample_w = FXAI_Clamp(FXAI_MoveEdgeWeight(move_points, cost), 0.15, 6.0);
      UpdateNativeQualityHeads(x, sample_w, FXAI_Clamp(hp.lr, 0.0002, 0.08), FXAI_Clamp(hp.l2, 0.0, 0.20));
      UpdateLinearPolicy(cls, z, signed_move, hp, cost, sample_w);
      UpdateRegimeModels(cls, z, sample_w);
      UpdateRF(cls, z, move_points, sample_w);
      UpdateKalmanPCAOU(cls, z, signed_move, sample_w);
      m_class_mass[cls] += sample_w;
      double am = MathAbs(move_points);
      if(!m_move_ready_local)
      {
         m_move_ema = am;
         m_move_ready_local = true;
      }
      else
      {
         m_move_ema = 0.97 * m_move_ema + 0.03 * am;
      }
      m_step++;
   }

   virtual bool SaveModelState(const int handle) const
   {
      FileWriteInteger(handle, 1);
      FileWriteInteger(handle, FrameworkKind());
      FileWriteInteger(handle, m_step);
      FileWriteDouble(handle, m_move_ema);
      FileWriteInteger(handle, (m_move_ready_local ? 1 : 0));
      FileWriteDouble(handle, m_ou_mean);
      FileWriteDouble(handle, m_ou_speed);
      FileWriteDouble(handle, m_ou_var);
      FileWriteDouble(handle, m_prev_spread);
      FileWriteInteger(handle, (m_prev_spread_ready ? 1 : 0));
      for(int c=0; c<3; c++)
      {
         FileWriteDouble(handle, m_class_mass[c]);
         FileWriteDouble(handle, m_policy_old[c]);
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
         {
            FileWriteDouble(handle, m_w[c][i]);
            FileWriteDouble(handle, m_v[c][i]);
         }
      }
      for(int r=0; r<FXAI_FW_REGIMES; r++)
      {
         FileWriteDouble(handle, m_hmm_pi[r]);
         FileWriteDouble(handle, m_regime_mu[r]);
         FileWriteDouble(handle, m_regime_var[r]);
         FileWriteDouble(handle, m_garch_h[r]);
         for(int q=0; q<FXAI_FW_REGIMES; q++)
            FileWriteDouble(handle, m_hmm_A[r][q]);
      }
      for(int i=0; i<FXAI_FW_STATE_DIM; i++)
      {
         FileWriteDouble(handle, m_kalman_state[i]);
         FileWriteDouble(handle, m_kalman_cov[i]);
         FileWriteDouble(handle, m_pca_mean[i]);
         FileWriteDouble(handle, m_value_w[i]);
         for(int pc=0; pc<3; pc++)
            FileWriteDouble(handle, m_pca_loading[pc][i]);
      }
      for(int pc=0; pc<3; pc++)
         FileWriteDouble(handle, m_pca_var[pc]);
      for(int t=0; t<FXAI_FW_RF_TREES; t++)
      {
         for(int d=0; d<3; d++)
         {
            FileWriteInteger(handle, m_rf_feature[t][d]);
            FileWriteDouble(handle, m_rf_threshold[t][d]);
         }
         for(int l=0; l<FXAI_FW_RF_LEAVES; l++)
         {
            FileWriteDouble(handle, m_rf_leaf_move[t][l]);
            for(int c=0; c<3; c++)
               FileWriteDouble(handle, m_rf_leaf_mass[t][l][c]);
         }
      }
      return true;
   }

   virtual bool LoadModelState(const int handle,
                               const int version)
   {
      int local_version = FileReadInteger(handle);
      int saved_kind = FileReadInteger(handle);
      if(local_version != 1 || saved_kind != FrameworkKind())
         return false;
      m_step = FileReadInteger(handle);
      m_move_ema = FileReadDouble(handle);
      m_move_ready_local = (FileReadInteger(handle) != 0);
      m_ou_mean = FileReadDouble(handle);
      m_ou_speed = FileReadDouble(handle);
      m_ou_var = FileReadDouble(handle);
      m_prev_spread = FileReadDouble(handle);
      m_prev_spread_ready = (FileReadInteger(handle) != 0);
      for(int c=0; c<3; c++)
      {
         m_class_mass[c] = FileReadDouble(handle);
         m_policy_old[c] = FileReadDouble(handle);
         for(int i=0; i<FXAI_FW_STATE_DIM; i++)
         {
            m_w[c][i] = FileReadDouble(handle);
            m_v[c][i] = FileReadDouble(handle);
         }
      }
      for(int r=0; r<FXAI_FW_REGIMES; r++)
      {
         m_hmm_pi[r] = FileReadDouble(handle);
         m_regime_mu[r] = FileReadDouble(handle);
         m_regime_var[r] = FileReadDouble(handle);
         m_garch_h[r] = FileReadDouble(handle);
         for(int q=0; q<FXAI_FW_REGIMES; q++)
            m_hmm_A[r][q] = FileReadDouble(handle);
      }
      for(int i=0; i<FXAI_FW_STATE_DIM; i++)
      {
         m_kalman_state[i] = FileReadDouble(handle);
         m_kalman_cov[i] = FileReadDouble(handle);
         m_pca_mean[i] = FileReadDouble(handle);
         m_value_w[i] = FileReadDouble(handle);
         for(int pc=0; pc<3; pc++)
            m_pca_loading[pc][i] = FileReadDouble(handle);
      }
      for(int pc=0; pc<3; pc++)
         m_pca_var[pc] = FileReadDouble(handle);
      for(int t=0; t<FXAI_FW_RF_TREES; t++)
      {
         for(int d=0; d<3; d++)
         {
            m_rf_feature[t][d] = FileReadInteger(handle);
            m_rf_threshold[t][d] = FileReadDouble(handle);
         }
         for(int l=0; l<FXAI_FW_RF_LEAVES; l++)
         {
            m_rf_leaf_move[t][l] = FileReadDouble(handle);
            for(int c=0; c<3; c++)
               m_rf_leaf_mass[t][l][c] = FileReadDouble(handle);
         }
      }
      return true;
   }
};

#endif // __FXAI_FRAMEWORK_MODEL_MQH__
