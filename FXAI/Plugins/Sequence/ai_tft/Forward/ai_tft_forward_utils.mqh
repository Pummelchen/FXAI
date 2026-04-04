   int SessionBucket(const datetime ts) const
   {
      MqlDateTime md;
      TimeToStruct(ts, md);
      int h = md.hour;
      if(h >= 6 && h <= 12) return 1;  // Europe
      if(h >= 13 && h <= 20) return 2; // US
      if(h >= 21 || h <= 2) return 0;  // Asia
      return 3;                        // transition/off
   }

   uint SymbolHash(const string s) const
   {
      uint h = 2166136261U;
      int n = StringLen(s);
      for(int i=0; i<n; i++)
      {
         uint c = (uint)StringGetCharacter(s, i);
         h ^= c;
         h *= 16777619U;
      }
      return h;
   }

   double DropMask(const int idx,
                   const int salt,
                   const double rate,
                   const bool training) const
   {
      if(!training) return 1.0;
      if(rate <= 1e-9) return 1.0;

      uint h = (uint)(m_step * 2654435761U);
      h ^= (uint)((idx + 3) * 2246822519U);
      h ^= (uint)((salt + 11) * 3266489917U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < rate) return 0.0;
      return 1.0 / (1.0 - rate);
   }

   double StochScale(const int idx,
                     const bool training) const
   {
      if(!training) return 1.0;
      const double drop = 0.12;
      uint h = (uint)((m_step + 17) * 40503U);
      h ^= (uint)((idx + 19) * 2654435761U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < drop) return 0.0;
      return 1.0 / (1.0 - drop);
   }

   double ScheduledLR(const FXAIAIHyperParams &hp) const
   {
      double base = FXAI_Clamp(hp.lr, 0.00002, 0.25000);
      double lr = FXAI_LRScheduleCosineWarm(base, m_step, 128, 2048, 0.20);
      lr *= FXAI_LRScheduleInvSqrt(1.0, m_step, 128, 0.0012);
      return FXAI_Clamp(lr, 0.00001, 0.05000);
   }

   void AdamWStep(double &p,
                  double &m,
                  double &v,
                  const double g,
                  const double lr,
                  const double wd)
   {
      FXAI_OptAdamWStep(p, m, v, g, lr, 0.90, 0.999, wd, MathMax(m_adam_t, 1));
   }

   double HuberGrad(const double err,
                    const double delta) const
   {
      double d = (delta > 0.0 ? delta : 1.0);
      if(err > d) return d;
      if(err < -d) return -d;
      return err;
   }

   double PinballHuberGrad(const double target,
                           const double q,
                           const double tau,
                           const double kappa) const
   {
      double k = (kappa > 1e-6 ? kappa : 1.0);
      double u = target - q;
      if(u >= k) return -tau;
      if(u <= -k) return (1.0 - tau);

      double g = (0.5 - tau) - (u / (2.0 * k));
      return FXAI_Clamp(g, -tau, 1.0 - tau);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost_points,
                      const double sample_w) const
   {
      double sw = FXAI_Clamp(sample_w, 0.25, 4.00);
      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);

      if(cls == FXAI_TFT_SKIP)
      {
         if(edge <= 0.0) return FXAI_Clamp(sw * 1.40, 0.25, 6.00);
         return FXAI_Clamp(sw * 0.75, 0.25, 6.00);
      }

      if(edge <= 0.0) return FXAI_Clamp(sw * 0.65, 0.25, 6.00);
      return FXAI_Clamp(sw * (1.0 + 0.05 * MathMin(edge, 20.0)), 0.25, 6.00);
   }

   double MoveWeight(const double move_points,
                     const double cost_points,
                     const double sample_w) const
   {
      double sw = FXAI_Clamp(sample_w, 0.25, 4.00);
      double ew = FXAI_MoveEdgeWeight(move_points, cost_points);
      double edge = MathMax(0.0, MathAbs(move_points) - MathMax(cost_points, 0.0));
      return FXAI_Clamp(sw * ew * (1.0 + 0.05 * MathMin(edge, 20.0)), 0.25, 8.00);
   }

   int MapClass(const int y,
                const double &x[],
                const double move_points) const
   {
      if(y == FXAI_TFT_SELL || y == FXAI_TFT_BUY || y == FXAI_TFT_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return FXAI_TFT_SKIP;

      if(y > 0) return FXAI_TFT_BUY;
      if(y == 0) return FXAI_TFT_SELL;
      return (move_points >= 0.0 ? FXAI_TFT_BUY : FXAI_TFT_SELL);
   }

   void SoftmaxN(const double &logits[],
                 const int n,
                 double &probs[]) const
   {
      double m = logits[0];
      for(int i=1; i<n; i++)
         if(logits[i] > m) m = logits[i];

      double den = 0.0;
      for(int i=0; i<n; i++)
      {
         probs[i] = MathExp(FXAI_ClipSym(logits[i] - m, 30.0));
         den += probs[i];
      }
      if(den <= 0.0) den = 1.0;
      for(int i=0; i<n; i++)
         probs[i] /= den;
   }

   void ResetNorm(void)
   {
      m_norm_ready = false;
      m_norm_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void ResetHistory(void)
   {
      m_hist_len = 0;
      m_hist_ptr = 0;
      for(int t=0; t<FXAI_TFT_SEQ; t++)
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_hist_x[t][i] = 0.0;
   }

   void ResetTrainBuffer(void)
   {
      m_train_len = 0;
      for(int t=0; t<FXAI_TFT_TBPTT; t++)
      {
         m_train_cls[t] = FXAI_TFT_SKIP;
         m_train_move[t] = 0.0;
         m_train_cost[t] = 0.0;
         m_train_w[t] = 1.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_train_x[t][i] = 0.0;
      }
   }

   void ResetWalkForward(void)
   {
      m_wf_len = 0;
      m_wf_ptr = 0;
      for(int i=0; i<FXAI_TFT_WF; i++)
      {
         m_wf_pup[i] = 0.5;
         m_wf_pskip[i] = 0.5;
         m_wf_move[i] = 0.0;
         m_wf_cost[i] = 0.0;
         m_wf_cls[i] = FXAI_TFT_SKIP;
         m_wf_sess[i] = 3;
      }
   }

   void ResetSessionCalibration(void)
   {
      for(int s=0; s<FXAI_TFT_SESSIONS; s++)
      {
         m_sess_a[s] = 1.0;
         m_sess_b[s] = 0.0;
         m_sess_steps[s] = 0;
         m_sess_ready[s] = false;
      }
   }

   void InitStaticMask(void)
   {
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_static_mask[i] = 0.0;

      m_static_mask[0] = 1.0;
      // feature index = i-1; mark more stable/context features.
      m_static_mask[5] = 1.0;
      m_static_mask[7] = 1.0;
      m_static_mask[11] = 1.0;
      m_static_mask[12] = 1.0;
      m_static_mask[13] = 1.0;
      m_static_mask[14] = 1.0;
      m_static_mask[15] = 1.0;
   }

   void ZeroMoments(void)
   {
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_m_vsn_gate_b[i] = 0.0;
         m_v_vsn_gate_b[i] = 0.0;
         for(int j=0; j<FXAI_AI_WEIGHTS; j++)
         {
            m_m_vsn_gate_w[i][j] = 0.0;
            m_v_vsn_gate_w[i][j] = 0.0;
         }
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_m_vsn_proj_w[i][h] = 0.0;
            m_v_vsn_proj_w[i][h] = 0.0;
            m_m_vsn_proj_b[i][h] = 0.0;
            m_v_vsn_proj_b[i][h] = 0.0;
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_m_static_b[h] = 0.0;
         m_v_static_b[h] = 0.0;

         m_m_enc_h0_b[h] = 0.0; m_v_enc_h0_b[h] = 0.0;
         m_m_enc_c0_b[h] = 0.0; m_v_enc_c0_b[h] = 0.0;
         m_m_dec_h0_b[h] = 0.0; m_v_dec_h0_b[h] = 0.0;
         m_m_dec_c0_b[h] = 0.0; m_v_dec_c0_b[h] = 0.0;

         m_m_e_bi[h] = 0.0; m_v_e_bi[h] = 0.0;
         m_m_e_bf[h] = 0.0; m_v_e_bf[h] = 0.0;
         m_m_e_bo[h] = 0.0; m_v_e_bo[h] = 0.0;
         m_m_e_bg[h] = 0.0; m_v_e_bg[h] = 0.0;

         m_m_d_bi[h] = 0.0; m_v_d_bi[h] = 0.0;
         m_m_d_bf[h] = 0.0; m_v_d_bf[h] = 0.0;
         m_m_d_bo[h] = 0.0; m_v_d_bo[h] = 0.0;
         m_m_d_bg[h] = 0.0; m_v_d_bg[h] = 0.0;

         m_m_ff1_b[h] = 0.0; m_v_ff1_b[h] = 0.0;
         m_m_ff2_b[h] = 0.0; m_v_ff2_b[h] = 0.0;

         m_m_w_mu[h] = 0.0; m_v_w_mu[h] = 0.0;
         m_m_w_logv[h] = 0.0; m_v_w_logv[h] = 0.0;
         m_m_w_q25[h] = 0.0; m_v_w_q25[h] = 0.0;
         m_m_w_q75[h] = 0.0; m_v_w_q75[h] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_m_static_w[h][i] = 0.0;
            m_v_static_w[h][i] = 0.0;
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            m_m_enc_h0_w[h][j] = 0.0; m_v_enc_h0_w[h][j] = 0.0;
            m_m_enc_c0_w[h][j] = 0.0; m_v_enc_c0_w[h][j] = 0.0;
            m_m_dec_h0_s_w[h][j] = 0.0; m_v_dec_h0_s_w[h][j] = 0.0;
            m_m_dec_h0_e_w[h][j] = 0.0; m_v_dec_h0_e_w[h][j] = 0.0;
            m_m_dec_c0_s_w[h][j] = 0.0; m_v_dec_c0_s_w[h][j] = 0.0;
            m_m_dec_c0_e_w[h][j] = 0.0; m_v_dec_c0_e_w[h][j] = 0.0;

            m_m_e_wi_x[h][j] = 0.0; m_v_e_wi_x[h][j] = 0.0;
            m_m_e_wf_x[h][j] = 0.0; m_v_e_wf_x[h][j] = 0.0;
            m_m_e_wo_x[h][j] = 0.0; m_v_e_wo_x[h][j] = 0.0;
            m_m_e_wg_x[h][j] = 0.0; m_v_e_wg_x[h][j] = 0.0;
            m_m_e_wi_h[h][j] = 0.0; m_v_e_wi_h[h][j] = 0.0;
            m_m_e_wf_h[h][j] = 0.0; m_v_e_wf_h[h][j] = 0.0;
            m_m_e_wo_h[h][j] = 0.0; m_v_e_wo_h[h][j] = 0.0;
            m_m_e_wg_h[h][j] = 0.0; m_v_e_wg_h[h][j] = 0.0;

            m_m_d_wi_x[h][j] = 0.0; m_v_d_wi_x[h][j] = 0.0;
            m_m_d_wf_x[h][j] = 0.0; m_v_d_wf_x[h][j] = 0.0;
            m_m_d_wo_x[h][j] = 0.0; m_v_d_wo_x[h][j] = 0.0;
            m_m_d_wg_x[h][j] = 0.0; m_v_d_wg_x[h][j] = 0.0;
            m_m_d_wi_h[h][j] = 0.0; m_v_d_wi_h[h][j] = 0.0;
            m_m_d_wf_h[h][j] = 0.0; m_v_d_wf_h[h][j] = 0.0;
            m_m_d_wo_h[h][j] = 0.0; m_v_d_wo_h[h][j] = 0.0;
            m_m_d_wg_h[h][j] = 0.0; m_v_d_wg_h[h][j] = 0.0;

            m_m_ff1_w[h][j] = 0.0; m_v_ff1_w[h][j] = 0.0;
            m_m_ff2_w[h][j] = 0.0; m_v_ff2_w[h][j] = 0.0;
         }

         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         {
            m_m_w_cls[c][h] = 0.0;
            m_v_w_cls[c][h] = 0.0;
         }
      }

      for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
      {
         m_m_b_cls[c] = 0.0;
         m_v_b_cls[c] = 0.0;
      }

      for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
      {
         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               m_m_wq[hd][d][h] = 0.0; m_v_wq[hd][d][h] = 0.0;
               m_m_wk[hd][d][h] = 0.0; m_v_wk[hd][d][h] = 0.0;
               m_m_wv[hd][d][h] = 0.0; m_v_wv[hd][d][h] = 0.0;
            }
            for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
            {
               m_m_wo[hd][o][d] = 0.0;
               m_v_wo[hd][o][d] = 0.0;
            }
         }
      }

      for(int i=0; i<FXAI_TFT_SEQ; i++)
      {
         m_m_rel_bias[i] = 0.0;
         m_v_rel_bias[i] = 0.0;
      }

      m_m_b_mu = 0.0; m_v_b_mu = 0.0;
      m_m_b_logv = 0.0; m_v_b_logv = 0.0;
      m_m_b_q25 = 0.0; m_v_b_q25 = 0.0;
      m_m_b_q75 = 0.0; m_v_b_q75 = 0.0;
   }

   void InitParams(void)
   {
      InitStaticMask();
      ResetNorm();
      ResetHistory();
      ResetTrainBuffer();
      ResetWalkForward();
      ResetSessionCalibration();
      m_step = 0;
      m_seen = 0;
      m_adam_t = 0;

      m_thr_buy = 0.62;
      m_thr_sell = 0.38;
      m_thr_skip = 0.58;

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_vsn_gate_b[i] = 0.0;
         for(int j=0; j<FXAI_AI_WEIGHTS; j++)
         {
            double s = (double)((i + 1) * (j + 3));
            m_vsn_gate_w[i][j] = 0.05 * MathSin(0.63 * s);
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double s2 = (double)((i + 2) * (h + 1));
            m_vsn_proj_w[i][h] = 0.06 * MathCos(0.71 * s2);
            m_vsn_proj_b[i][h] = 0.0;
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_static_b[h] = 0.0;
         m_enc_h0_b[h] = 0.0;
         m_enc_c0_b[h] = 0.0;
         m_dec_h0_b[h] = 0.0;
         m_dec_c0_b[h] = 0.0;

         m_e_bi[h] = 0.0;
         m_e_bf[h] = 0.40;
         m_e_bo[h] = 0.0;
         m_e_bg[h] = 0.0;

         m_d_bi[h] = 0.0;
         m_d_bf[h] = 0.35;
         m_d_bo[h] = 0.0;
         m_d_bg[h] = 0.0;

         m_ff1_b[h] = 0.0;
         m_ff2_b[h] = 0.0;

         m_w_mu[h] = 0.04 * MathSin((double)(h + 1) * 0.91);
         m_w_logv[h] = 0.03 * MathCos((double)(h + 1) * 0.99);
         m_w_q25[h] = 0.03 * MathSin((double)(h + 1) * 1.07);
         m_w_q75[h] = 0.03 * MathCos((double)(h + 1) * 1.13);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double s = (double)((h + 1) * (i + 2));
            m_static_w[h][i] = 0.04 * MathSin(0.67 * s);
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            double s2 = (double)((h + 2) * (j + 3));
            m_enc_h0_w[h][j] = 0.05 * MathSin(0.59 * s2);
            m_enc_c0_w[h][j] = 0.05 * MathCos(0.61 * s2);
            m_dec_h0_s_w[h][j] = 0.05 * MathCos(0.63 * s2);
            m_dec_h0_e_w[h][j] = 0.05 * MathSin(0.65 * s2);
            m_dec_c0_s_w[h][j] = 0.05 * MathSin(0.69 * s2);
            m_dec_c0_e_w[h][j] = 0.05 * MathCos(0.73 * s2);

            m_e_wi_x[h][j] = 0.05 * MathSin(0.77 * s2);
            m_e_wf_x[h][j] = 0.05 * MathCos(0.79 * s2);
            m_e_wo_x[h][j] = 0.05 * MathSin(0.81 * s2);
            m_e_wg_x[h][j] = 0.05 * MathCos(0.83 * s2);

            m_e_wi_h[h][j] = 0.04 * MathSin(0.85 * s2);
            m_e_wf_h[h][j] = 0.04 * MathCos(0.87 * s2);
            m_e_wo_h[h][j] = 0.04 * MathSin(0.89 * s2);
            m_e_wg_h[h][j] = 0.04 * MathCos(0.91 * s2);

            m_d_wi_x[h][j] = 0.05 * MathCos(0.93 * s2);
            m_d_wf_x[h][j] = 0.05 * MathSin(0.95 * s2);
            m_d_wo_x[h][j] = 0.05 * MathCos(0.97 * s2);
            m_d_wg_x[h][j] = 0.05 * MathSin(0.99 * s2);

            m_d_wi_h[h][j] = 0.04 * MathCos(1.01 * s2);
            m_d_wf_h[h][j] = 0.04 * MathSin(1.03 * s2);
            m_d_wo_h[h][j] = 0.04 * MathCos(1.07 * s2);
            m_d_wg_h[h][j] = 0.04 * MathSin(1.09 * s2);

            m_ff1_w[h][j] = 0.05 * MathCos(1.11 * s2);
            m_ff2_w[h][j] = 0.05 * MathSin(1.13 * s2);
         }

         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         {
            double s3 = (double)((c + 2) * (h + 1));
            m_w_cls[c][h] = 0.05 * MathSin(0.74 * s3);
         }
      }

      for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
      {
         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               double s = (double)((hd + 1) * (d + 2) * (h + 3));
               m_wq[hd][d][h] = 0.05 * MathSin(0.67 * s);
               m_wk[hd][d][h] = 0.05 * MathCos(0.71 * s);
               m_wv[hd][d][h] = 0.05 * MathSin(0.73 * s);
               m_wo[hd][h][d] = 0.05 * MathCos(0.79 * s);
            }
         }
      }

      for(int i=0; i<FXAI_TFT_SEQ; i++)
         m_rel_bias[i] = 0.0;

      for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         m_b_cls[c] = 0.0;
      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.5;

      ZeroMoments();
      SyncShadow(true);
      m_shadow_ready = false;
      m_initialized = true;
   }

   void SyncShadow(const bool hard)
   {
      double a = (hard ? 0.0 : 0.995);
      double b = (hard ? 1.0 : (1.0 - a));

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_s_vsn_gate_b[i] = (hard ? m_vsn_gate_b[i] : a * m_s_vsn_gate_b[i] + b * m_vsn_gate_b[i]);
         for(int j=0; j<FXAI_AI_WEIGHTS; j++)
            m_s_vsn_gate_w[i][j] = (hard ? m_vsn_gate_w[i][j] : a * m_s_vsn_gate_w[i][j] + b * m_vsn_gate_w[i][j]);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_s_vsn_proj_w[i][h] = (hard ? m_vsn_proj_w[i][h] : a * m_s_vsn_proj_w[i][h] + b * m_vsn_proj_w[i][h]);
            m_s_vsn_proj_b[i][h] = (hard ? m_vsn_proj_b[i][h] : a * m_s_vsn_proj_b[i][h] + b * m_vsn_proj_b[i][h]);
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_s_static_b[h] = (hard ? m_static_b[h] : a * m_s_static_b[h] + b * m_static_b[h]);

         m_s_enc_h0_b[h] = (hard ? m_enc_h0_b[h] : a * m_s_enc_h0_b[h] + b * m_enc_h0_b[h]);
         m_s_enc_c0_b[h] = (hard ? m_enc_c0_b[h] : a * m_s_enc_c0_b[h] + b * m_enc_c0_b[h]);
         m_s_dec_h0_b[h] = (hard ? m_dec_h0_b[h] : a * m_s_dec_h0_b[h] + b * m_dec_h0_b[h]);
         m_s_dec_c0_b[h] = (hard ? m_dec_c0_b[h] : a * m_s_dec_c0_b[h] + b * m_dec_c0_b[h]);

         m_s_e_bi[h] = (hard ? m_e_bi[h] : a * m_s_e_bi[h] + b * m_e_bi[h]);
         m_s_e_bf[h] = (hard ? m_e_bf[h] : a * m_s_e_bf[h] + b * m_e_bf[h]);
         m_s_e_bo[h] = (hard ? m_e_bo[h] : a * m_s_e_bo[h] + b * m_e_bo[h]);
         m_s_e_bg[h] = (hard ? m_e_bg[h] : a * m_s_e_bg[h] + b * m_e_bg[h]);

         m_s_d_bi[h] = (hard ? m_d_bi[h] : a * m_s_d_bi[h] + b * m_d_bi[h]);
         m_s_d_bf[h] = (hard ? m_d_bf[h] : a * m_s_d_bf[h] + b * m_d_bf[h]);
         m_s_d_bo[h] = (hard ? m_d_bo[h] : a * m_s_d_bo[h] + b * m_d_bo[h]);
         m_s_d_bg[h] = (hard ? m_d_bg[h] : a * m_s_d_bg[h] + b * m_d_bg[h]);

         m_s_ff1_b[h] = (hard ? m_ff1_b[h] : a * m_s_ff1_b[h] + b * m_ff1_b[h]);
         m_s_ff2_b[h] = (hard ? m_ff2_b[h] : a * m_s_ff2_b[h] + b * m_ff2_b[h]);

         m_s_w_mu[h] = (hard ? m_w_mu[h] : a * m_s_w_mu[h] + b * m_w_mu[h]);
         m_s_w_logv[h] = (hard ? m_w_logv[h] : a * m_s_w_logv[h] + b * m_w_logv[h]);
         m_s_w_q25[h] = (hard ? m_w_q25[h] : a * m_s_w_q25[h] + b * m_w_q25[h]);
         m_s_w_q75[h] = (hard ? m_w_q75[h] : a * m_s_w_q75[h] + b * m_w_q75[h]);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_s_static_w[h][i] = (hard ? m_static_w[h][i] : a * m_s_static_w[h][i] + b * m_static_w[h][i]);

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            m_s_enc_h0_w[h][j] = (hard ? m_enc_h0_w[h][j] : a * m_s_enc_h0_w[h][j] + b * m_enc_h0_w[h][j]);
            m_s_enc_c0_w[h][j] = (hard ? m_enc_c0_w[h][j] : a * m_s_enc_c0_w[h][j] + b * m_enc_c0_w[h][j]);
            m_s_dec_h0_s_w[h][j] = (hard ? m_dec_h0_s_w[h][j] : a * m_s_dec_h0_s_w[h][j] + b * m_dec_h0_s_w[h][j]);
            m_s_dec_h0_e_w[h][j] = (hard ? m_dec_h0_e_w[h][j] : a * m_s_dec_h0_e_w[h][j] + b * m_dec_h0_e_w[h][j]);
            m_s_dec_c0_s_w[h][j] = (hard ? m_dec_c0_s_w[h][j] : a * m_s_dec_c0_s_w[h][j] + b * m_dec_c0_s_w[h][j]);
            m_s_dec_c0_e_w[h][j] = (hard ? m_dec_c0_e_w[h][j] : a * m_s_dec_c0_e_w[h][j] + b * m_dec_c0_e_w[h][j]);

            m_s_e_wi_x[h][j] = (hard ? m_e_wi_x[h][j] : a * m_s_e_wi_x[h][j] + b * m_e_wi_x[h][j]);
            m_s_e_wf_x[h][j] = (hard ? m_e_wf_x[h][j] : a * m_s_e_wf_x[h][j] + b * m_e_wf_x[h][j]);
            m_s_e_wo_x[h][j] = (hard ? m_e_wo_x[h][j] : a * m_s_e_wo_x[h][j] + b * m_e_wo_x[h][j]);
            m_s_e_wg_x[h][j] = (hard ? m_e_wg_x[h][j] : a * m_s_e_wg_x[h][j] + b * m_e_wg_x[h][j]);
            m_s_e_wi_h[h][j] = (hard ? m_e_wi_h[h][j] : a * m_s_e_wi_h[h][j] + b * m_e_wi_h[h][j]);
            m_s_e_wf_h[h][j] = (hard ? m_e_wf_h[h][j] : a * m_s_e_wf_h[h][j] + b * m_e_wf_h[h][j]);
            m_s_e_wo_h[h][j] = (hard ? m_e_wo_h[h][j] : a * m_s_e_wo_h[h][j] + b * m_e_wo_h[h][j]);
            m_s_e_wg_h[h][j] = (hard ? m_e_wg_h[h][j] : a * m_s_e_wg_h[h][j] + b * m_e_wg_h[h][j]);

            m_s_d_wi_x[h][j] = (hard ? m_d_wi_x[h][j] : a * m_s_d_wi_x[h][j] + b * m_d_wi_x[h][j]);
            m_s_d_wf_x[h][j] = (hard ? m_d_wf_x[h][j] : a * m_s_d_wf_x[h][j] + b * m_d_wf_x[h][j]);
            m_s_d_wo_x[h][j] = (hard ? m_d_wo_x[h][j] : a * m_s_d_wo_x[h][j] + b * m_d_wo_x[h][j]);
            m_s_d_wg_x[h][j] = (hard ? m_d_wg_x[h][j] : a * m_s_d_wg_x[h][j] + b * m_d_wg_x[h][j]);
            m_s_d_wi_h[h][j] = (hard ? m_d_wi_h[h][j] : a * m_s_d_wi_h[h][j] + b * m_d_wi_h[h][j]);
            m_s_d_wf_h[h][j] = (hard ? m_d_wf_h[h][j] : a * m_s_d_wf_h[h][j] + b * m_d_wf_h[h][j]);
            m_s_d_wo_h[h][j] = (hard ? m_d_wo_h[h][j] : a * m_s_d_wo_h[h][j] + b * m_d_wo_h[h][j]);
            m_s_d_wg_h[h][j] = (hard ? m_d_wg_h[h][j] : a * m_s_d_wg_h[h][j] + b * m_d_wg_h[h][j]);

            m_s_ff1_w[h][j] = (hard ? m_ff1_w[h][j] : a * m_s_ff1_w[h][j] + b * m_ff1_w[h][j]);
            m_s_ff2_w[h][j] = (hard ? m_ff2_w[h][j] : a * m_s_ff2_w[h][j] + b * m_ff2_w[h][j]);
         }

         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
            m_s_w_cls[c][h] = (hard ? m_w_cls[c][h] : a * m_s_w_cls[c][h] + b * m_w_cls[c][h]);
      }

      for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
      {
         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               m_s_wq[hd][d][h] = (hard ? m_wq[hd][d][h] : a * m_s_wq[hd][d][h] + b * m_wq[hd][d][h]);
               m_s_wk[hd][d][h] = (hard ? m_wk[hd][d][h] : a * m_s_wk[hd][d][h] + b * m_wk[hd][d][h]);
               m_s_wv[hd][d][h] = (hard ? m_wv[hd][d][h] : a * m_s_wv[hd][d][h] + b * m_wv[hd][d][h]);
               m_s_wo[hd][h][d] = (hard ? m_wo[hd][h][d] : a * m_s_wo[hd][h][d] + b * m_wo[hd][h][d]);
            }
         }
      }

      for(int i=0; i<FXAI_TFT_SEQ; i++)
         m_s_rel_bias[i] = (hard ? m_rel_bias[i] : a * m_s_rel_bias[i] + b * m_rel_bias[i]);

      for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         m_s_b_cls[c] = (hard ? m_b_cls[c] : a * m_s_b_cls[c] + b * m_b_cls[c]);
      m_s_b_mu = (hard ? m_b_mu : a * m_s_b_mu + b * m_b_mu);
      m_s_b_logv = (hard ? m_b_logv : a * m_s_b_logv + b * m_b_logv);
      m_s_b_q25 = (hard ? m_b_q25 : a * m_s_b_q25 + b * m_b_q25);
      m_s_b_q75 = (hard ? m_b_q75 : a * m_s_b_q75 + b * m_b_q75);
   }

   void UpdateNorm(const double &x[])
   {
      double a = (m_norm_steps < 128 ? 0.05 : 0.015);
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double d = x[i] - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = x[i] - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * dv * dv;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;
      }
      m_norm_steps++;
      if(m_norm_steps >= 32) m_norm_ready = true;
   }

   void Normalize(const double &x[],
                  double &xn[]) const
   {
      xn[0] = 1.0;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         if(!m_norm_ready)
         {
            xn[i] = FXAI_ClipSym(x[i], 8.0);
            continue;
         }
         double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FXAI_ClipSym((x[i] - m_x_mean[i]) * inv, 8.0);
      }
   }

   void PushHistory(const double &x[])
   {
      int p = m_hist_ptr;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_hist_x[p][i] = x[i];

      m_hist_ptr++;
      if(m_hist_ptr >= FXAI_TFT_SEQ) m_hist_ptr = 0;
      if(m_hist_len < FXAI_TFT_SEQ) m_hist_len++;
   }

   int HistIndexBack(const int back) const
   {
      if(m_hist_len <= 0) return -1;
      int b = back;
      if(b < 0) b = 0;
      if(b >= m_hist_len) b = m_hist_len - 1;
      int idx = m_hist_ptr - 1 - b;
      while(idx < 0) idx += FXAI_TFT_SEQ;
      while(idx >= FXAI_TFT_SEQ) idx -= FXAI_TFT_SEQ;
      return idx;
   }

