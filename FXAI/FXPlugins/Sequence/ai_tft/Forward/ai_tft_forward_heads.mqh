      double g_d_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_d_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_d_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_d_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_d_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_d_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_d_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_d_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_d_bi[FXAI_AI_MLP_HIDDEN], g_d_bf[FXAI_AI_MLP_HIDDEN], g_d_bo[FXAI_AI_MLP_HIDDEN], g_d_bg[FXAI_AI_MLP_HIDDEN];

      double g_wq[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN], g_wk[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN], g_wv[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
      double g_wo[FXAI_TFT_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_TFT_D_HEAD];
      double g_rel_bias[FXAI_TFT_SEQ];

      double g_ff1_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_ff1_b[FXAI_AI_MLP_HIDDEN];
      double g_ff2_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_ff2_b[FXAI_AI_MLP_HIDDEN];

      double g_w_cls[FXAI_TFT_CLASS_COUNT][FXAI_AI_MLP_HIDDEN], g_b_cls[FXAI_TFT_CLASS_COUNT];
      double g_w_mu[FXAI_AI_MLP_HIDDEN], g_b_mu;
      double g_w_logv[FXAI_AI_MLP_HIDDEN], g_b_logv;
      double g_w_q25[FXAI_AI_MLP_HIDDEN], g_b_q25;
      double g_w_q75[FXAI_AI_MLP_HIDDEN], g_b_q75;

      // Zero gradients.
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         g_vsn_gate_b[i] = 0.0;
         for(int j=0; j<FXAI_AI_WEIGHTS; j++)
            g_vsn_gate_w[i][j] = 0.0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g_vsn_proj_w[i][h] = 0.0;
            g_vsn_proj_b[i][h] = 0.0;
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         g_static_b[h] = 0.0;
         g_enc_h0_b[h] = 0.0;
         g_enc_c0_b[h] = 0.0;
         g_dec_h0_b[h] = 0.0;
         g_dec_c0_b[h] = 0.0;

         g_e_bi[h] = 0.0; g_e_bf[h] = 0.0; g_e_bo[h] = 0.0; g_e_bg[h] = 0.0;
         g_d_bi[h] = 0.0; g_d_bf[h] = 0.0; g_d_bo[h] = 0.0; g_d_bg[h] = 0.0;

         g_ff1_b[h] = 0.0;
         g_ff2_b[h] = 0.0;

         g_w_mu[h] = 0.0;
         g_w_logv[h] = 0.0;
         g_w_q25[h] = 0.0;
         g_w_q75[h] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            g_static_w[h][i] = 0.0;

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            g_enc_h0_w[h][j] = 0.0; g_enc_c0_w[h][j] = 0.0;
            g_dec_h0_s_w[h][j] = 0.0; g_dec_h0_e_w[h][j] = 0.0;
            g_dec_c0_s_w[h][j] = 0.0; g_dec_c0_e_w[h][j] = 0.0;

            g_e_wi_x[h][j] = 0.0; g_e_wf_x[h][j] = 0.0; g_e_wo_x[h][j] = 0.0; g_e_wg_x[h][j] = 0.0;
            g_e_wi_h[h][j] = 0.0; g_e_wf_h[h][j] = 0.0; g_e_wo_h[h][j] = 0.0; g_e_wg_h[h][j] = 0.0;

            g_d_wi_x[h][j] = 0.0; g_d_wf_x[h][j] = 0.0; g_d_wo_x[h][j] = 0.0; g_d_wg_x[h][j] = 0.0;
            g_d_wi_h[h][j] = 0.0; g_d_wf_h[h][j] = 0.0; g_d_wo_h[h][j] = 0.0; g_d_wg_h[h][j] = 0.0;

            g_ff1_w[h][j] = 0.0;
            g_ff2_w[h][j] = 0.0;
         }

         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
            g_w_cls[c][h] = 0.0;
      }
      for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         g_b_cls[c] = 0.0;
      g_b_mu = 0.0;
      g_b_logv = 0.0;
      g_b_q25 = 0.0;
      g_b_q75 = 0.0;

      for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
      {
         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               g_wq[hd][d][h] = 0.0;
               g_wk[hd][d][h] = 0.0;
               g_wv[hd][d][h] = 0.0;
               g_wo[hd][h][d] = 0.0;
            }
         }
      }
      for(int i=0; i<FXAI_TFT_SEQ; i++)
         g_rel_bias[i] = 0.0;

      // Backprop buffers.
      double d_emb_total[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
      double d_enc_attn[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
      for(int t=0; t<n; t++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            d_emb_total[t][h] = 0.0;
            d_enc_attn[t][h] = 0.0;
         }
      }

      double dh_dec_next[FXAI_AI_MLP_HIDDEN], dc_dec_next[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         dh_dec_next[h] = 0.0;
         dc_dec_next[h] = 0.0;
      }

      // Decoder-time reverse pass.
      for(int t=n - 1; t>=0; t--)
      {
         int cls = m_train_cls[t];
         double mv = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = FXAI_Clamp(m_train_w[t], 0.25, 4.00);

         double w_cls = ClassWeight(cls, mv, cost, sw);
         double w_mv = MoveWeight(mv, cost, sw);

         double pt = FXAI_Clamp(c_probs[t][cls], 0.001, 0.999);
         double focal = MathPow(FXAI_Clamp(1.0 - pt, 0.02, 1.0), 1.50);

         double dlogit[FXAI_TFT_CLASS_COUNT];
         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         {
            double yk = (c == cls ? 1.0 : 0.0);
            dlogit[c] = FXAI_ClipSym((c_probs[t][c] - yk) * w_cls * focal, 4.0);
         }

         double target = MathAbs(mv);
         double mu = c_mu[t];
         double logv = c_logv[t];
         double q25 = c_q25[t];
         double q75 = c_q75[t];
         double var = FXAI_Clamp(MathExp(logv), 0.05, 100.0);

         double gmu = FXAI_ClipSym((HuberGrad(mu - target, 6.0) / MathMax(var, 0.25)) * w_mv, 5.0);
         double glv = FXAI_ClipSym(0.5 * w_mv * (1.0 - ((mu - target) * (mu - target)) / MathMax(var, 0.25)), 4.0);
         double gq25 = FXAI_ClipSym(PinballHuberGrad(target, q25, 0.25, 1.5) * w_mv, 3.0);
         double gq75 = FXAI_ClipSym(PinballHuberGrad(target, q75, 0.75, 1.5) * w_mv, 3.0);
         if(q25 > q75)
         {
            double pen = 0.20 * FXAI_ClipSym(q25 - q75, 4.0);
            gq25 += pen;
            gq75 -= pen;
         }

         double d_final[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double d = dh_dec_next[h];
            for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
               d += dlogit[c] * m_w_cls[c][h];
            d += gmu * m_w_mu[h] + glv * m_w_logv[h] + gq25 * m_w_q25[h] + gq75 * m_w_q75[h];
            d_final[h] = d;

            g_w_mu[h] += gmu * c_final[t][h];
            g_w_logv[h] += glv * c_final[t][h];
            g_w_q25[h] += gq25 * c_final[t][h];
            g_w_q75[h] += gq75 * c_final[t][h];

            for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
               g_w_cls[c][h] += dlogit[c] * c_final[t][h];
         }
         g_b_mu += gmu;
         g_b_logv += glv;
         g_b_q25 += gq25;
         g_b_q75 += gq75;
         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
            g_b_cls[c] += dlogit[c];

         // FFN backward: final = pre + stoch * ff2.
         double d_pre[FXAI_AI_MLP_HIDDEN];
         double d_ff2[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            d_pre[h] = d_final[h];
            d_ff2[h] = d_final[h] * c_stoch_scale[t];
            g_ff2_b[h] += d_ff2[h];
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               g_ff2_w[h][i] += d_ff2[h] * c_ff1[t][i];
         }

         double d_ff1[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            double s = 0.0;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               s += m_ff2_w[h][i] * d_ff2[h];
            d_ff1[i] = s;
         }

         double d_z1[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            d_z1[h] = d_ff1[h] * c_ff1_mask[t][h] * (1.0 - c_ff1_raw[t][h] * c_ff1_raw[t][h]);
            g_ff1_b[h] += d_z1[h];
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               g_ff1_w[h][i] += d_z1[h] * c_pre[t][i];
         }

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            double s = 0.0;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               s += m_ff1_w[h][i] * d_z1[h];
            d_pre[i] += s;
         }

         // Split pre = dec_h + attn_out.
         double d_dec[FXAI_AI_MLP_HIDDEN];
         double d_attn_out[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            d_dec[h] = d_pre[h];
            d_attn_out[h] = d_pre[h];
         }

         // Attention backward.
         double inv_scale = 1.0 / MathSqrt((double)FXAI_TFT_D_HEAD);
         for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
         {
            double d_ctx[FXAI_TFT_D_HEAD];
            for(int d=0; d<FXAI_TFT_D_HEAD; d++) d_ctx[d] = 0.0;

            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               for(int d=0; d<FXAI_TFT_D_HEAD; d++)
               {
                  g_wo[hd][h][d] += d_attn_out[h] * c_attn_ctx[t][hd][d];
                  d_ctx[d] += d_attn_out[h] * m_wo[hd][h][d];
               }
            }

            double d_a[FXAI_TFT_TBPTT];
            for(int j=0; j<n; j++) d_a[j] = 0.0;
            double d_v[FXAI_TFT_TBPTT][FXAI_TFT_D_HEAD];
            for(int j=0; j<n; j++)
               for(int d=0; d<FXAI_TFT_D_HEAD; d++)
                  d_v[j][d] = 0.0;

            for(int j=0; j<=t && j<n; j++)
            {
               for(int d=0; d<FXAI_TFT_D_HEAD; d++)
               {
                  d_a[j] += d_ctx[d] * c_attn_v[t][hd][j][d];
                  d_v[j][d] += d_ctx[d] * c_attn_w[t][hd][j];
               }
            }

            double sum_ad = 0.0;
            for(int j=0; j<=t && j<n; j++)
               sum_ad += c_attn_w[t][hd][j] * d_a[j];

            double d_s[FXAI_TFT_TBPTT];
            for(int j=0; j<n; j++) d_s[j] = 0.0;
            for(int j=0; j<=t && j<n; j++)
               d_s[j] = c_attn_w[t][hd][j] * (d_a[j] - sum_ad);

            double d_q[FXAI_TFT_D_HEAD];
            for(int d=0; d<FXAI_TFT_D_HEAD; d++) d_q[d] = 0.0;

            for(int j=0; j<=t && j<n; j++)
            {
               int lag = t - j;
               if(lag < 0) lag = 0;
               if(lag >= FXAI_TFT_SEQ) lag = FXAI_TFT_SEQ - 1;
               g_rel_bias[lag] += d_s[j];

               for(int d=0; d<FXAI_TFT_D_HEAD; d++)
               {
                  d_q[d] += d_s[j] * inv_scale * c_attn_k[t][hd][j][d];

                  double dk = d_s[j] * inv_scale * c_attn_q[t][hd][d];
                  g_wk[hd][d][0] += 0.0; // keep compiler from optimizing away dimensions in strict builds.
                  for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
                  {
                     g_wk[hd][d][h] += dk * c_e_h[j][h];
                     d_enc_attn[j][h] += m_wk[hd][d][h] * dk;
                  }

                  for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
                  {
                     g_wv[hd][d][h] += d_v[j][d] * c_e_h[j][h];
                     d_enc_attn[j][h] += m_wv[hd][d][h] * d_v[j][d];
                  }
               }
            }

            for(int d=0; d<FXAI_TFT_D_HEAD; d++)
            {
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               {
                  g_wq[hd][d][h] += d_q[d] * c_d_h[t][h];
                  d_dec[h] += m_wq[hd][d][h] * d_q[d];
               }
            }
         }

         // Decoder LSTM backward.
         double dh_prev[FXAI_AI_MLP_HIDDEN];
         double dc_prev[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            dh_prev[h] = 0.0;
            dc_prev[h] = 0.0;
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double tanh_c = FXAI_Tanh(c_d_c[t][h]);
            double dh = d_dec[h];
            double dc = dc_dec_next[h] + dh * c_d_o[t][h] * (1.0 - tanh_c * tanh_c);

            double doo = (dh * tanh_c) * c_d_o[t][h] * (1.0 - c_d_o[t][h]);
            double di = (dc * c_d_g[t][h]) * c_d_i[t][h] * (1.0 - c_d_i[t][h]);
            double df = (dc * c_d_c_prev[t][h]) * c_d_f[t][h] * (1.0 - c_d_f[t][h]);
            double dg = (dc * c_d_i[t][h]) * (1.0 - c_d_g[t][h] * c_d_g[t][h]);

            g_d_bi[h] += di;
            g_d_bf[h] += df;
            g_d_bo[h] += doo;
            g_d_bg[h] += dg;

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               g_d_wi_x[h][i] += di * c_emb[t][i];
               g_d_wf_x[h][i] += df * c_emb[t][i];
               g_d_wo_x[h][i] += doo * c_emb[t][i];
               g_d_wg_x[h][i] += dg * c_emb[t][i];

               d_emb_total[t][i] += m_d_wi_x[h][i] * di +
                                    m_d_wf_x[h][i] * df +
                                    m_d_wo_x[h][i] * doo +
                                    m_d_wg_x[h][i] * dg;

               g_d_wi_h[h][i] += di * c_d_h_prev[t][i];
               g_d_wf_h[h][i] += df * c_d_h_prev[t][i];
               g_d_wo_h[h][i] += doo * c_d_h_prev[t][i];
               g_d_wg_h[h][i] += dg * c_d_h_prev[t][i];

               dh_prev[i] += m_d_wi_h[h][i] * di +
                             m_d_wf_h[h][i] * df +
                             m_d_wo_h[h][i] * doo +
                             m_d_wg_h[h][i] * dg;
            }

            dc_prev[h] = dc * c_d_f[t][h];
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            dh_dec_next[h] = dh_prev[h];
            dc_dec_next[h] = dc_prev[h];
         }

         // Calibration + walk-forward updates with pre-calibration raw outputs.
         double den = c_probs[t][FXAI_TFT_BUY] + c_probs[t][FXAI_TFT_SELL];
         if(den < 1e-9) den = 1e-9;
         double p_dir_raw = c_probs[t][FXAI_TFT_BUY] / den;

         int ydir = (cls == FXAI_TFT_BUY ? 1 : 0);
         if(cls == FXAI_TFT_SKIP)
            ydir = (mv >= 0.0 ? 1 : 0);

         int sess = SessionBucket(ResolveContextTime());
         UpdateSessionCalibration(sess, p_dir_raw, ydir, sw);

         if(cls == FXAI_TFT_SKIP)
            UpdateCalibration(p_dir_raw, ydir, 0.25 * sw);
         else
            UpdateCalibration(p_dir_raw, ydir, sw);

         FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, mv, 0.05);
         double x_train_t[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            x_train_t[i] = m_train_x[t][i];
         UpdateMoveHead(x_train_t, mv, hp, sw);

         AddWalkForwardSample(p_dir_raw,
                              c_probs[t][FXAI_TFT_SKIP],
                              cls,
                              mv,
                              cost,
                              sess);
      }

      // Decoder init gradients.
      double d_static[FXAI_AI_MLP_HIDDEN];
      double d_enc_last_from_dec_init[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         d_static[h] = 0.0;
         d_enc_last_from_dec_init[h] = 0.0;
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double dz_h = dh_dec_next[h] * (1.0 - c_dec_h0[h] * c_dec_h0[h]);
         double dz_c = dc_dec_next[h] * (1.0 - c_dec_c0[h] * c_dec_c0[h]);

         g_dec_h0_b[h] += dz_h;
         g_dec_c0_b[h] += dz_c;

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            g_dec_h0_s_w[h][j] += dz_h * c_static[j];
            g_dec_h0_e_w[h][j] += dz_h * c_e_h[n - 1][j];
            g_dec_c0_s_w[h][j] += dz_c * c_static[j];
            g_dec_c0_e_w[h][j] += dz_c * c_e_h[n - 1][j];

            d_static[j] += m_dec_h0_s_w[h][j] * dz_h + m_dec_c0_s_w[h][j] * dz_c;
            d_enc_last_from_dec_init[j] += m_dec_h0_e_w[h][j] * dz_h + m_dec_c0_e_w[h][j] * dz_c;
         }
      }

      // Encoder reverse pass.
      double dh_enc_next[FXAI_AI_MLP_HIDDEN], dc_enc_next[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         dh_enc_next[h] = d_enc_last_from_dec_init[h];
         dc_enc_next[h] = 0.0;
      }

      for(int t=n - 1; t>=0; t--)
      {
         double dh_prev[FXAI_AI_MLP_HIDDEN];
         double dc_prev[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            dh_prev[h] = 0.0;
            dc_prev[h] = 0.0;
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double dh = dh_enc_next[h] + d_enc_attn[t][h];
            double tanh_c = FXAI_Tanh(c_e_c[t][h]);
            double dc = dc_enc_next[h] + dh * c_e_o[t][h] * (1.0 - tanh_c * tanh_c);

            double doo = (dh * tanh_c) * c_e_o[t][h] * (1.0 - c_e_o[t][h]);
            double di = (dc * c_e_g[t][h]) * c_e_i[t][h] * (1.0 - c_e_i[t][h]);
            double df = (dc * c_e_c_prev[t][h]) * c_e_f[t][h] * (1.0 - c_e_f[t][h]);
            double dg = (dc * c_e_i[t][h]) * (1.0 - c_e_g[t][h] * c_e_g[t][h]);

            g_e_bi[h] += di;
            g_e_bf[h] += df;
            g_e_bo[h] += doo;
            g_e_bg[h] += dg;

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               g_e_wi_x[h][i] += di * c_emb[t][i];
               g_e_wf_x[h][i] += df * c_emb[t][i];
               g_e_wo_x[h][i] += doo * c_emb[t][i];
               g_e_wg_x[h][i] += dg * c_emb[t][i];

               d_emb_total[t][i] += m_e_wi_x[h][i] * di +
                                    m_e_wf_x[h][i] * df +
                                    m_e_wo_x[h][i] * doo +
                                    m_e_wg_x[h][i] * dg;

               g_e_wi_h[h][i] += di * c_e_h_prev[t][i];
               g_e_wf_h[h][i] += df * c_e_h_prev[t][i];
               g_e_wo_h[h][i] += doo * c_e_h_prev[t][i];
               g_e_wg_h[h][i] += dg * c_e_h_prev[t][i];

               dh_prev[i] += m_e_wi_h[h][i] * di +
                             m_e_wf_h[h][i] * df +
                             m_e_wo_h[h][i] * doo +
                             m_e_wg_h[h][i] * dg;
            }

            dc_prev[h] = dc * c_e_f[t][h];
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            dh_enc_next[h] = dh_prev[h];
            dc_enc_next[h] = dc_prev[h];
         }
      }

      // Encoder init gradients.
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double dz_h = dh_enc_next[h] * (1.0 - c_enc_h0[h] * c_enc_h0[h]);
         double dz_c = dc_enc_next[h] * (1.0 - c_enc_c0[h] * c_enc_c0[h]);

         g_enc_h0_b[h] += dz_h;
         g_enc_c0_b[h] += dz_c;

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            g_enc_h0_w[h][j] += dz_h * c_static[j];
            g_enc_c0_w[h][j] += dz_c * c_static[j];
            d_static[j] += m_enc_h0_w[h][j] * dz_h + m_enc_c0_w[h][j] * dz_c;
         }
      }

      // Static context gradients to static projection.
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double dz = d_static[h] * (1.0 - c_static[h] * c_static[h]);
         g_static_b[h] += dz;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            g_static_w[h][i] += dz * (c_xn[n - 1][i] * m_static_mask[i]);
      }

      // VSN backward (full softmax + per-feature projection gradients).
      for(int t=n - 1; t>=0; t--)
      {
         double d_alpha[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) d_alpha[i] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               double d_feat = d_emb_total[t][h] * c_alpha[t][i];
               double zgrad = d_feat * (1.0 - c_feat[t][i][h] * c_feat[t][i][h]);
               g_vsn_proj_w[i][h] += zgrad * c_xn[t][i];
               g_vsn_proj_b[i][h] += zgrad;
               d_alpha[i] += d_emb_total[t][h] * c_feat[t][i][h];
            }
         }

         double sumad = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            sumad += c_alpha[t][i] * d_alpha[i];

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double dlogit = c_alpha[t][i] * (d_alpha[i] - sumad);
            g_vsn_gate_b[i] += dlogit;
            for(int j=0; j<FXAI_AI_WEIGHTS; j++)
               g_vsn_gate_w[i][j] += dlogit * c_xn[t][j];
         }
      }

      // Global gradient norm clip.
      double g2 = 0.0;

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         g2 += g_vsn_gate_b[i] * g_vsn_gate_b[i];
         for(int j=0; j<FXAI_AI_WEIGHTS; j++)
            g2 += g_vsn_gate_w[i][j] * g_vsn_gate_w[i][j];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g2 += g_vsn_proj_w[i][h] * g_vsn_proj_w[i][h];
            g2 += g_vsn_proj_b[i][h] * g_vsn_proj_b[i][h];
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         g2 += g_static_b[h] * g_static_b[h];
         g2 += g_enc_h0_b[h] * g_enc_h0_b[h] + g_enc_c0_b[h] * g_enc_c0_b[h] + g_dec_h0_b[h] * g_dec_h0_b[h] + g_dec_c0_b[h] * g_dec_c0_b[h];
         g2 += g_e_bi[h] * g_e_bi[h] + g_e_bf[h] * g_e_bf[h] + g_e_bo[h] * g_e_bo[h] + g_e_bg[h] * g_e_bg[h];
         g2 += g_d_bi[h] * g_d_bi[h] + g_d_bf[h] * g_d_bf[h] + g_d_bo[h] * g_d_bo[h] + g_d_bg[h] * g_d_bg[h];
         g2 += g_ff1_b[h] * g_ff1_b[h] + g_ff2_b[h] * g_ff2_b[h];
         g2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] + g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            g2 += g_static_w[h][i] * g_static_w[h][i];

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            g2 += g_enc_h0_w[h][j] * g_enc_h0_w[h][j] + g_enc_c0_w[h][j] * g_enc_c0_w[h][j];
            g2 += g_dec_h0_s_w[h][j] * g_dec_h0_s_w[h][j] + g_dec_h0_e_w[h][j] * g_dec_h0_e_w[h][j];
            g2 += g_dec_c0_s_w[h][j] * g_dec_c0_s_w[h][j] + g_dec_c0_e_w[h][j] * g_dec_c0_e_w[h][j];

            g2 += g_e_wi_x[h][j] * g_e_wi_x[h][j] + g_e_wf_x[h][j] * g_e_wf_x[h][j] + g_e_wo_x[h][j] * g_e_wo_x[h][j] + g_e_wg_x[h][j] * g_e_wg_x[h][j];
            g2 += g_e_wi_h[h][j] * g_e_wi_h[h][j] + g_e_wf_h[h][j] * g_e_wf_h[h][j] + g_e_wo_h[h][j] * g_e_wo_h[h][j] + g_e_wg_h[h][j] * g_e_wg_h[h][j];

            g2 += g_d_wi_x[h][j] * g_d_wi_x[h][j] + g_d_wf_x[h][j] * g_d_wf_x[h][j] + g_d_wo_x[h][j] * g_d_wo_x[h][j] + g_d_wg_x[h][j] * g_d_wg_x[h][j];
            g2 += g_d_wi_h[h][j] * g_d_wi_h[h][j] + g_d_wf_h[h][j] * g_d_wf_h[h][j] + g_d_wo_h[h][j] * g_d_wo_h[h][j] + g_d_wg_h[h][j] * g_d_wg_h[h][j];

            g2 += g_ff1_w[h][j] * g_ff1_w[h][j] + g_ff2_w[h][j] * g_ff2_w[h][j];
         }

         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
            g2 += g_w_cls[c][h] * g_w_cls[c][h];
      }

      for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         g2 += g_b_cls[c] * g_b_cls[c];
      g2 += g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;

      for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
      {
         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               g2 += g_wq[hd][d][h] * g_wq[hd][d][h];
               g2 += g_wk[hd][d][h] * g_wk[hd][d][h];
               g2 += g_wv[hd][d][h] * g_wv[hd][d][h];
               g2 += g_wo[hd][h][d] * g_wo[hd][h][d];
            }
         }
      }
      for(int i=0; i<FXAI_TFT_SEQ; i++)
         g2 += g_rel_bias[i] * g_rel_bias[i];

      double gnorm = MathSqrt(g2 + 1e-12);
      double gscale = (gnorm > 5.0 ? (5.0 / gnorm) : 1.0);

      // AdamW update.
      double lr = ScheduledLR(hp);
      double wd = FXAI_Clamp(0.50 * hp.l2, 0.0, 0.05);
      m_adam_t++;
      m_step += n;

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         AdamWStep(m_vsn_gate_b[i], m_m_vsn_gate_b[i], m_v_vsn_gate_b[i], gscale * g_vsn_gate_b[i], lr, 0.0);
         for(int j=0; j<FXAI_AI_WEIGHTS; j++)
            AdamWStep(m_vsn_gate_w[i][j], m_m_vsn_gate_w[i][j], m_v_vsn_gate_w[i][j], gscale * g_vsn_gate_w[i][j], lr, wd);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            AdamWStep(m_vsn_proj_w[i][h], m_m_vsn_proj_w[i][h], m_v_vsn_proj_w[i][h], gscale * g_vsn_proj_w[i][h], lr, wd);
            AdamWStep(m_vsn_proj_b[i][h], m_m_vsn_proj_b[i][h], m_v_vsn_proj_b[i][h], gscale * g_vsn_proj_b[i][h], lr, 0.0);
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         AdamWStep(m_static_b[h], m_m_static_b[h], m_v_static_b[h], gscale * g_static_b[h], lr, 0.0);

         AdamWStep(m_enc_h0_b[h], m_m_enc_h0_b[h], m_v_enc_h0_b[h], gscale * g_enc_h0_b[h], lr, 0.0);
         AdamWStep(m_enc_c0_b[h], m_m_enc_c0_b[h], m_v_enc_c0_b[h], gscale * g_enc_c0_b[h], lr, 0.0);
         AdamWStep(m_dec_h0_b[h], m_m_dec_h0_b[h], m_v_dec_h0_b[h], gscale * g_dec_h0_b[h], lr, 0.0);
         AdamWStep(m_dec_c0_b[h], m_m_dec_c0_b[h], m_v_dec_c0_b[h], gscale * g_dec_c0_b[h], lr, 0.0);

         AdamWStep(m_e_bi[h], m_m_e_bi[h], m_v_e_bi[h], gscale * g_e_bi[h], lr, 0.0);
         AdamWStep(m_e_bf[h], m_m_e_bf[h], m_v_e_bf[h], gscale * g_e_bf[h], lr, 0.0);
         AdamWStep(m_e_bo[h], m_m_e_bo[h], m_v_e_bo[h], gscale * g_e_bo[h], lr, 0.0);
         AdamWStep(m_e_bg[h], m_m_e_bg[h], m_v_e_bg[h], gscale * g_e_bg[h], lr, 0.0);

         AdamWStep(m_d_bi[h], m_m_d_bi[h], m_v_d_bi[h], gscale * g_d_bi[h], lr, 0.0);
         AdamWStep(m_d_bf[h], m_m_d_bf[h], m_v_d_bf[h], gscale * g_d_bf[h], lr, 0.0);
         AdamWStep(m_d_bo[h], m_m_d_bo[h], m_v_d_bo[h], gscale * g_d_bo[h], lr, 0.0);
         AdamWStep(m_d_bg[h], m_m_d_bg[h], m_v_d_bg[h], gscale * g_d_bg[h], lr, 0.0);

         AdamWStep(m_ff1_b[h], m_m_ff1_b[h], m_v_ff1_b[h], gscale * g_ff1_b[h], lr, 0.0);
         AdamWStep(m_ff2_b[h], m_m_ff2_b[h], m_v_ff2_b[h], gscale * g_ff2_b[h], lr, 0.0);

         AdamWStep(m_w_mu[h], m_m_w_mu[h], m_v_w_mu[h], gscale * g_w_mu[h], lr, wd);
         AdamWStep(m_w_logv[h], m_m_w_logv[h], m_v_w_logv[h], gscale * g_w_logv[h], lr, wd);
         AdamWStep(m_w_q25[h], m_m_w_q25[h], m_v_w_q25[h], gscale * g_w_q25[h], lr, wd);
         AdamWStep(m_w_q75[h], m_m_w_q75[h], m_v_w_q75[h], gscale * g_w_q75[h], lr, wd);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            AdamWStep(m_static_w[h][i], m_m_static_w[h][i], m_v_static_w[h][i], gscale * g_static_w[h][i], lr, wd);

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            AdamWStep(m_enc_h0_w[h][j], m_m_enc_h0_w[h][j], m_v_enc_h0_w[h][j], gscale * g_enc_h0_w[h][j], lr, wd);
            AdamWStep(m_enc_c0_w[h][j], m_m_enc_c0_w[h][j], m_v_enc_c0_w[h][j], gscale * g_enc_c0_w[h][j], lr, wd);
            AdamWStep(m_dec_h0_s_w[h][j], m_m_dec_h0_s_w[h][j], m_v_dec_h0_s_w[h][j], gscale * g_dec_h0_s_w[h][j], lr, wd);
            AdamWStep(m_dec_h0_e_w[h][j], m_m_dec_h0_e_w[h][j], m_v_dec_h0_e_w[h][j], gscale * g_dec_h0_e_w[h][j], lr, wd);
            AdamWStep(m_dec_c0_s_w[h][j], m_m_dec_c0_s_w[h][j], m_v_dec_c0_s_w[h][j], gscale * g_dec_c0_s_w[h][j], lr, wd);
            AdamWStep(m_dec_c0_e_w[h][j], m_m_dec_c0_e_w[h][j], m_v_dec_c0_e_w[h][j], gscale * g_dec_c0_e_w[h][j], lr, wd);

            AdamWStep(m_e_wi_x[h][j], m_m_e_wi_x[h][j], m_v_e_wi_x[h][j], gscale * g_e_wi_x[h][j], lr, wd);
            AdamWStep(m_e_wf_x[h][j], m_m_e_wf_x[h][j], m_v_e_wf_x[h][j], gscale * g_e_wf_x[h][j], lr, wd);
            AdamWStep(m_e_wo_x[h][j], m_m_e_wo_x[h][j], m_v_e_wo_x[h][j], gscale * g_e_wo_x[h][j], lr, wd);
            AdamWStep(m_e_wg_x[h][j], m_m_e_wg_x[h][j], m_v_e_wg_x[h][j], gscale * g_e_wg_x[h][j], lr, wd);
            AdamWStep(m_e_wi_h[h][j], m_m_e_wi_h[h][j], m_v_e_wi_h[h][j], gscale * g_e_wi_h[h][j], lr, wd);
            AdamWStep(m_e_wf_h[h][j], m_m_e_wf_h[h][j], m_v_e_wf_h[h][j], gscale * g_e_wf_h[h][j], lr, wd);
            AdamWStep(m_e_wo_h[h][j], m_m_e_wo_h[h][j], m_v_e_wo_h[h][j], gscale * g_e_wo_h[h][j], lr, wd);
            AdamWStep(m_e_wg_h[h][j], m_m_e_wg_h[h][j], m_v_e_wg_h[h][j], gscale * g_e_wg_h[h][j], lr, wd);

            AdamWStep(m_d_wi_x[h][j], m_m_d_wi_x[h][j], m_v_d_wi_x[h][j], gscale * g_d_wi_x[h][j], lr, wd);
            AdamWStep(m_d_wf_x[h][j], m_m_d_wf_x[h][j], m_v_d_wf_x[h][j], gscale * g_d_wf_x[h][j], lr, wd);
            AdamWStep(m_d_wo_x[h][j], m_m_d_wo_x[h][j], m_v_d_wo_x[h][j], gscale * g_d_wo_x[h][j], lr, wd);
            AdamWStep(m_d_wg_x[h][j], m_m_d_wg_x[h][j], m_v_d_wg_x[h][j], gscale * g_d_wg_x[h][j], lr, wd);
            AdamWStep(m_d_wi_h[h][j], m_m_d_wi_h[h][j], m_v_d_wi_h[h][j], gscale * g_d_wi_h[h][j], lr, wd);
            AdamWStep(m_d_wf_h[h][j], m_m_d_wf_h[h][j], m_v_d_wf_h[h][j], gscale * g_d_wf_h[h][j], lr, wd);
            AdamWStep(m_d_wo_h[h][j], m_m_d_wo_h[h][j], m_v_d_wo_h[h][j], gscale * g_d_wo_h[h][j], lr, wd);
            AdamWStep(m_d_wg_h[h][j], m_m_d_wg_h[h][j], m_v_d_wg_h[h][j], gscale * g_d_wg_h[h][j], lr, wd);

            AdamWStep(m_ff1_w[h][j], m_m_ff1_w[h][j], m_v_ff1_w[h][j], gscale * g_ff1_w[h][j], lr, wd);
            AdamWStep(m_ff2_w[h][j], m_m_ff2_w[h][j], m_v_ff2_w[h][j], gscale * g_ff2_w[h][j], lr, wd);
         }

         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
            AdamWStep(m_w_cls[c][h], m_m_w_cls[c][h], m_v_w_cls[c][h], gscale * g_w_cls[c][h], lr, wd);
      }

      for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         AdamWStep(m_b_cls[c], m_m_b_cls[c], m_v_b_cls[c], gscale * g_b_cls[c], lr, 0.0);
      AdamWStep(m_b_mu, m_m_b_mu, m_v_b_mu, gscale * g_b_mu, lr, 0.0);
      AdamWStep(m_b_logv, m_m_b_logv, m_v_b_logv, gscale * g_b_logv, lr, 0.0);
      AdamWStep(m_b_q25, m_m_b_q25, m_v_b_q25, gscale * g_b_q25, lr, 0.0);
      AdamWStep(m_b_q75, m_m_b_q75, m_v_b_q75, gscale * g_b_q75, lr, 0.0);

      for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
      {
         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               AdamWStep(m_wq[hd][d][h], m_m_wq[hd][d][h], m_v_wq[hd][d][h], gscale * g_wq[hd][d][h], lr, wd);
               AdamWStep(m_wk[hd][d][h], m_m_wk[hd][d][h], m_v_wk[hd][d][h], gscale * g_wk[hd][d][h], lr, wd);
               AdamWStep(m_wv[hd][d][h], m_m_wv[hd][d][h], m_v_wv[hd][d][h], gscale * g_wv[hd][d][h], lr, wd);
               AdamWStep(m_wo[hd][h][d], m_m_wo[hd][h][d], m_v_wo[hd][h][d], gscale * g_wo[hd][h][d], lr, wd);
            }
         }
      }
      for(int i=0; i<FXAI_TFT_SEQ; i++)
         AdamWStep(m_rel_bias[i], m_m_rel_bias[i], m_v_rel_bias[i], gscale * g_rel_bias[i], lr, 0.0);

      m_b_logv = FXAI_Clamp(m_b_logv, -4.0, 4.0);
      if(m_b_q75 < m_b_q25 + 1e-4) m_b_q75 = m_b_q25 + 1e-4;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_w_mu[h] = FXAI_ClipSym(m_w_mu[h], 8.0);
         m_w_logv[h] = FXAI_ClipSym(m_w_logv[h], 8.0);
         m_w_q25[h] = FXAI_ClipSym(m_w_q25[h], 8.0);
         m_w_q75[h] = FXAI_ClipSym(m_w_q75[h], 8.0);
      }

      SyncShadow(false);
      if(!m_shadow_ready && m_adam_t >= 64)
         m_shadow_ready = true;

      if((m_step % 32) == 0)
         OptimizeThresholds();
   }

   void ForwardInference(const double &x[],
                         const bool use_shadow,
                         double &p_sell,
                         double &p_buy,
                         double &p_skip,
                         double &mu,
                         double &logv,
                         double &q25,
                         double &q75)
   {
      double seq[FXAI_TFT_SEQ][FXAI_AI_WEIGHTS];
      int n = 0;
      BuildInferenceSequence(x, seq, n);
      if(n < 1) n = 1;

      double xn[FXAI_TFT_SEQ][FXAI_AI_WEIGHTS];
      double emb[FXAI_TFT_SEQ][FXAI_AI_MLP_HIDDEN];
      for(int t=0; t<n; t++)
      {
         double seq_t[FXAI_AI_WEIGHTS];
         double xn_t[FXAI_AI_WEIGHTS];
         double emb_t[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            seq_t[i] = seq[t][i];

         Normalize(seq_t, xn_t);
         VSNForward(xn_t, use_shadow, false, -1, emb_t);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            xn[t][i] = xn_t[i];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            emb[t][h] = emb_t[h];
      }

      double sctx[FXAI_AI_MLP_HIDDEN];
      double xn_last[FXAI_AI_WEIGHTS];
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         xn_last[i] = xn[n - 1][i];
      StaticContext(xn_last, use_shadow, sctx);

      double eh[FXAI_AI_MLP_HIDDEN], ec[FXAI_AI_MLP_HIDDEN];
      InitEncoderState(sctx, use_shadow, eh, ec);

      double enc_h[FXAI_TFT_SEQ][FXAI_AI_MLP_HIDDEN];
      for(int t=0; t<n; t++)
      {
         double ig[FXAI_AI_MLP_HIDDEN], fg[FXAI_AI_MLP_HIDDEN], og[FXAI_AI_MLP_HIDDEN], gg[FXAI_AI_MLP_HIDDEN];
         double cnew[FXAI_AI_MLP_HIDDEN], hnew[FXAI_AI_MLP_HIDDEN];
         double emb_t[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            emb_t[h] = emb[t][h];
         EncoderStep(emb_t, eh, ec, use_shadow, ig, fg, og, gg, cnew, hnew);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            eh[h] = hnew[h];
            ec[h] = cnew[h];
            enc_h[t][h] = hnew[h];
         }
      }

      double dh[FXAI_AI_MLP_HIDDEN], dc[FXAI_AI_MLP_HIDDEN];
      double enc_last[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         enc_last[h] = enc_h[n - 1][h];
      InitDecoderState(sctx, enc_last, use_shadow, dh, dc);

      for(int t=0; t<n; t++)
      {
         double ig[FXAI_AI_MLP_HIDDEN], fg[FXAI_AI_MLP_HIDDEN], og[FXAI_AI_MLP_HIDDEN], gg[FXAI_AI_MLP_HIDDEN];
         double cnew[FXAI_AI_MLP_HIDDEN], hnew[FXAI_AI_MLP_HIDDEN];
         double emb_t[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            emb_t[h] = emb[t][h];
         DecoderStep(emb_t, dh, dc, use_shadow, ig, fg, og, gg, cnew, hnew);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            dh[h] = hnew[h];
            dc[h] = cnew[h];
         }

         double attn[FXAI_AI_MLP_HIDDEN];
         AttentionStep(t, n, use_shadow, dh, enc_h, false, attn);

         double pre[FXAI_AI_MLP_HIDDEN], fin[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            pre[h] = FXAI_ClipSym(dh[h] + attn[h], 8.0);
         FFNStep(-1, use_shadow, false, pre, fin);

         double dump;
         HeadsStep(-1, use_shadow, fin, dump, mu, logv, q25, q75, p_sell, p_buy, p_skip);
      }
   }

