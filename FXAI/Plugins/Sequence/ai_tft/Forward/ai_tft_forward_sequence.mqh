   void BuildInferenceSequence(const double &x_current[],
                               double &seq[][FXAI_AI_WEIGHTS],
                               int &n) const
   {
      int keep = m_hist_len;
      if(keep > FXAI_TFT_SEQ - 1) keep = FXAI_TFT_SEQ - 1;

      for(int i=0; i<keep; i++)
      {
         int back = keep - 1 - i;
         int idx = HistIndexBack(back);
         if(idx < 0) continue;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            seq[i][k] = m_hist_x[idx][k];
      }

      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         seq[keep][k] = x_current[k];
      n = keep + 1;
      if(n < 1) n = 1;
   }

   void VSNForward(const double &xn[],
                   const bool use_shadow,
                   const bool training,
                   const int cache_t,
                   double &emb[])
   {
      double logits[FXAI_AI_WEIGHTS];
      double alpha[FXAI_AI_WEIGHTS];
      double feat[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double s = (use_shadow ? m_s_vsn_gate_b[i] : m_vsn_gate_b[i]);
         for(int j=0; j<FXAI_AI_WEIGHTS; j++)
            s += (use_shadow ? m_s_vsn_gate_w[i][j] : m_vsn_gate_w[i][j]) * xn[j];
         logits[i] = FXAI_ClipSym(s, 20.0);
      }
      SoftmaxN(logits, FXAI_AI_WEIGHTS, alpha);

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double z = (use_shadow ? m_s_vsn_proj_b[i][h] : m_vsn_proj_b[i][h]);
            z += (use_shadow ? m_s_vsn_proj_w[i][h] : m_vsn_proj_w[i][h]) * xn[i];
            feat[i][h] = FXAI_Tanh(FXAI_ClipSym(z, 8.0));
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double s = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            s += alpha[i] * feat[i][h];
         emb[h] = s;
      }

      if(training)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            emb[h] *= DropMask(h, 111 + cache_t, 0.05, true);
      }

      if(cache_t >= 0 && cache_t < FXAI_TFT_TBPTT)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            c_alpha[cache_t][i] = alpha[i];
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               c_feat[cache_t][i][h] = feat[i][h];
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            c_emb[cache_t][h] = emb[h];
      }
   }

   void StaticContext(const double &xn_last[],
                      const bool use_shadow,
                      double &s[]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double z = (use_shadow ? m_s_static_b[h] : m_static_b[h]);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            z += (use_shadow ? m_s_static_w[h][i] : m_static_w[h][i]) * (xn_last[i] * m_static_mask[i]);
         s[h] = FXAI_Tanh(FXAI_ClipSym(z, 8.0));
      }
   }

   void InitEncoderState(const double &s[],
                         const bool use_shadow,
                         double &h0[],
                         double &c0[]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double zh = (use_shadow ? m_s_enc_h0_b[h] : m_enc_h0_b[h]);
         double zc = (use_shadow ? m_s_enc_c0_b[h] : m_enc_c0_b[h]);
         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            zh += (use_shadow ? m_s_enc_h0_w[h][j] : m_enc_h0_w[h][j]) * s[j];
            zc += (use_shadow ? m_s_enc_c0_w[h][j] : m_enc_c0_w[h][j]) * s[j];
         }
         h0[h] = FXAI_Tanh(FXAI_ClipSym(zh, 8.0));
         c0[h] = FXAI_Tanh(FXAI_ClipSym(zc, 8.0));
      }
   }

   void InitDecoderState(const double &s[],
                         const double &enc_last[],
                         const bool use_shadow,
                         double &h0[],
                         double &c0[]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double zh = (use_shadow ? m_s_dec_h0_b[h] : m_dec_h0_b[h]);
         double zc = (use_shadow ? m_s_dec_c0_b[h] : m_dec_c0_b[h]);
         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            zh += (use_shadow ? m_s_dec_h0_s_w[h][j] : m_dec_h0_s_w[h][j]) * s[j];
            zh += (use_shadow ? m_s_dec_h0_e_w[h][j] : m_dec_h0_e_w[h][j]) * enc_last[j];

            zc += (use_shadow ? m_s_dec_c0_s_w[h][j] : m_dec_c0_s_w[h][j]) * s[j];
            zc += (use_shadow ? m_s_dec_c0_e_w[h][j] : m_dec_c0_e_w[h][j]) * enc_last[j];
         }
         h0[h] = FXAI_Tanh(FXAI_ClipSym(zh, 8.0));
         c0[h] = FXAI_Tanh(FXAI_ClipSym(zc, 8.0));
      }
   }

   void EncoderStep(const double &x_in[],
                    const double &h_prev[],
                    const double &c_prev[],
                    const bool use_shadow,
                    double &ig[],
                    double &fg[],
                    double &og[],
                    double &gg[],
                    double &c_new[],
                    double &h_new[]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double zi = (use_shadow ? m_s_e_bi[h] : m_e_bi[h]);
         double zf = (use_shadow ? m_s_e_bf[h] : m_e_bf[h]);
         double zo = (use_shadow ? m_s_e_bo[h] : m_e_bo[h]);
         double zg = (use_shadow ? m_s_e_bg[h] : m_e_bg[h]);

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            zi += (use_shadow ? m_s_e_wi_x[h][i] : m_e_wi_x[h][i]) * x_in[i];
            zf += (use_shadow ? m_s_e_wf_x[h][i] : m_e_wf_x[h][i]) * x_in[i];
            zo += (use_shadow ? m_s_e_wo_x[h][i] : m_e_wo_x[h][i]) * x_in[i];
            zg += (use_shadow ? m_s_e_wg_x[h][i] : m_e_wg_x[h][i]) * x_in[i];

            zi += (use_shadow ? m_s_e_wi_h[h][i] : m_e_wi_h[h][i]) * h_prev[i];
            zf += (use_shadow ? m_s_e_wf_h[h][i] : m_e_wf_h[h][i]) * h_prev[i];
            zo += (use_shadow ? m_s_e_wo_h[h][i] : m_e_wo_h[h][i]) * h_prev[i];
            zg += (use_shadow ? m_s_e_wg_h[h][i] : m_e_wg_h[h][i]) * h_prev[i];
         }

         ig[h] = FXAI_Sigmoid(FXAI_ClipSym(zi, 20.0));
         fg[h] = FXAI_Sigmoid(FXAI_ClipSym(zf, 20.0));
         og[h] = FXAI_Sigmoid(FXAI_ClipSym(zo, 20.0));
         gg[h] = FXAI_Tanh(FXAI_ClipSym(zg, 8.0));

         c_new[h] = fg[h] * c_prev[h] + ig[h] * gg[h];
         h_new[h] = og[h] * FXAI_Tanh(c_new[h]);
      }
   }

   void DecoderStep(const double &x_in[],
                    const double &h_prev[],
                    const double &c_prev[],
                    const bool use_shadow,
                    double &ig[],
                    double &fg[],
                    double &og[],
                    double &gg[],
                    double &c_new[],
                    double &h_new[]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double zi = (use_shadow ? m_s_d_bi[h] : m_d_bi[h]);
         double zf = (use_shadow ? m_s_d_bf[h] : m_d_bf[h]);
         double zo = (use_shadow ? m_s_d_bo[h] : m_d_bo[h]);
         double zg = (use_shadow ? m_s_d_bg[h] : m_d_bg[h]);

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            zi += (use_shadow ? m_s_d_wi_x[h][i] : m_d_wi_x[h][i]) * x_in[i];
            zf += (use_shadow ? m_s_d_wf_x[h][i] : m_d_wf_x[h][i]) * x_in[i];
            zo += (use_shadow ? m_s_d_wo_x[h][i] : m_d_wo_x[h][i]) * x_in[i];
            zg += (use_shadow ? m_s_d_wg_x[h][i] : m_d_wg_x[h][i]) * x_in[i];

            zi += (use_shadow ? m_s_d_wi_h[h][i] : m_d_wi_h[h][i]) * h_prev[i];
            zf += (use_shadow ? m_s_d_wf_h[h][i] : m_d_wf_h[h][i]) * h_prev[i];
            zo += (use_shadow ? m_s_d_wo_h[h][i] : m_d_wo_h[h][i]) * h_prev[i];
            zg += (use_shadow ? m_s_d_wg_h[h][i] : m_d_wg_h[h][i]) * h_prev[i];
         }

         ig[h] = FXAI_Sigmoid(FXAI_ClipSym(zi, 20.0));
         fg[h] = FXAI_Sigmoid(FXAI_ClipSym(zf, 20.0));
         og[h] = FXAI_Sigmoid(FXAI_ClipSym(zo, 20.0));
         gg[h] = FXAI_Tanh(FXAI_ClipSym(zg, 8.0));

         c_new[h] = fg[h] * c_prev[h] + ig[h] * gg[h];
         h_new[h] = og[h] * FXAI_Tanh(c_new[h]);
      }
   }

   void AttentionStep(const int t,
                      const int n,
                      const bool use_shadow,
                      const double &dec_h[],
                      const double &enc_h[][FXAI_AI_MLP_HIDDEN],
                      const bool cache,
                      double &attn_out[])
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         attn_out[h] = 0.0;

      double inv_scale = 1.0 / MathSqrt((double)FXAI_TFT_D_HEAD);

      int nn = n;
      if(nn < 1) nn = 1;
      if(nn > FXAI_TFT_SEQ) nn = FXAI_TFT_SEQ;
      if(cache)
      {
         if(t < 0 || t >= FXAI_TFT_TBPTT) return;
         if(nn > FXAI_TFT_TBPTT) nn = FXAI_TFT_TBPTT;
      }


      for(int hd=0; hd<FXAI_TFT_HEADS; hd++)
      {
         double q[FXAI_TFT_D_HEAD];
         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
         {
            double s = 0.0;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               s += (use_shadow ? m_s_wq[hd][d][h] : m_wq[hd][d][h]) * dec_h[h];
            q[d] = s;
            if(cache) c_attn_q[t][hd][d] = s;
         }

         double logits[FXAI_TFT_SEQ];
         double den = 0.0;
         double maxv = -1e100;

         for(int j=0; j<=t && j<nn; j++)
         {
            double score = 0.0;
            for(int d=0; d<FXAI_TFT_D_HEAD; d++)
            {
               double k = 0.0;
               double v = 0.0;
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               {
                  k += (use_shadow ? m_s_wk[hd][d][h] : m_wk[hd][d][h]) * enc_h[j][h];
                  v += (use_shadow ? m_s_wv[hd][d][h] : m_wv[hd][d][h]) * enc_h[j][h];
               }
               if(cache)
               {
                  c_attn_k[t][hd][j][d] = k;
                  c_attn_v[t][hd][j][d] = v;
               }
               score += q[d] * k;
            }

            int lag = t - j;
            if(lag < 0) lag = 0;
            if(lag >= FXAI_TFT_SEQ) lag = FXAI_TFT_SEQ - 1;
            score = score * inv_scale + (use_shadow ? m_s_rel_bias[lag] : m_rel_bias[lag]);
            logits[j] = score;
            if(score > maxv) maxv = score;
         }

         for(int j=0; j<=t && j<nn; j++)
         {
            logits[j] = MathExp(FXAI_ClipSym(logits[j] - maxv, 30.0));
            den += logits[j];
         }
         if(den <= 0.0) den = 1.0;

         double ctx[FXAI_TFT_D_HEAD];
         for(int d=0; d<FXAI_TFT_D_HEAD; d++) ctx[d] = 0.0;

         for(int j=0; j<=t && j<nn; j++)
         {
            double a = logits[j] / den;
            if(cache) c_attn_w[t][hd][j] = a;
            for(int d=0; d<FXAI_TFT_D_HEAD; d++)
               ctx[d] += a * (cache ? c_attn_v[t][hd][j][d] : 0.0);
         }

      if(!cache)
      {
         // If no cache requested, recompute context values directly.
         for(int d=0; d<FXAI_TFT_D_HEAD; d++) ctx[d] = 0.0;
         for(int j=0; j<=t && j<nn; j++)
         {
            double a = logits[j] / den;
            for(int d=0; d<FXAI_TFT_D_HEAD; d++)
            {
               double vv = 0.0;
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
                  vv += (use_shadow ? m_s_wv[hd][d][h] : m_wv[hd][d][h]) * enc_h[j][h];
               ctx[d] += a * vv;
            }
         }
      }

         for(int d=0; d<FXAI_TFT_D_HEAD; d++)
            if(cache) c_attn_ctx[t][hd][d] = ctx[d];

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double s = 0.0;
            for(int d=0; d<FXAI_TFT_D_HEAD; d++)
               s += (use_shadow ? m_s_wo[hd][h][d] : m_wo[hd][h][d]) * ctx[d];
            attn_out[h] += s;
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         attn_out[h] = FXAI_ClipSym(attn_out[h], 8.0);

      if(cache)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            c_attn_out[t][h] = attn_out[h];
      }
   }

   void FFNStep(const int t,
                const bool use_shadow,
                const bool training,
                const double &pre[],
                double &out[])
   {
      double ff1_raw[FXAI_AI_MLP_HIDDEN];
      double ff1[FXAI_AI_MLP_HIDDEN];

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double z = (use_shadow ? m_s_ff1_b[h] : m_ff1_b[h]);
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            z += (use_shadow ? m_s_ff1_w[h][i] : m_ff1_w[h][i]) * pre[i];

         ff1_raw[h] = FXAI_Tanh(FXAI_ClipSym(z, 8.0));
         double m = DropMask(h, 190 + t, 0.10, training);
         ff1[h] = ff1_raw[h] * m;
      }

      double ff2[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double z = (use_shadow ? m_s_ff2_b[h] : m_ff2_b[h]);
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            z += (use_shadow ? m_s_ff2_w[h][i] : m_ff2_w[h][i]) * ff1[i];
         ff2[h] = FXAI_ClipSym(z, 8.0);
      }

      double sc = StochScale(t, training);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         out[h] = FXAI_ClipSym(pre[h] + sc * ff2[h], 8.0);

      if(t >= 0 && t < FXAI_TFT_TBPTT)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            c_pre[t][h] = pre[h];
            c_ff1_raw[t][h] = ff1_raw[h];
            c_ff1[t][h] = ff1[h];
            c_ff1_mask[t][h] = (MathAbs(ff1_raw[h]) > 1e-9 ? ff1[h] / ff1_raw[h] : 0.0);
            c_ff2[t][h] = ff2[h];
            c_final[t][h] = out[h];
         }
         c_stoch_scale[t] = sc;
      }
   }

   void HeadsStep(const int t,
                  const bool use_shadow,
                  const double &state[],
                  double &probs,
                  double &mu,
                  double &logv,
                  double &q25,
                  double &q75,
                  double &p_sell,
                  double &p_buy,
                  double &p_skip)
   {
      double logits[FXAI_TFT_CLASS_COUNT];
      double probs3[FXAI_TFT_CLASS_COUNT];

      for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
      {
         double z = (use_shadow ? m_s_b_cls[c] : m_b_cls[c]);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            z += (use_shadow ? m_s_w_cls[c][h] : m_w_cls[c][h]) * state[h];
         logits[c] = FXAI_ClipSym(z, 20.0);
      }

      SoftmaxN(logits, FXAI_TFT_CLASS_COUNT, probs3);

      mu = (use_shadow ? m_s_b_mu : m_b_mu);
      logv = (use_shadow ? m_s_b_logv : m_b_logv);
      q25 = (use_shadow ? m_s_b_q25 : m_b_q25);
      q75 = (use_shadow ? m_s_b_q75 : m_b_q75);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         mu += (use_shadow ? m_s_w_mu[h] : m_w_mu[h]) * state[h];
         logv += (use_shadow ? m_s_w_logv[h] : m_w_logv[h]) * state[h];
         q25 += (use_shadow ? m_s_w_q25[h] : m_w_q25[h]) * state[h];
         q75 += (use_shadow ? m_s_w_q75[h] : m_w_q75[h]) * state[h];
      }

      logv = FXAI_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double tmp = q25;
         q25 = q75;
         q75 = tmp;
      }

      p_sell = probs3[FXAI_TFT_SELL];
      p_buy = probs3[FXAI_TFT_BUY];
      p_skip = probs3[FXAI_TFT_SKIP];
      probs = p_buy;

      if(t >= 0 && t < FXAI_TFT_TBPTT)
      {
         for(int c=0; c<FXAI_TFT_CLASS_COUNT; c++)
         {
            c_logits[t][c] = logits[c];
            c_probs[t][c] = probs3[c];
         }
         c_mu[t] = mu;
         c_logv[t] = logv;
         c_q25[t] = q25;
         c_q75[t] = q75;
      }
   }

   void AppendTrainSample(const int cls,
                          const double &x[],
                          const double move_points,
                          const double cost_points,
                          const double sample_w)
   {
      if(m_train_len < FXAI_TFT_TBPTT)
      {
         int p = m_train_len;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_train_x[p][i] = x[i];
         m_train_cls[p] = cls;
         m_train_move[p] = move_points;
         m_train_cost[p] = cost_points;
         m_train_w[p] = sample_w;
         m_train_len++;
         return;
      }

      for(int t=1; t<FXAI_TFT_TBPTT; t++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_train_x[t - 1][i] = m_train_x[t][i];
         m_train_cls[t - 1] = m_train_cls[t];
         m_train_move[t - 1] = m_train_move[t];
         m_train_cost[t - 1] = m_train_cost[t];
         m_train_w[t - 1] = m_train_w[t];
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_train_x[FXAI_TFT_TBPTT - 1][i] = x[i];
      m_train_cls[FXAI_TFT_TBPTT - 1] = cls;
      m_train_move[FXAI_TFT_TBPTT - 1] = move_points;
      m_train_cost[FXAI_TFT_TBPTT - 1] = cost_points;
      m_train_w[FXAI_TFT_TBPTT - 1] = sample_w;
   }

   void ForwardTrainSequence(const int n)
   {
      double sctx[FXAI_AI_MLP_HIDDEN];
      double eh[FXAI_AI_MLP_HIDDEN], ec[FXAI_AI_MLP_HIDDEN];
      double dh[FXAI_AI_MLP_HIDDEN], dc[FXAI_AI_MLP_HIDDEN];

      for(int t=0; t<n; t++)
      {
         double xin[FXAI_AI_WEIGHTS];
         double xout[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            xin[i] = m_train_x[t][i];
         Normalize(xin, xout);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            c_xn[t][i] = xout[i];
      }

      double xn_last[FXAI_AI_WEIGHTS];
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         xn_last[i] = c_xn[n - 1][i];
      StaticContext(xn_last, false, sctx);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         c_static[h] = sctx[h];

      InitEncoderState(sctx, false, eh, ec);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         c_enc_h0[h] = eh[h];
         c_enc_c0[h] = ec[h];
      }

      // Encoder unroll.
      for(int t=0; t<n; t++)
      {
         double emb[FXAI_AI_MLP_HIDDEN];
         double ig[FXAI_AI_MLP_HIDDEN], fg[FXAI_AI_MLP_HIDDEN], og[FXAI_AI_MLP_HIDDEN], gg[FXAI_AI_MLP_HIDDEN];
         double cnew[FXAI_AI_MLP_HIDDEN], hnew[FXAI_AI_MLP_HIDDEN];

         double xn_t[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            xn_t[i] = c_xn[t][i];
         VSNForward(xn_t, false, true, t, emb);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            c_e_h_prev[t][h] = eh[h];
            c_e_c_prev[t][h] = ec[h];
         }

         EncoderStep(emb, eh, ec, false, ig, fg, og, gg, cnew, hnew);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            c_e_i[t][h] = ig[h];
            c_e_f[t][h] = fg[h];
            c_e_o[t][h] = og[h];
            c_e_g[t][h] = gg[h];
            c_e_c[t][h] = cnew[h];
            c_e_h[t][h] = hnew[h];
            eh[h] = hnew[h];
            ec[h] = cnew[h];
         }
      }

      double enc_last[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         enc_last[h] = c_e_h[n - 1][h];
      InitDecoderState(sctx, enc_last, false, dh, dc);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         c_dec_h0[h] = dh[h];
         c_dec_c0[h] = dc[h];
      }

      // Decoder + attention + FFN + heads.
      for(int t=0; t<n; t++)
      {
         double ig[FXAI_AI_MLP_HIDDEN], fg[FXAI_AI_MLP_HIDDEN], og[FXAI_AI_MLP_HIDDEN], gg[FXAI_AI_MLP_HIDDEN];
         double cnew[FXAI_AI_MLP_HIDDEN], hnew[FXAI_AI_MLP_HIDDEN];

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            c_d_h_prev[t][h] = dh[h];
            c_d_c_prev[t][h] = dc[h];
         }

         double emb_t[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            emb_t[h] = c_emb[t][h];
         DecoderStep(emb_t, dh, dc, false, ig, fg, og, gg, cnew, hnew);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            c_d_i[t][h] = ig[h];
            c_d_f[t][h] = fg[h];
            c_d_o[t][h] = og[h];
            c_d_g[t][h] = gg[h];
            c_d_c[t][h] = cnew[h];
            c_d_h[t][h] = hnew[h];
            dh[h] = hnew[h];
            dc[h] = cnew[h];
         }

         double attn[FXAI_AI_MLP_HIDDEN];
         double dec_h_t[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            dec_h_t[h] = c_d_h[t][h];
         AttentionStep(t, n, false, dec_h_t, c_e_h, true, attn);

         double pre[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            pre[h] = FXAI_ClipSym(c_d_h[t][h] + attn[h], 8.0);

         double finalv[FXAI_AI_MLP_HIDDEN];
         FFNStep(t, false, true, pre, finalv);

         double dump, mu, logv, q25, q75, ps, pb, pk;
         HeadsStep(t, false, finalv, dump, mu, logv, q25, q75, ps, pb, pk);
      }
   }

   void UpdateSessionCalibration(const int sess,
                                 const double p_raw,
                                 const int y,
                                 const double sample_w)
   {
      int s = sess;
      if(s < 0) s = 0;
      if(s >= FXAI_TFT_SESSIONS) s = FXAI_TFT_SESSIONS - 1;

      double pr = FXAI_Clamp(p_raw, 0.001, 0.999);
      double z = FXAI_Logit(pr);
      double py = FXAI_Sigmoid(m_sess_a[s] * z + m_sess_b[s]);
      double e = (double)y - py;

      double lr = FXAI_Clamp(0.010 * sample_w, 0.001, 0.030);
      m_sess_a[s] += lr * (e * z - 0.001 * (m_sess_a[s] - 1.0));
      m_sess_b[s] += lr * e;
      m_sess_a[s] = FXAI_Clamp(m_sess_a[s], 0.20, 5.00);
      m_sess_b[s] = FXAI_Clamp(m_sess_b[s], -4.0, 4.0);

      m_sess_steps[s]++;
      if(m_sess_steps[s] >= 20)
         m_sess_ready[s] = true;
   }

   double ApplySessionCalibration(const int sess,
                                  const double p_raw) const
   {
      int s = sess;
      if(s < 0) s = 0;
      if(s >= FXAI_TFT_SESSIONS) s = FXAI_TFT_SESSIONS - 1;

      double p = FXAI_Clamp(p_raw, 0.001, 0.999);
      double z = FXAI_Logit(p);
      double pc = FXAI_Sigmoid(m_sess_a[s] * z + m_sess_b[s]);
      return FXAI_Clamp(pc, 0.001, 0.999);
   }

   void AddWalkForwardSample(const double p_up,
                             const double p_skip,
                             const int cls,
                             const double move_points,
                             const double cost_points,
                             const int sess)
   {
      int p = m_wf_ptr;
      m_wf_pup[p] = FXAI_Clamp(p_up, 0.001, 0.999);
      m_wf_pskip[p] = FXAI_Clamp(p_skip, 0.001, 0.999);
      m_wf_cls[p] = cls;
      m_wf_move[p] = move_points;
      m_wf_cost[p] = cost_points;
      m_wf_sess[p] = sess;

      m_wf_ptr++;
      if(m_wf_ptr >= FXAI_TFT_WF) m_wf_ptr = 0;
      if(m_wf_len < FXAI_TFT_WF) m_wf_len++;
   }

   double EvalDecisionUtility(const int pred,
                              const int cls,
                              const double move_points,
                              const double cost_points) const
   {
      double cost = MathMax(cost_points, 0.0);
      double edge = MathMax(0.0, MathAbs(move_points) - cost);

      if(pred == FXAI_TFT_SKIP)
      {
         if(cls == FXAI_TFT_SKIP) return 0.10;
         return 0.00;
      }

      if(pred == cls)
         return edge;

      if(cls == FXAI_TFT_SKIP)
         return -0.35 * cost;

      return -(edge + 0.5 * cost);
   }

   void OptimizeThresholds(void)
   {
      if(m_wf_len < 48) return;

      double best_score = -1e100;
      double best_buy = m_thr_buy;
      double best_sell = m_thr_sell;
      double best_skip = m_thr_skip;

      for(double th_buy=0.52; th_buy<=0.80; th_buy+=0.04)
      {
         for(double th_sell=0.20; th_sell<=0.48; th_sell+=0.04)
         {
            if(th_sell >= th_buy) continue;
            for(double th_skip=0.45; th_skip<=0.75; th_skip+=0.05)
            {
               double score = 0.0;
               for(int k=0; k<m_wf_len; k++)
               {
                  int idx = m_wf_ptr - 1 - k;
                  while(idx < 0) idx += FXAI_TFT_WF;
                  while(idx >= FXAI_TFT_WF) idx -= FXAI_TFT_WF;

                  int pred = FXAI_TFT_SKIP;
                  if(m_wf_pskip[idx] < th_skip)
                  {
                     if(m_wf_pup[idx] >= th_buy) pred = FXAI_TFT_BUY;
                     else if(m_wf_pup[idx] <= th_sell) pred = FXAI_TFT_SELL;
                  }

                  score += EvalDecisionUtility(pred,
                                               m_wf_cls[idx],
                                               m_wf_move[idx],
                                               m_wf_cost[idx]);
               }

               if(score > best_score)
               {
                  best_score = score;
                  best_buy = th_buy;
                  best_sell = th_sell;
                  best_skip = th_skip;
               }
            }
         }
      }

      m_thr_buy = FXAI_Clamp(best_buy, 0.50, 0.90);
      m_thr_sell = FXAI_Clamp(best_sell, 0.10, 0.50);
      m_thr_skip = FXAI_Clamp(best_skip, 0.35, 0.90);
   }

   void TrainTBPTT(const FXAIAIHyperParams &hp)
   {
      int n = m_train_len;
      if(n <= 0) return;
      if(n > FXAI_TFT_TBPTT) n = FXAI_TFT_TBPTT;
      if(n < 4) return;

      ForwardTrainSequence(n);

      // Gradients.
      double g_vsn_gate_w[FXAI_AI_WEIGHTS][FXAI_AI_WEIGHTS];
      double g_vsn_gate_b[FXAI_AI_WEIGHTS];
      double g_vsn_proj_w[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
      double g_vsn_proj_b[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];

      double g_static_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
      double g_static_b[FXAI_AI_MLP_HIDDEN];

      double g_enc_h0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_enc_h0_b[FXAI_AI_MLP_HIDDEN];
      double g_enc_c0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_enc_c0_b[FXAI_AI_MLP_HIDDEN];
      double g_dec_h0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_dec_h0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_dec_h0_b[FXAI_AI_MLP_HIDDEN];
      double g_dec_c0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_dec_c0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_dec_c0_b[FXAI_AI_MLP_HIDDEN];

      double g_e_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_e_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_e_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_e_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_e_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_e_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_e_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], g_e_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_e_bi[FXAI_AI_MLP_HIDDEN], g_e_bf[FXAI_AI_MLP_HIDDEN], g_e_bo[FXAI_AI_MLP_HIDDEN], g_e_bg[FXAI_AI_MLP_HIDDEN];

