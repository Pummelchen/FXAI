   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      // Tick-level observation cadence: always update state / sequence, but only train once per new M1 bar.
      m_obs_step++;

      // Controlled reset policy to reduce state bleed across sharp regime jumps.
      if((m_obs_step % 4096) == 0)
         ResetSequence();
      if(MathAbs(x[1]) > 9.0 || MathAbs(x[2]) > 9.0)
         ResetSequence();

      // Always adapt input normalization to current regime.
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      UpdateInputStats(xa);

      // Always push the tick observation into the rolling sequence buffer.
      double xn_tick[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, xn_tick);
      PushSequence(xn_tick);

      // Only retrain on the first tick of each new M1 bar.
      if(!ShouldTrainOnNewM1Bar())
         return;

      // Training step counters.
      m_step++;
      m_adam_t++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (int)FXAI_LABEL_SKIP;

      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FXAI_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double base_lr = ScheduledLR(h);
      double l2 = FXAI_Clamp(h.l2, 0.0, 0.0800);

      double cost = InputCostProxyPoints(x);
      double sample_w = MoveSampleWeight(x, move_points);
      sample_w = FXAI_Clamp(sample_w * cls_bal, 0.10, 6.00);

      double rep[FXAI_TFM_D_MODEL];
      double p_raw[FXAI_TFM_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FXAI_TFM_HORIZONS];
      double patch_stat[FXAI_AI_FEATURES][FXAI_TFM_PATCH_LEN];
      double token_hist[FXAI_TFM_CODEBOOK];
      int token_target = 0;
      double layer_in_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
      double layer_ctx_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
      double layer_ff_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_FF];
      double mem_attn[FXAI_TFM_MEMORY];
      int token_count = 0;

      ForwardPass(xa,
                  true,
                  rep,
                  p_raw,
                  mu,
                  logv,
                  q25,
                  q75,
                  mu_h,
                  patch_stat,
                  token_hist,
                  token_target,
                  layer_in_mean,
                  layer_ctx_mean,
                  layer_ff_mean,
                  mem_attn,
                  token_count);

      double cal_lr = FXAI_Clamp(0.02 + 0.12 * base_lr, 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, sample_w, cal_lr);

      // Keep binary calibrator aligned for legacy paths.
      double den_dir = p_raw[(int)FXAI_LABEL_BUY] + p_raw[(int)FXAI_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FXAI_LABEL_BUY] / den_dir;
      if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, sample_w);
      else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, sample_w);

      double target_cls[FXAI_TFM_CLASS_COUNT];
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
         target_cls[c] = (c == cls ? 1.0 : 0.0);

      // Cross-entropy gradient.
      double err_cls[FXAI_TFM_CLASS_COUNT];
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
         err_cls[c] = (p_raw[c] - target_cls[c]);

      double g_rep[FXAI_TFM_D_MODEL];
      for(int d=0; d<FXAI_TFM_D_MODEL; d++) g_rep[d] = 0.0;

      double lr_head = AdamGroupLR(0, MathAbs(err_cls[0]) + MathAbs(err_cls[1]) + MathAbs(err_cls[2]), base_lr);
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
      {
         m_b_cls[c] -= lr_head * sample_w * err_cls[c];
         m_b_cls[c] = FXAI_ClipSym(m_b_cls[c], 4.0);

         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double w_old = m_w_cls[c][d];
            double grad = err_cls[c] * rep[d] + l2 * 0.20 * m_w_cls[c][d];
            m_w_cls[c][d] -= lr_head * sample_w * grad;
            g_rep[d] += err_cls[c] * w_old;
         }
      }

      // Distributional move head gradients.
      double move_tgt = MathAbs(move_points);
      double sigma = MathExp(0.5 * FXAI_Clamp(logv, -4.0, 4.0));
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double sig2 = sigma * sigma;

      double diff = mu - move_tgt;
      double g_mu = FXAI_ClipSym(diff / (sig2 + 1e-6), 4.0);
      double g_logv = FXAI_ClipSym(0.5 * (1.0 - (diff * diff) / (sig2 + 1e-6)), 4.0);
      double e25 = move_tgt - q25;
      double e75 = move_tgt - q75;
      double g_q25 = (e25 >= 0.0 ? -0.25 : 0.75);
      double g_q75 = (e75 >= 0.0 ? -0.75 : 0.25);

      double edge = MathAbs(move_points) - cost;
      double move_w = FXAI_Clamp(sample_w * (0.50 + edge / MathMax(cost, 1.0)), 0.10, 8.00);
      double lr_move = AdamGroupLR(1, MathAbs(g_mu) + MathAbs(g_logv), base_lr * 0.70);

      m_b_mu -= lr_move * move_w * g_mu;
      m_b_logv -= lr_move * move_w * g_logv;
      m_b_q25 -= lr_move * move_w * g_q25;
      m_b_q75 -= lr_move * move_w * g_q75;

      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         double wmu_old = m_w_mu[d];
         double wlv_old = m_w_logv[d];
         double w25_old = m_w_q25[d];
         double w75_old = m_w_q75[d];

         m_w_mu[d] -= lr_move * move_w * (g_mu * rep[d] + l2 * 0.10 * m_w_mu[d]);
         m_w_logv[d] -= lr_move * move_w * (g_logv * rep[d] + l2 * 0.10 * m_w_logv[d]);
         m_w_q25[d] -= lr_move * move_w * (g_q25 * rep[d] + l2 * 0.10 * m_w_q25[d]);
         m_w_q75[d] -= lr_move * move_w * (g_q75 * rep[d] + l2 * 0.10 * m_w_q75[d]);

         g_rep[d] += move_w * (g_mu * wmu_old + 0.35 * g_logv * wlv_old + 0.20 * g_q25 * w25_old + 0.20 * g_q75 * w75_old);
      }

      // Multi-horizon move heads.
      double horizon_tgt[FXAI_TFM_HORIZONS];
      horizon_tgt[0] = move_tgt;
      horizon_tgt[1] = 0.75 * move_tgt;
      horizon_tgt[2] = 0.55 * move_tgt;
      double lr_h = AdamGroupLR(2, MathAbs(diff), base_lr * 0.55);
      for(int hidx=0; hidx<FXAI_TFM_HORIZONS; hidx++)
      {
         double gh = FXAI_ClipSym(mu_h[hidx] - horizon_tgt[hidx], 4.0);
         m_b_mu_h[hidx] -= lr_h * move_w * gh;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double wold = m_w_mu_h[hidx][d];
            m_w_mu_h[hidx][d] -= lr_h * move_w * (gh * rep[d] + l2 * 0.04 * m_w_mu_h[hidx][d]);
            g_rep[d] += 0.20 * move_w * gh * wold;
         }
      }

      // Chronos token-likelihood objective.
      double tok_prob[FXAI_TFM_CODEBOOK];
      int tok_top = 0;
      TokenHead(rep, tok_prob, tok_top);
      double lr_tok = AdamGroupLR(3, 1.0 - tok_prob[token_target], base_lr * 0.45);
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++)
      {
         double target = (t == token_target ? 1.0 : 0.0);
         double err = tok_prob[t] - target;
         m_b_tok[t] -= lr_tok * sample_w * err;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double wold = m_w_tok[t][d];
            m_w_tok[t][d] -= lr_tok * sample_w * (err * rep[d] + l2 * 0.03 * m_w_tok[t][d]);
            g_rep[d] += 0.15 * sample_w * err * wold;
         }
      }

      // Codebook adaptation from token histogram.
      double hist_sum = 0.0;
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++) hist_sum += token_hist[t];
      if(hist_sum <= 0.0) hist_sum = 1.0;
      double lr_cb = AdamGroupLR(6, 1.0, base_lr * 0.22);
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++)
      {
         double usage = token_hist[t] / hist_sum;
         m_codebook_usage[t] = 0.995 * m_codebook_usage[t] + 0.005 * usage;
         if(usage <= 0.0) continue;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double tgt = rep[d] * usage;
            double e = m_codebook[t][d] - tgt;
            m_codebook[t][d] -= lr_cb * move_w * (e + l2 * 0.02 * m_codebook[t][d]);
         }
      }

      // Gradient clipping on shared representation gradient.
      double gnorm2 = 0.0;
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         gnorm2 += g_rep[d] * g_rep[d];
      double gnorm = MathSqrt(gnorm2);
      if(gnorm > 3.0)
      {
         double s = 3.0 / MathMax(gnorm, 1e-9);
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            g_rep[d] *= s;
      }

      // Update patch embedding + channel gates.
      double lr_patch = AdamGroupLR(2, gnorm, base_lr * 0.45);
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            for(int t=0; t<FXAI_TFM_PATCH_LEN; t++)
            {
               double grad = (g_rep[d] * patch_stat[f][t] / (double)MathMax(token_count, 1)) + l2 * 0.10 * m_w_patch[d][f][t];
               m_w_patch[d][f][t] -= lr_patch * move_w * grad;
            }
         }
         m_b_patch[d] -= lr_patch * move_w * g_rep[d] * 0.15;
      }

      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         double pm = 0.0;
         for(int t=0; t<FXAI_TFM_PATCH_LEN; t++)
            pm += patch_stat[f][t];
         pm /= (double)FXAI_TFM_PATCH_LEN;

         double gf = 0.0;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            gf += g_rep[d] * pm;

         m_ch_gate[f] -= lr_patch * move_w * (0.25 * gf + l2 * 0.02 * (m_ch_gate[f] - 1.0));
         m_ch_gate[f] = FXAI_Clamp(m_ch_gate[f], 0.10, 4.00);
      }

      // Update positional embeddings with recency focus.
      double lr_pos = AdamGroupLR(3, gnorm, base_lr * 0.20);
      for(int p=0; p<token_count; p++)
      {
         double rw = (p == token_count - 1 ? 0.40 : 0.12);
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double grad = rw * g_rep[d] / (double)MathMax(token_count, 1);
            m_pos[p][d] -= lr_pos * move_w * (grad + l2 * 0.02 * m_pos[p][d]);
         }
      }

      // Encoder-weight updates using layer summary statistics.
      for(int l=0; l<FXAI_TFM_LAYERS; l++)
      {
         double lr_enc = AdamGroupLR(4 + l, gnorm, base_lr * 0.25);

         // Output projection from attention contexts.
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            for(int od=0; od<FXAI_TFM_D_MODEL; od++)
            {
               double grad = g_rep[d] * layer_ctx_mean[l][od] + l2 * 0.05 * m_wo[l][d][od];
               m_wo[l][d][od] -= lr_enc * move_w * grad;
            }
         }

         // FFN2.
         double dff[FXAI_TFM_D_FF];
         for(int r=0; r<FXAI_TFM_D_FF; r++) dff[r] = 0.0;

         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            m_bff2[l][d] -= lr_enc * move_w * g_rep[d] * 0.30;
            for(int r=0; r<FXAI_TFM_D_FF; r++)
            {
               double grad = g_rep[d] * layer_ff_mean[l][r] + l2 * 0.05 * m_wff2[l][d][r];
               m_wff2[l][d][r] -= lr_enc * move_w * grad;
               dff[r] += g_rep[d] * m_wff2[l][d][r];
            }
         }

         // FFN1 (approximate backprop through GELU).
         for(int r=0; r<FXAI_TFM_D_FF; r++)
         {
            double act = layer_ff_mean[l][r];
            double dg = GELUDerivApprox(act);
            double dr = dff[r] * dg;
            m_bff1[l][r] -= lr_enc * move_w * dr * 0.25;
            for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            {
               double grad = dr * layer_in_mean[l][d] + l2 * 0.05 * m_wff1[l][r][d];
               m_wff1[l][r][d] -= lr_enc * move_w * grad;
            }
         }

         // Q/K/V tiny corrective step keeps attention adaptable without heavy backprop.
         for(int hdx=0; hdx<FXAI_TFM_HEADS; hdx++)
         {
            for(int dh=0; dh<FXAI_TFM_D_HEAD; dh++)
            {
               for(int d=0; d<FXAI_TFM_D_MODEL; d++)
               {
                  double corr = g_rep[d] * layer_in_mean[l][d] * 0.015;
                  m_wq[l][hdx][dh][d] -= lr_enc * move_w * (corr + l2 * 0.02 * m_wq[l][hdx][dh][d]);
                  m_wk[l][hdx][dh][d] -= lr_enc * move_w * (corr + l2 * 0.02 * m_wk[l][hdx][dh][d]);
                  m_wv[l][hdx][dh][d] -= lr_enc * move_w * (corr + l2 * 0.02 * m_wv[l][hdx][dh][d]);
               }
            }
         }
      }

      // Retrieval-memory and token-gate updates.
      double lr_mem = AdamGroupLR(7, gnorm, base_lr * 0.18);
      double best_attn = mem_attn[0];
      for(int m=0; m<FXAI_TFM_MEMORY; m++)
      {
         m_mem_usage[m] = 0.995 * m_mem_usage[m] + 0.005 * mem_attn[m];
         if(mem_attn[m] > best_attn)
            best_attn = mem_attn[m];

         double mix = FXAI_Clamp(mem_attn[m], 0.0, 1.0);
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            m_mem_k[m][d] = (1.0 - lr_mem * mix) * m_mem_k[m][d] + (lr_mem * mix) * rep[d];
            m_mem_v[m][d] = (1.0 - lr_mem * mix) * m_mem_v[m][d] + (lr_mem * mix) * rep[d];
            m_mem_k[m][d] = FXAI_ClipSym(m_mem_k[m][d], 8.0);
            m_mem_v[m][d] = FXAI_ClipSym(m_mem_v[m][d], 8.0);
         }
      }

      // Refresh least-used memory slot periodically.
      if((m_step % 128) == 0)
      {
         int least = 0;
         for(int m=1; m<FXAI_TFM_MEMORY; m++)
         {
            if(m_mem_usage[m] < m_mem_usage[least])
               least = m;
         }
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            m_mem_k[least][d] = rep[d];
            m_mem_v[least][d] = rep[d];
         }
         m_mem_usage[least] = 1.0;
      }

      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         int cb = CodebookIndex(f, QuantizeFeatureValue(f, patch_stat[f][0]));
         double align = 0.0;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            align += rep[d] * m_codebook[cb][d];
         m_codebook_gate[f] += lr_mem * 0.01 * FXAI_ClipSym(align, 5.0);
         m_codebook_gate[f] = FXAI_Clamp(m_codebook_gate[f], 0.10, 4.00);
      }

      // Update shared move estimators in base plugin.
      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(xa, move_points, h, sample_w);
      UpdateNativeQualityHeads(xa, sample_w, h.lr, h.l2);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double expected_move = 0.0;
      if(!PredictModelCore(x, hp, probs, expected_move))
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
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      return MathMax(PredictMoveHeadRaw(xa), m_move_ema_abs);
   }
