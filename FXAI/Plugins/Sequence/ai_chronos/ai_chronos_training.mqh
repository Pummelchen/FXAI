   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      // Keep normalization stats responsive even if training is throttled.
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      UpdateInputStats(xa);

      // Retrain only once per new M1 bar (first tick of the bar).
      // Also reset state on obvious feature blow-ups to avoid leaking bad history.
      if(MathAbs(xa[1]) > 9.0 || MathAbs(xa[2]) > 9.0)
         ResetSequence();

      if(!ShouldTrainOnNewM1Bar())
         return;

      m_step++;
      m_adam_t++;

      // Controlled reset policy to reduce state bleed across sharp regime jumps.
      if((m_step % 4096) == 0)
         ResetSequence();
      if(MathAbs(xa[1]) > 9.0 || MathAbs(xa[2]) > 9.0)
         ResetSequence();

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (int)FXAI_LABEL_SKIP;

      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FXAI_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double base_lr = ScheduledLR(h);
      double l2 = FXAI_Clamp(h.l2, 0.0, 0.0800);

      double cost = InputCostProxyPoints(x);
      double sample_w = MoveSampleWeight(x, move_points);
      sample_w = FXAI_Clamp(sample_w * cls_bal, 0.10, 6.00);

      double rep[FXAI_CHR_D_MODEL];
      double p_raw[FXAI_CHR_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FXAI_CHR_HORIZONS];
      double patch_stat[FXAI_AI_FEATURES][FXAI_CHR_PATCH_LEN];
      double token_hist[FXAI_CHR_CODEBOOK];
      int token_target = 0;
      double layer_in_mean[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];
      double layer_ctx_mean[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];
      double layer_ff_mean[FXAI_CHR_LAYERS][FXAI_CHR_D_FF];
      double mem_attn[FXAI_CHR_MEMORY];
      int token_count = 0;

      ForwardPass(xa,
                  true,
                  true,
                  cls,
                  move_points,
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

      double p_cal_now[FXAI_CHR_CLASS_COUNT];
      Calibrate3(p_raw, p_cal_now);
      double ev_now = ExpectedMoveFromHeads(mu, logv, q25, q75, mu_h, p_cal_now[(int)FXAI_LABEL_SKIP]);
      UpdateValidationMetrics(cls, p_cal_now, ev_now, cost);
      if(!m_reference_ready && m_val_steps > 64)
         base_lr *= 0.85;

      double cal_lr = FXAI_Clamp(0.02 + 0.12 * base_lr, 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, sample_w, cal_lr);

      // Keep binary calibrator aligned for legacy paths.
      double den_dir = p_raw[(int)FXAI_LABEL_BUY] + p_raw[(int)FXAI_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FXAI_LABEL_BUY] / den_dir;
      if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, sample_w);
      else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, sample_w);

      double target_cls[FXAI_CHR_CLASS_COUNT];
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
         target_cls[c] = (c == cls ? 1.0 : 0.0);

      // Cross-entropy + teacher distillation gradient.
      double t_logits[FXAI_CHR_CLASS_COUNT];
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         double z = m_t_b_cls[c];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            z += m_t_w_cls[c][d] * rep[d];
         t_logits[c] = z;
      }
      double p_teacher[FXAI_CHR_CLASS_COUNT];
      Softmax3(t_logits, p_teacher);

      double err_cls[FXAI_CHR_CLASS_COUNT];
      double kd = 0.18;
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
         err_cls[c] = (p_raw[c] - target_cls[c]) + kd * (p_raw[c] - p_teacher[c]);

      double g_rep[FXAI_CHR_D_MODEL];
      for(int d=0; d<FXAI_CHR_D_MODEL; d++) g_rep[d] = 0.0;

      double lr_head = AdamGroupLR(0, MathAbs(err_cls[0]) + MathAbs(err_cls[1]) + MathAbs(err_cls[2]), base_lr);
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         m_b_cls[c] -= lr_head * sample_w * err_cls[c];
         m_b_cls[c] = FXAI_ClipSym(m_b_cls[c], 4.0);

         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
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

      double q_pred[FXAI_CHR_QUANTILES];
      double g_q[FXAI_CHR_QUANTILES];
      for(int q=0; q<FXAI_CHR_QUANTILES; q++)
      {
         double z = m_b_q[q];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            z += m_w_q[q][d] * rep[d];
         q_pred[q] = z;
         double tau = QuantileLevel(q);
         double ind = (move_tgt <= q_pred[q] ? 1.0 : 0.0);
         g_q[q] = (ind - tau);
      }
      // Monotonic quantile penalty.
      const double mono_lambda = 0.08;
      for(int q=1; q<FXAI_CHR_QUANTILES; q++)
      {
         double gap = q_pred[q - 1] - q_pred[q];
         if(gap > 0.0)
         {
            g_q[q - 1] += mono_lambda * gap;
            g_q[q] -= mono_lambda * gap;
         }
      }

      double edge = MathAbs(move_points) - cost;
      double move_w = FXAI_Clamp(sample_w * (0.50 + edge / MathMax(cost, 1.0)), 0.10, 8.00);
      double qg = 0.0;
      for(int q=0; q<FXAI_CHR_QUANTILES; q++) qg += MathAbs(g_q[q]);
      double lr_move = AdamGroupLR(1, MathAbs(g_mu) + MathAbs(g_logv) + 0.30 * qg, base_lr * 0.70);

      m_b_mu -= lr_move * move_w * g_mu;
      m_b_logv -= lr_move * move_w * g_logv;
      for(int q=0; q<FXAI_CHR_QUANTILES; q++)
         m_b_q[q] -= lr_move * move_w * g_q[q];

      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         double wmu_old = m_w_mu[d];
         double wlv_old = m_w_logv[d];

         m_w_mu[d] -= lr_move * move_w * (g_mu * rep[d] + l2 * 0.10 * m_w_mu[d]);
         m_w_logv[d] -= lr_move * move_w * (g_logv * rep[d] + l2 * 0.10 * m_w_logv[d]);
         double gq_rep = 0.0;
         for(int q=0; q<FXAI_CHR_QUANTILES; q++)
         {
            double wq_old = m_w_q[q][d];
            m_w_q[q][d] -= lr_move * move_w * (g_q[q] * rep[d] + l2 * 0.06 * m_w_q[q][d]);
            gq_rep += g_q[q] * wq_old;
         }
         g_rep[d] += move_w * (g_mu * wmu_old + 0.35 * g_logv * wlv_old + 0.16 * gq_rep);
      }

      // Multi-horizon move heads.
      double horizon_tgt[FXAI_CHR_HORIZONS];
      horizon_tgt[0] = move_tgt;
      horizon_tgt[1] = 0.80 * move_tgt;
      horizon_tgt[2] = 0.60 * move_tgt;
      horizon_tgt[3] = 0.40 * move_tgt;
      double lr_h = AdamGroupLR(2, MathAbs(diff), base_lr * 0.55);
      for(int hidx=0; hidx<FXAI_CHR_HORIZONS; hidx++)
      {
         double gh = FXAI_ClipSym(mu_h[hidx] - horizon_tgt[hidx], 4.0);
         m_b_mu_h[hidx] -= lr_h * move_w * gh;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double wold = m_w_mu_h[hidx][d];
            m_w_mu_h[hidx][d] -= lr_h * move_w * (gh * rep[d] + l2 * 0.04 * m_w_mu_h[hidx][d]);
            g_rep[d] += 0.20 * move_w * gh * wold;
         }
      }

      // Chronos token-likelihood objective.
      double tok_prob[FXAI_CHR_CODEBOOK];
      int tok_top = 0;
      TokenHead(rep, tok_prob, tok_top);
      double lr_tok = AdamGroupLR(3, 1.0 - tok_prob[token_target], base_lr * 0.45);
      for(int t=0; t<FXAI_CHR_CODEBOOK; t++)
      {
         double target = (t == token_target ? 1.0 : 0.0);
         double err = tok_prob[t] - target;
         m_b_tok[t] -= lr_tok * sample_w * err;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double wold = m_w_tok[t][d];
            m_w_tok[t][d] -= lr_tok * sample_w * (err * rep[d] + l2 * 0.03 * m_w_tok[t][d]);
            g_rep[d] += 0.15 * sample_w * err * wold;
         }
      }

      // Codebook adaptation from token histogram.
      double hist_sum = 0.0;
      for(int t=0; t<FXAI_CHR_CODEBOOK; t++) hist_sum += token_hist[t];
      if(hist_sum <= 0.0) hist_sum = 1.0;
      double lr_cb = AdamGroupLR(6, 1.0, base_lr * 0.22);
      for(int t=0; t<FXAI_CHR_CODEBOOK; t++)
      {
         double usage = token_hist[t] / hist_sum;
         m_codebook_usage[t] = 0.995 * m_codebook_usage[t] + 0.005 * usage;
         if(usage <= 0.0) continue;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double tgt = rep[d] * usage;
            double e = m_codebook[t][d] - tgt;
            m_codebook[t][d] -= lr_cb * move_w * (e + l2 * 0.02 * m_codebook[t][d]);
         }
      }

      // Gradient clipping on shared representation gradient.
      double gnorm2 = 0.0;
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         gnorm2 += g_rep[d] * g_rep[d];
      double gnorm = MathSqrt(gnorm2);
      if(gnorm > 3.0)
      {
         double s = 3.0 / MathMax(gnorm, 1e-9);
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            g_rep[d] *= s;
      }

      // Update patch embedding + channel gates.
      double lr_patch = AdamGroupLR(2, gnorm, base_lr * 0.45);
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
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
         for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
            pm += patch_stat[f][t];
         pm /= (double)FXAI_CHR_PATCH_LEN;

         double gf = 0.0;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            gf += g_rep[d] * pm;

         m_ch_gate[f] -= lr_patch * move_w * (0.25 * gf + l2 * 0.02 * (m_ch_gate[f] - 1.0));
         m_ch_gate[f] = FXAI_Clamp(m_ch_gate[f], 0.10, 4.00);
      }

      // Update positional embeddings with recency focus.
      double lr_pos = AdamGroupLR(3, gnorm, base_lr * 0.20);
      for(int p=0; p<token_count; p++)
      {
         double rw = (p == token_count - 1 ? 0.40 : 0.12);
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double grad = rw * g_rep[d] / (double)MathMax(token_count, 1);
            m_pos[p][d] -= lr_pos * move_w * (grad + l2 * 0.02 * m_pos[p][d]);
         }
      }

      // Encoder-weight updates using cached token activations (causal TBPTT-lite).
      double tok_w_den = ((token_count > 0) ? ((double)MathMax(token_count - 1, 0) + 1.6) : 1.0);
      if(tok_w_den <= 0.0) tok_w_den = 1.0;

      for(int l=0; l<FXAI_CHR_LAYERS; l++)
      {
         double lr_enc = AdamGroupLR(4 + l, gnorm, base_lr * 0.25);

         // Output projection from cached attention context.
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            for(int od=0; od<FXAI_CHR_D_MODEL; od++)
            {
               int hdx = od / FXAI_CHR_D_HEAD;
               int dh = od % FXAI_CHR_D_HEAD;
               double grad = l2 * 0.05 * m_wo[l][d][od];
               for(int t=0; t<token_count; t++)
               {
                  double tw = ((t == token_count - 1 ? 1.6 : 1.0) / tok_w_den);
                  grad += tw * g_rep[d] * m_cache_ctx[l][hdx][t][dh];
               }
               m_wo[l][d][od] -= lr_enc * move_w * grad;
            }
         }

         // FFN2 -> FFN1 path.
         double dff[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_FF];
         for(int t=0; t<token_count; t++)
            for(int r=0; r<FXAI_CHR_D_FF; r++)
               dff[t][r] = 0.0;

         for(int t=0; t<token_count; t++)
         {
            double tw = ((t == token_count - 1 ? 1.6 : 1.0) / tok_w_den);
            for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            {
               double gt = tw * g_rep[d];
               m_bff2[l][d] -= lr_enc * move_w * gt * 0.30;
               for(int r=0; r<FXAI_CHR_D_FF; r++)
               {
                  double w_old = m_wff2[l][d][r];
                  double grad = gt * m_cache_ff1[l][t][r] + l2 * 0.05 * m_wff2[l][d][r];
                  m_wff2[l][d][r] -= lr_enc * move_w * grad;
                  dff[t][r] += gt * w_old;
               }
            }
         }

         for(int r=0; r<FXAI_CHR_D_FF; r++)
         {
            double bgrad = 0.0;
            for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            {
               double grad = l2 * 0.05 * m_wff1[l][r][d];
               for(int t=0; t<token_count; t++)
               {
                  double tw = ((t == token_count - 1 ? 1.6 : 1.0) / tok_w_den);
                  double dr = dff[t][r] * GELUDerivApprox(m_cache_ff1[l][t][r]);
                  grad += tw * dr * m_cache_u[l][t][d];
                  bgrad += tw * dr * 0.25;
               }
               m_wff1[l][r][d] -= lr_enc * move_w * grad;
            }
            m_bff1[l][r] -= lr_enc * move_w * bgrad;
         }

         // Q/K/V corrective step from cached token inputs and attention shape.
         for(int hdx=0; hdx<FXAI_CHR_HEADS; hdx++)
         {
            for(int dh=0; dh<FXAI_CHR_D_HEAD; dh++)
            {
               int od = hdx * FXAI_CHR_D_HEAD + dh;
               for(int d=0; d<FXAI_CHR_D_MODEL; d++)
               {
                  double grad_q = l2 * 0.02 * m_wq[l][hdx][dh][d];
                  double grad_k = l2 * 0.02 * m_wk[l][hdx][dh][d];
                  double grad_v = l2 * 0.02 * m_wv[l][hdx][dh][d];
                  for(int t=0; t<token_count; t++)
                  {
                     double tw = ((t == token_count - 1 ? 1.6 : 1.0) / tok_w_den);
                     double xin = m_cache_layer_in[l][t][d];
                     double gt = tw * g_rep[od];
                     double diag_att = m_cache_att[l][hdx][t][t];
                     grad_q += 0.018 * gt * xin;
                     grad_k += 0.014 * gt * xin * (1.0 - diag_att);
                     grad_v += 0.016 * gt * xin;
                  }
                  m_wq[l][hdx][dh][d] -= lr_enc * move_w * grad_q;
                  m_wk[l][hdx][dh][d] -= lr_enc * move_w * grad_k;
                  m_wv[l][hdx][dh][d] -= lr_enc * move_w * grad_v;
               }
            }
         }
      }

      // Retrieval-memory and token-gate updates.
      double lr_mem = AdamGroupLR(7, gnorm, base_lr * 0.18);
      double best_attn = mem_attn[0];
      for(int m=0; m<FXAI_CHR_MEMORY; m++)
      {
         m_mem_usage[m] = 0.995 * m_mem_usage[m] + 0.005 * mem_attn[m];
         if(mem_attn[m] > best_attn)
            best_attn = mem_attn[m];

         double mix = FXAI_Clamp(mem_attn[m], 0.0, 1.0);
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            m_mem_k[m][d] = (1.0 - lr_mem * mix) * m_mem_k[m][d] + (lr_mem * mix) * rep[d];
            m_mem_v[m][d] = (1.0 - lr_mem * mix) * m_mem_v[m][d] + (lr_mem * mix) * rep[d];
            m_mem_k[m][d] = FXAI_ClipSym(m_mem_k[m][d], 8.0);
            m_mem_v[m][d] = FXAI_ClipSym(m_mem_v[m][d], 8.0);
         }
      }
      m_b_mem_gate = FXAI_ClipSym(0.995 * m_b_mem_gate + 0.005 * (best_attn - 0.25), 2.0);

      // Refresh least-used memory slot periodically.
      if((m_step % 128) == 0)
      {
         int least = 0;
         for(int m=1; m<FXAI_CHR_MEMORY; m++)
         {
            if(m_mem_usage[m] < m_mem_usage[least])
               least = m;
         }
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
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
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            align += rep[d] * m_codebook[cb][d];
         m_codebook_gate[f] += lr_mem * 0.01 * FXAI_ClipSym(align, 5.0);
         m_codebook_gate[f] = FXAI_Clamp(m_codebook_gate[f], 0.10, 4.00);
      }

      // Update shared move estimators in base plugin.
      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(xa, move_points, h, sample_w);
      UpdateNativeQualityHeads(xa, sample_w, h.lr, h.l2);

      // Replay consolidation to reduce forgetting on volatile regimes.
      ReplayPush(cls, xa, move_points, cost, sample_w);
      int replay_steps = 0;
      if(m_chr_replay_size >= 192) replay_steps = 2;
      else if(m_chr_replay_size >= 64) replay_steps = 1;
      for(int rs=0; rs<replay_steps; rs++)
      {
         int slot = ReplaySampleSlot();
         if(slot < 0) break;

         double rep_r[FXAI_CHR_D_MODEL];
         double p_raw_r[FXAI_CHR_CLASS_COUNT];
         double mu_r = 0.0, logv_r = 0.0, q25_r = 0.0, q75_r = 0.0;
         double mu_h_r[FXAI_CHR_HORIZONS];
         double patch_stat_r[FXAI_AI_FEATURES][FXAI_CHR_PATCH_LEN];
         double token_hist_r[FXAI_CHR_CODEBOOK];
         int token_target_r = 0;
         double layer_in_mean_r[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];
         double layer_ctx_mean_r[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];
         double layer_ff_mean_r[FXAI_CHR_LAYERS][FXAI_CHR_D_FF];
         double mem_attn_r[FXAI_CHR_MEMORY];
         int token_count_r = 0;
         double xr[FXAI_AI_WEIGHTS];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            xr[k] = m_chr_replay_x[slot][k];

         ForwardPass(xr,
                     false,
                     true,
                     m_replay_cls[slot],
                     m_chr_replay_move[slot],
                     rep_r,
                     p_raw_r,
                     mu_r,
                     logv_r,
                     q25_r,
                     q75_r,
                     mu_h_r,
                     patch_stat_r,
                     token_hist_r,
                     token_target_r,
                     layer_in_mean_r,
                     layer_ctx_mean_r,
                     layer_ff_mean_r,
                     mem_attn_r,
                     token_count_r);

         int cls_r = ClampI(m_replay_cls[slot], 0, FXAI_CHR_CLASS_COUNT - 1);
         double wr = FXAI_Clamp(0.35 * m_replay_w[slot], 0.05, 1.50);
         double err_r[FXAI_CHR_CLASS_COUNT];
         for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
            err_r[c] = p_raw_r[c] - (c == cls_r ? 1.0 : 0.0);

         double lr_r = AdamGroupLR(13, MathAbs(err_r[0]) + MathAbs(err_r[1]) + MathAbs(err_r[2]), base_lr * 0.20);
         for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
         {
            m_b_cls[c] -= lr_r * wr * err_r[c];
            for(int d=0; d<FXAI_CHR_D_MODEL; d++)
               m_w_cls[c][d] -= lr_r * wr * (err_r[c] * rep_r[d] + l2 * 0.08 * m_w_cls[c][d]);
         }
         UpdateCalibrator3(p_raw_r, cls_r, wr, FXAI_Clamp(0.50 * cal_lr, 0.0002, 0.0120));
      }

      UpdateTeacherHeads();
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
};
