   bool   m_initialized;
   int    m_step;
   int    m_adam_t;


   // Observation cadence / training gate.
   int      m_obs_step;
   datetime m_last_m1_train_bar;

   // Rolling multivariate sequence state.
   int    m_seq_ptr;
   int    m_seq_len;
   double m_seq[FXAI_TFM_SEQ][FXAI_AI_FEATURES];

   // Input normalization.
   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   // Class balance EMA.
   double m_cls_ema[FXAI_TFM_CLASS_COUNT];

   // Tokenization statistics and foundation vocabulary.
   double m_feat_mean[FXAI_AI_FEATURES];
   double m_feat_var[FXAI_AI_FEATURES];
   bool   m_feat_stats_ready;
   int    m_feat_stats_steps;

   double m_codebook[FXAI_TFM_CODEBOOK][FXAI_TFM_D_MODEL];
   double m_codebook_usage[FXAI_TFM_CODEBOOK];
   double m_codebook_gate[FXAI_AI_FEATURES];

   // Patch embedding + channel gating.
   double m_w_patch[FXAI_TFM_D_MODEL][FXAI_AI_FEATURES][FXAI_TFM_PATCH_LEN];
   double m_b_patch[FXAI_TFM_D_MODEL];
   double m_ch_gate[FXAI_AI_FEATURES];

   // Positional embedding per patch token.
   double m_pos[FXAI_TFM_MAX_PATCHES][FXAI_TFM_D_MODEL];

   // Encoder stack.
   double m_wq[FXAI_TFM_LAYERS][FXAI_TFM_HEADS][FXAI_TFM_D_HEAD][FXAI_TFM_D_MODEL];
   double m_wk[FXAI_TFM_LAYERS][FXAI_TFM_HEADS][FXAI_TFM_D_HEAD][FXAI_TFM_D_MODEL];
   double m_wv[FXAI_TFM_LAYERS][FXAI_TFM_HEADS][FXAI_TFM_D_HEAD][FXAI_TFM_D_MODEL];
   double m_wo[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL][FXAI_TFM_D_MODEL];

   double m_wff1[FXAI_TFM_LAYERS][FXAI_TFM_D_FF][FXAI_TFM_D_MODEL];
   double m_bff1[FXAI_TFM_LAYERS][FXAI_TFM_D_FF];
   double m_wff2[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL][FXAI_TFM_D_FF];
   double m_bff2[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];

   double m_ln1_g[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
   double m_ln1_b[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
   double m_ln2_g[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
   double m_ln2_b[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];

   // Output heads.
   double m_w_cls[FXAI_TFM_CLASS_COUNT][FXAI_TFM_D_MODEL];
   double m_b_cls[FXAI_TFM_CLASS_COUNT];

   double m_w_mu[FXAI_TFM_D_MODEL];
   double m_b_mu;
   double m_w_logv[FXAI_TFM_D_MODEL];
   double m_b_logv;
   double m_w_q25[FXAI_TFM_D_MODEL];
   double m_b_q25;
   double m_w_q75[FXAI_TFM_D_MODEL];
   double m_b_q75;
   double m_w_mu_h[FXAI_TFM_HORIZONS][FXAI_TFM_D_MODEL];
   double m_b_mu_h[FXAI_TFM_HORIZONS];

   // Token-level language modeling head (TimesFM-style discrete forecasting objective).
   double m_w_tok[FXAI_TFM_CODEBOOK][FXAI_TFM_D_MODEL];
   double m_b_tok[FXAI_TFM_CODEBOOK];

   // Retrieval memory bank to emulate foundation priors and decoder cache.
   double m_mem_k[FXAI_TFM_MEMORY][FXAI_TFM_D_MODEL];
   double m_mem_v[FXAI_TFM_MEMORY][FXAI_TFM_D_MODEL];
   double m_mem_usage[FXAI_TFM_MEMORY];
   int    m_mem_ptr;
   double m_w_mem_q[FXAI_TFM_D_MODEL][FXAI_TFM_D_MODEL];
   double m_w_mem_gate[FXAI_TFM_D_MODEL];
   double m_b_mem_gate;

   // Native 3-class calibration.
   double m_cal_temp;
   double m_cal_bias[FXAI_TFM_CLASS_COUNT];
   double m_cal_iso_pos[FXAI_TFM_CLASS_COUNT][FXAI_TFM_CAL_BINS];
   double m_cal_iso_cnt[FXAI_TFM_CLASS_COUNT][FXAI_TFM_CAL_BINS];
   int    m_cal3_steps;

   // Lightweight adaptive optimizer moments.
   double m_opt_m[8];
   double m_opt_v[8];

   int ClampI(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
   }



   datetime CurrentM1BarOpenTime(void) const
   {
      datetime ctx_time = ResolveContextTime();
      if(ctx_time > 0)
         return (datetime)(ctx_time - (ctx_time % 60));

      datetime t = 0;
      if(!FXAI_MarketDataBarTime(Symbol(), PERIOD_M1, 0, t))
         t = 0;
      if(t <= 0)
      {
         datetime now = TimeCurrent();
         return (datetime)(now - (now % 60));
      }
      return t;
   }

   bool ShouldTrainOnNewM1Bar(void)
   {
      datetime b = CurrentM1BarOpenTime();
      if(b <= 0) return false;

      if(m_last_m1_train_bar == 0)
      {
         m_last_m1_train_bar = b;
         return true;
      }

      if(b != m_last_m1_train_bar)
      {
         m_last_m1_train_bar = b;
         return true;
      }

      return false;
   }
   double GELU(const double x) const
   {
      double x3 = x * x * x;
      double t = 0.7978845608 * (x + 0.044715 * x3);
      return 0.5 * x * (1.0 + FXAI_Tanh(t));
   }

   double GELUDerivApprox(const double x) const
   {
      // Lightweight smooth derivative approximation.
      double s = FXAI_Sigmoid(1.702 * x);
      return FXAI_Clamp(s * (1.0 + 1.702 * x * (1.0 - s)), 0.02, 1.20);
   }

   void LayerNormAffine(double &v[],
                        const double &g[],
                        const double &b[]) const
   {
      FXAI_ModuleLayerNormAffine(v, FXAI_TFM_D_MODEL, g, b);
   }

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double e0 = MathExp(FXAI_Clamp(logits[0] - m, -30.0, 30.0));
      double e1 = MathExp(FXAI_Clamp(logits[1] - m, -30.0, 30.0));
      double e2 = MathExp(FXAI_Clamp(logits[2] - m, -30.0, 30.0));
      double s = e0 + e1 + e2;
      if(s <= 0.0)
      {
         probs[0] = 0.3333333;
         probs[1] = 0.3333333;
         probs[2] = 0.3333333;
         return;
      }

      probs[0] = e0 / s;
      probs[1] = e1 / s;
      probs[2] = e2 / s;
   }

   double ScheduledLR(const FXAIAIHyperParams &hp) const
   {
      double base = FXAI_Clamp(hp.lr, 0.0002, 0.0800);
      double lr = FXAI_LRScheduleInvSqrt(base, m_step, 160, 0.0011);
      return FXAI_Clamp(lr, 0.00005, 0.05000);
   }

   double AdamGroupLR(const int group_idx,
                      const double grad_mag,
                      const double base_lr)
   {
      int g = ClampI(group_idx, 0, 7);
      const double b1 = 0.90;
      const double b2 = 0.999;
      const double eps = 1e-8;

      double gm = MathAbs(grad_mag);
      m_opt_m[g] = b1 * m_opt_m[g] + (1.0 - b1) * gm;
      m_opt_v[g] = b2 * m_opt_v[g] + (1.0 - b2) * gm * gm;

      double t = (double)MathMax(m_adam_t, 1);
      double mh = m_opt_m[g] / (1.0 - MathPow(b1, t));
      double vh = m_opt_v[g] / (1.0 - MathPow(b2, t));
      double scale = mh / (MathSqrt(vh) + eps);
      return FXAI_Clamp(base_lr * (0.60 + 0.40 * scale), 0.000003, 0.100000);
   }

   void ResetInputNorm(void)
   {
      m_x_norm_ready = false;
      m_x_norm_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void UpdateInputStats(const double &x[])
   {
      double a = (m_x_norm_steps < 160 ? 0.04 : 0.012);
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double d = x[i] - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = x[i] - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * dv * dv;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;
      }
      m_x_norm_steps++;
      if(m_x_norm_steps >= 32) m_x_norm_ready = true;
   }

   void NormalizeInput(const double &x[], double &xn[]) const
   {
      xn[0] = 1.0;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         if(!m_x_norm_ready)
         {
            xn[i] = FXAI_ClipSym(x[i], 8.0);
            continue;
         }

         double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FXAI_ClipSym((x[i] - m_x_mean[i]) * inv, 8.0);
      }
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         xa[i] = x[i];

      int win_n = CurrentWindowSize();
      if(win_n <= 1) return;

      double mean1 = CurrentWindowFeatureMean(0);
      double mean2 = CurrentWindowFeatureMean(1);
      double mean4 = CurrentWindowFeatureMean(3);
      double first1 = CurrentWindowValue(0, 1);
      double last1 = CurrentWindowValue(win_n - 1, 1);
      double first2 = CurrentWindowValue(0, 2);
      double last2 = CurrentWindowValue(win_n - 1, 2);
      double trend1 = (first1 - last1) / (double)(win_n - 1);
      double trend2 = (first2 - last2) / (double)(win_n - 1);
      double var1 = 0.0;
      for(int i=0; i<win_n; i++)
      {
         double d1 = CurrentWindowValue(i, 1) - mean1;
         var1 += d1 * d1;
      }
      var1 = MathSqrt(var1 / (double)win_n);
      double attn[];
      double conv_fast[];
      double conv_slow[];
      double block[];
      double k_fast[3] = {0.58, 0.27, 0.15};
      double k_slow[5] = {0.34, 0.24, 0.18, 0.14, 0.10};
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_TRANSFORMER, SequenceContextSpan());
      dims.stride = MathMax(dims.stride, 2);
      dims.patch_size = MathMax(dims.patch_size, 2);
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildSequenceBlockSummaries(x, dims, seq_cfg, k_fast, 3, k_slow, 5, attn, conv_fast, conv_slow, block);

      xa[1] = FXAI_ClipSym(0.48 * xa[1] + 0.22 * mean1 + 0.14 * trend1 + 0.08 * attn[1] + 0.08 * block[1], 8.0);
      xa[2] = FXAI_ClipSym(0.48 * xa[2] + 0.22 * mean2 + 0.14 * trend2 + 0.08 * attn[2] + 0.08 * block[2], 8.0);
      xa[4] = FXAI_ClipSym(0.58 * xa[4] + 0.20 * mean4 + 0.08 * conv_slow[4] + 0.14 * block[4], 8.0);
      xa[5] = FXAI_ClipSym(0.60 * xa[5] + 0.18 * var1 + 0.10 * MathAbs(conv_fast[5]) + 0.12 * MathAbs(block[5]), 8.0);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(FXAI_TFM_SEQ, 72);
   }

   void ResetFeatureStats(void)
   {
      m_feat_stats_ready = false;
      m_feat_stats_steps = 0;
      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         m_feat_mean[f] = 0.0;
         m_feat_var[f] = 1.0;
         m_codebook_gate[f] = 1.0;
      }
   }

   void UpdateFeatureStats(const double &xn[])
   {
      double a = (m_feat_stats_steps < 192 ? 0.040 : 0.012);
      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         double v = xn[f + 1];
         double d = v - m_feat_mean[f];
         m_feat_mean[f] += a * d;
         double dv = v - m_feat_mean[f];
         m_feat_var[f] = (1.0 - a) * m_feat_var[f] + a * dv * dv;
         if(m_feat_var[f] < 1e-6) m_feat_var[f] = 1e-6;
      }
      m_feat_stats_steps++;
      if(m_feat_stats_steps >= 48) m_feat_stats_ready = true;
   }

   int QuantizeFeatureValue(const int f, const double v) const
   {
      if(f < 0 || f >= FXAI_AI_FEATURES) return 0;
      double mu = m_feat_mean[f];
      double sd = MathSqrt(m_feat_var[f] + 1e-6);
      if(sd < 1e-6) sd = 1e-6;
      double z = (v - mu) / sd;
      z = FXAI_Clamp(z, -4.0, 4.0);
      double u = (z + 4.0) / 8.0;
      int b = (int)MathFloor(u * (double)FXAI_TFM_VALUE_BINS);
      if(b < 0) b = 0;
      if(b >= FXAI_TFM_VALUE_BINS) b = FXAI_TFM_VALUE_BINS - 1;
      return b;
   }

   int CodebookIndex(const int feature, const int bin) const
   {
      int f = ClampI(feature, 0, FXAI_AI_FEATURES - 1);
      int b = ClampI(bin, 0, FXAI_TFM_VALUE_BINS - 1);
      int idx = (f * FXAI_TFM_VALUE_BINS + b) % FXAI_TFM_CODEBOOK;
      return idx;
   }

   void BuildCodebookPatchEmbedding(const double &patch_mean[],
                                    double &emb[]) const
   {
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         emb[d] = 0.0;

      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         int b = QuantizeFeatureValue(f, patch_mean[f]);
         int cb = CodebookIndex(f, b);
         double g = FXAI_Clamp(m_codebook_gate[f], 0.10, 4.00);
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            emb[d] += g * m_codebook[cb][d];
      }

      double inv = 1.0 / (double)FXAI_AI_FEATURES;
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         emb[d] *= inv;
   }

   void ApplyMemoryRetrieval(const double &rep_in[],
                             double &rep_out[],
                             double &mem_attn[]) const
   {
      double q[FXAI_TFM_D_MODEL];
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         double s = 0.0;
         for(int j=0; j<FXAI_TFM_D_MODEL; j++)
            s += m_w_mem_q[d][j] * rep_in[j];
         q[d] = s;
      }

      double score[FXAI_TFM_MEMORY];
      double mx = -1e100;
      double inv_scale = 1.0 / MathSqrt((double)FXAI_TFM_D_MODEL);
      for(int m=0; m<FXAI_TFM_MEMORY; m++)
      {
         double s = 0.0;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            s += q[d] * m_mem_k[m][d];
         score[m] = s * inv_scale;
         if(score[m] > mx) mx = score[m];
      }

      double den = 0.0;
      for(int m=0; m<FXAI_TFM_MEMORY; m++)
      {
         score[m] = MathExp(FXAI_Clamp(score[m] - mx, -30.0, 30.0));
         den += score[m];
      }
      if(den <= 0.0) den = 1.0;

      double ctx[FXAI_TFM_D_MODEL];
      for(int d=0; d<FXAI_TFM_D_MODEL; d++) ctx[d] = 0.0;

      for(int m=0; m<FXAI_TFM_MEMORY; m++)
      {
         mem_attn[m] = score[m] / den;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            ctx[d] += mem_attn[m] * m_mem_v[m][d];
      }

      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         double gate = FXAI_Sigmoid(m_w_mem_gate[d] * rep_in[d] + m_b_mem_gate);
         rep_out[d] = FXAI_ClipSym(rep_in[d] + gate * ctx[d], 8.0);
      }
   }

   void TokenHead(const double &rep[],
                  double &tok_prob[],
                  int &top_idx) const
   {
      double logits[FXAI_TFM_CODEBOOK];
      double mx = -1e100;
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++)
      {
         double z = m_b_tok[t];
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            z += m_w_tok[t][d] * rep[d];
         logits[t] = z;
         if(z > mx) mx = z;
      }

      double den = 0.0;
      top_idx = 0;
      double best = -1.0;
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++)
      {
         tok_prob[t] = MathExp(FXAI_Clamp(logits[t] - mx, -30.0, 30.0));
         den += tok_prob[t];
      }
      if(den <= 0.0) den = 1.0;
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++)
      {
         tok_prob[t] /= den;
         if(tok_prob[t] > best)
         {
            best = tok_prob[t];
            top_idx = t;
         }
      }
   }

   void ResetSequence(void)
   {
      m_seq_ptr = -1;
      m_seq_len = 0;
      for(int t=0; t<FXAI_TFM_SEQ; t++)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
            m_seq[t][f] = 0.0;
      }
   }

   void PushSequence(const double &xn[])
   {
      // Deduplicate: avoid pushing the same normalized observation twice.
      if(m_seq_len > 0 && m_seq_ptr >= 0)
      {
         bool same = true;
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            if(MathAbs(m_seq[m_seq_ptr][f] - xn[f + 1]) > 1e-9)
            {
               same = false;
               break;
            }
         }
         if(same)
            return;
      }

      int next = m_seq_ptr + 1;
      if(next >= FXAI_TFM_SEQ) next = 0;

      for(int f=0; f<FXAI_AI_FEATURES; f++)
         m_seq[next][f] = xn[f + 1];

      m_seq_ptr = next;
      if(m_seq_len < FXAI_TFM_SEQ) m_seq_len++;
   }

   void BuildTemporalMatrix(const double &xn[],
                            double &seq_out[][FXAI_AI_FEATURES],
                            int &out_len) const
   {
      out_len = 0;
      if(m_seq_len > 0 && m_seq_ptr >= 0)
      {
         int start = m_seq_ptr - (m_seq_len - 1);
         while(start < 0) start += FXAI_TFM_SEQ;

         for(int i=0; i<m_seq_len; i++)
         {
            int idx = start + i;
            while(idx >= FXAI_TFM_SEQ) idx -= FXAI_TFM_SEQ;
            for(int f=0; f<FXAI_AI_FEATURES; f++)
               seq_out[out_len][f] = m_seq[idx][f];
            out_len++;
         }
      }

      // Append current normalized observation unless it's already the newest sequence element.
bool append_cur = true;
if(out_len > 0)
{
   append_cur = false;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      if(MathAbs(seq_out[out_len - 1][f] - xn[f + 1]) > 1e-9)
      {
         append_cur = true;
         break;
      }
   }
}

if(append_cur)
{
   if(out_len < FXAI_TFM_SEQ)
   {
      for(int f=0; f<FXAI_AI_FEATURES; f++)
         seq_out[out_len][f] = xn[f + 1];
      out_len++;
   }
   else
   {
      // Overwrite the newest slot if full.
      for(int i=0; i<FXAI_TFM_SEQ - 1; i++)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
            seq_out[i][f] = seq_out[i + 1][f];
      }
      for(int f=0; f<FXAI_AI_FEATURES; f++)
         seq_out[FXAI_TFM_SEQ - 1][f] = xn[f + 1];
      out_len = FXAI_TFM_SEQ;
   }
}
   }

   void BuildPatchTokens(const double &seq_mat[][FXAI_AI_FEATURES],
                         const int seq_len,
                         double &tokens[][FXAI_TFM_D_MODEL],
                         int &token_count,
                         double &patch_stat[][FXAI_TFM_PATCH_LEN],
                         double &token_hist[],
                         int &last_token_target) const
   {
      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         for(int t=0; t<FXAI_TFM_PATCH_LEN; t++)
            patch_stat[f][t] = 0.0;
      }
      for(int i=0; i<FXAI_TFM_CODEBOOK; i++)
         token_hist[i] = 0.0;
      last_token_target = 0;

      int count = 1;
      if(seq_len >= FXAI_TFM_PATCH_LEN)
         count = 1 + (seq_len - FXAI_TFM_PATCH_LEN) / FXAI_TFM_STRIDE;
      if(count < 1) count = 1;
      if(count > FXAI_TFM_MAX_PATCHES) count = FXAI_TFM_MAX_PATCHES;
      token_count = count;

      for(int p=0; p<count; p++)
      {
         int start;
         if(seq_len >= FXAI_TFM_PATCH_LEN)
         {
            int base_start = seq_len - FXAI_TFM_PATCH_LEN - (count - 1 - p) * FXAI_TFM_STRIDE;
            start = ClampI(base_start, 0, MathMax(seq_len - FXAI_TFM_PATCH_LEN, 0));
         }
         else
         {
            start = 0;
         }

         double pmean[FXAI_AI_FEATURES];
         for(int f=0; f<FXAI_AI_FEATURES; f++) pmean[f] = 0.0;
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            for(int t=0; t<FXAI_TFM_PATCH_LEN; t++)
            {
               int idx = start + t;
               if(idx >= seq_len) idx = seq_len - 1;
               if(idx < 0) idx = 0;
               double xv = seq_mat[idx][f];
               pmean[f] += xv;
               patch_stat[f][t] += xv;
            }
            pmean[f] /= (double)FXAI_TFM_PATCH_LEN;
         }

         double cb[FXAI_TFM_D_MODEL];
         BuildCodebookPatchEmbedding(pmean, cb);

         int tok_primary = CodebookIndex(0, QuantizeFeatureValue(0, pmean[0]));
         token_hist[tok_primary] += 1.0;
         if(p == count - 1) last_token_target = tok_primary;

         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double z = m_b_patch[d];
            for(int f=0; f<FXAI_AI_FEATURES; f++)
            {
               double g = FXAI_Clamp(m_ch_gate[f], 0.10, 4.00);
               for(int t=0; t<FXAI_TFM_PATCH_LEN; t++)
               {
                  int idx = start + t;
                  if(idx >= seq_len) idx = seq_len - 1;
                  if(idx < 0) idx = 0;
                  double xv = seq_mat[idx][f];
                  z += m_w_patch[d][f][t] * (g * xv);
               }
            }
            tokens[p][d] = FXAI_ClipSym(FXAI_Tanh(z + cb[d]) + m_pos[p][d], 8.0);
         }
      }

      double inv = 1.0 / (double)MathMax(count, 1);
      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         for(int t=0; t<FXAI_TFM_PATCH_LEN; t++)
            patch_stat[f][t] *= inv;
      }
   }

   void AttentionLayer(const int layer,
                       const double &in_tokens[][FXAI_TFM_D_MODEL],
                       const int n_tokens,
                       double &out_tokens[][FXAI_TFM_D_MODEL],
                       double &mean_in[],
                       double &mean_ctx[],
                       double &mean_ff[])
   {
      double ln1g[FXAI_TFM_D_MODEL];
      double ln1b[FXAI_TFM_D_MODEL];
      double ln2g[FXAI_TFM_D_MODEL];
      double ln2b[FXAI_TFM_D_MODEL];
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         ln1g[d] = m_ln1_g[layer][d];
         ln1b[d] = m_ln1_b[layer][d];
         ln2g[d] = m_ln2_g[layer][d];
         ln2b[d] = m_ln2_b[layer][d];
      }

      double inv_tok = 1.0 / (double)MathMax(n_tokens, 1);
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         mean_in[d] = 0.0;
         mean_ctx[d] = 0.0;
      }
      for(int r=0; r<FXAI_TFM_D_FF; r++)
         mean_ff[r] = 0.0;

      for(int i=0; i<n_tokens; i++)
      {
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            mean_in[d] += in_tokens[i][d] * inv_tok;
      }

      double q[FXAI_TFM_D_HEAD];
double score[FXAI_TFM_MAX_PATCHES];
double ctx_head[FXAI_TFM_HEADS][FXAI_TFM_D_HEAD];

// Cache K/V projections once per (head, token) to avoid redundant O(n_tokens^2) work.
double Kcache[FXAI_TFM_HEADS][FXAI_TFM_MAX_PATCHES][FXAI_TFM_D_HEAD];
double Vcache[FXAI_TFM_HEADS][FXAI_TFM_MAX_PATCHES][FXAI_TFM_D_HEAD];

double inv_scale = 1.0 / MathSqrt((double)FXAI_TFM_D_HEAD);

for(int h=0; h<FXAI_TFM_HEADS; h++)
{
   for(int t=0; t<n_tokens; t++)
   {
      for(int d=0; d<FXAI_TFM_D_HEAD; d++)
      {
         double ks = 0.0;
         double vs = 0.0;
         for(int j=0; j<FXAI_TFM_D_MODEL; j++)
         {
            double xj = in_tokens[t][j];
            ks += m_wk[layer][h][d][j] * xj;
            vs += m_wv[layer][h][d][j] * xj;
         }
         Kcache[h][t][d] = ks;
         Vcache[h][t][d] = vs;
      }
   }
}

for(int i=0; i<n_tokens; i++)
{
   for(int h=0; h<FXAI_TFM_HEADS; h++)
   {
      for(int d=0; d<FXAI_TFM_D_HEAD; d++)
      {
         double s = 0.0;
         for(int j=0; j<FXAI_TFM_D_MODEL; j++)
            s += m_wq[layer][h][d][j] * in_tokens[i][j];
         q[d] = s;
      }

      double mx = -1e100;
      for(int t=0; t<n_tokens; t++)
      {
         double sc = 0.0;
         for(int d=0; d<FXAI_TFM_D_HEAD; d++)
            sc += q[d] * Kcache[h][t][d];
         score[t] = sc * inv_scale;
         if(score[t] > mx) mx = score[t];
      }

      double den = 0.0;
      for(int t=0; t<n_tokens; t++)
      {
         score[t] = MathExp(FXAI_Clamp(score[t] - mx, -30.0, 30.0));
         den += score[t];
      }
      if(den <= 0.0) den = 1.0;

      for(int d=0; d<FXAI_TFM_D_HEAD; d++)
         ctx_head[h][d] = 0.0;

      for(int t=0; t<n_tokens; t++)
      {
         double a = score[t] / den;
         for(int d=0; d<FXAI_TFM_D_HEAD; d++)
            ctx_head[h][d] += a * Vcache[h][t][d];
      }
   }

double att[FXAI_TFM_D_MODEL];
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double s = 0.0;
            for(int h=0; h<FXAI_TFM_HEADS; h++)
            {
               for(int hd=0; hd<FXAI_TFM_D_HEAD; hd++)
               {
                  int od = h * FXAI_TFM_D_HEAD + hd;
                  s += m_wo[layer][d][od] * ctx_head[h][hd];
               }
            }
            att[d] = s;
         }

         double u[FXAI_TFM_D_MODEL];
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            u[d] = in_tokens[i][d] + att[d];
         LayerNormAffine(u, ln1g, ln1b);

         double ff1[FXAI_TFM_D_FF];
         for(int r=0; r<FXAI_TFM_D_FF; r++)
         {
            double z = m_bff1[layer][r];
            for(int d=0; d<FXAI_TFM_D_MODEL; d++)
               z += m_wff1[layer][r][d] * u[d];
            ff1[r] = GELU(z);
            mean_ff[r] += ff1[r] * inv_tok;
         }

         double v2[FXAI_TFM_D_MODEL];
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double z = m_bff2[layer][d];
            for(int r=0; r<FXAI_TFM_D_FF; r++)
               z += m_wff2[layer][d][r] * ff1[r];
            v2[d] = u[d] + z;
         }
         LayerNormAffine(v2, ln2g, ln2b);

         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            out_tokens[i][d] = v2[d];
            mean_ctx[d] += att[d] * inv_tok;
         }
      }
   }

   void PoolRepresentation(const double &tokens[][FXAI_TFM_D_MODEL],
                           const int n_tokens,
                           double &rep[]) const
   {
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         rep[d] = 0.0;

      if(n_tokens <= 0)
         return;

      for(int t=0; t<n_tokens; t++)
      {
         double w = 1.0;
         if(t == n_tokens - 1) w = 1.75;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            rep[d] += w * tokens[t][d];
      }

      double den = (double)n_tokens + 0.75;
      if(den <= 0.0) den = 1.0;
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         rep[d] = FXAI_ClipSym(rep[d] / den, 8.0);
   }

   void ComputeHeads(const double &rep[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75,
                     double &mu_h[]) const
   {
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            z += m_w_cls[c][d] * rep[d];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      q25 = m_b_q25;
      q75 = m_b_q75;
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         mu += m_w_mu[d] * rep[d];
         logv += m_w_logv[d] * rep[d];
         q25 += m_w_q25[d] * rep[d];
         q75 += m_w_q75[d] * rep[d];
      }

      for(int h=0; h<FXAI_TFM_HORIZONS; h++)
      {
         double z = m_b_mu_h[h];
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            z += m_w_mu_h[h][d] * rep[d];
         mu_h[h] = z;
      }

      logv = FXAI_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double t = q25;
         q25 = q75;
         q75 = t;
      }
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FXAI_Clamp(m_cal_temp, 0.50, 3.00);
      double logits[FXAI_TFM_CLASS_COUNT];
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
      {
         double pr = FXAI_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_TFM_CLASS_COUNT];
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_TFM_CAL_BINS; b++) total += m_cal_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_TFM_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_TFM_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal_iso_cnt[c][b] > 1e-9)
               r = m_cal_iso_pos[c][b] / m_cal_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_TFM_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_TFM_CAL_BINS) bi = FXAI_TFM_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
         p_cal[c] = FXAI_Clamp(0.75 * p_cal[c] + 0.25 * p_iso[c], 0.0005, 0.9990);

      double s = p_cal[0] + p_cal[1] + p_cal[2];
      if(s <= 0.0) s = 1.0;
      p_cal[0] /= s;
      p_cal[1] /= s;
      p_cal[2] /= s;
   }

   void UpdateCalibrator3(const double &p_raw[],
                          const int cls,
                          const double sample_w,
                          const double lr)
   {
      double inv_temp = 1.0 / FXAI_Clamp(m_cal_temp, 0.50, 3.00);
      double logits[FXAI_TFM_CLASS_COUNT];
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
      {
         double pr = FXAI_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal_bias[c];
      }

      double p_cal[FXAI_TFM_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FXAI_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FXAI_Clamp(0.20 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_bias[c] = FXAI_ClipSym(m_cal_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_TFM_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_TFM_CAL_BINS) bi = FXAI_TFM_CAL_BINS - 1;
         m_cal_iso_cnt[c][bi] += w;
         m_cal_iso_pos[c][bi] += w * target;
      }

      m_cal_temp = FXAI_Clamp(m_cal_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
      m_cal3_steps++;
   }

   void InitWeights(void)
   {
      ResetSequence();
      ResetInputNorm();
      ResetFeatureStats();
      m_step = 0;
      m_adam_t = 0;
      m_obs_step = 0;
      m_last_m1_train_bar = 0;
      m_mem_ptr = 0;
      m_b_mem_gate = 0.0;

      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
      {
         m_cls_ema[c] = 1.0;
         m_cal_bias[c] = 0.0;
         m_b_cls[c] = 0.0;
         for(int b=0; b<FXAI_TFM_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      m_cal_temp = 1.0;
      m_cal3_steps = 0;

      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      for(int f=0; f<FXAI_AI_FEATURES; f++)
         m_ch_gate[f] = 1.0;

      for(int cb=0; cb<FXAI_TFM_CODEBOOK; cb++)
      {
         m_codebook_usage[cb] = 1.0;
         m_b_tok[cb] = 0.0;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double s = (double)((cb + 1) * (d + 2));
            m_codebook[cb][d] = 0.02 * MathSin(0.41 * s);
            m_w_tok[cb][d] = 0.02 * MathCos(0.37 * s);
         }
      }

      for(int m=0; m<FXAI_TFM_MEMORY; m++)
      {
         m_mem_usage[m] = 1.0;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            double s = (double)((m + 2) * (d + 3));
            m_mem_k[m][d] = 0.03 * MathSin(0.29 * s);
            m_mem_v[m][d] = 0.03 * MathCos(0.31 * s);
         }
      }

      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         m_w_mem_gate[d] = 0.10 * MathCos((double)(d + 1) * 0.71);
         for(int j=0; j<FXAI_TFM_D_MODEL; j++)
         {
            double s = (double)((d + 2) * (j + 3));
            m_w_mem_q[d][j] = 0.03 * MathSin(0.33 * s);
         }
      }

      for(int p=0; p<FXAI_TFM_MAX_PATCHES; p++)
      {
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            m_pos[p][d] = 0.015 * MathSin((double)((p + 1) * (d + 2)) * 0.43);
         }
      }

      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
      {
         m_b_patch[d] = 0.0;
         m_w_mu[d] = 0.03 * MathSin((double)(d + 2) * 0.83);
         m_w_logv[d] = 0.03 * MathCos((double)(d + 3) * 0.89);
         m_w_q25[d] = 0.03 * MathSin((double)(d + 4) * 0.97);
         m_w_q75[d] = 0.03 * MathCos((double)(d + 5) * 1.03);

         for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
            m_w_cls[c][d] = 0.03 * MathSin((double)((c + 2) * (d + 1)) * 0.79);

         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            for(int t=0; t<FXAI_TFM_PATCH_LEN; t++)
            {
               double s = (double)((d + 1) * (f + 2) * (t + 3));
               m_w_patch[d][f][t] = 0.025 * MathSin(0.61 * s);
            }
         }
      }

      for(int l=0; l<FXAI_TFM_LAYERS; l++)
      {
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            m_bff2[l][d] = 0.0;
            m_ln1_g[l][d] = 1.0;
            m_ln1_b[l][d] = 0.0;
            m_ln2_g[l][d] = 1.0;
            m_ln2_b[l][d] = 0.0;

            for(int od=0; od<FXAI_TFM_D_MODEL; od++)
            {
               double s = (double)((l + 1) * (d + 2) * (od + 3));
               m_wo[l][d][od] = 0.02 * MathCos(0.53 * s);
            }

            for(int r=0; r<FXAI_TFM_D_FF; r++)
            {
               double s2 = (double)((l + 1) * (d + 1) * (r + 2));
               m_wff2[l][d][r] = 0.02 * MathSin(0.57 * s2);
            }
         }

         for(int r=0; r<FXAI_TFM_D_FF; r++)
         {
            m_bff1[l][r] = 0.0;
            for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            {
               double s = (double)((l + 2) * (r + 1) * (d + 3));
               m_wff1[l][r][d] = 0.02 * MathCos(0.59 * s);
            }
         }

         for(int h=0; h<FXAI_TFM_HEADS; h++)
         {
            for(int dh=0; dh<FXAI_TFM_D_HEAD; dh++)
            {
               for(int d=0; d<FXAI_TFM_D_MODEL; d++)
               {
                  double s = (double)((l + 1) * (h + 2) * (dh + 3) * (d + 1));
                  m_wq[l][h][dh][d] = 0.02 * MathSin(0.47 * s);
                  m_wk[l][h][dh][d] = 0.02 * MathCos(0.49 * s);
                  m_wv[l][h][dh][d] = 0.02 * MathSin(0.51 * s);
               }
            }
         }
      }

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;
      for(int h=0; h<FXAI_TFM_HORIZONS; h++)
      {
         m_b_mu_h[h] = 0.0;
         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
            m_w_mu_h[h][d] = 0.02 * MathSin((double)((h + 2) * (d + 1)) * 0.67);
      }

      // Slight skip prior before calibration settles.
      m_b_cls[(int)FXAI_LABEL_SKIP] = 0.20;

      m_initialized = true;
   }

   void ForwardPass(const double &x[],
                    const bool commit,
                    double &rep[],
                    double &p_raw[],
                    double &mu,
                    double &logv,
                    double &q25,
                    double &q75,
                    double &mu_h[],
                    double &patch_stat[][FXAI_TFM_PATCH_LEN],
                    double &token_hist[],
                    int &token_target,
                    double &layer_in_mean[][FXAI_TFM_D_MODEL],
                    double &layer_ctx_mean[][FXAI_TFM_D_MODEL],
                    double &layer_ff_mean[][FXAI_TFM_D_FF],
                    double &mem_attn[],
                    int &token_count)
   {
      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(x, xn);
      UpdateFeatureStats(xn);

      double seq_mat[FXAI_TFM_SEQ][FXAI_AI_FEATURES];
      int seq_len = 0;
      BuildTemporalMatrix(xn, seq_mat, seq_len);

      double tokens_a[FXAI_TFM_MAX_PATCHES][FXAI_TFM_D_MODEL];
      double tokens_b[FXAI_TFM_MAX_PATCHES][FXAI_TFM_D_MODEL];
      BuildPatchTokens(seq_mat, seq_len, tokens_a, token_count, patch_stat, token_hist, token_target);

      if(token_count < 1) token_count = 1;
      if(token_count > FXAI_TFM_MAX_PATCHES) token_count = FXAI_TFM_MAX_PATCHES;

      for(int l=0; l<FXAI_TFM_LAYERS; l++)
      {
         double mean_in_l[FXAI_TFM_D_MODEL];
         double mean_ctx_l[FXAI_TFM_D_MODEL];
         double mean_ff_l[FXAI_TFM_D_FF];

         AttentionLayer(l,
                        tokens_a,
                        token_count,
                        tokens_b,
                        mean_in_l,
                        mean_ctx_l,
                        mean_ff_l);

         for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         {
            layer_in_mean[l][d] = mean_in_l[d];
            layer_ctx_mean[l][d] = mean_ctx_l[d];
         }
         for(int r=0; r<FXAI_TFM_D_FF; r++)
            layer_ff_mean[l][r] = mean_ff_l[r];

         for(int t=0; t<token_count; t++)
         {
            for(int d=0; d<FXAI_TFM_D_MODEL; d++)
               tokens_a[t][d] = tokens_b[t][d];
         }
      }

      PoolRepresentation(tokens_a, token_count, rep);
      double rep_mem[FXAI_TFM_D_MODEL];
      ApplyMemoryRetrieval(rep, rep_mem, mem_attn);
      for(int d=0; d<FXAI_TFM_D_MODEL; d++)
         rep[d] = rep_mem[d];

      double logits[FXAI_TFM_CLASS_COUNT];
      ComputeHeads(rep, logits, p_raw, mu, logv, q25, q75, mu_h);

      if(commit)
         PushSequence(xn);
   }

   double ExpectedMoveFromHeads(const double mu,
                                const double logv,
                                const double q25,
                                const double q75,
                                const double &mu_h[],
                                const double skip_prob) const
   {
      double sigma = MathExp(0.5 * FXAI_Clamp(logv, -4.0, 4.0));
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double h_ev = 0.0;
      for(int h=0; h<FXAI_TFM_HORIZONS; h++)
         h_ev += MathAbs(mu_h[h]) * (h == 0 ? 0.50 : (h == 1 ? 0.30 : 0.20));
      double ev = (0.52 * MathAbs(mu) + 0.26 * h_ev + 0.14 * sigma + 0.08 * iqr) *
                  FXAI_Clamp(1.0 - skip_prob, 0.0, 1.0);
      return ev;
   }
