   int ClampI(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
   }

   double QuantileLevel(const int qi) const
   {
      static const double qv[FXAI_CHR_QUANTILES] = {0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95};
      int q = ClampI(qi, 0, FXAI_CHR_QUANTILES - 1);
      return qv[q];
   }

   int SessionBucket(const datetime t) const
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      int h = dt.hour;
      if(h < 8) return 0;
      if(h < 13) return 1;
      if(h < 20) return 2;
      return 3;
   }

   int RegimeBucket(void) const
   {
      if(!m_move_ready) return 0;
      return (m_move_ema_abs > 5.0 ? 1 : 0);
   }

   void BuildCalLogits(const double &p_raw[], const int sess, const int reg, double &logits[]) const
   {
      double lraw[FXAI_CHR_CLASS_COUNT];
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         double z = m_cal_vs_b[c] + m_cal_session_b[sess][c] + m_cal_regime_b[reg][c];
         for(int j=0; j<FXAI_CHR_CLASS_COUNT; j++)
            z += m_cal_vs_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

   void ReplayPush(const int cls,
                   const double &x[],
                   const double move_points,
                   const double cost_points,
                   const double sample_w)
   {
      int p = m_chr_replay_head;
      m_replay_pos[p] = (m_seq_ptr >= 0 ? m_seq_ptr : 0);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_chr_replay_x[p][k] = x[k];
      m_replay_cls[p] = cls;
      m_chr_replay_move[p] = move_points;
      m_chr_replay_cost[p] = cost_points;
      m_replay_w[p] = sample_w;
      m_chr_replay_time[p] = ResolveContextTime();
      m_chr_replay_head++;
      if(m_chr_replay_head >= FXAI_CHR_REPLAY) m_chr_replay_head = 0;
      if(m_chr_replay_size < FXAI_CHR_REPLAY) m_chr_replay_size++;
   }

   int ReplaySampleSlot(void)
   {
      if(m_chr_replay_size <= 0) return -1;
      double u = PluginRand01();
      int age = (int)MathFloor(u * (double)m_chr_replay_size);
      if(age < 0) age = 0;
      if(age >= m_chr_replay_size) age = m_chr_replay_size - 1;

      int slot = m_chr_replay_head - 1 - age;
      while(slot < 0) slot += FXAI_CHR_REPLAY;
      while(slot >= FXAI_CHR_REPLAY) slot -= FXAI_CHR_REPLAY;
      return slot;
   }

   void UpdateTeacherHeads(void)
   {
      const double a = 0.995;
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         m_t_b_cls[c] = a * m_t_b_cls[c] + (1.0 - a) * m_b_cls[c];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            m_t_w_cls[c][d] = a * m_t_w_cls[c][d] + (1.0 - a) * m_w_cls[c][d];
      }
   }

   void UpdateValidationMetrics(const int cls,
                                const double &p_cal[],
                                const double expected_move_points,
                                const double cost_points)
   {
      int y = ClampI(cls, 0, FXAI_CHR_CLASS_COUNT - 1);
      double ce = -MathLog(FXAI_Clamp(p_cal[y], 1e-6, 1.0));
      double brier = 0.0;
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         double t = (c == y ? 1.0 : 0.0);
         double d = p_cal[c] - t;
         brier += d * d;
      }
      brier /= 3.0;

      double conf = p_cal[0];
      int pred = 0;
      for(int c=1; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         if(p_cal[c] > conf) { conf = p_cal[c]; pred = c; }
      }
      double acc = (pred == y ? 1.0 : 0.0);
      int bi = (int)MathFloor(conf * (double)FXAI_CHR_ECE_BINS);
      if(bi < 0) bi = 0;
      if(bi >= FXAI_CHR_ECE_BINS) bi = FXAI_CHR_ECE_BINS - 1;
      for(int b=0; b<FXAI_CHR_ECE_BINS; b++)
      {
         m_ece_mass[b] *= 0.997;
         m_ece_acc[b] *= 0.997;
         m_ece_conf[b] *= 0.997;
      }
      m_ece_mass[bi] += 1.0;
      m_ece_acc[bi] += acc;
      m_ece_conf[bi] += conf;
      double ece_num = 0.0, ece_den = 0.0;
      for(int b=0; b<FXAI_CHR_ECE_BINS; b++)
      {
         if(m_ece_mass[b] <= 1e-9) continue;
         double ba = m_ece_acc[b] / m_ece_mass[b];
         double bc = m_ece_conf[b] / m_ece_mass[b];
         ece_num += m_ece_mass[b] * MathAbs(ba - bc);
         ece_den += m_ece_mass[b];
      }
      double ece = (ece_den > 0.0 ? ece_num / ece_den : 0.0);

      double ev_after_cost = expected_move_points - MathMax(cost_points, 0.0);
      if(!m_val_ready)
      {
         m_val_nll_fast = m_val_nll_slow = ce;
         m_val_brier_fast = m_val_brier_slow = brier;
         m_val_ece_fast = m_val_ece_slow = ece;
         m_val_ev_after_cost_fast = m_val_ev_after_cost_slow = ev_after_cost;
         m_val_ready = true;
      }
      else
      {
         m_val_nll_fast = 0.92 * m_val_nll_fast + 0.08 * ce;
         m_val_nll_slow = 0.995 * m_val_nll_slow + 0.005 * ce;
         m_val_brier_fast = 0.92 * m_val_brier_fast + 0.08 * brier;
         m_val_brier_slow = 0.995 * m_val_brier_slow + 0.005 * brier;
         m_val_ece_fast = 0.92 * m_val_ece_fast + 0.08 * ece;
         m_val_ece_slow = 0.995 * m_val_ece_slow + 0.005 * ece;
         m_val_ev_after_cost_fast = 0.92 * m_val_ev_after_cost_fast + 0.08 * ev_after_cost;
         m_val_ev_after_cost_slow = 0.995 * m_val_ev_after_cost_slow + 0.005 * ev_after_cost;
      }
      m_val_steps++;

      m_reference_ready =
         (m_val_steps >= 256 &&
          m_val_nll_fast <= 0.95 &&
          m_val_brier_fast <= 0.22 &&
          m_val_ece_fast <= 0.12 &&
          m_val_ev_after_cost_fast > 0.0 &&
          m_val_nll_fast <= 1.25 * MathMax(0.10, m_val_nll_slow));
   }

   datetime CurrentM1BarOpenTime(void) const
   {
      datetime ctx_time = ResolveContextTime();
      if(ctx_time > 0)
         return (datetime)(ctx_time - (ctx_time % 60));

      datetime bt = 0;
      if(!FXAI_MarketDataBarTime(Symbol(), PERIOD_M1, 0, bt))
         bt = 0;
      if(bt <= 0)
      {
         datetime tc = TimeCurrent();
         bt = (datetime)(tc - (tc % 60));
      }
      return bt;
   }

   bool ShouldTrainOnNewM1Bar(void)
   {
      datetime bt = CurrentM1BarOpenTime();
      if(bt == m_last_m1_train_bar)
         return false;
      m_last_m1_train_bar = bt;
      return true;
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
      FXAI_ModuleLayerNormAffine(v, FXAI_CHR_D_MODEL, g, b);
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
      int g = ClampI(group_idx, 0, 15);
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

      xa[1] = FXAI_ClipSym(0.44 * xa[1] + 0.24 * mean1 + 0.14 * trend1 + 0.08 * attn[1] + 0.10 * block[1], 8.0);
      xa[2] = FXAI_ClipSym(0.44 * xa[2] + 0.24 * mean2 + 0.14 * trend2 + 0.08 * attn[2] + 0.10 * block[2], 8.0);
      xa[4] = FXAI_ClipSym(0.58 * xa[4] + 0.20 * mean4 + 0.10 * conv_slow[4] + 0.12 * block[4], 8.0);
      xa[5] = FXAI_ClipSym(0.56 * xa[5] + 0.20 * var1 + 0.10 * MathAbs(conv_fast[5]) + 0.14 * MathAbs(block[5]), 8.0);
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
      int b = (int)MathFloor(u * (double)FXAI_CHR_VALUE_BINS);
      if(b < 0) b = 0;
      if(b >= FXAI_CHR_VALUE_BINS) b = FXAI_CHR_VALUE_BINS - 1;
      return b;
   }

   int CodebookIndex(const int feature, const int bin) const
   {
      int f = ClampI(feature, 0, FXAI_AI_FEATURES - 1);
      int b = ClampI(bin, 0, FXAI_CHR_VALUE_BINS - 1);
      int idx = (f * FXAI_CHR_VALUE_BINS + b) % FXAI_CHR_CODEBOOK;
      return idx;
   }

   int BuildFutureTokenTarget(const double &xn[],
                              const int label_cls,
                              const double move_points) const
   {
      int level = QuantizeFeatureValue(0, xn[1]);
      double mv = FXAI_Clamp(move_points, -40.0, 40.0);
      double mvu = (mv + 40.0) / 80.0;
      int delta = (int)MathFloor(mvu * (double)FXAI_CHR_VALUE_BINS);
      if(delta < 0) delta = 0;
      if(delta >= FXAI_CHR_VALUE_BINS) delta = FXAI_CHR_VALUE_BINS - 1;

      double vol_src = MathAbs(xn[4]) + 0.5 * MathAbs(xn[5]);
      int vol = QuantizeFeatureValue(3, vol_src);
      int reg = ClampI(label_cls, 0, FXAI_CHR_CLASS_COUNT - 1);

      int idx = (((level * FXAI_CHR_VALUE_BINS + delta) * FXAI_CHR_VALUE_BINS + vol) * FXAI_CHR_CLASS_COUNT + reg) % FXAI_CHR_CODEBOOK;
      if(idx < 0) idx += FXAI_CHR_CODEBOOK;
      return idx;
   }

   void BuildCodebookPatchEmbedding(const double &patch_mean[],
                                    double &emb[]) const
   {
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         emb[d] = 0.0;

      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         int b = QuantizeFeatureValue(f, patch_mean[f]);
         int cb = CodebookIndex(f, b);
         double g = FXAI_Clamp(m_codebook_gate[f], 0.10, 4.00);
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            emb[d] += g * m_codebook[cb][d];
      }

      double inv = 1.0 / (double)FXAI_AI_FEATURES;
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         emb[d] *= inv;
   }

   void ApplyMemoryRetrieval(const double &rep_in[],
                             double &rep_out[],
                             double &mem_attn[]) const
   {
      double q[FXAI_CHR_D_MODEL];
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         double s = 0.0;
         for(int j=0; j<FXAI_CHR_D_MODEL; j++)
            s += m_w_mem_q[d][j] * rep_in[j];
         q[d] = s;
      }

      double score[FXAI_CHR_MEMORY];
      double mx = -1e100;
      double inv_scale = 1.0 / MathSqrt((double)FXAI_CHR_D_MODEL);
      for(int m=0; m<FXAI_CHR_MEMORY; m++)
      {
         double s = 0.0;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            s += q[d] * m_mem_k[m][d];
         score[m] = s * inv_scale;
         if(score[m] > mx) mx = score[m];
      }

      double den = 0.0;
      for(int m=0; m<FXAI_CHR_MEMORY; m++)
      {
         score[m] = MathExp(FXAI_Clamp(score[m] - mx, -30.0, 30.0));
         den += score[m];
      }
      if(den <= 0.0) den = 1.0;

      double ctx[FXAI_CHR_D_MODEL];
      for(int d=0; d<FXAI_CHR_D_MODEL; d++) ctx[d] = 0.0;

      for(int m=0; m<FXAI_CHR_MEMORY; m++)
      {
         mem_attn[m] = score[m] / den;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            ctx[d] += mem_attn[m] * m_mem_v[m][d];
      }

      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         double gate = FXAI_Sigmoid(m_w_mem_gate[d] * rep_in[d] + m_b_mem_gate);
         rep_out[d] = FXAI_ClipSym(rep_in[d] + gate * ctx[d], 8.0);
      }
   }

   void TokenHead(const double &rep[],
                  double &tok_prob[],
                  int &top_idx) const
   {
      double logits[FXAI_CHR_CODEBOOK];
      double mx = -1e100;
      for(int t=0; t<FXAI_CHR_CODEBOOK; t++)
      {
         double z = m_b_tok[t];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            z += m_w_tok[t][d] * rep[d];
         logits[t] = z;
         if(z > mx) mx = z;
      }

      double den = 0.0;
      top_idx = 0;
      double best = -1.0;
      for(int t=0; t<FXAI_CHR_CODEBOOK; t++)
      {
         tok_prob[t] = MathExp(FXAI_Clamp(logits[t] - mx, -30.0, 30.0));
         den += tok_prob[t];
      }
      if(den <= 0.0) den = 1.0;
      for(int t=0; t<FXAI_CHR_CODEBOOK; t++)
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
      for(int t=0; t<FXAI_CHR_SEQ; t++)
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
            m_seq[t][f] = 0.0;
      }
   }

   void PushSequence(const double &xn[])
   {
      int next = m_seq_ptr + 1;
      if(next >= FXAI_CHR_SEQ) next = 0;

      for(int f=0; f<FXAI_AI_FEATURES; f++)
         m_seq[next][f] = xn[f + 1];

      m_seq_ptr = next;
      if(m_seq_len < FXAI_CHR_SEQ) m_seq_len++;
   }

   void BuildTemporalMatrix(const double &xn[],
                            double &seq_out[][FXAI_AI_FEATURES],
                            int &out_len,
                            const bool include_current) const
   {
      out_len = 0;
      if(m_seq_len > 0 && m_seq_ptr >= 0)
      {
         int start = m_seq_ptr - (m_seq_len - 1);
         while(start < 0) start += FXAI_CHR_SEQ;

         for(int i=0; i<m_seq_len; i++)
         {
            int idx = start + i;
            while(idx >= FXAI_CHR_SEQ) idx -= FXAI_CHR_SEQ;
            for(int f=0; f<FXAI_AI_FEATURES; f++)
               seq_out[out_len][f] = m_seq[idx][f];
            out_len++;
         }
      }

      if(!include_current)
         return;

      // Append current normalized observation for prediction step.
      if(out_len >= FXAI_CHR_SEQ)
      {
         // Overwrite the newest slot if full.
         for(int i=0; i<FXAI_CHR_SEQ - 1; i++)
         {
            for(int f=0; f<FXAI_AI_FEATURES; f++)
               seq_out[i][f] = seq_out[i + 1][f];
         }
         for(int f=0; f<FXAI_AI_FEATURES; f++)
            seq_out[FXAI_CHR_SEQ - 1][f] = xn[f + 1];
         out_len = FXAI_CHR_SEQ;
         return;
      }

      for(int f=0; f<FXAI_AI_FEATURES; f++)
         seq_out[out_len][f] = xn[f + 1];
      out_len++;
   }

   void BuildPatchTokens(const double &seq_mat[][FXAI_AI_FEATURES],
                         const int seq_len,
                         double &tokens[][FXAI_CHR_D_MODEL],
                         int &token_count,
                         double &patch_stat[][FXAI_CHR_PATCH_LEN],
                         double &token_hist[],
                         int &last_token_target) const
   {
      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
            patch_stat[f][t] = 0.0;
      }
      for(int i=0; i<FXAI_CHR_CODEBOOK; i++)
         token_hist[i] = 0.0;
      last_token_target = 0;

      int count = 1;
      if(seq_len >= FXAI_CHR_PATCH_LEN)
         count = 1 + (seq_len - FXAI_CHR_PATCH_LEN) / FXAI_CHR_STRIDE;
      if(count < 1) count = 1;
      if(count > FXAI_CHR_MAX_PATCHES) count = FXAI_CHR_MAX_PATCHES;
      token_count = count;

      for(int p=0; p<count; p++)
      {
         int start;
         if(seq_len >= FXAI_CHR_PATCH_LEN)
         {
            int base_start = seq_len - FXAI_CHR_PATCH_LEN - (count - 1 - p) * FXAI_CHR_STRIDE;
            start = ClampI(base_start, 0, MathMax(seq_len - FXAI_CHR_PATCH_LEN, 0));
         }
         else
         {
            start = 0;
         }

         double pmean[FXAI_AI_FEATURES];
         for(int f=0; f<FXAI_AI_FEATURES; f++) pmean[f] = 0.0;
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
            {
               int idx = start + t;
               if(idx >= seq_len) idx = seq_len - 1;
               if(idx < 0) idx = 0;
               double xv = seq_mat[idx][f];
               pmean[f] += xv;
               patch_stat[f][t] += xv;
            }
            pmean[f] /= (double)FXAI_CHR_PATCH_LEN;
         }

         double cb[FXAI_CHR_D_MODEL];
         BuildCodebookPatchEmbedding(pmean, cb);

         int tok_primary = CodebookIndex(0, QuantizeFeatureValue(0, pmean[0]));
         token_hist[tok_primary] += 1.0;
         if(p == count - 1) last_token_target = tok_primary;

         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double z = m_b_patch[d];
            for(int f=0; f<FXAI_AI_FEATURES; f++)
            {
               double g = FXAI_Clamp(m_ch_gate[f], 0.10, 4.00);
               for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
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
         for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
            patch_stat[f][t] *= inv;
      }
   }

   void AttentionLayer(const int layer,
                       const double &in_tokens[][FXAI_CHR_D_MODEL],
                       const int n_tokens,
                       double &out_tokens[][FXAI_CHR_D_MODEL],
                       const bool causal_mask,
                       const bool store_cache,
                       double &mean_in[],
                       double &mean_ctx[],
                       double &mean_ff[])
   {
      double ln1g[FXAI_CHR_D_MODEL];
      double ln1b[FXAI_CHR_D_MODEL];
      double ln2g[FXAI_CHR_D_MODEL];
      double ln2b[FXAI_CHR_D_MODEL];
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         ln1g[d] = m_ln1_g[layer][d];
         ln1b[d] = m_ln1_b[layer][d];
         ln2g[d] = m_ln2_g[layer][d];
         ln2b[d] = m_ln2_b[layer][d];
      }

      double inv_tok = 1.0 / (double)MathMax(n_tokens, 1);
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         mean_in[d] = 0.0;
         mean_ctx[d] = 0.0;
      }
      for(int r=0; r<FXAI_CHR_D_FF; r++)
         mean_ff[r] = 0.0;

      for(int i=0; i<n_tokens; i++)
      {
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            mean_in[d] += in_tokens[i][d] * inv_tok;
            if(store_cache) m_cache_layer_in[layer][i][d] = in_tokens[i][d];
         }
      }

      // Precompute K,V once per head/token (critical for Strategy Tester speed).
      double K[FXAI_CHR_HEADS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_HEAD];
      double Vh[FXAI_CHR_HEADS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_HEAD];

      for(int h=0; h<FXAI_CHR_HEADS; h++)
      {
         for(int t=0; t<n_tokens; t++)
         {
            for(int d=0; d<FXAI_CHR_D_HEAD; d++)
            {
               double ks = 0.0;
               double vs = 0.0;
               for(int j=0; j<FXAI_CHR_D_MODEL; j++)
               {
                  double xj = in_tokens[t][j];
                  ks += m_wk[layer][h][d][j] * xj;
                  vs += m_wv[layer][h][d][j] * xj;
               }
               K[h][t][d]  = ks;
               Vh[h][t][d] = vs;
               if(store_cache)
               {
                  m_cache_k[layer][h][t][d] = ks;
                  m_cache_v[layer][h][t][d] = vs;
               }
            }
         }
      }

      double q[FXAI_CHR_D_HEAD];
      double score[FXAI_CHR_MAX_PATCHES];
      double ctx_head[FXAI_CHR_HEADS][FXAI_CHR_D_HEAD];

      double inv_scale = 1.0 / MathSqrt((double)FXAI_CHR_D_HEAD);

      for(int i=0; i<n_tokens; i++)
      {
         for(int h=0; h<FXAI_CHR_HEADS; h++)
         {
            // Query for this token/head.
            for(int d=0; d<FXAI_CHR_D_HEAD; d++)
            {
               double s = 0.0;
               for(int j=0; j<FXAI_CHR_D_MODEL; j++)
                  s += m_wq[layer][h][d][j] * in_tokens[i][j];
               q[d] = s;
               if(store_cache) m_cache_q[layer][h][i][d] = s;
            }

            // Scores against cached keys.
            double mx = -1e100;
            int tmax = (causal_mask ? i : (n_tokens - 1));
            if(tmax < 0) tmax = 0;
            for(int t=0; t<=tmax; t++)
            {
               double sc = 0.0;
               for(int d=0; d<FXAI_CHR_D_HEAD; d++)
                  sc += q[d] * K[h][t][d];
               score[t] = sc * inv_scale;
               if(score[t] > mx) mx = score[t];
            }

            double den = 0.0;
            for(int t=0; t<=tmax; t++)
            {
               score[t] = MathExp(FXAI_Clamp(score[t] - mx, -30.0, 30.0));
               den += score[t];
            }
            if(den <= 0.0) den = 1.0;

            for(int d=0; d<FXAI_CHR_D_HEAD; d++)
               ctx_head[h][d] = 0.0;

            // Context using cached values.
            for(int t=0; t<=tmax; t++)
            {
               double a = score[t] / den;
               if(store_cache) m_cache_att[layer][h][i][t] = a;
               for(int d=0; d<FXAI_CHR_D_HEAD; d++)
               {
                  ctx_head[h][d] += a * Vh[h][t][d];
               }
            }
            if(store_cache)
            {
               for(int t=tmax+1; t<n_tokens; t++)
                  m_cache_att[layer][h][i][t] = 0.0;
            }
            for(int d=0; d<FXAI_CHR_D_HEAD; d++)
            {
               if(store_cache) m_cache_ctx[layer][h][i][d] = ctx_head[h][d];
            }
         }

         double att[FXAI_CHR_D_MODEL];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double s = 0.0;
            for(int h=0; h<FXAI_CHR_HEADS; h++)
            {
               for(int hd=0; hd<FXAI_CHR_D_HEAD; hd++)
               {
                  int od = h * FXAI_CHR_D_HEAD + hd;
                  s += m_wo[layer][d][od] * ctx_head[h][hd];
               }
            }
            att[d] = s;
         }

         double u[FXAI_CHR_D_MODEL];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            u[d] = in_tokens[i][d] + att[d];
         LayerNormAffine(u, ln1g, ln1b);
         if(store_cache)
            for(int d=0; d<FXAI_CHR_D_MODEL; d++) m_cache_u[layer][i][d] = u[d];

         double ff1[FXAI_CHR_D_FF];
         for(int r=0; r<FXAI_CHR_D_FF; r++)
         {
            double z = m_bff1[layer][r];
            for(int d=0; d<FXAI_CHR_D_MODEL; d++)
               z += m_wff1[layer][r][d] * u[d];
            ff1[r] = GELU(z);
            mean_ff[r] += ff1[r] * inv_tok;
            if(store_cache) m_cache_ff1[layer][i][r] = ff1[r];
         }

         double v2[FXAI_CHR_D_MODEL];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double z = m_bff2[layer][d];
            for(int r=0; r<FXAI_CHR_D_FF; r++)
               z += m_wff2[layer][d][r] * ff1[r];
            v2[d] = u[d] + z;
         }
         LayerNormAffine(v2, ln2g, ln2b);

         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            out_tokens[i][d] = v2[d];
            mean_ctx[d] += att[d] * inv_tok;
            if(store_cache && layer == FXAI_CHR_LAYERS - 1)
               m_cache_x_out[i][d] = v2[d];
         }
      }
   }


   void PoolRepresentation(const double &tokens[][FXAI_CHR_D_MODEL],
                           const int n_tokens,
                           double &rep[]) const
   {
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         rep[d] = 0.0;

      if(n_tokens <= 0)
         return;

      for(int t=0; t<n_tokens; t++)
      {
         double w = 1.0;
         if(t == n_tokens - 1) w = 1.75;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            rep[d] += w * tokens[t][d];
      }

      double den = (double)n_tokens + 0.75;
      if(den <= 0.0) den = 1.0;
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         rep[d] = FXAI_ClipSym(rep[d] / den, 8.0);
   }

   void ComputeHeads(const double &rep[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75,
                     double &q_all[],
                     double &mu_h[]) const
   {
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            z += m_w_cls[c][d] * rep[d];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         mu += m_w_mu[d] * rep[d];
         logv += m_w_logv[d] * rep[d];
      }

      for(int q=0; q<FXAI_CHR_QUANTILES; q++)
      {
         double z = m_b_q[q];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            z += m_w_q[q][d] * rep[d];
         q_all[q] = z;
      }

      for(int h=0; h<FXAI_CHR_HORIZONS; h++)
      {
         double z = m_b_mu_h[h];
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            z += m_w_mu_h[h][d] * rep[d];
         mu_h[h] = z;
      }

      logv = FXAI_Clamp(logv, -4.0, 4.0);
      // Enforce monotonic quantiles in output path.
      for(int q=1; q<FXAI_CHR_QUANTILES; q++)
      {
         if(q_all[q] < q_all[q - 1]) q_all[q] = q_all[q - 1];
      }
      q25 = q_all[2];
      q75 = q_all[4];
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
      int sess = SessionBucket(ResolveContextTime());
      int reg = RegimeBucket();
      double logits[FXAI_CHR_CLASS_COUNT];
      BuildCalLogits(p_raw, sess, reg, logits);
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_CHR_CLASS_COUNT];
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_CHR_CAL_BINS; b++) total += m_cal_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_CHR_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_CHR_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal_iso_cnt[c][b] > 1e-9)
               r = m_cal_iso_pos[c][b] / m_cal_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_CHR_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_CHR_CAL_BINS) bi = FXAI_CHR_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
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
      int sess = SessionBucket(ResolveContextTime());
      int reg = RegimeBucket();
      double logits[FXAI_CHR_CLASS_COUNT];
      BuildCalLogits(p_raw, sess, reg, logits);

      double p_cal[FXAI_CHR_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double lraw[FXAI_CHR_CLASS_COUNT];
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FXAI_Clamp(0.20 * lr * w, 0.0002, 0.0200);
      double reg_l2 = 0.0005;
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_vs_b[c] = FXAI_ClipSym(m_cal_vs_b[c] + cal_lr * e, 4.0);
         m_cal_session_b[sess][c] = FXAI_ClipSym(m_cal_session_b[sess][c] + 0.7 * cal_lr * e, 3.0);
         m_cal_regime_b[reg][c] = FXAI_ClipSym(m_cal_regime_b[reg][c] + 0.6 * cal_lr * e, 3.0);
         for(int j=0; j<FXAI_CHR_CLASS_COUNT; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double grad = e * lraw[j] - reg_l2 * (m_cal_vs_w[c][j] - target_w);
            m_cal_vs_w[c][j] = FXAI_ClipSym(m_cal_vs_w[c][j] + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_CHR_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_CHR_CAL_BINS) bi = FXAI_CHR_CAL_BINS - 1;
         m_cal_iso_cnt[c][bi] += w;
         m_cal_iso_pos[c][bi] += w * target;
      }

      m_cal3_steps++;
   }

   void InitWeights(void)
   {
      ResetSequence();
      ResetInputNorm();
      ResetFeatureStats();
      m_step = 0;
      m_adam_t = 0;
      m_mem_ptr = 0;
      m_last_m1_train_bar = 0;
      m_b_mem_gate = 0.0;

      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         m_cls_ema[c] = 1.0;
         m_b_cls[c] = 0.0;
         m_cal_vs_b[c] = 0.0;
         m_t_b_cls[c] = 0.0;
         for(int j=0; j<FXAI_CHR_CLASS_COUNT; j++)
            m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<FXAI_CHR_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      for(int s=0; s<4; s++)
         for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
            m_cal_session_b[s][c] = 0.0;
      for(int r=0; r<2; r++)
         for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
            m_cal_regime_b[r][c] = 0.0;
      m_cal3_steps = 0;

      for(int g=0; g<16; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      m_chr_replay_head = 0;
      m_chr_replay_size = 0;
      for(int r=0; r<FXAI_CHR_REPLAY; r++)
      {
         m_replay_pos[r] = 0;
         m_replay_cls[r] = (int)FXAI_LABEL_SKIP;
         m_chr_replay_move[r] = 0.0;
         m_chr_replay_cost[r] = 0.0;
         m_replay_w[r] = 1.0;
         m_chr_replay_time[r] = 0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_chr_replay_x[r][k] = 0.0;
      }

      m_val_ready = false;
      m_val_steps = 0;
      m_val_nll_fast = 0.0;
      m_val_nll_slow = 0.0;
      m_val_brier_fast = 0.0;
      m_val_brier_slow = 0.0;
      m_val_ece_fast = 0.0;
      m_val_ece_slow = 0.0;
      m_val_ev_after_cost_fast = 0.0;
      m_val_ev_after_cost_slow = 0.0;
      for(int b=0; b<FXAI_CHR_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
      m_reference_ready = false;

      for(int f=0; f<FXAI_AI_FEATURES; f++)
         m_ch_gate[f] = 1.0;

      for(int cb=0; cb<FXAI_CHR_CODEBOOK; cb++)
      {
         m_codebook_usage[cb] = 1.0;
         m_b_tok[cb] = 0.0;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double s = (double)((cb + 1) * (d + 2));
            m_codebook[cb][d] = 0.02 * MathSin(0.41 * s);
            m_w_tok[cb][d] = 0.02 * MathCos(0.37 * s);
         }
      }

      for(int m=0; m<FXAI_CHR_MEMORY; m++)
      {
         m_mem_usage[m] = 1.0;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            double s = (double)((m + 2) * (d + 3));
            m_mem_k[m][d] = 0.03 * MathSin(0.29 * s);
            m_mem_v[m][d] = 0.03 * MathCos(0.31 * s);
         }
      }

      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         m_w_mem_gate[d] = 0.10 * MathCos((double)(d + 1) * 0.71);
         for(int j=0; j<FXAI_CHR_D_MODEL; j++)
         {
            double s = (double)((d + 2) * (j + 3));
            m_w_mem_q[d][j] = 0.03 * MathSin(0.33 * s);
         }
      }

      for(int p=0; p<FXAI_CHR_MAX_PATCHES; p++)
      {
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            m_pos[p][d] = 0.015 * MathSin((double)((p + 1) * (d + 2)) * 0.43);
         }
      }

      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
      {
         m_b_patch[d] = 0.0;
         m_w_mu[d] = 0.03 * MathSin((double)(d + 2) * 0.83);
         m_w_logv[d] = 0.03 * MathCos((double)(d + 3) * 0.89);

         for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
         {
            m_w_cls[c][d] = 0.03 * MathSin((double)((c + 2) * (d + 1)) * 0.79);
            m_t_w_cls[c][d] = m_w_cls[c][d];
         }

         for(int q=0; q<FXAI_CHR_QUANTILES; q++)
            m_w_q[q][d] = 0.03 * MathSin((double)((q + 2) * (d + 3)) * 0.61);

         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
            {
               double s = (double)((d + 1) * (f + 2) * (t + 3));
               m_w_patch[d][f][t] = 0.025 * MathSin(0.61 * s);
            }
         }
      }

      for(int l=0; l<FXAI_CHR_LAYERS; l++)
      {
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            m_bff2[l][d] = 0.0;
            m_ln1_g[l][d] = 1.0;
            m_ln1_b[l][d] = 0.0;
            m_ln2_g[l][d] = 1.0;
            m_ln2_b[l][d] = 0.0;

            for(int od=0; od<FXAI_CHR_D_MODEL; od++)
            {
               double s = (double)((l + 1) * (d + 2) * (od + 3));
               m_wo[l][d][od] = 0.02 * MathCos(0.53 * s);
            }

            for(int r=0; r<FXAI_CHR_D_FF; r++)
            {
               double s2 = (double)((l + 1) * (d + 1) * (r + 2));
               m_wff2[l][d][r] = 0.02 * MathSin(0.57 * s2);
            }
         }

         for(int r=0; r<FXAI_CHR_D_FF; r++)
         {
            m_bff1[l][r] = 0.0;
            for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            {
               double s = (double)((l + 2) * (r + 1) * (d + 3));
               m_wff1[l][r][d] = 0.02 * MathCos(0.59 * s);
            }
         }

         for(int h=0; h<FXAI_CHR_HEADS; h++)
         {
            for(int dh=0; dh<FXAI_CHR_D_HEAD; dh++)
            {
               for(int d=0; d<FXAI_CHR_D_MODEL; d++)
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
      for(int q=0; q<FXAI_CHR_QUANTILES; q++) m_b_q[q] = 0.0;
      for(int h=0; h<FXAI_CHR_HORIZONS; h++)
      {
         m_b_mu_h[h] = 0.0;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
            m_w_mu_h[h][d] = 0.02 * MathSin((double)((h + 2) * (d + 1)) * 0.67);
      }

      // Slight skip prior before calibration settles.
      m_b_cls[(int)FXAI_LABEL_SKIP] = 0.20;

      // Distilled warm-start anchors (embedded constants, no disk dependency).
      static const double distill_cls[FXAI_CHR_CLASS_COUNT][4] =
      {
         {-0.22, -0.11, 0.08, 0.15},
         { 0.24,  0.10, 0.06, -0.09},
         {-0.06,  0.02, -0.04, 0.20}
      };
      static const double distill_tok[8] = {0.12, -0.09, 0.07, -0.05, 0.03, -0.02, 0.01, -0.01};
      for(int c=0; c<FXAI_CHR_CLASS_COUNT; c++)
      {
         for(int d=0; d<4 && d<FXAI_CHR_D_MODEL; d++)
         {
            m_w_cls[c][d] = 0.7 * m_w_cls[c][d] + 0.3 * distill_cls[c][d];
            m_t_w_cls[c][d] = m_w_cls[c][d];
         }
      }
      for(int t=0; t<FXAI_CHR_CODEBOOK; t++)
      {
         for(int d=0; d<8 && d<FXAI_CHR_D_MODEL; d++)
            m_w_tok[t][d] = 0.85 * m_w_tok[t][d] + 0.15 * distill_tok[d];
      }

      m_initialized = true;
   }

   void ForwardPass(const double &x[],
                    const bool commit,
                    const bool training_mode,
                    const int label_cls,
                    const double move_points,
                    double &rep[],
                    double &p_raw[],
                    double &mu,
                    double &logv,
                    double &q25,
                    double &q75,
                    double &mu_h[],
                    double &patch_stat[][FXAI_CHR_PATCH_LEN],
                    double &token_hist[],
                    int &token_target,
                    double &layer_in_mean[][FXAI_CHR_D_MODEL],
                    double &layer_ctx_mean[][FXAI_CHR_D_MODEL],
                    double &layer_ff_mean[][FXAI_CHR_D_FF],
                    double &mem_attn[],
                    int &token_count)
   {
      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(x, xn);
      UpdateFeatureStats(xn);

      double seq_mat[FXAI_CHR_SEQ][FXAI_AI_FEATURES];
      int seq_len = 0;
      BuildTemporalMatrix(xn, seq_mat, seq_len, !training_mode);

      double tokens_a[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];
      double tokens_b[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];
      BuildPatchTokens(seq_mat, seq_len, tokens_a, token_count, patch_stat, token_hist, token_target);

      if(token_count < 1) token_count = 1;
      if(token_count > FXAI_CHR_MAX_PATCHES) token_count = FXAI_CHR_MAX_PATCHES;

      if(training_mode)
         token_target = BuildFutureTokenTarget(xn, label_cls, move_points);

      m_cache_token_count = token_count;
      m_cache_token_target = token_target;
      for(int p=0; p<token_count; p++)
      {
         m_cache_patch_start[p] = 0;
         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            m_cache_x0[p][d] = tokens_a[p][d];
            m_cache_patch_code[p][d] = 0.0;
            m_cache_patch_z[p][d] = 0.0;
         }
      }
      for(int f=0; f<FXAI_AI_FEATURES; f++)
         for(int t=0; t<FXAI_CHR_PATCH_LEN; t++)
            m_cache_patch_stat[f][t] = patch_stat[f][t];

      for(int l=0; l<FXAI_CHR_LAYERS; l++)
      {
         double mean_in_l[FXAI_CHR_D_MODEL];
         double mean_ctx_l[FXAI_CHR_D_MODEL];
         double mean_ff_l[FXAI_CHR_D_FF];

         AttentionLayer(l,
                        tokens_a,
                        token_count,
                        tokens_b,
                        true,
                        training_mode,
                        mean_in_l,
                        mean_ctx_l,
                        mean_ff_l);

         for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         {
            layer_in_mean[l][d] = mean_in_l[d];
            layer_ctx_mean[l][d] = mean_ctx_l[d];
         }
         for(int r=0; r<FXAI_CHR_D_FF; r++)
            layer_ff_mean[l][r] = mean_ff_l[r];

         for(int t=0; t<token_count; t++)
         {
            for(int d=0; d<FXAI_CHR_D_MODEL; d++)
               tokens_a[t][d] = tokens_b[t][d];
         }
      }

      PoolRepresentation(tokens_a, token_count, rep);
      double rep_mem[FXAI_CHR_D_MODEL];
      ApplyMemoryRetrieval(rep, rep_mem, mem_attn);
      for(int d=0; d<FXAI_CHR_D_MODEL; d++)
         rep[d] = rep_mem[d];

      double logits[FXAI_CHR_CLASS_COUNT];
      double q_all[FXAI_CHR_QUANTILES];
      ComputeHeads(rep, logits, p_raw, mu, logv, q25, q75, q_all, mu_h);
      q25 += 0.0 * q_all[0];

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
      for(int h=0; h<FXAI_CHR_HORIZONS; h++)
         h_ev += MathAbs(mu_h[h]) * (h == 0 ? 0.50 : (h == 1 ? 0.30 : 0.20));
      double ev = (0.52 * MathAbs(mu) + 0.26 * h_ev + 0.14 * sigma + 0.08 * iqr) *
                  FXAI_Clamp(1.0 - skip_prob, 0.0, 1.0);
      return ev;
   }
