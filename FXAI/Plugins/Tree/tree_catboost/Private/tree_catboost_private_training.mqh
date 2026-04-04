               meanv += cand[l][c];
            }
            meanv /= (double)FXAI_CAT_CLASS_COUNT;
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               cand[l][c] -= meanv;
         }

         double loss_old = ComputeLeafLoss(n, leaf_idx, y_all, sample_w, base_margin, tree.leaf_value);
         double loss_new = ComputeLeafLoss(n, leaf_idx, y_all, sample_w, base_margin, cand);

         double step_scale = 1.0;
         if(loss_new > loss_old)
         {
            step_scale = 0.5;
            for(int l=0; l<leaf_count; l++)
            {
               for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
                  cand[l][c] = tree.leaf_value[l][c] + step_scale * (cand[l][c] - tree.leaf_value[l][c]);
            }
            loss_new = ComputeLeafLoss(n, leaf_idx, y_all, sample_w, base_margin, cand);
            if(loss_new > loss_old)
            {
               for(int l=0; l<leaf_count; l++)
                  for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
                     cand[l][c] = tree.leaf_value[l][c];
            }
         }

         for(int l=0; l<leaf_count; l++)
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               tree.leaf_value[l][c] = cand[l][c];
      }
   }

   void ApplyModelShrinkage(const double shrink)
   {
      double s = FXAI_Clamp(1.0 - shrink, 0.90, 1.00);
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_bias[c] *= s;

      for(int t=0; t<m_tree_count; t++)
      {
         int leaf_count = (m_trees[t].depth > 0 ? (1 << m_trees[t].depth) : 1);
         if(leaf_count < 1) leaf_count = 1;
         if(leaf_count > FXAI_CAT_MAX_LEAVES) leaf_count = FXAI_CAT_MAX_LEAVES;
         for(int l=0; l<leaf_count; l++)
         {
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               m_trees[t].leaf_value[l][c] *= s;
         }
      }
   }

   void BuildRawLogits(const double &p_raw[],
                       const int session_bucket,
                       const int regime_bucket,
                       double &logits[]) const
   {
      double lraw[FXAI_CAT_CLASS_COUNT];
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double z = m_cal_vs_b[c] + m_cal_session_b[session_bucket][c] + m_cal_regime_b[regime_bucket][c];
         for(int j=0; j<FXAI_CAT_CLASS_COUNT; j++)
            z += m_cal_vs_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      int session = SessionBucket(ResolveContextTime());
      int regime = RegimeBucket();

      double logits[FXAI_CAT_CLASS_COUNT];
      BuildRawLogits(p_raw, session, regime, logits);
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_CAT_CLASS_COUNT];
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_CAT_CAL_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 50.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_CAT_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_CAT_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_CAT_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_CAT_CAL_BINS) bi = FXAI_CAT_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         p_cal[c] = FXAI_Clamp(0.80 * p_cal[c] + 0.20 * p_iso[c], 0.0005, 0.9990);

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
      int session = SessionBucket(ResolveContextTime());
      int regime = RegimeBucket();

      double logits[FXAI_CAT_CLASS_COUNT];
      BuildRawLogits(p_raw, session, regime, logits);

      double p_cal[FXAI_CAT_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double lraw[FXAI_CAT_CLASS_COUNT];
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.20, 8.00);
      double cal_lr = FXAI_Clamp(0.15 * lr * w, 0.0001, 0.0200);
      double reg = 0.0005;

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_vs_b[c] = FXAI_ClipSym(m_cal_vs_b[c] + cal_lr * e, 4.0);
         m_cal_session_b[session][c] = FXAI_ClipSym(m_cal_session_b[session][c] + 0.7 * cal_lr * e, 3.0);
         m_cal_regime_b[regime][c] = FXAI_ClipSym(m_cal_regime_b[regime][c] + 0.6 * cal_lr * e, 3.0);

         for(int j=0; j<FXAI_CAT_CLASS_COUNT; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double wij = m_cal_vs_w[c][j];
            double grad = e * lraw[j] - reg * (wij - target_w);
            m_cal_vs_w[c][j] = FXAI_ClipSym(wij + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_CAT_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_CAT_CAL_BINS) bi = FXAI_CAT_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

      m_cal3_steps++;
   }

   void UpdateLossDrift(const double ce_loss)
   {
      if(!m_loss_ready)
      {
         m_loss_fast = ce_loss;
         m_loss_slow = ce_loss;
         m_loss_ready = true;
         return;
      }

      m_loss_fast = 0.90 * m_loss_fast + 0.10 * ce_loss;
      m_loss_slow = 0.995 * m_loss_slow + 0.005 * ce_loss;

      if(m_drift_cooldown > 0) m_drift_cooldown--;
      if(m_step < 256 || m_drift_cooldown > 0) return;

      if(m_loss_fast > 1.7 * MathMax(m_loss_slow, 0.10))
         m_drift_cooldown = 96;
   }

   void UpdateValidationHarness(const int cls,
                                const double &x[],
                                const double &p_cal[],
                                const double sample_w)
   {
      int y = cls;
      if(y < 0 || y >= FXAI_CAT_CLASS_COUNT) y = (int)FXAI_LABEL_SKIP;

      double ce = -MathLog(FXAI_Clamp(p_cal[y], 1e-6, 1.0));
      double brier = 0.0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double t = (c == y ? 1.0 : 0.0);
         double d = p_cal[c] - t;
         brier += d * d;
      }
      brier /= 3.0;

      double conf = p_cal[0];
      int pred = 0;
      for(int c=1; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         if(p_cal[c] > conf)
         {
            conf = p_cal[c];
            pred = c;
         }
      }
      double acc = (pred == y ? 1.0 : 0.0);

      double w = FXAI_Clamp(sample_w, 0.25, 4.0);
      if(!m_val_ready)
      {
         m_val_ce_fast = ce;
         m_val_ce_slow = ce;
         m_val_brier_fast = brier;
         m_val_brier_slow = brier;
         m_val_ece_fast = 0.0;
         m_val_ece_slow = 0.0;
         m_val_ready = true;
      }
      else
      {
         m_val_ce_fast = 0.92 * m_val_ce_fast + 0.08 * ce;
         m_val_ce_slow = 0.995 * m_val_ce_slow + 0.005 * ce;
         m_val_brier_fast = 0.92 * m_val_brier_fast + 0.08 * brier;
         m_val_brier_slow = 0.995 * m_val_brier_slow + 0.005 * brier;
      }

      for(int b=0; b<FXAI_CAT_ECE_BINS; b++)
      {
         m_ece_mass[b] *= 0.997;
         m_ece_acc[b] *= 0.997;
         m_ece_conf[b] *= 0.997;
      }
      int bi = (int)MathFloor(conf * (double)FXAI_CAT_ECE_BINS);
      if(bi < 0) bi = 0;
      if(bi >= FXAI_CAT_ECE_BINS) bi = FXAI_CAT_ECE_BINS - 1;
      m_ece_mass[bi] += w;
      m_ece_acc[bi] += w * acc;
      m_ece_conf[bi] += w * conf;

      double ece_num = 0.0;
      double ece_den = 0.0;
      for(int b=0; b<FXAI_CAT_ECE_BINS; b++)
      {
         if(m_ece_mass[b] <= 1e-9) continue;
         double ba = m_ece_acc[b] / m_ece_mass[b];
         double bc = m_ece_conf[b] / m_ece_mass[b];
         ece_num += m_ece_mass[b] * MathAbs(ba - bc);
         ece_den += m_ece_mass[b];
      }
      double ece = (ece_den > 0.0 ? ece_num / ece_den : 0.0);
      m_val_ece_fast = 0.92 * m_val_ece_fast + 0.08 * ece;
      m_val_ece_slow = 0.995 * m_val_ece_slow + 0.005 * ece;

      if(!m_feat_stats_ready)
      {
         for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
         {
            int f = TrackFeatIndex(k);
            double v = x[f];
            m_feat_ref_mean[k] = v;
            m_feat_cur_mean[k] = v;
            m_feat_ref_var[k] = 1.0;
            m_feat_cur_var[k] = 1.0;
         }
         m_feat_stats_ready = true;
      }
      else
      {
         for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
         {
            int f = TrackFeatIndex(k);
            double v = x[f];

            double d_ref = v - m_feat_ref_mean[k];
            m_feat_ref_mean[k] += 0.001 * d_ref;
            m_feat_ref_var[k] = 0.999 * m_feat_ref_var[k] + 0.001 * d_ref * d_ref;

            double d_cur = v - m_feat_cur_mean[k];
            m_feat_cur_mean[k] += 0.025 * d_cur;
            m_feat_cur_var[k] = 0.975 * m_feat_cur_var[k] + 0.025 * d_cur * d_cur;
         }
      }

      double drift = 0.0;
      for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
      {
         double sd = MathSqrt(MathMax(m_feat_ref_var[k], 1e-6));
         drift += MathAbs(m_feat_cur_mean[k] - m_feat_ref_mean[k]) / sd;
      }
      drift /= (double)FXAI_CAT_TRACK_FEATS;
      m_feat_drift_score = drift;

      bool bad = false;
      if(m_val_ce_fast > 1.45 * MathMax(m_val_ce_slow, 0.10) && m_val_ce_fast > 0.85) bad = true;
      if(m_val_brier_fast > 1.40 * MathMax(m_val_brier_slow, 0.02) && m_val_brier_fast > 0.18) bad = true;
      if(m_val_ece_fast > 0.18) bad = true;
      if(m_feat_drift_score > 4.0) bad = true;

      if(bad) m_quality_alarm++;
      else if(m_quality_alarm > 0) m_quality_alarm--;

      if(m_quality_alarm > 16)
      {
         m_quality_alarm = 16;
         if(m_drift_cooldown < 120) m_drift_cooldown = 120;
      }

      m_val_steps++;
   }

   bool BuildOneTree(const FXAIAIHyperParams &hp)
   {
      if(m_buf_size < FXAI_CAT_MIN_BUFFER) return false;

      int n = m_buf_size;
      double x_all[][FXAI_AI_WEIGHTS];
      int y_all[];
      double mv_all[];
      double w_all[];
      ArrayResize(x_all, n);
      ArrayResize(y_all, n);
      ArrayResize(mv_all, n);
      ArrayResize(w_all, n);

      for(int i=0; i<n; i++)
      {
         int p = BufPos(i);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            x_all[i][k] = m_buf_x[p][k];
         int cls = m_buf_y[p];
         if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;
         y_all[i] = cls;
         mv_all[i] = m_buf_move[p];
         w_all[i] = m_buf_w[p];
      }

      // Recommended defaults: lr 0.02..0.05, l2 3..8.
      double eta_base = FXAI_Clamp(hp.xgb_lr, 0.0200, 0.0500);
      double lambda = FXAI_Clamp(hp.xgb_l2, 3.0000, 8.0000);

      // LR schedule + drift adaptation.
      double schedule = MathPow(0.985, (double)m_tree_count / 16.0);
      double eta = FXAI_Clamp(eta_base * schedule, 0.0020, 0.0500);
      if(m_drift_cooldown > 0)
      {
         eta = FXAI_Clamp(eta * 0.75, 0.0020, 0.0500);
         lambda = FXAI_Clamp(lambda * 1.20, 2.0, 12.0);
      }

      // Bootstrap policy + row subsampling.
      int bs_mode = (m_step % 3); // 0 Bayesian, 1 Bernoulli, 2 MVS-like.
      double subsample = (m_quality_alarm > 8 ? 0.90 : 0.75);
      double boot_w[];
      ArrayResize(boot_w, n);
      BuildBootstrapWeights(n, bs_mode, subsample, w_all, boot_w);

      // Ordered/CTR-expanded feature matrix and ordered bias.
      double x_ext[][FXAI_CAT_EXT_WEIGHTS];
      double ordered_bias[][FXAI_CAT_CLASS_COUNT];
      ArrayResize(x_ext, n);
      ArrayResize(ordered_bias, n);
      BuildExtendedTraining(n, x_all, y_all, x_ext, ordered_bias);

      // Build candidate split borders in extended space.
      BuildSplitCandidates(n, x_ext);

      // Optional feature subsampling (RSM).
      double rsm = 0.65;
      if(rsm < 0.20) rsm = 0.20;
      if(rsm > 1.00) rsm = 1.00;
      bool feat_use[];
      ArrayResize(feat_use, FXAI_CAT_EXT_WEIGHTS);
      feat_use[0] = false;
      int avail = 0;
      for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         if(m_split_border_count[f] <= 0 || m_split_valid_count[f] < 2 * FXAI_CAT_MIN_DATA)
         {
            feat_use[f] = false;
            continue;
         }
         feat_use[f] = (Rand01() <= rsm);
         if(feat_use[f]) avail++;
      }
      if(avail <= 0)
      {
         for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
         {
            if(m_split_border_count[f] > 0) { feat_use[f] = true; break; }
         }
      }

      // Base margins and ordered-safe gradients/hessians.
      double base_margin[][FXAI_CAT_CLASS_COUNT];
      double g[][FXAI_CAT_CLASS_COUNT];
      double h[][FXAI_CAT_CLASS_COUNT];
      ArrayResize(base_margin, n);
      ArrayResize(g, n);
      ArrayResize(h, n);

      double noise_sigma = FXAI_Clamp(0.010 * MathSqrt(eta), 0.0, 0.020);
      for(int i=0; i<n; i++)
      {
         double xloc[FXAI_CAT_EXT_WEIGHTS];
         for(int k=0; k<FXAI_CAT_EXT_WEIGHTS; k++) xloc[k] = x_ext[i][k];
         double mloc[FXAI_CAT_CLASS_COUNT];
         ModelMarginsExt(xloc, mloc);
         for(int c0=0; c0<FXAI_CAT_CLASS_COUNT; c0++) base_margin[i][c0] = mloc[c0];

         double z[FXAI_CAT_CLASS_COUNT];
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            z[c] = base_margin[i][c] + ordered_bias[i][c];

         double p[FXAI_CAT_CLASS_COUNT];
         Softmax3(z, p);

         double wi = FXAI_Clamp(boot_w[i], 0.0, 12.0);
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         {
            double target = (c == y_all[i] ? 1.0 : 0.0);
            double gc = (target - p[c]) * wi;
            if(noise_sigma > 0.0)
               gc += noise_sigma * RandGauss();
            double hc = FXAI_Clamp(p[c] * (1.0 - p[c]) * wi, 0.001, 8.0);
            g[i][c] = gc;
            h[i][c] = hc;
         }
      }

      // Bias Newton refinement.
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double G = 0.0, H = 0.0;
         for(int i=0; i<n; i++)
         {
            G += g[i][c];
            H += h[i][c];
         }
         if(H > 1e-9)
            m_bias[c] += 0.15 * eta * FXAI_ClipSym(G / (H + lambda), 3.0);
      }
      double mean_bias = (m_bias[0] + m_bias[1] + m_bias[2]) / 3.0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++) m_bias[c] -= mean_bias;

      FXAICatTree tree;
      InitTree(tree);

      int leaf_idx[];
      ArrayResize(leaf_idx, n);
      for(int i=0; i<n; i++) leaf_idx[i] = 0;

      int depth_used = 0;
      for(int d=0; d<FXAI_CAT_MAX_DEPTH; d++)
      {
         int best_f = -1;
         double best_thr = 0.0;
         bool best_default_left = true;
         double best_gain = 0.0;

         for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
         {
            if(!feat_use[f]) continue;
            int bc = m_split_border_count[f];
            if(bc <= 0) continue;

            for(int b=0; b<bc; b++)
            {
               double thr = m_split_borders[f][b];
               for(int pass=0; pass<2; pass++)
               {
                  bool default_left = (pass == 0);
                  double gain = 0.0;
                  if(!EvaluateSplit(d, n, leaf_idx, x_ext, g, h, boot_w, f, thr, default_left, lambda, gain))
                     continue;
                  if(gain > best_gain)
                  {
                     best_gain = gain;
                     best_f = f;
                     best_thr = thr;
                     best_default_left = default_left;
                  }
               }
            }
         }

         if(best_f < 1)
            break;

         tree.levels[d].feature = best_f;
         tree.levels[d].threshold = best_thr;
         tree.levels[d].default_left = best_default_left;
         m_feature_use[best_f]++;

         for(int i=0; i<n; i++)
         {
            double xv = x_ext[i][best_f];
            bool go_left;
            if(!MathIsValidNumber(xv)) go_left = best_default_left;
            else go_left = (xv <= best_thr);
            leaf_idx[i] = (leaf_idx[i] << 1) | (go_left ? 0 : 1);
         }

         depth_used = d + 1;
      }

      tree.depth = depth_used;
      BuildLeafValuesIterative(tree, n, leaf_idx, y_all, mv_all, boot_w, base_margin, eta, lambda);

      // Model shrinkage before appending new tree.
      double shrink = FXAI_Clamp(0.0015 + 0.0100 * eta, 0.0010, 0.0100);
      ApplyModelShrinkage(shrink);

      if(m_tree_count < FXAI_CAT_MAX_TREES)
      {
         m_trees[m_tree_count] = tree;
         m_tree_count++;
      }
      else
      {
         for(int t=1; t<FXAI_CAT_MAX_TREES; t++)
            m_trees[t - 1] = m_trees[t];
         m_trees[FXAI_CAT_MAX_TREES - 1] = tree;
         m_tree_count = FXAI_CAT_MAX_TREES;
      }

      return true;
   }

