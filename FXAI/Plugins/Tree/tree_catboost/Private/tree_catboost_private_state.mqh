   bool   m_initialized;
   int    m_step;
   int    m_tree_count;
   double m_bias[FXAI_CAT_CLASS_COUNT];
   CFXAINativeQualityHeads m_quality_heads;

   FXAICatTree m_trees[FXAI_CAT_MAX_TREES];

   // Ordered online sample buffer.
   double m_buf_x[FXAI_CAT_BUFFER][FXAI_AI_WEIGHTS];
   int    m_buf_y[FXAI_CAT_BUFFER];
   double m_buf_move[FXAI_CAT_BUFFER];
   double m_buf_w[FXAI_CAT_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   // Categorical/CTR settings and runtime stats.
   int    m_ctr_base_feat[FXAI_CAT_CTR_BASE];
   int    m_ctr_pair_a[FXAI_CAT_CTR_PAIR_COUNT];
   int    m_ctr_pair_b[FXAI_CAT_CTR_PAIR_COUNT];
   bool   m_ctr_ready;
   double m_ctr_global_class[FXAI_CAT_CLASS_COUNT];
   double m_ctr_bin_total[FXAI_CAT_CTR_BASE][FXAI_CAT_BINS];
   double m_ctr_bin_class[FXAI_CAT_CTR_BASE][FXAI_CAT_BINS][FXAI_CAT_CLASS_COUNT];
   double m_ctr_pair_total[FXAI_CAT_CTR_PAIR_COUNT][FXAI_CAT_CTR_PAIR_HASH];
   double m_ctr_pair_buy[FXAI_CAT_CTR_PAIR_COUNT][FXAI_CAT_CTR_PAIR_HASH];

   // Quantile borders for base feature quantization.
   int    m_base_border_count[FXAI_AI_WEIGHTS];
   double m_base_borders[FXAI_AI_WEIGHTS][FXAI_CAT_MAX_BORDERS];

   // Candidate borders for split search on extended feature space.
   int    m_split_border_count[FXAI_CAT_EXT_WEIGHTS];
   int    m_split_valid_count[FXAI_CAT_EXT_WEIGHTS];
   double m_split_borders[FXAI_CAT_EXT_WEIGHTS][FXAI_CAT_MAX_BORDERS];

   // Split reuse penalty / feature usage tracker.
   int    m_feature_use[FXAI_CAT_EXT_WEIGHTS];

   // Drift adaptation and class balance.
   double m_cls_ema[FXAI_CAT_CLASS_COUNT];
   bool   m_loss_ready;
   double m_loss_fast;
   double m_loss_slow;
   int    m_drift_cooldown;

   // Plugin-native multiclass calibration (vector scaling + isotonic-like bins).
   double m_cal_vs_w[FXAI_CAT_CLASS_COUNT][FXAI_CAT_CLASS_COUNT];
   double m_cal_vs_b[FXAI_CAT_CLASS_COUNT];
   double m_cal_session_b[4][FXAI_CAT_CLASS_COUNT];
   double m_cal_regime_b[2][FXAI_CAT_CLASS_COUNT];
   double m_cal3_iso_pos[FXAI_CAT_CLASS_COUNT][FXAI_CAT_CAL_BINS];
   double m_cal3_iso_cnt[FXAI_CAT_CLASS_COUNT][FXAI_CAT_CAL_BINS];
   int    m_cal3_steps;

   // Validation and drift harness (rolling CE/Brier/ECE + feature drift alarms).
   bool   m_val_ready;
   int    m_val_steps;
   double m_val_ce_fast;
   double m_val_ce_slow;
   double m_val_brier_fast;
   double m_val_brier_slow;
   double m_val_ece_fast;
   double m_val_ece_slow;
   double m_ece_mass[FXAI_CAT_ECE_BINS];
   double m_ece_acc[FXAI_CAT_ECE_BINS];
   double m_ece_conf[FXAI_CAT_ECE_BINS];

   bool   m_feat_stats_ready;
   double m_feat_ref_mean[FXAI_CAT_TRACK_FEATS];
   double m_feat_ref_var[FXAI_CAT_TRACK_FEATS];
   double m_feat_cur_mean[FXAI_CAT_TRACK_FEATS];
   double m_feat_cur_var[FXAI_CAT_TRACK_FEATS];
   double m_feat_drift_score;
   int    m_quality_alarm;

   int TrackFeatIndex(const int k) const
   {
      static const int ids[FXAI_CAT_TRACK_FEATS] = {1, 2, 3, 7, 12, 20};
      if(k < 0 || k >= FXAI_CAT_TRACK_FEATS) return 1;
      int f = ids[k];
      if(f < 1) f = 1;
      if(f >= FXAI_AI_WEIGHTS) f = FXAI_AI_WEIGHTS - 1;
      return f;
   }

   int SessionBucket(const datetime t) const
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      int h = dt.hour;
      if(h < 8) return 0;      // Asia session
      if(h < 13) return 1;     // Europe open/core
      if(h < 20) return 2;     // US overlap/core
      return 3;                // late/illiquid hours
   }

   int RegimeBucket(void) const
   {
      if(!m_move_ready) return 0;
      if(m_move_ema_abs > 5.0) return 1;
      return 0;
   }

   double Rand01(void)
   {
      return PluginRand01();
   }

   double RandGauss(void)
   {
      // Lightweight approx N(0,1) via CLT sum(U)-6.
      double s = 0.0;
      for(int i=0; i<12; i++) s += Rand01();
      return s - 6.0;
   }

   int PairHash(const int a, const int b, const int q) const
   {
      int aa = (a >= 0 ? a : 0);
      int bb = (b >= 0 ? b : 0);
      int h = (aa * 131 + bb * 17 + q * 53 + 97) % FXAI_CAT_CTR_PAIR_HASH;
      if(h < 0) h += FXAI_CAT_CTR_PAIR_HASH;
      return h;
   }

   void InitTree(FXAICatTree &tree) const
   {
      tree.depth = 0;
      for(int d=0; d<FXAI_CAT_MAX_DEPTH; d++)
      {
         tree.levels[d].feature = -1;
         tree.levels[d].threshold = 0.0;
         tree.levels[d].default_left = true;
      }

      for(int l=0; l<FXAI_CAT_MAX_LEAVES; l++)
      {
         tree.leaf_move_mean[l] = 0.0;
         tree.leaf_count[l] = 0;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            tree.leaf_value[l][c] = 0.0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FXAI_CAT_BUFFER;
      int p = start + logical_idx;
      while(p >= FXAI_CAT_BUFFER) p -= FXAI_CAT_BUFFER;
      return p;
   }

   void PushSample(const int y,
                   const double &x[],
                   const double move_points,
                   const double sample_w)
   {
      int pos = m_buf_head;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_buf_x[pos][i] = x[i];
      m_buf_y[pos] = y;
      m_buf_move[pos] = move_points;
      m_buf_w[pos] = sample_w;

      m_buf_head++;
      if(m_buf_head >= FXAI_CAT_BUFFER) m_buf_head = 0;
      if(m_buf_size < FXAI_CAT_BUFFER) m_buf_size++;
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
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

   void InitCTRLayout(void)
   {
      for(int b=0; b<FXAI_CAT_CTR_BASE; b++)
      {
         int f = 1 + b;
         if(f >= FXAI_AI_WEIGHTS) f = FXAI_AI_WEIGHTS - 1;
         m_ctr_base_feat[b] = f;
      }

      // Pair selected high-signal interactions from base list.
      m_ctr_pair_a[0] = 0; m_ctr_pair_b[0] = 1;
      m_ctr_pair_a[1] = 0; m_ctr_pair_b[1] = 2;
      m_ctr_pair_a[2] = 1; m_ctr_pair_b[2] = 3;
      m_ctr_pair_a[3] = 2; m_ctr_pair_b[3] = 4;
      m_ctr_pair_a[4] = 5; m_ctr_pair_b[4] = 6;
      m_ctr_pair_a[5] = 7; m_ctr_pair_b[5] = 8;
   }

   void ResetCTRState(void)
   {
      m_ctr_ready = false;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_ctr_global_class[c] = 1.0 / 3.0;

      for(int b=0; b<FXAI_CAT_CTR_BASE; b++)
      {
         for(int k=0; k<FXAI_CAT_BINS; k++)
         {
            m_ctr_bin_total[b][k] = 0.0;
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               m_ctr_bin_class[b][k][c] = 0.0;
         }
      }

      for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
      {
         for(int h=0; h<FXAI_CAT_CTR_PAIR_HASH; h++)
         {
            m_ctr_pair_total[q][h] = 0.0;
            m_ctr_pair_buy[q][h] = 0.0;
         }
      }
   }

   void ResetSplitState(void)
   {
      for(int f=0; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         m_feature_use[f] = 0;
         m_split_border_count[f] = 0;
         m_split_valid_count[f] = 0;
         for(int b=0; b<FXAI_CAT_MAX_BORDERS; b++)
            m_split_borders[f][b] = 0.0;
      }
      for(int f=0; f<FXAI_AI_WEIGHTS; f++)
      {
         m_base_border_count[f] = 0;
         for(int b=0; b<FXAI_CAT_MAX_BORDERS; b++)
            m_base_borders[f][b] = 0.0;
      }
   }

   void BuildBordersForFeature(const int n,
                               const double &x_all[][FXAI_AI_WEIGHTS],
                               const int feature)
   {
      if(feature < 1 || feature >= FXAI_AI_WEIGHTS)
      {
         m_base_border_count[feature] = 0;
         return;
      }

      double vals[];
      ArrayResize(vals, 0);
      for(int i=0; i<n; i++)
      {
         double v = x_all[i][feature];
         if(!MathIsValidNumber(v)) continue;
         int sz = ArraySize(vals);
         ArrayResize(vals, sz + 1);
         vals[sz] = v;
      }

      int valid = ArraySize(vals);
      if(valid < 2)
      {
         m_base_border_count[feature] = 0;
         return;
      }

      ArraySort(vals);
      int desired = FXAI_CAT_BINS - 1;
      if(desired > valid - 1) desired = valid - 1;

      int out = 0;
      double last = -DBL_MAX;
      for(int b=1; b<=desired; b++)
      {
         int qi = (int)MathFloor(((double)b / (double)(desired + 1)) * (double)(valid - 1));
         if(qi < 0) qi = 0;
         if(qi >= valid) qi = valid - 1;
         double thr = vals[qi];
         if(out == 0 || MathAbs(thr - last) > 1e-12)
         {
            m_base_borders[feature][out] = thr;
            last = thr;
            out++;
            if(out >= FXAI_CAT_MAX_BORDERS) break;
         }
      }
      m_base_border_count[feature] = out;
   }

   void BuildBaseBorders(const int n,
                         const double &x_all[][FXAI_AI_WEIGHTS])
   {
      for(int f=1; f<FXAI_AI_WEIGHTS; f++)
         BuildBordersForFeature(n, x_all, f);
   }

   int QuantizeBaseFeature(const int base_feature_idx,
                           const double v) const
   {
      int f = base_feature_idx;
      if(f < 1 || f >= FXAI_AI_WEIGHTS) return -1;
      if(!MathIsValidNumber(v)) return -1;
      int bc = m_base_border_count[f];
      if(bc <= 0) return 0;

      int lo = 0;
      int hi = bc - 1;
      int pos = bc;
      while(lo <= hi)
      {
         int mid = (lo + hi) >> 1;
         if(v <= m_base_borders[f][mid])
         {
            pos = mid;
            hi = mid - 1;
         }
         else
         {
            lo = mid + 1;
         }
      }

      if(pos < 0) pos = 0;
      if(pos >= FXAI_CAT_BINS) pos = FXAI_CAT_BINS - 1;
      return pos;
   }

   int TraverseLeafIndex(const FXAICatTree &tree,
                         const double &x_ext[]) const
   {
      if(tree.depth <= 0) return 0;
      int leaf = 0;
      for(int d=0; d<tree.depth; d++)
      {
         int f = tree.levels[d].feature;
         if(f < 1 || f >= FXAI_CAT_EXT_WEIGHTS) break;

         double xv = x_ext[f];
         bool go_left;
         if(!MathIsValidNumber(xv))
            go_left = tree.levels[d].default_left;
         else
            go_left = (xv <= tree.levels[d].threshold);

         leaf = (leaf << 1) | (go_left ? 0 : 1);
         if(leaf < 0 || leaf >= FXAI_CAT_MAX_LEAVES)
            return 0;
      }
      return leaf;
   }

   void TreeMargins(const FXAICatTree &tree,
                    const double &x_ext[],
                    double &margins[]) const
   {
      int leaf = TraverseLeafIndex(tree, x_ext);
      if(leaf < 0 || leaf >= FXAI_CAT_MAX_LEAVES) return;

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         margins[c] += tree.leaf_value[leaf][c];
   }

   void ModelMarginsExt(const double &x_ext[],
                        double &margins[]) const
   {
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         margins[c] = m_bias[c];

      for(int t=0; t<m_tree_count; t++)
         TreeMargins(m_trees[t], x_ext, margins);
   }

   void BuildSplitCandidates(const int n,
                             const double &x_ext[][FXAI_CAT_EXT_WEIGHTS])
   {
      for(int f=0; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         m_split_border_count[f] = 0;
         m_split_valid_count[f] = 0;
      }

      for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         double vals[];
         ArrayResize(vals, 0);
         for(int i=0; i<n; i++)
         {
            double v = x_ext[i][f];
            if(!MathIsValidNumber(v)) continue;
            int sz = ArraySize(vals);
            ArrayResize(vals, sz + 1);
            vals[sz] = v;
         }

         int valid = ArraySize(vals);
         m_split_valid_count[f] = valid;
         if(valid < 2)
         {
            m_split_border_count[f] = 0;
            continue;
         }

         ArraySort(vals);
         int desired = FXAI_CAT_BINS - 1;
         if(desired > valid - 1) desired = valid - 1;

         int out = 0;
         double last = -DBL_MAX;
         for(int b=1; b<=desired; b++)
         {
            int qi = (int)MathFloor(((double)b / (double)(desired + 1)) * (double)(valid - 1));
            if(qi < 0) qi = 0;
            if(qi >= valid) qi = valid - 1;
            double thr = vals[qi];
            if(out == 0 || MathAbs(thr - last) > 1e-12)
            {
               m_split_borders[f][out] = thr;
               last = thr;
               out++;
               if(out >= FXAI_CAT_MAX_BORDERS) break;
            }
         }
         m_split_border_count[f] = out;
      }
   }

   void BuildPermutations(const int n,
                          int &perm[][FXAI_CAT_ORDER_PERMS])
   {
      ArrayResize(perm, n);
      for(int i=0; i<n; i++)
      {
         for(int p=0; p<FXAI_CAT_ORDER_PERMS; p++)
            perm[i][p] = i;
      }

      for(int p=1; p<FXAI_CAT_ORDER_PERMS; p++)
      {
         for(int i=n-1; i>0; i--)
         {
            int j = (int)MathFloor(Rand01() * (double)(i + 1));
            if(j < 0) j = 0;
            if(j > i) j = i;
            int tmp = perm[i][p];
            perm[i][p] = perm[j][p];
            perm[j][p] = tmp;
         }
      }
   }

   void BuildGlobalCTRStats(const int n,
                            const int &y_all[],
                            const int &qbin[][FXAI_CAT_CTR_BASE])
   {
      ResetCTRState();

      double cls_cnt[FXAI_CAT_CLASS_COUNT] = {0.0, 0.0, 0.0};
      for(int i=0; i<n; i++)
      {
         int cls = y_all[i];
         if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;
         cls_cnt[cls] += 1.0;
      }

      double total = cls_cnt[0] + cls_cnt[1] + cls_cnt[2];
      if(total <= 0.0) total = 1.0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_ctr_global_class[c] = FXAI_Clamp(cls_cnt[c] / total, 0.001, 0.998);

      for(int i=0; i<n; i++)
      {
         int cls = y_all[i];
         if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;

         for(int bf=0; bf<FXAI_CAT_CTR_BASE; bf++)
         {
            int bin = qbin[i][bf];
            if(bin < 0 || bin >= FXAI_CAT_BINS) continue;
            m_ctr_bin_total[bf][bin] += 1.0;
            m_ctr_bin_class[bf][bin][cls] += 1.0;
         }

         for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
         {
            int ba = qbin[i][m_ctr_pair_a[q]];
            int bb = qbin[i][m_ctr_pair_b[q]];
            if(ba < 0 || bb < 0) continue;
            int h = PairHash(ba, bb, q);
            m_ctr_pair_total[q][h] += 1.0;
            double ydir = (cls == (int)FXAI_LABEL_BUY ? 1.0 : (cls == (int)FXAI_LABEL_SKIP ? 0.5 : 0.0));
            m_ctr_pair_buy[q][h] += ydir;
         }
      }

      m_ctr_ready = true;
   }

   void BuildOrderedCTRFeatures(const int n,
                                const int &y_all[],
                                const int &qbin[][FXAI_CAT_CTR_BASE],
                                double &ctr_all[][FXAI_CAT_CTR_FEATURES],
