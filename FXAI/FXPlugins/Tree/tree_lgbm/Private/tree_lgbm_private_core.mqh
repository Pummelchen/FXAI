   bool   m_initialized;
   int    m_step;
   double m_bias[FXAI_LGB_CLASS_COUNT];

   FXAILGBTree m_trees[FXAI_LGB_CLASS_COUNT][FXAI_LGB_MAX_TREES];
   int        m_tree_count[FXAI_LGB_CLASS_COUNT];

   // Drift-aware replay ring buffer.
   double m_buf_x[FXAI_LGB_BUFFER][FXAI_AI_WEIGHTS];
   int    m_buf_cls[FXAI_LGB_BUFFER];
   double m_buf_move[FXAI_LGB_BUFFER];
   double m_buf_cost[FXAI_LGB_BUFFER];
   double m_buf_w[FXAI_LGB_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   // Native 3-class calibration (vector scaling + isotonic bins).
   double m_cal_vs_w[FXAI_LGB_CLASS_COUNT][FXAI_LGB_CLASS_COUNT];
   double m_cal_vs_b[FXAI_LGB_CLASS_COUNT];
   double m_cal_iso_pos[FXAI_LGB_CLASS_COUNT][FXAI_LGB_CAL_BINS];
   double m_cal_iso_cnt[FXAI_LGB_CLASS_COUNT][FXAI_LGB_CAL_BINS];
   int    m_cal3_steps;

   // Online validation / reliability gating.
   bool   m_val_ready;
   int    m_val_steps;
   double m_val_nll_fast;
   double m_val_nll_slow;
   double m_val_brier_fast;
   double m_val_brier_slow;
   double m_val_ece_fast;
   double m_val_ece_slow;
   double m_val_ev_fast;
   double m_val_ev_slow;
   double m_ece_mass[FXAI_LGB_ECE_BINS];
   double m_ece_acc[FXAI_LGB_ECE_BINS];
   double m_ece_conf[FXAI_LGB_ECE_BINS];
   bool   m_quality_degraded;
   CFXAINativeQualityHeads m_quality_heads;

   int ClampI(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
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

   void InitTree(FXAILGBTree &tree) const
   {
      tree.node_count = 1;
      for(int n=0; n<FXAI_LGB_MAX_NODES; n++)
      {
         tree.nodes[n].is_leaf = true;
         tree.nodes[n].feature = -1;
         tree.nodes[n].threshold = 0.0;
         tree.nodes[n].default_left = true;
         tree.nodes[n].left = -1;
         tree.nodes[n].right = -1;
         tree.nodes[n].depth = 0;
         tree.nodes[n].leaf_value = 0.0;
         tree.nodes[n].move_mean = 0.0;
         tree.nodes[n].move_var = 0.0;
         tree.nodes[n].move_q10 = 0.0;
         tree.nodes[n].move_q50 = 0.0;
         tree.nodes[n].move_q90 = 0.0;
         tree.nodes[n].sample_count = 0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FXAI_LGB_BUFFER;
      int p = start + logical_idx;
      while(p >= FXAI_LGB_BUFFER) p -= FXAI_LGB_BUFFER;
      return p;
   }

   void PushSample(const int cls,
                   const double &x[],
                   const double move_points,
                   const double cost_points,
                   const double sample_w)
   {
      int pos = m_buf_head;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_buf_x[pos][i] = x[i];
      m_buf_cls[pos] = cls;
      m_buf_move[pos] = move_points;
      m_buf_cost[pos] = cost_points;
      m_buf_w[pos] = sample_w;

      m_buf_head++;
      if(m_buf_head >= FXAI_LGB_BUFFER) m_buf_head = 0;
      if(m_buf_size < FXAI_LGB_BUFFER) m_buf_size++;
   }

   int TraverseLeafIndex(const FXAILGBTree &tree, const double &x[]) const
   {
      int node = 0;
      int guard = 0;
      while(node >= 0 && node < tree.node_count && guard < FXAI_LGB_MAX_NODES)
      {
         if(tree.nodes[node].is_leaf) return node;

         int f = tree.nodes[node].feature;
         if(f < 1 || f >= FXAI_AI_WEIGHTS) return node;
         double xv = x[f];
         bool go_left;
         if(!MathIsValidNumber(xv)) go_left = tree.nodes[node].default_left;
         else go_left = (xv <= tree.nodes[node].threshold);

         int nxt = (go_left ? tree.nodes[node].left : tree.nodes[node].right);
         if(nxt < 0 || nxt >= tree.node_count) return node;
         node = nxt;
         guard++;
      }
      return 0;
   }

   double TreeOutput(const FXAILGBTree &tree, const double &x[]) const
   {
      int leaf = TraverseLeafIndex(tree, x);
      if(leaf < 0 || leaf >= tree.node_count) return 0.0;
      return tree.nodes[leaf].leaf_value;
   }

   void ModelRawLogits(const double &x[], double &logits[]) const
   {
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         double s = m_bias[c];
         for(int t=0; t<m_tree_count[c]; t++) s += TreeOutput(m_trees[c][t], x);
         logits[c] = s;
      }
   }

   void ModelRawLogitsDropClass(const double &x[],
                                const int drop_class,
                                const int &keep_mask[],
                                const double keep_scale,
                                double &logits[]) const
   {
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         double s = m_bias[c];
         for(int t=0; t<m_tree_count[c]; t++)
         {
            if(c == drop_class && ArraySize(keep_mask) == m_tree_count[c])
            {
               if(keep_mask[t] == 0) continue;
               s += keep_scale * TreeOutput(m_trees[c][t], x);
            }
            else s += TreeOutput(m_trees[c][t], x);
         }
         logits[c] = s;
      }
   }

   int BinByRange(const double x, const double minv, const double maxv) const
   {
      if(!MathIsValidNumber(x)) return -1;
      double range = maxv - minv;
      if(range <= 1e-12) return 0;
      double q = (x - minv) / range;
      if(q <= 0.0) return 0;
      if(q >= 1.0) return FXAI_LGB_BINS - 1;
      int b = (int)MathFloor(q * (double)FXAI_LGB_BINS);
      if(b < 0) b = 0;
      if(b >= FXAI_LGB_BINS) b = FXAI_LGB_BINS - 1;
      return b;
   }

   void BuildTargetDist(const int cls,
                        const double move_points,
                        const double cost_points,
                        double &target[]) const
   {
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++) target[c] = 0.0;
      int y = ClampI(cls, 0, FXAI_LGB_CLASS_COUNT - 1);
      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);

      if(y == (int)FXAI_LABEL_SKIP)
      {
         target[(int)FXAI_LABEL_SKIP] = 1.0;
         return;
      }

      int dir = (y == (int)FXAI_LABEL_BUY ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      if(edge <= 0.0)
      {
         target[dir] = 0.35;
         target[(int)FXAI_LABEL_SKIP] = 0.65;
         return;
      }

      double pdir = FXAI_Clamp(0.75 + 0.10 * edge / MathMax(cost_points, 1.0), 0.75, 0.95);
      target[dir] = pdir;
      target[(int)FXAI_LABEL_SKIP] = 1.0 - pdir;
   }

   double SampleTimeDecay(const int age) const
   {
      // Half-life in samples: recent bars dominate.
      const double half_life = 512.0;
      double a = (double)MathMax(age, 0);
      return MathExp(-0.69314718056 * a / half_life);
   }

   double EVWeight(const int cls,
                   const double move_points,
                   const double cost_points,
                   const double base_w) const
   {
      double w = FXAI_Clamp(base_w, 0.10, 6.0);
      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);
      if(cls == (int)FXAI_LABEL_SKIP)
      {
         if(edge <= 0.0) return FXAI_Clamp(1.50 * w, 0.10, 8.0);
         return FXAI_Clamp(0.80 * w, 0.10, 8.0);
      }

      if(edge <= 0.0) return FXAI_Clamp(0.55 * w, 0.10, 8.0);
      return FXAI_Clamp(w * (1.0 + 0.08 * MathMin(edge, 30.0)), 0.10, 8.0);
   }

   void SetLeafFromAssign(FXAILGBTree &tree,
                          const int node_idx,
                          const int &assign[],
                          const int tag,
                          const double &g[],
                          const double &h[],
                          const double &mv[],
                          const double eta,
                          const double lambda) const
   {
      double G = 0.0, H = 0.0;
      double sum_mv = 0.0, sum_mv2 = 0.0;
      int n = ArraySize(assign);
      int cnt = 0;

      double vals[];
      ArrayResize(vals, 0);

      for(int i=0; i<n; i++)
      {
         if(assign[i] != tag) continue;
         G += g[i];
         H += h[i];

         double av = MathAbs(mv[i]);
         sum_mv += av;
         sum_mv2 += av * av;
         int sz = ArraySize(vals);
         ArrayResize(vals, sz + 1);
         vals[sz] = av;
         cnt++;
      }

      tree.nodes[node_idx].is_leaf = true;
      tree.nodes[node_idx].feature = -1;
      tree.nodes[node_idx].threshold = 0.0;
      tree.nodes[node_idx].default_left = true;
      tree.nodes[node_idx].left = -1;
      tree.nodes[node_idx].right = -1;
      tree.nodes[node_idx].sample_count = cnt;

      double mean = (cnt > 0 ? sum_mv / (double)cnt : 0.0);
      double var = 0.0;
      if(cnt > 0)
      {
         var = (sum_mv2 / (double)cnt) - mean * mean;
         if(var < 0.0) var = 0.0;
      }
      tree.nodes[node_idx].move_mean = mean;
      tree.nodes[node_idx].move_var = var;

      tree.nodes[node_idx].move_q10 = mean;
      tree.nodes[node_idx].move_q50 = mean;
      tree.nodes[node_idx].move_q90 = mean;
      if(cnt > 0)
      {
         ArraySort(vals);
         int i10 = (int)MathFloor(0.10 * (double)(cnt - 1));
         int i50 = (int)MathFloor(0.50 * (double)(cnt - 1));
         int i90 = (int)MathFloor(0.90 * (double)(cnt - 1));
         i10 = ClampI(i10, 0, cnt - 1);
         i50 = ClampI(i50, 0, cnt - 1);
         i90 = ClampI(i90, 0, cnt - 1);
         tree.nodes[node_idx].move_q10 = vals[i10];
         tree.nodes[node_idx].move_q50 = vals[i50];
         tree.nodes[node_idx].move_q90 = vals[i90];
      }

      double leaf = 0.0;
      if(H > 1e-9) leaf = eta * FXAI_ClipSym(G / (H + lambda), 6.0);
      tree.nodes[node_idx].leaf_value = leaf;
   }

   bool FindBestSplitForLeaf(const int &assign[],
                             const int leaf_tag,
                             const int depth,
                             const double &x_all[][FXAI_AI_WEIGHTS],
                             const double &g[],
                             const double &h[],
                             const double lambda,
                             const int &feat_active[],
                             int &best_feature,
                             double &best_thr,
                             bool &best_default_left,
                             double &best_gain) const
   {
      best_feature = -1;
      best_thr = 0.0;
      best_default_left = true;
      best_gain = 0.0;
      if(depth >= FXAI_LGB_MAX_DEPTH) return false;

