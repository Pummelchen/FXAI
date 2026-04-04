      int leaves = 1;
      while(leaves < FXAI_LGB_MAX_LEAVES)
      {
         int best_leaf = -1;
         int best_feature = -1;
         double best_thr = 0.0;
         bool best_default_left = true;
         double best_gain = 0.0;

         for(int node=0; node<tree.node_count; node++)
         {
            if(!tree.nodes[node].is_leaf) continue;
            if(tree.nodes[node].depth >= FXAI_LGB_MAX_DEPTH) continue;
            if(tree.nodes[node].sample_count < 2 * FXAI_LGB_MIN_DATA) continue;

            int f = -1;
            double thr = 0.0;
            bool def_left = true;
            double gain = 0.0;
            if(!FindBestSplitForLeaf(assign,
                                     node,
                                     tree.nodes[node].depth,
                                     x_use,
                                     g_use,
                                     h_use,
                                     lambda,
                                     feat_active,
                                     f,
                                     thr,
                                     def_left,
                                     gain))
               continue;

            if(gain > best_gain)
            {
               best_gain = gain;
               best_leaf = node;
               best_feature = f;
               best_thr = thr;
               best_default_left = def_left;
            }
         }

         if(best_leaf < 0 || best_feature < 1 || best_gain <= 0.0) break;
         if(tree.node_count + 2 > FXAI_LGB_MAX_NODES) break;

         int left = tree.node_count;
         int right = tree.node_count + 1;
         tree.node_count += 2;

         tree.nodes[best_leaf].is_leaf = false;
         tree.nodes[best_leaf].feature = best_feature;
         tree.nodes[best_leaf].threshold = best_thr;
         tree.nodes[best_leaf].default_left = best_default_left;
         tree.nodes[best_leaf].left = left;
         tree.nodes[best_leaf].right = right;

         tree.nodes[left].depth = tree.nodes[best_leaf].depth + 1;
         tree.nodes[right].depth = tree.nodes[best_leaf].depth + 1;

         for(int i=0; i<n_use; i++)
         {
            if(assign[i] != best_leaf) continue;
            double xv = x_use[i][best_feature];
            bool go_left = (!MathIsValidNumber(xv) ? best_default_left : (xv <= best_thr));
            assign[i] = (go_left ? left : right);
         }

         SetLeafFromAssign(tree, left, assign, left, g_use, h_use, mv_use, eta, lambda);
         SetLeafFromAssign(tree, right, assign, right, g_use, h_use, mv_use, eta, lambda);
         leaves++;
      }

      if(m_tree_count[class_id] < FXAI_LGB_MAX_TREES)
      {
         m_trees[class_id][m_tree_count[class_id]] = tree;
         m_tree_count[class_id]++;
      }
      else
      {
         for(int t=1; t<FXAI_LGB_MAX_TREES; t++) m_trees[class_id][t - 1] = m_trees[class_id][t];
         m_trees[class_id][FXAI_LGB_MAX_TREES - 1] = tree;
         m_tree_count[class_id] = FXAI_LGB_MAX_TREES;
      }

      return true;
   }

   void BuildCalLogits(const double &p_raw[], double &logits[]) const
   {
      double lraw[FXAI_LGB_CLASS_COUNT];
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         double z = m_cal_vs_b[c];
         for(int j=0; j<FXAI_LGB_CLASS_COUNT; j++) z += m_cal_vs_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

   void Calibrate3(const double &p_raw[], double &p_cal[]) const
   {
      double logits[FXAI_LGB_CLASS_COUNT];
      BuildCalLogits(p_raw, logits);
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_LGB_CLASS_COUNT];
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_LGB_CAL_BINS; b++) total += m_cal_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_LGB_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_LGB_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal_iso_cnt[c][b] > 1e-9) r = m_cal_iso_pos[c][b] / m_cal_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_LGB_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_LGB_CAL_BINS) bi = FXAI_LGB_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
         p_cal[c] = FXAI_Clamp(0.75 * p_cal[c] + 0.25 * p_iso[c], 0.0005, 0.9990);

      double s = p_cal[0] + p_cal[1] + p_cal[2];
      if(s <= 0.0) s = 1.0;
      p_cal[0] /= s;
      p_cal[1] /= s;
      p_cal[2] /= s;
   }

   void UpdateCalibrator3(const double &p_raw[], const int cls, const double sample_w, const double lr)
   {
      double logits[FXAI_LGB_CLASS_COUNT];
      BuildCalLogits(p_raw, logits);

      double p_cal[FXAI_LGB_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double lraw[FXAI_LGB_CLASS_COUNT];
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++) lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.20, 8.00);
      double cal_lr = FXAI_Clamp(0.25 * lr * w, 0.0002, 0.0200);
      double reg_l2 = 0.0005;

      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_vs_b[c] = FXAI_ClipSym(m_cal_vs_b[c] + cal_lr * e, 4.0);
         for(int j=0; j<FXAI_LGB_CLASS_COUNT; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double grad = e * lraw[j] - reg_l2 * (m_cal_vs_w[c][j] - target_w);
            m_cal_vs_w[c][j] = FXAI_ClipSym(m_cal_vs_w[c][j] + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_LGB_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_LGB_CAL_BINS) bi = FXAI_LGB_CAL_BINS - 1;
         m_cal_iso_cnt[c][bi] += w;
         m_cal_iso_pos[c][bi] += w * target;
      }
      m_cal3_steps++;
   }

   void UpdateValidationMetrics(const int cls,
                                const double &p_cal[],
                                const double expected_move_points,
                                const double cost_points)
   {
      int y = ClampI(cls, 0, FXAI_LGB_CLASS_COUNT - 1);
      double ce = -MathLog(FXAI_Clamp(p_cal[y], 1e-6, 1.0));
      double brier = 0.0;
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         double t = (c == y ? 1.0 : 0.0);
         double d = p_cal[c] - t;
         brier += d * d;
      }
      brier /= 3.0;

      double conf = p_cal[0];
      int pred = 0;
      for(int c=1; c<FXAI_LGB_CLASS_COUNT; c++) if(p_cal[c] > conf) { conf = p_cal[c]; pred = c; }
      double acc = (pred == y ? 1.0 : 0.0);

      int bi = (int)MathFloor(conf * (double)FXAI_LGB_ECE_BINS);
      if(bi < 0) bi = 0;
      if(bi >= FXAI_LGB_ECE_BINS) bi = FXAI_LGB_ECE_BINS - 1;
      for(int b=0; b<FXAI_LGB_ECE_BINS; b++)
      {
         m_ece_mass[b] *= 0.997;
         m_ece_acc[b] *= 0.997;
         m_ece_conf[b] *= 0.997;
      }
      m_ece_mass[bi] += 1.0;
      m_ece_acc[bi] += acc;
      m_ece_conf[bi] += conf;

      double ece_num = 0.0, ece_den = 0.0;
      for(int b=0; b<FXAI_LGB_ECE_BINS; b++)
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
         m_val_ev_fast = m_val_ev_slow = ev_after_cost;
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
         m_val_ev_fast = 0.92 * m_val_ev_fast + 0.08 * ev_after_cost;
         m_val_ev_slow = 0.995 * m_val_ev_slow + 0.005 * ev_after_cost;
      }

      m_val_steps++;
      m_quality_degraded = false;
      if(m_val_steps > 128)
      {
         if(m_val_nll_fast > 1.15 * MathMax(0.05, m_val_nll_slow)) m_quality_degraded = true;
         if(m_val_brier_fast > 1.20 * MathMax(0.03, m_val_brier_slow)) m_quality_degraded = true;
         if(m_val_ece_fast > 1.25 * MathMax(0.02, m_val_ece_slow)) m_quality_degraded = true;
         if(m_val_ev_fast < 0.85 * m_val_ev_slow) m_quality_degraded = true;
      }
   }

   double LeafExpectedMove(const FXAILGBNode &leaf) const
   {
      double sigma = MathSqrt(MathMax(0.0, leaf.move_var));
      return 0.50 * leaf.move_q50 + 0.30 * leaf.move_mean + 0.15 * leaf.move_q90 + 0.05 * sigma;
   }

   void ClassMoveStats(const int cls,
                       const double &x[],
                       double &mean,
                       double &q10,
                       double &q50,
                       double &q90,
                       double &support) const
   {
      mean = 0.0;
      q10 = 0.0;
      q50 = 0.0;
      q90 = 0.0;
      support = 0.0;
      double wsum = 0.0;
      for(int t=0; t<m_tree_count[cls]; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[cls][t], x);
         if(leaf < 0 || leaf >= m_trees[cls][t].node_count) continue;
         FXAILGBNode nd = m_trees[cls][t].nodes[leaf];
         if(nd.sample_count <= 0) continue;
         double lw = MathAbs(nd.leaf_value) + 0.10;
         mean += lw * MathMax(0.0, nd.move_mean);
         q10 += lw * MathMax(0.0, nd.move_q10);
         q50 += lw * MathMax(0.0, nd.move_q50);
         q90 += lw * MathMax(0.0, nd.move_q90);
         support += lw * (double)nd.sample_count;
         wsum += lw;
      }
      if(wsum > 0.0)
      {
         mean /= wsum;
         q10 /= wsum;
         q50 /= wsum;
         q90 /= wsum;
      }
   }

   double ClassExpectedMove(const int cls, const double &x[]) const
   {
      double sum = 0.0;
      double wsum = 0.0;
      for(int t=0; t<m_tree_count[cls]; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[cls][t], x);
         if(leaf < 0 || leaf >= m_trees[cls][t].node_count) continue;
         FXAILGBNode nd = m_trees[cls][t].nodes[leaf];
         if(nd.sample_count <= 0) continue;
         double mv = LeafExpectedMove(nd);
         if(mv <= 0.0) continue;
         double lw = MathAbs(nd.leaf_value) + 0.10;
         sum += lw * mv;
         wsum += lw;
      }
      if(wsum > 0.0) return sum / wsum;
      return 0.0;
   }

