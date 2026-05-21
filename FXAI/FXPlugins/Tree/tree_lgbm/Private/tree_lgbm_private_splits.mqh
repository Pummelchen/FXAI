      int n = ArraySize(assign);
      double Gtot = 0.0, Htot = 0.0;
      int Ctot = 0;
      for(int i=0; i<n; i++)
      {
         if(assign[i] != leaf_tag) continue;
         Gtot += g[i];
         Htot += h[i];
         Ctot++;
      }

      if(Ctot < 2 * FXAI_LGB_MIN_DATA) return false;
      if(Htot < 2.0 * FXAI_LGB_MIN_CHILD_HESS) return false;

      double parent_score = (Gtot * Gtot) / (Htot + lambda + 1e-9);

      for(int f=1; f<FXAI_AI_WEIGHTS; f++)
      {
         if(ArraySize(feat_active) == FXAI_AI_WEIGHTS && feat_active[f] == 0) continue;

         double minv = DBL_MAX;
         double maxv = -DBL_MAX;
         int c_valid = 0, c_miss = 0;
         double Gmiss = 0.0, Hmiss = 0.0;

         for(int i=0; i<n; i++)
         {
            if(assign[i] != leaf_tag) continue;
            double xv = x_all[i][f];
            if(!MathIsValidNumber(xv))
            {
               c_miss++;
               Gmiss += g[i];
               Hmiss += h[i];
               continue;
            }
            if(xv < minv) minv = xv;
            if(xv > maxv) maxv = xv;
            c_valid++;
         }

         if(c_valid < 2 * FXAI_LGB_MIN_DATA) continue;
         if(maxv - minv < 1e-9) continue;

         double gbin[FXAI_LGB_BINS];
         double hbin[FXAI_LGB_BINS];
         int cbin[FXAI_LGB_BINS];
         for(int b=0; b<FXAI_LGB_BINS; b++)
         {
            gbin[b] = 0.0;
            hbin[b] = 0.0;
            cbin[b] = 0;
         }

         for(int i=0; i<n; i++)
         {
            if(assign[i] != leaf_tag) continue;
            double xv = x_all[i][f];
            int bi = BinByRange(xv, minv, maxv);
            if(bi < 0) continue;
            gbin[bi] += g[i];
            hbin[bi] += h[i];
            cbin[bi]++;
         }

         double GL = 0.0, HL = 0.0;
         int CL = 0;
         for(int sb=0; sb<FXAI_LGB_BINS - 1; sb++)
         {
            GL += gbin[sb];
            HL += hbin[sb];
            CL += cbin[sb];

            // missing -> left
            {
               double gL = GL + Gmiss;
               double hL = HL + Hmiss;
               int cL = CL + c_miss;
               double gR = Gtot - gL;
               double hR = Htot - hL;
               int cR = Ctot - cL;

               if(cL >= FXAI_LGB_MIN_DATA && cR >= FXAI_LGB_MIN_DATA &&
                  hL >= FXAI_LGB_MIN_CHILD_HESS && hR >= FXAI_LGB_MIN_CHILD_HESS)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FXAI_LGB_GAMMA;
                  if(gain > best_gain)
                  {
                     best_gain = gain;
                     best_feature = f;
                     best_default_left = true;
                     best_thr = minv + (maxv - minv) * ((double)(sb + 1) / (double)FXAI_LGB_BINS);
                  }
               }
            }

            // missing -> right
            {
               double gL = GL;
               double hL = HL;
               int cL = CL;
               double gR = Gtot - gL;
               double hR = Htot - hL;
               int cR = Ctot - cL;

               if(cL >= FXAI_LGB_MIN_DATA && cR >= FXAI_LGB_MIN_DATA &&
                  hL >= FXAI_LGB_MIN_CHILD_HESS && hR >= FXAI_LGB_MIN_CHILD_HESS)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FXAI_LGB_GAMMA;
                  if(gain > best_gain)
                  {
                     best_gain = gain;
                     best_feature = f;
                     best_default_left = false;
                     best_thr = minv + (maxv - minv) * ((double)(sb + 1) / (double)FXAI_LGB_BINS);
                  }
               }
            }
         }
      }

      return (best_feature >= 1 && best_gain > 0.0);
   }

   double GOSSThreshold(const double &abs_g[], const int n, const double top_rate) const
   {
      if(n <= 0) return DBL_MAX;
      double minv = abs_g[0], maxv = abs_g[0];
      for(int i=1; i<n; i++)
      {
         if(abs_g[i] < minv) minv = abs_g[i];
         if(abs_g[i] > maxv) maxv = abs_g[i];
      }
      if(maxv - minv <= 1e-12) return minv;

      int hist[FXAI_LGB_GOSS_BINS];
      for(int b=0; b<FXAI_LGB_GOSS_BINS; b++) hist[b] = 0;
      for(int i=0; i<n; i++)
      {
         double q = (abs_g[i] - minv) / (maxv - minv);
         int bi = (int)MathFloor(q * (double)FXAI_LGB_GOSS_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_LGB_GOSS_BINS) bi = FXAI_LGB_GOSS_BINS - 1;
         hist[bi]++;
      }

      int need = (int)MathCeil(FXAI_Clamp(top_rate, 0.01, 0.95) * (double)n);
      if(need < 1) need = 1;

      int cum = 0;
      for(int bi=FXAI_LGB_GOSS_BINS - 1; bi>=0; bi--)
      {
         cum += hist[bi];
         if(cum >= need)
         {
            double lo = minv + (maxv - minv) * ((double)bi / (double)FXAI_LGB_GOSS_BINS);
            return lo;
         }
      }
      return minv;
   }

   bool BuildOneTreeClass(const int class_id, const FXAIAIHyperParams &hp)
   {
      if(m_buf_size < FXAI_LGB_MIN_BUFFER) return false;

      int n = m_buf_size;
      double x_all[][FXAI_AI_WEIGHTS];
      int cls_all[];
      double mv_all[];
      double cost_all[];
      double w_all[];
      ArrayResize(x_all, n);
      ArrayResize(cls_all, n);
      ArrayResize(mv_all, n);
      ArrayResize(cost_all, n);
      ArrayResize(w_all, n);

      // Chronological extraction from replay ring.
      for(int i=0; i<n; i++)
      {
         int p = BufPos(i);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) x_all[i][k] = m_buf_x[p][k];
         cls_all[i] = m_buf_cls[p];
         mv_all[i] = m_buf_move[p];
         cost_all[i] = m_buf_cost[p];
         w_all[i] = m_buf_w[p];
      }

      // Class-balanced factors.
      int cls_cnt[FXAI_LGB_CLASS_COUNT];
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++) cls_cnt[c] = 0;
      for(int i=0; i<n; i++) cls_cnt[ClampI(cls_all[i], 0, FXAI_LGB_CLASS_COUNT - 1)]++;
      double mean_cnt = (double)(cls_cnt[0] + cls_cnt[1] + cls_cnt[2]) / 3.0;
      if(mean_cnt < 1.0) mean_cnt = 1.0;

      // DART-like dropout mask on existing trees of this class.
      int keep_mask[];
      ArrayResize(keep_mask, m_tree_count[class_id]);
      const double drop_rate = 0.05;
      int kept = 0;
      for(int t=0; t<m_tree_count[class_id]; t++)
      {
         int k = 1;
         if(t >= 4)
         {
            double u = PluginRand01();
            if(u < drop_rate) k = 0;
         }
         keep_mask[t] = k;
         if(k == 1) kept++;
      }
      if(m_tree_count[class_id] > 0 && kept <= 0)
      {
         int r = PluginRandIndex(m_tree_count[class_id]);
         keep_mask[r] = 1;
         kept = 1;
      }
      double keep_scale = (drop_rate < 0.99 ? 1.0 / MathMax(1e-6, 1.0 - drop_rate) : 1.0);

      // First pass: bagging + gradient/hessian construction for one-vs-rest class tree.
      int cand_idx[];
      double cand_g[];
      double cand_h[];
      double cand_abs[];
      ArrayResize(cand_idx, 0);
      ArrayResize(cand_g, 0);
      ArrayResize(cand_h, 0);
      ArrayResize(cand_abs, 0);

      const double bagging_fraction = 0.80;
      for(int i=0; i<n; i++)
      {
         double ub = PluginRand01();
         if(ub > bagging_fraction) continue;

         double xloc[FXAI_AI_WEIGHTS];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) xloc[k] = x_all[i][k];

         double logits[FXAI_LGB_CLASS_COUNT];
         ModelRawLogitsDropClass(xloc, class_id, keep_mask, keep_scale, logits);
         double probs[FXAI_LGB_CLASS_COUNT];
         Softmax3(logits, probs);

         double target[FXAI_LGB_CLASS_COUNT];
         BuildTargetDist(cls_all[i], mv_all[i], cost_all[i], target);

         int yi = ClampI(cls_all[i], 0, FXAI_LGB_CLASS_COUNT - 1);
         double bal = FXAI_Clamp(mean_cnt / MathMax(1.0, (double)cls_cnt[yi]), 0.50, 2.50);

         int age = n - 1 - i;
         double decay = SampleTimeDecay(age);
         double base_w = w_all[i] * decay * bal;
         double wi = EVWeight(yi, mv_all[i], cost_all[i], base_w);

         double p = FXAI_Clamp(probs[class_id], 0.001, 0.999);
         double g = (target[class_id] - p) * wi;
         double h = FXAI_Clamp(p * (1.0 - p) * wi, 0.02, 6.0);

         int sz = ArraySize(cand_idx);
         ArrayResize(cand_idx, sz + 1);
         ArrayResize(cand_g, sz + 1);
         ArrayResize(cand_h, sz + 1);
         ArrayResize(cand_abs, sz + 1);
         cand_idx[sz] = i;
         cand_g[sz] = g;
         cand_h[sz] = h;
         cand_abs[sz] = MathAbs(g);
      }

      int cand_n = ArraySize(cand_idx);
      if(cand_n < FXAI_LGB_MIN_BUFFER / 2) return false;

      // GOSS: keep top gradients + random subsample of the rest.
      const double top_rate = 0.20;
      const double other_rate = 0.10;
      double thr_top = GOSSThreshold(cand_abs, cand_n, top_rate);
      double small_scale = ((1.0 - top_rate) > 0.0 && other_rate > 0.0 ? (1.0 - top_rate) / other_rate : 1.0);

      double x_use[][FXAI_AI_WEIGHTS];
      double g_use[];
      double h_use[];
      double mv_use[];
      ArrayResize(x_use, 0);
      ArrayResize(g_use, 0);
      ArrayResize(h_use, 0);
      ArrayResize(mv_use, 0);

      for(int i=0; i<cand_n; i++)
      {
         bool keep = (cand_abs[i] >= thr_top);
         double gg = cand_g[i];
         double hh = cand_h[i];

         if(!keep)
         {
            double ur = PluginRand01();
            if(ur <= other_rate)
            {
               keep = true;
               gg *= small_scale;
               hh *= small_scale;
            }
         }
         if(!keep) continue;

         int src = cand_idx[i];
         int sz = ArraySize(g_use);
         ArrayResize(x_use, sz + 1);
         ArrayResize(g_use, sz + 1);
         ArrayResize(h_use, sz + 1);
         ArrayResize(mv_use, sz + 1);

         for(int k=0; k<FXAI_AI_WEIGHTS; k++) x_use[sz][k] = x_all[src][k];
         g_use[sz] = gg;
         h_use[sz] = hh;
         mv_use[sz] = mv_all[src];
      }

      int n_use = ArraySize(g_use);
      if(n_use < FXAI_LGB_MIN_BUFFER / 3) return false;

      // Feature subsampling mask.
      int feat_active[];
      ArrayResize(feat_active, FXAI_AI_WEIGHTS);
      feat_active[0] = 0;
      int feat_kept = 0;
      for(int f=1; f<FXAI_AI_WEIGHTS; f++)
      {
         double uf = PluginRand01();
         feat_active[f] = (uf <= 0.70 ? 1 : 0);
         if(feat_active[f] == 1) feat_kept++;
      }
      if(feat_kept <= 0)
      {
         int rf = 1 + PluginRandIndex(FXAI_AI_WEIGHTS - 1);
         feat_active[rf] = 1;
      }

      double G = 0.0, H = 0.0;
      for(int i=0; i<n_use; i++) { G += g_use[i]; H += h_use[i]; }

      double lambda = FXAI_Clamp(hp.xgb_l2, 0.0001, 10.0000);
      double eta = FXAI_Clamp(hp.xgb_lr, 0.0001, 0.5000);
      if(m_quality_degraded) eta *= 0.80;
      if(H > 1e-9)
         m_bias[class_id] += 0.15 * eta * FXAI_ClipSym(G / (H + lambda), 5.0);

      FXAILGBTree tree;
      InitTree(tree);

      int assign[];
      ArrayResize(assign, n_use);
      for(int i=0; i<n_use; i++) assign[i] = 0;
      SetLeafFromAssign(tree, 0, assign, 0, g_use, h_use, mv_use, eta, lambda);

