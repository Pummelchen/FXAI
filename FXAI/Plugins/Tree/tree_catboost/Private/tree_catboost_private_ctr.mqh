                                double &ordered_bias[][FXAI_CAT_CLASS_COUNT])
   {
      double global_prior[FXAI_CAT_CLASS_COUNT];
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         global_prior[c] = m_ctr_global_class[c];
      double global_dir = global_prior[(int)FXAI_LABEL_BUY] + 0.5 * global_prior[(int)FXAI_LABEL_SKIP];

      int perm[][FXAI_CAT_ORDER_PERMS];
      BuildPermutations(n, perm);

      for(int i=0; i<n; i++)
      {
         for(int k=0; k<FXAI_CAT_CTR_FEATURES; k++) ctr_all[i][k] = 0.0;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++) ordered_bias[i][c] = 0.0;
      }

      const double prior = 2.0;
      for(int p=0; p<FXAI_CAT_ORDER_PERMS; p++)
      {
         double cls_prefix[FXAI_CAT_CLASS_COUNT] = {0.0, 0.0, 0.0};
         double cnt_base[FXAI_CAT_CTR_BASE][FXAI_CAT_BINS];
         double cls_base[FXAI_CAT_CTR_BASE][FXAI_CAT_BINS][FXAI_CAT_CLASS_COUNT];
         double cnt_pair[FXAI_CAT_CTR_PAIR_COUNT][FXAI_CAT_CTR_PAIR_HASH];
         double buy_pair[FXAI_CAT_CTR_PAIR_COUNT][FXAI_CAT_CTR_PAIR_HASH];

         for(int bf=0; bf<FXAI_CAT_CTR_BASE; bf++)
         {
            for(int b=0; b<FXAI_CAT_BINS; b++)
            {
               cnt_base[bf][b] = 0.0;
               for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
                  cls_base[bf][b][c] = 0.0;
            }
         }
         for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
         {
            for(int h=0; h<FXAI_CAT_CTR_PAIR_HASH; h++)
            {
               cnt_pair[q][h] = 0.0;
               buy_pair[q][h] = 0.0;
            }
         }

         for(int s=0; s<n; s++)
         {
            int i = perm[s][p];
            int cls = y_all[i];
            if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;

            double prefix_total = cls_prefix[0] + cls_prefix[1] + cls_prefix[2];
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            {
               double pr = (cls_prefix[c] + prior * global_prior[c]) / (prefix_total + prior);
               pr = FXAI_Clamp(pr, 0.001, 0.999);
               ordered_bias[i][c] += MathLog(pr);
            }

            int off = 0;
            for(int bf=0; bf<FXAI_CAT_CTR_BASE; bf++)
            {
               int bin = qbin[i][bf];
               if(bin < 0 || bin >= FXAI_CAT_BINS)
               {
                  ctr_all[i][off + 0] += global_prior[(int)FXAI_LABEL_SELL];
                  ctr_all[i][off + 1] += global_prior[(int)FXAI_LABEL_BUY];
                  ctr_all[i][off + 2] += global_prior[(int)FXAI_LABEL_SKIP];
               }
               else
               {
                  double total = cnt_base[bf][bin];
                  for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
                  {
                     double pr = (cls_base[bf][bin][c] + prior * global_prior[c]) / (total + prior);
                     ctr_all[i][off + c] += FXAI_Clamp(pr, 0.001, 0.999);
                  }
               }
               off += FXAI_CAT_CTR_FEAT_PER_BASE;
            }

            for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
            {
               int ba = qbin[i][m_ctr_pair_a[q]];
               int bb = qbin[i][m_ctr_pair_b[q]];
               double pr = global_dir;
               if(ba >= 0 && bb >= 0)
               {
                  int h = PairHash(ba, bb, q);
                  double total = cnt_pair[q][h];
                  pr = (buy_pair[q][h] + prior * global_dir) / (total + prior);
               }
               ctr_all[i][off + q] += FXAI_Clamp(pr, 0.001, 0.999);
            }

            cls_prefix[cls] += 1.0;
            for(int bf=0; bf<FXAI_CAT_CTR_BASE; bf++)
            {
               int bin = qbin[i][bf];
               if(bin < 0 || bin >= FXAI_CAT_BINS) continue;
               cnt_base[bf][bin] += 1.0;
               cls_base[bf][bin][cls] += 1.0;
            }
            for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
            {
               int ba = qbin[i][m_ctr_pair_a[q]];
               int bb = qbin[i][m_ctr_pair_b[q]];
               if(ba < 0 || bb < 0) continue;
               int h = PairHash(ba, bb, q);
               cnt_pair[q][h] += 1.0;
               double ydir = (cls == (int)FXAI_LABEL_BUY ? 1.0 : (cls == (int)FXAI_LABEL_SKIP ? 0.5 : 0.0));
               buy_pair[q][h] += ydir;
            }
         }
      }

      double inv_perm = 1.0 / (double)FXAI_CAT_ORDER_PERMS;
      for(int i=0; i<n; i++)
      {
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            ordered_bias[i][c] *= inv_perm;
         double meanb = (ordered_bias[i][0] + ordered_bias[i][1] + ordered_bias[i][2]) / 3.0;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            ordered_bias[i][c] -= meanb;

         for(int k=0; k<FXAI_CAT_CTR_FEATURES; k++)
         {
            double pr = ctr_all[i][k] * inv_perm;
            ctr_all[i][k] = FXAI_Clamp(2.0 * pr - 1.0, -1.0, 1.0);
         }
      }
   }

   void BuildExtendedTraining(const int n,
                              const double &x_all[][FXAI_AI_WEIGHTS],
                              const int &y_all[],
                              double &x_ext[][FXAI_CAT_EXT_WEIGHTS],
                              double &ordered_bias[][FXAI_CAT_CLASS_COUNT])
   {
      BuildBaseBorders(n, x_all);

      int qbin[][FXAI_CAT_CTR_BASE];
      ArrayResize(qbin, n);
      for(int i=0; i<n; i++)
      {
         for(int b=0; b<FXAI_CAT_CTR_BASE; b++)
         {
            int f = m_ctr_base_feat[b];
            qbin[i][b] = QuantizeBaseFeature(f, x_all[i][f]);
         }
      }

      BuildGlobalCTRStats(n, y_all, qbin);

      double ctr_all[][FXAI_CAT_CTR_FEATURES];
      ArrayResize(ctr_all, n);
      BuildOrderedCTRFeatures(n, y_all, qbin, ctr_all, ordered_bias);

      for(int i=0; i<n; i++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            x_ext[i][k] = x_all[i][k];

         int off = FXAI_AI_WEIGHTS;
         for(int k=0; k<FXAI_CAT_CTR_FEATURES; k++)
            x_ext[i][off + k] = ctr_all[i][k];
      }
   }

   void BuildInferenceExtended(const double &x[],
                               double &x_ext[]) const
   {
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_ext[k] = x[k];

      int off = FXAI_AI_WEIGHTS;
      if(!m_ctr_ready)
      {
         for(int k=0; k<FXAI_CAT_CTR_FEATURES; k++)
            x_ext[off + k] = 0.0;
         return;
      }

      int qbin[FXAI_CAT_CTR_BASE];
      for(int bf=0; bf<FXAI_CAT_CTR_BASE; bf++)
      {
         int f = m_ctr_base_feat[bf];
         qbin[bf] = QuantizeBaseFeature(f, x[f]);
      }

      const double prior = 2.0;
      int p = 0;
      for(int bf=0; bf<FXAI_CAT_CTR_BASE; bf++)
      {
         int bin = qbin[bf];
         if(bin < 0 || bin >= FXAI_CAT_BINS)
         {
            x_ext[off + p + 0] = 2.0 * m_ctr_global_class[(int)FXAI_LABEL_SELL] - 1.0;
            x_ext[off + p + 1] = 2.0 * m_ctr_global_class[(int)FXAI_LABEL_BUY] - 1.0;
            x_ext[off + p + 2] = 2.0 * m_ctr_global_class[(int)FXAI_LABEL_SKIP] - 1.0;
         }
         else
         {
            double total = m_ctr_bin_total[bf][bin];
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            {
               double pr = (m_ctr_bin_class[bf][bin][c] + prior * m_ctr_global_class[c]) / (total + prior);
               x_ext[off + p + c] = FXAI_Clamp(2.0 * pr - 1.0, -1.0, 1.0);
            }
         }
         p += FXAI_CAT_CTR_FEAT_PER_BASE;
      }

      double global_dir = m_ctr_global_class[(int)FXAI_LABEL_BUY] + 0.5 * m_ctr_global_class[(int)FXAI_LABEL_SKIP];
      for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
      {
         double pr = global_dir;
         int ba = qbin[m_ctr_pair_a[q]];
         int bb = qbin[m_ctr_pair_b[q]];
         if(ba >= 0 && bb >= 0)
         {
            int h = PairHash(ba, bb, q);
            double total = m_ctr_pair_total[q][h];
            pr = (m_ctr_pair_buy[q][h] + prior * global_dir) / (total + prior);
         }
         x_ext[off + p + q] = FXAI_Clamp(2.0 * pr - 1.0, -1.0, 1.0);
      }
   }

   double CoupledNewtonScore(const double &g_mat[][FXAI_CAT_CLASS_COUNT],
                             const double &h_mat[][FXAI_CAT_CLASS_COUNT],
                             const int row,
                             const double lambda) const
   {
      double score = 0.0;
      double gs = 0.0;
      double hs = 0.0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double g = g_mat[row][c];
         double hh = h_mat[row][c];
         score += (g * g) / (hh + lambda + 1e-9);
         gs += g;
         hs += hh;
      }
      score -= 0.25 * (gs * gs) / (hs + 3.0 * lambda + 1e-9);
      return score;
   }

   bool EvaluateSplit(const int depth,
                      const int n,
                      const int &leaf_idx[],
                      const double &x_ext[][FXAI_CAT_EXT_WEIGHTS],
                      const double &g[][FXAI_CAT_CLASS_COUNT],
                      const double &h[][FXAI_CAT_CLASS_COUNT],
                      const double &sample_w[],
                      const int feature,
                      const double threshold,
                      const bool default_left,
                      const double lambda,
                      double &gain_out) const
   {
      gain_out = 0.0;
      if(depth < 0 || depth >= FXAI_CAT_MAX_DEPTH) return false;
      if(feature < 1 || feature >= FXAI_CAT_EXT_WEIGHTS) return false;

      int old_leaf_count = (1 << depth);
      int new_leaf_count = (1 << (depth + 1));

      double parent_g[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
      double parent_h[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
      double parent_w[FXAI_CAT_MAX_LEAVES];

      double child_g[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
      double child_h[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
      double child_w[FXAI_CAT_MAX_LEAVES];

      for(int l=0; l<FXAI_CAT_MAX_LEAVES; l++)
      {
         parent_w[l] = 0.0;
         child_w[l] = 0.0;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         {
            parent_g[l][c] = 0.0;
            parent_h[l][c] = 0.0;
            child_g[l][c] = 0.0;
            child_h[l][c] = 0.0;
         }
      }

      for(int i=0; i<n; i++)
      {
         double wi = sample_w[i];
         if(wi <= 0.0) continue;

         int old_leaf = leaf_idx[i];
         if(old_leaf < 0 || old_leaf >= old_leaf_count) continue;

         parent_w[old_leaf] += wi;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         {
            parent_g[old_leaf][c] += g[i][c];
            parent_h[old_leaf][c] += h[i][c];
         }

         double xv = x_ext[i][feature];
         bool go_left;
         if(!MathIsValidNumber(xv))
            go_left = default_left;
         else
            go_left = (xv <= threshold);

         int child_leaf = (old_leaf << 1) | (go_left ? 0 : 1);
         if(child_leaf < 0 || child_leaf >= new_leaf_count) continue;

         child_w[child_leaf] += wi;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         {
            child_g[child_leaf][c] += g[i][c];
            child_h[child_leaf][c] += h[i][c];
         }
      }

      double gain = 0.0;
      for(int old_leaf=0; old_leaf<old_leaf_count; old_leaf++)
      {
         int left_leaf = (old_leaf << 1);
         int right_leaf = left_leaf + 1;
         if(left_leaf < 0 || right_leaf >= new_leaf_count) return false;

         if(parent_w[old_leaf] < (double)FXAI_CAT_MIN_DATA) continue;
         if(child_w[left_leaf] < (double)FXAI_CAT_MIN_DATA || child_w[right_leaf] < (double)FXAI_CAT_MIN_DATA)
            return false;

         double hsum_left = 0.0;
         double hsum_right = 0.0;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         {
            hsum_left += child_h[left_leaf][c];
            hsum_right += child_h[right_leaf][c];
         }
         if(hsum_left < FXAI_CAT_MIN_CHILD_HESS || hsum_right < FXAI_CAT_MIN_CHILD_HESS)
            return false;

         double parent_score = CoupledNewtonScore(parent_g, parent_h, old_leaf, lambda);
         double left_score = CoupledNewtonScore(child_g, child_h, left_leaf, lambda);
         double right_score = CoupledNewtonScore(child_g, child_h, right_leaf, lambda);

         gain += 0.5 * ((left_score + right_score) - parent_score);
      }

      // Stronger regularization: split gamma + depth and feature reuse penalty.
      gain -= FXAI_CAT_GAMMA;
      gain -= 0.005 * (double)(depth + 1);
      gain -= 0.002 * (double)m_feature_use[feature];

      gain_out = gain;
      return (gain > 1e-7);
   }

   void BuildBootstrapWeights(const int n,
                              const int mode,
                              const double base_subsample,
                              const double &w_all[],
                              double &boot_w[])
   {
      double subsample = FXAI_Clamp(base_subsample, 0.45, 1.00);
      double avg_w = 0.0;
      for(int i=0; i<n; i++) avg_w += MathAbs(w_all[i]);
      avg_w /= MathMax(1.0, (double)n);
      if(avg_w <= 0.0) avg_w = 1.0;

      for(int i=0; i<n; i++)
      {
         double wi = FXAI_Clamp(MathAbs(w_all[i]), 0.05, 10.0);
         double bw = 0.0;

         if(mode == 0)
         {
            // Bayesian bootstrap.
            double u = MathMax(1e-6, Rand01());
            bw = wi * (-MathLog(u));
         }
         else if(mode == 1)
         {
            // Bernoulli subsample.
            if(Rand01() <= subsample) bw = wi;
            else bw = 0.0;
         }
         else
         {
            // MVS-like: probability by sample magnitude, importance corrected.
            double p = FXAI_Clamp(subsample * (0.50 + 0.50 * wi / avg_w), 0.10, 1.0);
            if(Rand01() <= p) bw = wi / p;
            else bw = 0.0;
         }

         boot_w[i] = FXAI_Clamp(bw, 0.0, 12.0);
      }
   }

   double ComputeLeafLoss(const int n,
                          const int &leaf_idx[],
                          const int &y_all[],
                          const double &sample_w[],
                          const double &base_margin[][FXAI_CAT_CLASS_COUNT],
                          const double &leaf_value[][FXAI_CAT_CLASS_COUNT]) const
   {
      double loss = 0.0;
      double wsum = 0.0;
      for(int i=0; i<n; i++)
      {
         int leaf = leaf_idx[i];
         if(leaf < 0 || leaf >= FXAI_CAT_MAX_LEAVES) leaf = 0;

         double z[FXAI_CAT_CLASS_COUNT];
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            z[c] = base_margin[i][c] + leaf_value[leaf][c];

         double p[FXAI_CAT_CLASS_COUNT];
         Softmax3(z, p);

         int cls = y_all[i];
         if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;
         double wi = FXAI_Clamp(sample_w[i], 0.0, 12.0);
         if(wi <= 0.0) continue;

         loss += wi * (-MathLog(FXAI_Clamp(p[cls], 1e-7, 1.0)));
         wsum += wi;
      }
      if(wsum <= 0.0) return 0.0;
      return loss / wsum;
   }

   void BuildLeafValuesIterative(FXAICatTree &tree,
                                 const int n,
                                 const int &leaf_idx[],
                                 const int &y_all[],
                                 const double &mv_all[],
                                 const double &sample_w[],
                                 const double &base_margin[][FXAI_CAT_CLASS_COUNT],
                                 const double eta,
                                 const double lambda) const
   {
      int leaf_count = (tree.depth > 0 ? (1 << tree.depth) : 1);
      if(leaf_count < 1) leaf_count = 1;
      if(leaf_count > FXAI_CAT_MAX_LEAVES) leaf_count = FXAI_CAT_MAX_LEAVES;

      for(int l=0; l<FXAI_CAT_MAX_LEAVES; l++)
      {
         tree.leaf_count[l] = 0;
         tree.leaf_move_mean[l] = 0.0;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            tree.leaf_value[l][c] = 0.0;
      }

      double mv_sum[FXAI_CAT_MAX_LEAVES];
      for(int l=0; l<FXAI_CAT_MAX_LEAVES; l++) mv_sum[l] = 0.0;
      for(int i=0; i<n; i++)
      {
         int leaf = (tree.depth > 0 ? leaf_idx[i] : 0);
         if(leaf < 0 || leaf >= leaf_count) continue;
         tree.leaf_count[leaf]++;
         mv_sum[leaf] += MathAbs(mv_all[i]);
      }
      for(int l=0; l<leaf_count; l++)
      {
         if(tree.leaf_count[l] > 0)
            tree.leaf_move_mean[l] = mv_sum[l] / (double)tree.leaf_count[l];
      }

      double cand[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
      for(int iter=0; iter<FXAI_CAT_LEAF_NEWTON_STEPS; iter++)
      {
         double g_leaf[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
         double h_leaf[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
         for(int l=0; l<FXAI_CAT_MAX_LEAVES; l++)
         {
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            {
               g_leaf[l][c] = 0.0;
               h_leaf[l][c] = 0.0;
               cand[l][c] = tree.leaf_value[l][c];
            }
         }

         for(int i=0; i<n; i++)
         {
            double wi = sample_w[i];
            if(wi <= 0.0) continue;

            int leaf = (tree.depth > 0 ? leaf_idx[i] : 0);
            if(leaf < 0 || leaf >= leaf_count) leaf = 0;

            double z[FXAI_CAT_CLASS_COUNT];
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               z[c] = base_margin[i][c] + tree.leaf_value[leaf][c];

            double p[FXAI_CAT_CLASS_COUNT];
            Softmax3(z, p);

            int cls = y_all[i];
            if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            {
               double target = (c == cls ? 1.0 : 0.0);
               double gc = (target - p[c]) * wi;
               double hc = FXAI_Clamp(p[c] * (1.0 - p[c]) * wi, 0.005, 8.0);
               g_leaf[leaf][c] += gc;
               h_leaf[leaf][c] += hc;
            }
         }

         for(int l=0; l<leaf_count; l++)
         {
            if(tree.leaf_count[l] <= 0) continue;

            double meanv = 0.0;
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            {
               double step = eta * FXAI_ClipSym(g_leaf[l][c] / (h_leaf[l][c] + lambda), 5.0);
               cand[l][c] = tree.leaf_value[l][c] + step;
