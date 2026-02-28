// FXAI v1
#ifndef __FX6_AI_CATBOOST_MQH__
#define __FX6_AI_CATBOOST_MQH__

#include "..\plugin_base.mqh"

#define FX6_CAT_CLASS_COUNT 3
#define FX6_CAT_MAX_TREES 96
#define FX6_CAT_MAX_DEPTH 5
#define FX6_CAT_MAX_LEAVES (1 << FX6_CAT_MAX_DEPTH)
#define FX6_CAT_BINS 20
#define FX6_CAT_BUFFER 4096
#define FX6_CAT_MIN_BUFFER 192
#define FX6_CAT_BUILD_EVERY 64
#define FX6_CAT_MIN_DATA 10
#define FX6_CAT_MIN_CHILD_HESS 0.05
#define FX6_CAT_GAMMA 0.01
#define FX6_CAT_CAL_BINS 10

struct FX6CatLevelSplit
{
   int    feature;
   double threshold;
   bool   default_left;
};

struct FX6CatTree
{
   int depth;
   FX6CatLevelSplit levels[FX6_CAT_MAX_DEPTH];
   double leaf_value[FX6_CAT_MAX_LEAVES][FX6_CAT_CLASS_COUNT];
   double leaf_move_mean[FX6_CAT_MAX_LEAVES];
   int    leaf_count[FX6_CAT_MAX_LEAVES];
};

class CFX6AICatBoost : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   int    m_tree_count;
   double m_bias[FX6_CAT_CLASS_COUNT];

   FX6CatTree m_trees[FX6_CAT_MAX_TREES];

   // Ordered online sample buffer.
   double m_buf_x[FX6_CAT_BUFFER][FX6_AI_WEIGHTS];
   int    m_buf_y[FX6_CAT_BUFFER];
   double m_buf_move[FX6_CAT_BUFFER];
   double m_buf_w[FX6_CAT_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   // Drift adaptation and class balance.
   double m_cls_ema[FX6_CAT_CLASS_COUNT];
   bool   m_loss_ready;
   double m_loss_fast;
   double m_loss_slow;
   int    m_drift_cooldown;

   // Plugin-native multiclass calibration.
   double m_cal3_temp;
   double m_cal3_bias[FX6_CAT_CLASS_COUNT];
   double m_cal3_iso_pos[FX6_CAT_CLASS_COUNT][FX6_CAT_CAL_BINS];
   double m_cal3_iso_cnt[FX6_CAT_CLASS_COUNT][FX6_CAT_CAL_BINS];
   int    m_cal3_steps;

   void InitTree(FX6CatTree &tree) const
   {
      tree.depth = 0;
      for(int d=0; d<FX6_CAT_MAX_DEPTH; d++)
      {
         tree.levels[d].feature = -1;
         tree.levels[d].threshold = 0.0;
         tree.levels[d].default_left = true;
      }

      for(int l=0; l<FX6_CAT_MAX_LEAVES; l++)
      {
         tree.leaf_move_mean[l] = 0.0;
         tree.leaf_count[l] = 0;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
            tree.leaf_value[l][c] = 0.0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FX6_CAT_BUFFER;
      int p = start + logical_idx;
      while(p >= FX6_CAT_BUFFER) p -= FX6_CAT_BUFFER;
      return p;
   }

   void PushSample(const int y,
                   const double &x[],
                   const double move_points,
                   const double sample_w)
   {
      int pos = m_buf_head;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_buf_x[pos][i] = x[i];
      m_buf_y[pos] = y;
      m_buf_move[pos] = move_points;
      m_buf_w[pos] = sample_w;

      m_buf_head++;
      if(m_buf_head >= FX6_CAT_BUFFER) m_buf_head = 0;
      if(m_buf_size < FX6_CAT_BUFFER) m_buf_size++;
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double e0 = MathExp(FX6_Clamp(logits[0] - m, -30.0, 30.0));
      double e1 = MathExp(FX6_Clamp(logits[1] - m, -30.0, 30.0));
      double e2 = MathExp(FX6_Clamp(logits[2] - m, -30.0, 30.0));
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

   int TraverseLeafIndex(const FX6CatTree &tree,
                         const double &x[]) const
   {
      if(tree.depth <= 0) return 0;
      int leaf = 0;
      for(int d=0; d<tree.depth; d++)
      {
         int f = tree.levels[d].feature;
         if(f < 1 || f >= FX6_AI_WEIGHTS) break;

         double xv = x[f];
         bool go_left;
         if(!MathIsValidNumber(xv))
            go_left = tree.levels[d].default_left;
         else
            go_left = (xv <= tree.levels[d].threshold);

         leaf = (leaf << 1) | (go_left ? 0 : 1);
         if(leaf < 0 || leaf >= FX6_CAT_MAX_LEAVES)
            return 0;
      }
      return leaf;
   }

   void TreeMargins(const FX6CatTree &tree,
                    const double &x[],
                    double &margins[]) const
   {
      int leaf = TraverseLeafIndex(tree, x);
      if(leaf < 0 || leaf >= FX6_CAT_MAX_LEAVES) return;

      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         margins[c] += tree.leaf_value[leaf][c];
   }

   void ModelMargins(const double &x[],
                     double &margins[]) const
   {
      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         margins[c] = m_bias[c];

      for(int t=0; t<m_tree_count; t++)
         TreeMargins(m_trees[t], x, margins);
   }

   void BuildFeatureRanges(const int n,
                           const double &x_all[][FX6_AI_WEIGHTS],
                           double &minv[],
                           double &maxv[],
                           int &valid_cnt[]) const
   {
      for(int f=0; f<FX6_AI_WEIGHTS; f++)
      {
         minv[f] = DBL_MAX;
         maxv[f] = -DBL_MAX;
         valid_cnt[f] = 0;
      }

      for(int i=0; i<n; i++)
      {
         for(int f=1; f<FX6_AI_WEIGHTS; f++)
         {
            double v = x_all[i][f];
            if(!MathIsValidNumber(v)) continue;
            if(v < minv[f]) minv[f] = v;
            if(v > maxv[f]) maxv[f] = v;
            valid_cnt[f]++;
         }
      }
   }

   bool EvaluateSplit(const int depth,
                      const int n,
                      const int &leaf_idx[],
                      const double &x_all[][FX6_AI_WEIGHTS],
                      const double &g[][FX6_CAT_CLASS_COUNT],
                      const double &h[][FX6_CAT_CLASS_COUNT],
                      const int feature,
                      const double threshold,
                      const bool default_left,
                      const double lambda,
                      double &gain_out) const
   {
      gain_out = 0.0;
      if(depth < 0 || depth >= FX6_CAT_MAX_DEPTH) return false;
      if(feature < 1 || feature >= FX6_AI_WEIGHTS) return false;

      int old_leaf_count = (1 << depth);
      int new_leaf_count = (1 << (depth + 1));

      double parent_g[FX6_CAT_MAX_LEAVES][FX6_CAT_CLASS_COUNT];
      double parent_h[FX6_CAT_MAX_LEAVES][FX6_CAT_CLASS_COUNT];
      int    parent_cnt[FX6_CAT_MAX_LEAVES];

      double child_g[FX6_CAT_MAX_LEAVES][FX6_CAT_CLASS_COUNT];
      double child_h[FX6_CAT_MAX_LEAVES][FX6_CAT_CLASS_COUNT];
      int    child_cnt[FX6_CAT_MAX_LEAVES];

      for(int l=0; l<FX6_CAT_MAX_LEAVES; l++)
      {
         parent_cnt[l] = 0;
         child_cnt[l] = 0;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            parent_g[l][c] = 0.0;
            parent_h[l][c] = 0.0;
            child_g[l][c] = 0.0;
            child_h[l][c] = 0.0;
         }
      }

      for(int i=0; i<n; i++)
      {
         int old_leaf = leaf_idx[i];
         if(old_leaf < 0 || old_leaf >= old_leaf_count) continue;

         parent_cnt[old_leaf]++;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            parent_g[old_leaf][c] += g[i][c];
            parent_h[old_leaf][c] += h[i][c];
         }

         double xv = x_all[i][feature];
         bool go_left;
         if(!MathIsValidNumber(xv))
            go_left = default_left;
         else
            go_left = (xv <= threshold);

         int child_leaf = (old_leaf << 1) | (go_left ? 0 : 1);
         if(child_leaf < 0 || child_leaf >= new_leaf_count) continue;

         child_cnt[child_leaf]++;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
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

         if(parent_cnt[old_leaf] <= 0) continue;
         if(child_cnt[left_leaf] < FX6_CAT_MIN_DATA || child_cnt[right_leaf] < FX6_CAT_MIN_DATA)
            return false;

         double hsum_left = 0.0;
         double hsum_right = 0.0;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            hsum_left += child_h[left_leaf][c];
            hsum_right += child_h[right_leaf][c];
         }
         if(hsum_left < FX6_CAT_MIN_CHILD_HESS || hsum_right < FX6_CAT_MIN_CHILD_HESS)
            return false;

         double parent_score = 0.0;
         double child_score = 0.0;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            parent_score += (parent_g[old_leaf][c] * parent_g[old_leaf][c]) /
                            (parent_h[old_leaf][c] + lambda + 1e-9);
            child_score += (child_g[left_leaf][c] * child_g[left_leaf][c]) /
                           (child_h[left_leaf][c] + lambda + 1e-9);
            child_score += (child_g[right_leaf][c] * child_g[right_leaf][c]) /
                           (child_h[right_leaf][c] + lambda + 1e-9);
         }

         gain += 0.5 * (child_score - parent_score) - FX6_CAT_GAMMA;
      }

      gain_out = gain;
      return (gain > 0.0);
   }

   void BuildLeafValues(FX6CatTree &tree,
                        const int n,
                        const int &leaf_idx[],
                        const double &g[][FX6_CAT_CLASS_COUNT],
                        const double &h[][FX6_CAT_CLASS_COUNT],
                        const double &mv[],
                        const double eta,
                        const double lambda) const
   {
      int leaf_count = (tree.depth > 0 ? (1 << tree.depth) : 1);
      if(leaf_count < 1) leaf_count = 1;
      if(leaf_count > FX6_CAT_MAX_LEAVES) leaf_count = FX6_CAT_MAX_LEAVES;

      double leaf_g[FX6_CAT_MAX_LEAVES][FX6_CAT_CLASS_COUNT];
      double leaf_h[FX6_CAT_MAX_LEAVES][FX6_CAT_CLASS_COUNT];
      double leaf_mv_sum[FX6_CAT_MAX_LEAVES];
      int    leaf_cnt[FX6_CAT_MAX_LEAVES];

      for(int l=0; l<FX6_CAT_MAX_LEAVES; l++)
      {
         leaf_mv_sum[l] = 0.0;
         leaf_cnt[l] = 0;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            leaf_g[l][c] = 0.0;
            leaf_h[l][c] = 0.0;
         }
      }

      for(int i=0; i<n; i++)
      {
         int leaf = (tree.depth > 0 ? leaf_idx[i] : 0);
         if(leaf < 0 || leaf >= leaf_count) continue;

         leaf_cnt[leaf]++;
         leaf_mv_sum[leaf] += MathAbs(mv[i]);
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            leaf_g[leaf][c] += g[i][c];
            leaf_h[leaf][c] += h[i][c];
         }
      }

      for(int l=0; l<leaf_count; l++)
      {
         tree.leaf_count[l] = leaf_cnt[l];
         tree.leaf_move_mean[l] = (leaf_cnt[l] > 0 ? leaf_mv_sum[l] / (double)leaf_cnt[l] : 0.0);

         double mean_val = 0.0;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            double v = 0.0;
            if(leaf_cnt[l] > 0 && leaf_h[l][c] > 1e-9)
               v = eta * FX6_ClipSym(leaf_g[l][c] / (leaf_h[l][c] + lambda), 5.0);
            tree.leaf_value[l][c] = v;
            mean_val += v;
         }
         mean_val /= (double)FX6_CAT_CLASS_COUNT;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
            tree.leaf_value[l][c] -= mean_val;
      }

      for(int l=leaf_count; l<FX6_CAT_MAX_LEAVES; l++)
      {
         tree.leaf_count[l] = 0;
         tree.leaf_move_mean[l] = 0.0;
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
            tree.leaf_value[l][c] = 0.0;
      }
   }

   bool BuildOneTree(const FX6AIHyperParams &hp)
   {
      if(m_buf_size < FX6_CAT_MIN_BUFFER) return false;

      int n = m_buf_size;
      double x_all[][FX6_AI_WEIGHTS];
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
         for(int k=0; k<FX6_AI_WEIGHTS; k++)
            x_all[i][k] = m_buf_x[p][k];
         y_all[i] = m_buf_y[p];
         mv_all[i] = m_buf_move[p];
         w_all[i] = m_buf_w[p];
      }

      double eta = FX6_Clamp(hp.xgb_lr, 0.0005, 0.3000);
      double lambda = FX6_Clamp(hp.xgb_l2, 0.0001, 10.0000);
      if(m_drift_cooldown > 0)
      {
         eta = FX6_Clamp(eta * 0.70, 0.0005, 0.3000);
         lambda = FX6_Clamp(lambda * 1.25, 0.0001, 10.0000);
      }

      double g[][FX6_CAT_CLASS_COUNT];
      double h[][FX6_CAT_CLASS_COUNT];
      ArrayResize(g, n);
      ArrayResize(h, n);

      for(int i=0; i<n; i++)
      {
         double xloc[FX6_AI_WEIGHTS];
         for(int k=0; k<FX6_AI_WEIGHTS; k++)
            xloc[k] = x_all[i][k];

         double margins[FX6_CAT_CLASS_COUNT];
         double probs[FX6_CAT_CLASS_COUNT];
         ModelMargins(xloc, margins);
         Softmax3(margins, probs);

         double wi = FX6_Clamp(w_all[i], 0.10, 6.00);
         for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         {
            double target = (c == y_all[i] ? 1.0 : 0.0);
            g[i][c] = (target - probs[c]) * wi;
            h[i][c] = FX6_Clamp(probs[c] * (1.0 - probs[c]) * wi, 0.01, 4.00);
         }
      }

      // Bias refinement with per-class Newton step.
      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
      {
         double G = 0.0, H = 0.0;
         for(int i=0; i<n; i++)
         {
            G += g[i][c];
            H += h[i][c];
         }
         if(H > 1e-9)
            m_bias[c] += 0.20 * eta * FX6_ClipSym(G / (H + lambda), 4.0);
      }
      double mean_bias = (m_bias[0] + m_bias[1] + m_bias[2]) / 3.0;
      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         m_bias[c] -= mean_bias;

      FX6CatTree tree;
      InitTree(tree);

      int leaf_idx[];
      ArrayResize(leaf_idx, n);
      for(int i=0; i<n; i++)
         leaf_idx[i] = 0;

      double minv[FX6_AI_WEIGHTS];
      double maxv[FX6_AI_WEIGHTS];
      int valid_cnt[FX6_AI_WEIGHTS];
      BuildFeatureRanges(n, x_all, minv, maxv, valid_cnt);

      int depth_used = 0;
      for(int d=0; d<FX6_CAT_MAX_DEPTH; d++)
      {
         int best_f = -1;
         double best_thr = 0.0;
         bool best_default_left = true;
         double best_gain = 0.0;

         for(int f=1; f<FX6_AI_WEIGHTS; f++)
         {
            if(valid_cnt[f] < 2 * FX6_CAT_MIN_DATA) continue;
            double range = maxv[f] - minv[f];
            if(range < 1e-12) continue;

            for(int b=1; b<FX6_CAT_BINS; b++)
            {
               double thr = minv[f] + range * ((double)b / (double)FX6_CAT_BINS);
               for(int pass=0; pass<2; pass++)
               {
                  bool default_left = (pass == 0);
                  double gain = 0.0;
                  if(!EvaluateSplit(d, n, leaf_idx, x_all, g, h, f, thr, default_left, lambda, gain))
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

         for(int i=0; i<n; i++)
         {
            double xv = x_all[i][best_f];
            bool go_left;
            if(!MathIsValidNumber(xv))
               go_left = best_default_left;
            else
               go_left = (xv <= best_thr);
            leaf_idx[i] = (leaf_idx[i] << 1) | (go_left ? 0 : 1);
         }

         depth_used = d + 1;
      }

      tree.depth = depth_used;
      BuildLeafValues(tree, n, leaf_idx, g, h, mv_all, eta, lambda);

      if(m_tree_count < FX6_CAT_MAX_TREES)
      {
         m_trees[m_tree_count] = tree;
         m_tree_count++;
      }
      else
      {
         for(int t=1; t<FX6_CAT_MAX_TREES; t++)
            m_trees[t - 1] = m_trees[t];
         m_trees[FX6_CAT_MAX_TREES - 1] = tree;
         m_tree_count = FX6_CAT_MAX_TREES;
      }

      return true;
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FX6_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FX6_CAT_CLASS_COUNT];
      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal3_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FX6_CAT_CLASS_COUNT];
      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FX6_CAT_CAL_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FX6_CAT_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FX6_CAT_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FX6_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_CAT_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_CAT_CAL_BINS) bi = FX6_CAT_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         p_cal[c] = FX6_Clamp(0.75 * p_cal[c] + 0.25 * p_iso[c], 0.0005, 0.9990);

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
      double inv_temp = 1.0 / FX6_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FX6_CAT_CLASS_COUNT];
      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal3_bias[c];
      }

      double p_cal[FX6_CAT_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FX6_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FX6_Clamp(0.18 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal3_bias[c] = FX6_ClipSym(m_cal3_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FX6_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_CAT_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_CAT_CAL_BINS) bi = FX6_CAT_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

      m_cal3_temp = FX6_Clamp(m_cal3_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
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

public:
   CFX6AICatBoost(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_CATBOOST; }
   virtual string AIName(void) const { return "catboost"; }
   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_tree_count = 0;
      m_buf_head = 0;
      m_buf_size = 0;

      m_loss_ready = false;
      m_loss_fast = 0.0;
      m_loss_slow = 0.0;
      m_drift_cooldown = 0;

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;

      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
      {
         m_bias[c] = 0.0;
         m_cls_ema[c] = 1.0;
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FX6_CAT_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      for(int i=0; i<FX6_CAT_BUFFER; i++)
      {
         m_buf_y[i] = (int)FX6_LABEL_SKIP;
         m_buf_move[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FX6_AI_WEIGHTS; k++)
            m_buf_x[i][k] = 0.0;
      }

      for(int t=0; t<FX6_CAT_MAX_TREES; t++)
         InitTree(m_trees[t]);
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(m_initialized) return;
      m_initialized = true;
      // Slight skip prior reduces early over-trading before first trees are built.
      m_bias[(int)FX6_LABEL_SKIP] = 0.12;
   }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double margins[FX6_CAT_CLASS_COUNT];
      double p_raw[FX6_CAT_CLASS_COUNT];
      ModelMargins(x, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, class_probs);

      expected_move_points = PredictExpectedMovePoints(x, hp);
      if(expected_move_points <= 0.0)
         expected_move_points = ResolveMinMovePoints();
      if(expected_move_points <= 0.0)
         expected_move_points = 0.10;
      return true;
   }

   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      UpdateWithMove(cls, x, hp, pseudo_move);
   }

protected:
   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
         cls = (int)FX6_LABEL_SKIP;

      for(int c=0; c<FX6_CAT_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FX6_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cost = InputCostProxyPoints(x);
      double abs_move = MathAbs(move_points);
      double edge = MathMax(0.0, abs_move - cost);
      double ev_w = FX6_Clamp(0.35 + (edge / MathMax(cost, 0.50)), 0.10, 6.00);
      if(cls == (int)FX6_LABEL_SKIP) ev_w *= 0.85;
      double w = FX6_Clamp(ev_w * cls_bal, 0.10, 6.00);

      double margins[FX6_CAT_CLASS_COUNT];
      double p_raw[FX6_CAT_CLASS_COUNT];
      ModelMargins(x, margins);
      Softmax3(margins, p_raw);

      double ce = -MathLog(FX6_Clamp(p_raw[cls], 1e-6, 1.0));
      UpdateLossDrift(ce);

      double cal_lr = FX6_Clamp(0.01 + 0.10 * FX6_Clamp(h.xgb_lr, 0.0005, 0.3000), 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, w, cal_lr);

      // Keep legacy binary calibrator aligned for compatibility paths.
      double den_dir = p_raw[(int)FX6_LABEL_BUY] + p_raw[(int)FX6_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FX6_LABEL_BUY] / den_dir;
      if(cls == (int)FX6_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, w);
      else if(cls == (int)FX6_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, w);

      PushSample(cls, x, move_points, w);

      int build_every = FX6_CAT_BUILD_EVERY;
      if(m_drift_cooldown > 0) build_every = FX6_CAT_BUILD_EVERY / 2;
      if(build_every < 16) build_every = 16;
      if(m_buf_size >= FX6_CAT_MIN_BUFFER && (m_step % build_every) == 0)
         BuildOneTree(h);

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, w);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double margins[FX6_CAT_CLASS_COUNT];
      double p_raw[FX6_CAT_CLASS_COUNT];
      double p_cal[FX6_CAT_CLASS_COUNT];
      ModelMargins(x, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, p_cal);

      double den = p_cal[(int)FX6_LABEL_BUY] + p_cal[(int)FX6_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FX6_Clamp(p_cal[(int)FX6_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double sum = 0.0;
      double wsum = 0.0;
      for(int t=0; t<m_tree_count; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[t], x);
         if(leaf < 0 || leaf >= FX6_CAT_MAX_LEAVES) continue;

         double mv = m_trees[t].leaf_move_mean[leaf];
         if(mv <= 0.0) continue;

         double conf = MathAbs(m_trees[t].leaf_value[leaf][(int)FX6_LABEL_BUY] -
                               m_trees[t].leaf_value[leaf][(int)FX6_LABEL_SELL]) + 0.15;
         sum += conf * mv;
         wsum += conf;
      }

      double tree_est = (wsum > 0.0 ? sum / wsum : -1.0);
      double base_est = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);

      if(tree_est > 0.0 && base_est > 0.0) return 0.65 * tree_est + 0.35 * base_est;
      if(tree_est > 0.0) return tree_est;
      return base_est;
   }
};

#endif // __FX6_AI_CATBOOST_MQH__
