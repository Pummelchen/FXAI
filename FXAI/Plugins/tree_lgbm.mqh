#ifndef __FXAI_AI_LIGHTGBM_MQH__
#define __FXAI_AI_LIGHTGBM_MQH__

#include "..\API\plugin_base.mqh"

// Reference-grade LightGBM-style plugin (3-class, histogram trees, 2nd-order, online calibrated).
#define FXAI_LGB_BINS 80
#define FXAI_LGB_MAX_LEAVES 63
#define FXAI_LGB_MAX_DEPTH 10
#define FXAI_LGB_MAX_NODES (2 * FXAI_LGB_MAX_LEAVES - 1)
#define FXAI_LGB_MAX_TREES 192
#define FXAI_LGB_BUFFER 4096
#define FXAI_LGB_MIN_DATA 20
#define FXAI_LGB_MIN_CHILD_HESS 0.20
#define FXAI_LGB_GAMMA 0.02
#define FXAI_LGB_BUILD_EVERY 16
#define FXAI_LGB_MIN_BUFFER 256
#define FXAI_LGB_CLASS_COUNT 3
#define FXAI_LGB_CAL_BINS 16
#define FXAI_LGB_ECE_BINS 12
#define FXAI_LGB_GOSS_BINS 64

struct FXAILGBNode
{
   bool   is_leaf;
   int    feature;
   double threshold;
   bool   default_left;
   int    left;
   int    right;
   int    depth;
   double leaf_value;

   // Leaf distribution stats for expected-move head.
   double move_mean;
   double move_var;
   double move_q10;
   double move_q50;
   double move_q90;
   int    sample_count;
};

struct FXAILGBTree
{
   int node_count;
   FXAILGBNode nodes[FXAI_LGB_MAX_NODES];
};

class CFXAIAILightGBM : public CFXAIAIPlugin
{
private:
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

public:
   CFXAIAILightGBM(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_LIGHTGBM; }
   virtual string AIName(void) const { return "tree_lgbm"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TREE, caps, 1, 1);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double logits[FXAI_LGB_CLASS_COUNT];
      ModelRawLogits(x, logits);
      double p_raw[FXAI_LGB_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, class_probs);

      double ev_buy = ClassExpectedMove((int)FXAI_LABEL_BUY, x);
      double ev_sell = ClassExpectedMove((int)FXAI_LABEL_SELL, x);
      double ev = class_probs[(int)FXAI_LABEL_BUY] * ev_buy + class_probs[(int)FXAI_LABEL_SELL] * ev_sell;

      double cost = ResolveCostPoints(x);
      if(cost < 0.0) cost = 0.0;
      ev = MathMax(0.0, ev - 0.35 * cost);

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0) expected_move_points = 0.75 * ev + 0.25 * m_move_ema_abs;
      else if(ev > 0.0) expected_move_points = ev;
      else expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      if(expected_move_points < 0.0) expected_move_points = 0.0;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double logits[FXAI_LGB_CLASS_COUNT];
      ModelRawLogits(x, logits);
      double p_raw[FXAI_LGB_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double ev_buy = 0.0, q10_buy = 0.0, q50_buy = 0.0, q90_buy = 0.0, sup_buy = 0.0;
      double ev_sell = 0.0, q10_sell = 0.0, q50_sell = 0.0, q90_sell = 0.0, sup_sell = 0.0;
      ClassMoveStats((int)FXAI_LABEL_BUY, x, ev_buy, q10_buy, q50_buy, q90_buy, sup_buy);
      ClassMoveStats((int)FXAI_LABEL_SELL, x, ev_sell, q10_sell, q50_sell, q90_sell, sup_sell);
      if(ev_buy <= 0.0) ev_buy = ClassExpectedMove((int)FXAI_LABEL_BUY, x);
      if(ev_sell <= 0.0) ev_sell = ClassExpectedMove((int)FXAI_LABEL_SELL, x);
      double ev = out.class_probs[(int)FXAI_LABEL_BUY] * ev_buy + out.class_probs[(int)FXAI_LABEL_SELL] * ev_sell;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      double mix_q10 = out.class_probs[(int)FXAI_LABEL_BUY] * q10_buy + out.class_probs[(int)FXAI_LABEL_SELL] * q10_sell;
      double mix_q50 = out.class_probs[(int)FXAI_LABEL_BUY] * q50_buy + out.class_probs[(int)FXAI_LABEL_SELL] * q50_sell;
      double mix_q90 = out.class_probs[(int)FXAI_LABEL_BUY] * q90_buy + out.class_probs[(int)FXAI_LABEL_SELL] * q90_sell;
      double sigma = MathMax(0.10, 0.25 * out.move_mean_points + 0.20 * (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, mix_q10 > 0.0 ? mix_q10 : (out.move_mean_points - 0.55 * sigma));
      out.move_q50_points = MathMax(out.move_q25_points, mix_q50 > 0.0 ? mix_q50 : out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, mix_q90 > 0.0 ? mix_q90 : (out.move_mean_points + 0.55 * sigma));
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      double support_rel = FXAI_Clamp((sup_buy + sup_sell) / 240.0, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.40 + 0.20 * (m_move_ready ? 1.0 : 0.0) + 0.20 * MathMin((double)m_tree_count[(int)FXAI_LABEL_BUY] / 32.0, 1.0) + 0.20 * support_rel, 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      double bank_mfe = 0.0, bank_mae = 0.0, bank_hit = 1.0, bank_path = 0.5, bank_fill = 0.5, bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_quality_heads.Predict(x,
                              out.move_mean_points,
                              FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                              out.reliability,
                              out.confidence,
                              bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust,
                              out);
      return true;
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_buf_head = 0;
      m_buf_size = 0;
      m_quality_heads.Reset();

      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         m_bias[c] = 0.0;
         m_tree_count[c] = 0;
         for(int t=0; t<FXAI_LGB_MAX_TREES; t++) InitTree(m_trees[c][t]);
      }

      for(int i=0; i<FXAI_LGB_BUFFER; i++)
      {
         m_buf_cls[i] = (int)FXAI_LABEL_SKIP;
         m_buf_move[i] = 0.0;
         m_buf_cost[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_buf_x[i][k] = 0.0;
      }

      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_LGB_CLASS_COUNT; j++) m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<FXAI_LGB_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      m_cal3_steps = 0;

      m_val_ready = false;
      m_val_steps = 0;
      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;
      m_quality_degraded = false;
      for(int b=0; b<FXAI_LGB_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) m_initialized = true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP) cls = (int)FXAI_LABEL_SKIP;

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double sample_w = MoveSampleWeight(x, move_points);
      sample_w = FXAI_Clamp(sample_w, 0.10, 6.00);
      m_quality_heads.Update(x,
                             sample_w,
                             TargetMFEPoints(),
                             FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                             TargetHitTimeFrac(),
                             TargetPathRisk(),
                             TargetFillRisk(),
                             TargetMaskedStep(),
                             TargetNextVol(),
                             TargetRegimeShift(),
                             TargetContextLead(),
                             h.lr,
                             h.l2);
      double cost = InputCostProxyPoints(x);

      // Online pre-update for calibration/metrics before structure update.
      double logits[FXAI_LGB_CLASS_COUNT];
      ModelRawLogits(x, logits);
      double p_raw[FXAI_LGB_CLASS_COUNT];
      Softmax3(logits, p_raw);
      double p_cal[FXAI_LGB_CLASS_COUNT];
      Calibrate3(p_raw, p_cal);

      double ev_buy = ClassExpectedMove((int)FXAI_LABEL_BUY, x);
      double ev_sell = ClassExpectedMove((int)FXAI_LABEL_SELL, x);
      double ev_now = p_cal[(int)FXAI_LABEL_BUY] * ev_buy + p_cal[(int)FXAI_LABEL_SELL] * ev_sell;
      UpdateValidationMetrics(cls, p_cal, ev_now, cost);

      double lr_cal = FXAI_Clamp(h.xgb_lr * 0.30, 0.0002, 0.0200);
      if(m_quality_degraded) lr_cal *= 0.80;
      UpdateCalibrator3(p_raw, cls, sample_w, lr_cal);

      // Legacy binary calibrator stays aligned for compatibility paths.
      double den_dir = p_raw[(int)FXAI_LABEL_BUY] + p_raw[(int)FXAI_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FXAI_LABEL_BUY] / den_dir;
      if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, sample_w);
      else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, sample_w);

      PushSample(cls, x, move_points, cost, sample_w);

      if(m_buf_size >= FXAI_LGB_MIN_BUFFER && (m_step % FXAI_LGB_BUILD_EVERY) == 0)
      {
         BuildOneTreeClass((int)FXAI_LABEL_SELL, h);
         BuildOneTreeClass((int)FXAI_LABEL_BUY, h);
         BuildOneTreeClass((int)FXAI_LABEL_SKIP, h);
      }

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, sample_w);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double probs[3];
      double expected_move = 0.0;
      if(!PredictModelCore(x, hp, probs, expected_move)) return 0.5;
      double den = probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) den = 1e-9;
      return FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double probs[3];
      double ev = -1.0;
      if(PredictModelCore(x, hp, probs, ev) && ev > 0.0) return ev;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
};

#endif // __FXAI_AI_LIGHTGBM_MQH__
