// FXAI v1
#ifndef __FX6_AI_XGB_FAST_MQH__
#define __FX6_AI_XGB_FAST_MQH__

#include "..\plugin_base.mqh"

#define FX6_XGBF_CLASS_COUNT 3
#define FX6_XGBF_SELL 0
#define FX6_XGBF_BUY  1
#define FX6_XGBF_SKIP 2

#define FX6_XGBF_MAX_TREES 128
#define FX6_XGBF_MAX_DEPTH 4
#define FX6_XGBF_MAX_NODES 31
#define FX6_XGBF_MAX_BINS  32
#define FX6_XGBF_BUFFER 4096

#define FX6_XGBF_MIN_SPLIT_SAMPLES_MIN 8
#define FX6_XGBF_MIN_SPLIT_SAMPLES_MAX 32
#define FX6_XGBF_MIN_CHILD_WEIGHT_MIN  0.05
#define FX6_XGBF_MIN_CHILD_WEIGHT_MAX  1.50

struct FX6XGBFastNode
{
   bool   is_leaf;
   int    feature;
   double threshold;
   bool   default_left;
   int    left;
   int    right;

   double leaf_value;
   double move_mean;
   double move_var;
   double move_q50;
   int    sample_count;
   double gain;
};

struct FX6XGBFastTree
{
   int    node_count;
   int    age;
   double weight;
   FX6XGBFastNode nodes[FX6_XGBF_MAX_NODES];
};

class CFX6AIXGBFast : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   int    m_step;

   double m_bias[FX6_XGBF_CLASS_COUNT];

   FX6XGBFastTree m_trees[FX6_XGBF_CLASS_COUNT][FX6_XGBF_MAX_TREES];
   int            m_tree_count[FX6_XGBF_CLASS_COUNT];

   // Ring buffer.
   double m_buf_x[FX6_XGBF_BUFFER][FX6_AI_WEIGHTS];
   int    m_buf_class[FX6_XGBF_BUFFER];
   double m_buf_move[FX6_XGBF_BUFFER];
   double m_buf_cost[FX6_XGBF_BUFFER];
   double m_buf_w[FX6_XGBF_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   // Runtime configuration (derived from hp so it is tunable via existing inputs/hyperparams).
   int    m_cfg_depth;
   int    m_cfg_bins;
   int    m_cfg_max_trees;
   int    m_cfg_build_every;
   int    m_cfg_min_buffer;
   int    m_cfg_min_split_samples;

   double m_cfg_subsample;
   double m_cfg_colsample;
   double m_cfg_lambda;
   double m_cfg_alpha;
   double m_cfg_gamma;
   double m_cfg_min_child_weight;
   double m_cfg_max_delta;
   double m_cfg_leaf_clip;

   double m_cfg_tree_decay;
   int    m_cfg_refresh_every;
   double m_cfg_refresh_lr;

   int ClampInt(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
   }

   void InitTree(FX6XGBFastTree &tree) const
   {
      tree.node_count = 1;
      tree.age = 0;
      tree.weight = 1.0;
      for(int n=0; n<FX6_XGBF_MAX_NODES; n++)
      {
         tree.nodes[n].is_leaf = true;
         tree.nodes[n].feature = -1;
         tree.nodes[n].threshold = 0.0;
         tree.nodes[n].default_left = true;
         tree.nodes[n].left = -1;
         tree.nodes[n].right = -1;

         tree.nodes[n].leaf_value = 0.0;
         tree.nodes[n].move_mean = 0.0;
         tree.nodes[n].move_var = 0.0;
         tree.nodes[n].move_q50 = 0.0;
         tree.nodes[n].sample_count = 0;
         tree.nodes[n].gain = 0.0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FX6_XGBF_BUFFER;
      int p = start + logical_idx;
      while(p >= FX6_XGBF_BUFFER) p -= FX6_XGBF_BUFFER;
      return p;
   }

   void PushSample(const int cls,
                   const double &x[],
                   const double move_points,
                   const double sample_w)
   {
      int pos = m_buf_head;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_buf_x[pos][i] = x[i];

      m_buf_class[pos] = cls;
      m_buf_move[pos] = move_points;
      m_buf_cost[pos] = InputCostProxyPoints(x);
      m_buf_w[pos] = sample_w;

      m_buf_head++;
      if(m_buf_head >= FX6_XGBF_BUFFER) m_buf_head = 0;
      if(m_buf_size < FX6_XGBF_BUFFER) m_buf_size++;
   }

   int MapUpdateClass(const int y,
                      const double &x[],
                      const double move_points) const
   {
      if(y == FX6_XGBF_SELL || y == FX6_XGBF_BUY || y == FX6_XGBF_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + (0.25 * MathMax(cost, 0.0));

      // If edge is too small after costs, label as skip.
      if(edge <= skip_band) return FX6_XGBF_SKIP;

      if(y > 0) return FX6_XGBF_BUY;
      if(y == 0) return FX6_XGBF_SELL;
      return (move_points >= 0.0 ? FX6_XGBF_BUY : FX6_XGBF_SELL);
   }

   void ConfigureRuntime(const FX6AIHyperParams &hp)
   {
      double complexity = FX6_Clamp(MathAbs(hp.xgb_split), 0.0, 1.0);

      // 1) Capacity controls (tunable via hp.xgb_split and hp.xgb_lr).
      m_cfg_depth = (complexity >= 0.55 ? 4 : 3);
      m_cfg_bins = ClampInt(16 + (int)MathRound(16.0 * complexity), 16, FX6_XGBF_MAX_BINS);
      m_cfg_max_trees = ClampInt(64 + (int)MathRound(64.0 * complexity), 64, FX6_XGBF_MAX_TREES);
      m_cfg_build_every = ClampInt(64 - (int)MathRound(24.0 * complexity), 24, 96);
      m_cfg_min_buffer = ClampInt(128 + (int)MathRound(256.0 * complexity), 96, 768);

      // 3) Subsampling / column sampling.
      m_cfg_subsample = FX6_Clamp(0.70 + (0.20 * (1.0 - complexity)) + (0.10 * hp.xgb_lr), 0.55, 1.00);
      m_cfg_colsample = FX6_Clamp(0.60 + (0.35 * complexity), 0.50, 1.00);

      // 4) Regularization controls.
      m_cfg_lambda = FX6_Clamp(hp.xgb_l2, 0.0001, 20.0);
      m_cfg_alpha = FX6_Clamp(0.02 + 0.80 * hp.l2 + 0.08 * complexity, 0.0, 2.0);
      m_cfg_gamma = FX6_Clamp(0.005 + 0.035 * complexity + 0.02 * hp.l2, 0.0, 0.25);
      m_cfg_min_child_weight = FX6_Clamp(0.06 + 0.35 * hp.l2, FX6_XGBF_MIN_CHILD_WEIGHT_MIN, FX6_XGBF_MIN_CHILD_WEIGHT_MAX);
      m_cfg_min_split_samples = ClampInt(10 + (int)MathRound(10.0 * hp.l2), FX6_XGBF_MIN_SPLIT_SAMPLES_MIN, FX6_XGBF_MIN_SPLIT_SAMPLES_MAX);
      m_cfg_max_delta = FX6_Clamp(0.80 + 2.40 * complexity, 0.25, 4.00);
      m_cfg_leaf_clip = FX6_Clamp(3.00 + 3.00 * complexity, 2.00, 8.00);

      // 8) Drift controls.
      m_cfg_tree_decay = FX6_Clamp(0.9985 - 0.0020 * complexity, 0.9900, 0.9999);
      m_cfg_refresh_every = ClampInt(256 - (int)MathRound(128.0 * complexity), 64, 512);
      m_cfg_refresh_lr = FX6_Clamp(0.03 + 0.08 * complexity, 0.01, 0.25);
   }

   double ScoreGH(const double G,
                  const double H,
                  const double lambda,
                  const double alpha) const
   {
      if(H <= 1e-12) return 0.0;
      double s = MathAbs(G) - alpha;
      if(s <= 0.0) return 0.0;
      return (s * s) / (H + lambda + 1e-9);
   }

   double LeafFromGH(const double G,
                     const double H,
                     const double lambda,
                     const double alpha) const
   {
      if(H <= 1e-12) return 0.0;
      double ag = MathAbs(G);
      if(ag <= alpha) return 0.0;
      double sign = (G >= 0.0 ? 1.0 : -1.0);
      return sign * (ag - alpha) / (H + lambda + 1e-9);
   }

   bool FeatureSelected(const int feature,
                        const int node_seed,
                        const double colsample) const
   {
      if(colsample >= 0.999) return true;
      uint h = (uint)(feature * 2654435761U);
      h ^= (uint)(node_seed * 2246822519U);
      h ^= (uint)(m_step * 3266489917U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      return (r <= colsample);
   }

   double ApproxAbsMedian(const int &sidx[], const double &mv[]) const
   {
      int n = ArraySize(sidx);
      if(n <= 0) return 0.0;

      double max_abs = 0.0;
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         double a = MathAbs(mv[id]);
         if(a > max_abs) max_abs = a;
      }
      if(max_abs <= 1e-12) return 0.0;

      int hist[9];
      for(int b=0; b<9; b++) hist[b] = 0;

      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         double a = MathAbs(mv[id]);
         int bi = (int)MathFloor((a / max_abs) * 9.0);
         if(bi < 0) bi = 0;
         if(bi >= 9) bi = 8;
         hist[bi]++;
      }

      int need = (n + 1) / 2;
      int csum = 0;
      for(int b=0; b<9; b++)
      {
         csum += hist[b];
         if(csum >= need)
            return ((double)b + 0.5) / 9.0 * max_abs;
      }
      return max_abs;
   }

   int TraverseLeafIndex(const FX6XGBFastTree &tree, const double &x[]) const
   {
      int node = 0;
      int guard = 0;
      while(node >= 0 && node < tree.node_count && guard < FX6_XGBF_MAX_NODES)
      {
         if(tree.nodes[node].is_leaf) return node;

         int f = tree.nodes[node].feature;
         if(f < 1 || f >= FX6_AI_WEIGHTS) return node;
         double xv = x[f];
         bool go_left = (!MathIsValidNumber(xv) ? tree.nodes[node].default_left : (xv <= tree.nodes[node].threshold));

         int nxt = (go_left ? tree.nodes[node].left : tree.nodes[node].right);
         if(nxt < 0 || nxt >= tree.node_count) return node;
         node = nxt;
         guard++;
      }
      return 0;
   }

   double TreeOutput(const FX6XGBFastTree &tree, const double &x[]) const
   {
      int leaf = TraverseLeafIndex(tree, x);
      if(leaf < 0 || leaf >= tree.node_count) return 0.0;
      return tree.weight * tree.nodes[leaf].leaf_value;
   }

   double ClassMargin(const int cls, const double &x[]) const
   {
      double s = m_bias[cls];
      int cnt = m_tree_count[cls];
      for(int t=0; t<cnt; t++)
         s += TreeOutput(m_trees[cls][t], x);
      return s;
   }

   void PredictRawClassProbs(const double &x[], double &probs[]) const
   {
      double margins[FX6_XGBF_CLASS_COUNT];
      for(int c=0; c<FX6_XGBF_CLASS_COUNT; c++)
         margins[c] = ClassMargin(c, x);

      double m = margins[0];
      for(int c=1; c<FX6_XGBF_CLASS_COUNT; c++)
         if(margins[c] > m) m = margins[c];

      double sum = 0.0;
      for(int c=0; c<FX6_XGBF_CLASS_COUNT; c++)
      {
         probs[c] = MathExp(FX6_Clamp(margins[c] - m, -30.0, 30.0));
         sum += probs[c];
      }

      if(sum <= 0.0)
      {
         for(int c=0; c<FX6_XGBF_CLASS_COUNT; c++) probs[c] = 1.0 / 3.0;
         return;
      }

      for(int c=0; c<FX6_XGBF_CLASS_COUNT; c++)
         probs[c] /= sum;
   }

   void BuildCuts(const int &sidx[],
                  const double &x_all[][FX6_AI_WEIGHTS],
                  const int bins,
                  double &cuts[][FX6_XGBF_MAX_BINS - 1]) const
   {
      for(int f=0; f<FX6_AI_WEIGHTS; f++)
         for(int b=0; b<FX6_XGBF_MAX_BINS - 1; b++)
            cuts[f][b] = DBL_MAX;

      int n = ArraySize(sidx);
      if(n <= 0) return;

      for(int f=1; f<FX6_AI_WEIGHTS; f++)
      {
         double minv = DBL_MAX;
         double maxv = -DBL_MAX;
         int valid = 0;

         for(int i=0; i<n; i++)
         {
            int id = sidx[i];
            double xv = x_all[id][f];
            if(!MathIsValidNumber(xv)) continue;
            if(xv < minv) minv = xv;
            if(xv > maxv) maxv = xv;
            valid++;
         }

         if(valid < 2 || maxv - minv < 1e-12)
            continue;

         for(int b=0; b<bins - 1; b++)
            cuts[f][b] = minv + ((maxv - minv) * (double)(b + 1) / (double)bins);
      }
   }

   bool FindBestSplit(const int node_idx,
                      const int depth,
                      const int &sidx[],
                      const double &g[],
                      const double &h[],
                      const double &x_all[][FX6_AI_WEIGHTS],
                      const double &cuts[][FX6_XGBF_MAX_BINS - 1],
                      int &best_feature,
                      double &best_thr,
                      bool &best_default_left,
                      double &best_gain) const
   {
      best_feature = -1;
      best_thr = 0.0;
      best_default_left = true;
      best_gain = 0.0;

      int n = ArraySize(sidx);
      if(n < 2 * m_cfg_min_split_samples) return false;

      double Gtot = 0.0, Htot = 0.0;
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         Gtot += g[id];
         Htot += h[id];
      }
      if(Htot <= 2.0 * m_cfg_min_child_weight) return false;

      double parent_score = ScoreGH(Gtot, Htot, m_cfg_lambda, m_cfg_alpha);
      if(parent_score <= 0.0) return false;

      for(int f=1; f<FX6_AI_WEIGHTS; f++)
      {
         if(!FeatureSelected(f, node_idx + depth * 131, m_cfg_colsample))
            continue;
         if(cuts[f][0] >= DBL_MAX * 0.5)
            continue;

         double gbin[FX6_XGBF_MAX_BINS];
         double hbin[FX6_XGBF_MAX_BINS];
         int    cbin[FX6_XGBF_MAX_BINS];
         for(int b=0; b<m_cfg_bins; b++)
         {
            gbin[b] = 0.0;
            hbin[b] = 0.0;
            cbin[b] = 0;
         }

         double Gmiss = 0.0, Hmiss = 0.0;
         int Cmiss = 0;

         for(int i=0; i<n; i++)
         {
            int id = sidx[i];
            double xv = x_all[id][f];
            if(!MathIsValidNumber(xv))
            {
               Gmiss += g[id];
               Hmiss += h[id];
               Cmiss++;
               continue;
            }

            int bi = 0;
            while(bi < m_cfg_bins - 1 && xv > cuts[f][bi]) bi++;
            if(bi < 0) bi = 0;
            if(bi >= m_cfg_bins) bi = m_cfg_bins - 1;

            gbin[bi] += g[id];
            hbin[bi] += h[id];
            cbin[bi]++;
         }

         double GL = 0.0, HL = 0.0;
         int CL = 0;

         for(int b=0; b<m_cfg_bins - 1; b++)
         {
            GL += gbin[b];
            HL += hbin[b];
            CL += cbin[b];

            double thr = cuts[f][b];

            // missing -> left
            {
               double gL = GL + Gmiss;
               double hL = HL + Hmiss;
               int cL = CL + Cmiss;
               double gR = Gtot - gL;
               double hR = Htot - hL;
               int cR = n - cL;

               if(cL >= m_cfg_min_split_samples && cR >= m_cfg_min_split_samples &&
                  hL >= m_cfg_min_child_weight && hR >= m_cfg_min_child_weight)
               {
                  double gain = 0.5 * (ScoreGH(gL, hL, m_cfg_lambda, m_cfg_alpha) +
                                       ScoreGH(gR, hR, m_cfg_lambda, m_cfg_alpha) -
                                       parent_score) - m_cfg_gamma;

                  if(gain > best_gain)
                  {
                     best_gain = gain;
                     best_feature = f;
                     best_thr = thr;
                     best_default_left = true;
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
               int cR = n - cL;

               if(cL >= m_cfg_min_split_samples && cR >= m_cfg_min_split_samples &&
                  hL >= m_cfg_min_child_weight && hR >= m_cfg_min_child_weight)
               {
                  double gain = 0.5 * (ScoreGH(gL, hL, m_cfg_lambda, m_cfg_alpha) +
                                       ScoreGH(gR, hR, m_cfg_lambda, m_cfg_alpha) -
                                       parent_score) - m_cfg_gamma;

                  if(gain > best_gain)
                  {
                     best_gain = gain;
                     best_feature = f;
                     best_thr = thr;
                     best_default_left = false;
                  }
               }
            }
         }
      }

      return (best_feature >= 1 && best_gain > 0.0);
   }

   void SetLeaf(FX6XGBFastTree &tree,
                const int node_idx,
                const int &sidx[],
                const double &g[],
                const double &h[],
                const double &mv[],
                const double eta)
   {
      double G = 0.0, H = 0.0;
      double sum_mv = 0.0, sum_sq = 0.0;

      int n = ArraySize(sidx);
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         G += g[id];
         H += h[id];

         double m = mv[id];
         sum_mv += m;
         sum_sq += m * m;
      }

      tree.nodes[node_idx].is_leaf = true;
      tree.nodes[node_idx].feature = -1;
      tree.nodes[node_idx].threshold = 0.0;
      tree.nodes[node_idx].default_left = true;
      tree.nodes[node_idx].left = -1;
      tree.nodes[node_idx].right = -1;
      tree.nodes[node_idx].sample_count = n;
      tree.nodes[node_idx].gain = 0.0;

      double leaf = eta * LeafFromGH(G, H, m_cfg_lambda, m_cfg_alpha);
      leaf = FX6_ClipSym(leaf, m_cfg_max_delta);
      tree.nodes[node_idx].leaf_value = FX6_ClipSym(leaf, m_cfg_leaf_clip);

      double mean = (n > 0 ? sum_mv / (double)n : 0.0);
      double var = 0.0;
      if(n > 0)
      {
         var = (sum_sq / (double)n) - (mean * mean);
         if(var < 0.0) var = 0.0;
      }

      tree.nodes[node_idx].move_mean = mean;
      tree.nodes[node_idx].move_var = var;
      tree.nodes[node_idx].move_q50 = ApproxAbsMedian(sidx, mv);
   }

   void BuildNode(FX6XGBFastTree &tree,
                  const int node_idx,
                  const int depth,
                  const int &sidx[],
                  const double &g[],
                  const double &h[],
                  const double &x_all[][FX6_AI_WEIGHTS],
                  const double &mv[],
                  const double &cuts[][FX6_XGBF_MAX_BINS - 1],
                  const double eta)
   {
      if(node_idx < 0 || node_idx >= FX6_XGBF_MAX_NODES) return;

      int n = ArraySize(sidx);
      if(n < m_cfg_min_split_samples || depth >= m_cfg_depth)
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, eta);
         return;
      }

      int best_f = -1;
      double best_thr = 0.0;
      bool best_def_left = true;
      double best_gain = 0.0;

      if(!FindBestSplit(node_idx, depth, sidx, g, h, x_all, cuts, best_f, best_thr, best_def_left, best_gain))
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, eta);
         return;
      }

      int left_idx[];
      int right_idx[];
      ArrayResize(left_idx, 0);
      ArrayResize(right_idx, 0);

      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         double xv = x_all[id][best_f];
         bool go_left = (!MathIsValidNumber(xv) ? best_def_left : (xv <= best_thr));

         if(go_left)
         {
            int ls = ArraySize(left_idx);
            ArrayResize(left_idx, ls + 1);
            left_idx[ls] = id;
         }
         else
         {
            int rs = ArraySize(right_idx);
            ArrayResize(right_idx, rs + 1);
            right_idx[rs] = id;
         }
      }

      if(ArraySize(left_idx) < m_cfg_min_split_samples ||
         ArraySize(right_idx) < m_cfg_min_split_samples ||
         tree.node_count + 2 > FX6_XGBF_MAX_NODES)
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, eta);
         return;
      }

      if(best_gain <= 0.0)
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, eta);
         return;
      }

      int left_node = tree.node_count;
      int right_node = tree.node_count + 1;
      tree.node_count += 2;

      tree.nodes[node_idx].is_leaf = false;
      tree.nodes[node_idx].feature = best_f;
      tree.nodes[node_idx].threshold = best_thr;
      tree.nodes[node_idx].default_left = best_def_left;
      tree.nodes[node_idx].left = left_node;
      tree.nodes[node_idx].right = right_node;
      tree.nodes[node_idx].leaf_value = 0.0;
      tree.nodes[node_idx].sample_count = n;
      tree.nodes[node_idx].gain = best_gain;

      BuildNode(tree, left_node, depth + 1, left_idx, g, h, x_all, mv, cuts, eta);
      BuildNode(tree, right_node, depth + 1, right_idx, g, h, x_all, mv, cuts, eta);

      // 4) Post-prune small gain splits.
      if(tree.nodes[left_node].is_leaf && tree.nodes[right_node].is_leaf)
      {
         if(best_gain <= (0.5 * m_cfg_gamma + 1e-9))
         {
            SetLeaf(tree, node_idx, sidx, g, h, mv, eta);
            return;
         }
      }
   }

   void CompactClassTrees(const int cls)
   {
      int cnt = m_tree_count[cls];
      if(cnt <= 0) return;

      FX6XGBFastTree keep[FX6_XGBF_MAX_TREES];
      int k = 0;
      for(int i=0; i<cnt; i++)
      {
         if(m_trees[cls][i].weight < 0.02) continue;
         keep[k] = m_trees[cls][i];
         k++;
      }

      int lim = m_cfg_max_trees;
      if(lim > FX6_XGBF_MAX_TREES) lim = FX6_XGBF_MAX_TREES;
      if(k > lim)
      {
         int drop = k - lim;
         for(int i=0; i<lim; i++)
            m_trees[cls][i] = keep[i + drop];
         m_tree_count[cls] = lim;
         return;
      }

      for(int i=0; i<k; i++)
         m_trees[cls][i] = keep[i];
      m_tree_count[cls] = k;
   }

   void ApplyTreeDecay(void)
   {
      for(int c=0; c<FX6_XGBF_CLASS_COUNT; c++)
      {
         for(int t=0; t<m_tree_count[c]; t++)
         {
            m_trees[c][t].weight *= m_cfg_tree_decay;
            m_trees[c][t].age++;
         }
         CompactClassTrees(c);
      }
   }

   bool BuildOneClassTree(const int cls, const FX6AIHyperParams &hp)
   {
      if(m_buf_size < m_cfg_min_buffer) return false;

      int n = m_buf_size;
      double x_all[][FX6_AI_WEIGHTS];
      int class_all[];
      double mv_all[];
      double cost_all[];
      double w_all[];
      ArrayResize(x_all, n);
      ArrayResize(class_all, n);
      ArrayResize(mv_all, n);
      ArrayResize(cost_all, n);
      ArrayResize(w_all, n);

      for(int i=0; i<n; i++)
      {
         int p = BufPos(i);
         for(int k=0; k<FX6_AI_WEIGHTS; k++)
            x_all[i][k] = m_buf_x[p][k];
         class_all[i] = m_buf_class[p];
         mv_all[i] = m_buf_move[p];
         cost_all[i] = m_buf_cost[p];
         w_all[i] = m_buf_w[p];
      }

      // 3) Row subsampling.
      int root_idx[];
      ArrayResize(root_idx, 0);
      for(int i=0; i<n; i++)
      {
         uint hv = (uint)(i * 2654435761U) ^ (uint)(m_step * 2246822519U) ^ (uint)(cls * 3266489917U);
         double r = (double)(hv & 0xFFFF) / 65535.0;
         if(r <= m_cfg_subsample)
         {
            int rs = ArraySize(root_idx);
            ArrayResize(root_idx, rs + 1);
            root_idx[rs] = i;
         }
      }

      if(ArraySize(root_idx) < m_cfg_min_buffer)
      {
         ArrayResize(root_idx, 0);
         int need = m_cfg_min_buffer;
         if(need > n) need = n;
         int start = n - need;
         if(start < 0) start = 0;
         for(int i=start; i<n; i++)
         {
            int rs = ArraySize(root_idx);
            ArrayResize(root_idx, rs + 1);
            root_idx[rs] = i;
         }
      }

      if(ArraySize(root_idx) < m_cfg_min_split_samples)
         return false;

      // 2/6) OvR logistic objective with EV/cost-aware sample weighting.
      double g[];
      double h[];
      ArrayResize(g, n);
      ArrayResize(h, n);

      double Gbatch = 0.0, Hbatch = 0.0;
      for(int i=0; i<n; i++)
      {
         double xloc[FX6_AI_WEIGHTS];
         for(int k=0; k<FX6_AI_WEIGHTS; k++) xloc[k] = x_all[i][k];

         int yk = (class_all[i] == cls ? 1 : 0);
         double margin = ClassMargin(cls, xloc);
         double p = FX6_Sigmoid(margin);

         double edge = MathAbs(mv_all[i]) - cost_all[i];
         double edge_w = FX6_MoveEdgeWeight(mv_all[i], cost_all[i]);

         if(cls == FX6_XGBF_SKIP)
         {
            if(edge <= 0.0) edge_w *= 1.50;
            else            edge_w *= 0.70;
         }
         else
         {
            if(edge <= 0.0) edge_w *= 0.55;
            else            edge_w *= FX6_Clamp(1.0 + 0.05 * MathMin(edge, 20.0), 1.0, 2.0);
         }

         double wi = FX6_Clamp(w_all[i] * edge_w, 0.10, 8.00);
         g[i] = ((double)yk - p) * wi;
         h[i] = FX6_Clamp(p * (1.0 - p) * wi, 0.01, 8.0);
      }

      for(int i=0; i<ArraySize(root_idx); i++)
      {
         int id = root_idx[i];
         Gbatch += g[id];
         Hbatch += h[id];
      }

      // 4) Bias refinement with L1/L2 + max delta.
      if(Hbatch > 1e-9)
      {
         double bstep = 0.20 * hp.xgb_lr * LeafFromGH(Gbatch, Hbatch, m_cfg_lambda, m_cfg_alpha);
         bstep = FX6_ClipSym(bstep, m_cfg_max_delta);
         m_bias[cls] += bstep;
         m_bias[cls] = FX6_Clamp(m_bias[cls], -8.0, 8.0);
      }

      // 5) Histogram cuts reused through node growth.
      double cuts[][FX6_XGBF_MAX_BINS - 1];
      ArrayResize(cuts, FX6_AI_WEIGHTS);
      BuildCuts(root_idx, x_all, m_cfg_bins, cuts);

      FX6XGBFastTree tree;
      InitTree(tree);
      tree.weight = 1.0;
      tree.age = 0;

      BuildNode(tree, 0, 0, root_idx, g, h, x_all, mv_all, cuts, FX6_Clamp(hp.xgb_lr, 0.0001, 1.0000));
      if(tree.node_count <= 0) return false;

      int lim = m_cfg_max_trees;
      if(lim > FX6_XGBF_MAX_TREES) lim = FX6_XGBF_MAX_TREES;

      if(m_tree_count[cls] < lim)
      {
         m_trees[cls][m_tree_count[cls]] = tree;
         m_tree_count[cls]++;
      }
      else
      {
         // Drift-aware replacement: drop the oldest/weakest head tree.
         int drop_idx = 0;
         double worst = 1e18;
         for(int t=0; t<m_tree_count[cls]; t++)
         {
            double score = (0.7 * m_trees[cls][t].weight) + (0.3 / (double)(m_trees[cls][t].age + 1));
            if(score < worst)
            {
               worst = score;
               drop_idx = t;
            }
         }

         for(int t=drop_idx + 1; t<m_tree_count[cls]; t++)
            m_trees[cls][t - 1] = m_trees[cls][t];
         m_trees[cls][m_tree_count[cls] - 1] = tree;
      }

      return true;
   }

   void RefreshLeaves(void)
   {
      if(m_buf_size <= 0) return;

      int n_ref = m_buf_size;
      int max_ref = m_cfg_min_buffer * 2;
      if(n_ref > max_ref) n_ref = max_ref;
      if(n_ref <= 0) return;

      int start = m_buf_size - n_ref;
      if(start < 0) start = 0;

      for(int cls=0; cls<FX6_XGBF_CLASS_COUNT; cls++)
      {
         for(int t=0; t<m_tree_count[cls]; t++)
         {
            double cnt[FX6_XGBF_MAX_NODES];
            double pos[FX6_XGBF_MAX_NODES];
            double sum_mv[FX6_XGBF_MAX_NODES];
            double sum_sq[FX6_XGBF_MAX_NODES];
            double sum_abs[FX6_XGBF_MAX_NODES];

            for(int n=0; n<FX6_XGBF_MAX_NODES; n++)
            {
               cnt[n] = 0.0;
               pos[n] = 0.0;
               sum_mv[n] = 0.0;
               sum_sq[n] = 0.0;
               sum_abs[n] = 0.0;
            }

            for(int li=start; li<m_buf_size; li++)
            {
               int p = BufPos(li);
               double xloc[FX6_AI_WEIGHTS];
               for(int k=0; k<FX6_AI_WEIGHTS; k++) xloc[k] = m_buf_x[p][k];

               int leaf = TraverseLeafIndex(m_trees[cls][t], xloc);
               if(leaf < 0 || leaf >= m_trees[cls][t].node_count) continue;

               double mv = m_buf_move[p];
               cnt[leaf] += 1.0;
               sum_mv[leaf] += mv;
               sum_sq[leaf] += mv * mv;
               sum_abs[leaf] += MathAbs(mv);
               if(m_buf_class[p] == cls)
                  pos[leaf] += 1.0;
            }

            for(int n=0; n<m_trees[cls][t].node_count; n++)
            {
               if(!m_trees[cls][t].nodes[n].is_leaf) continue;
               if(cnt[n] <= 0.0) continue;

               double mean = sum_mv[n] / cnt[n];
               double var = (sum_sq[n] / cnt[n]) - (mean * mean);
               if(var < 0.0) var = 0.0;
               double q50 = 0.80 * (sum_abs[n] / cnt[n]);

               m_trees[cls][t].nodes[n].move_mean = 0.70 * m_trees[cls][t].nodes[n].move_mean + 0.30 * mean;
               m_trees[cls][t].nodes[n].move_var = 0.70 * m_trees[cls][t].nodes[n].move_var + 0.30 * var;
               m_trees[cls][t].nodes[n].move_q50 = 0.70 * m_trees[cls][t].nodes[n].move_q50 + 0.30 * q50;
               m_trees[cls][t].nodes[n].sample_count = (int)cnt[n];

               double p_cls = (pos[n] + 1.0) / (cnt[n] + 2.0);
               double target_logit = FX6_Logit(FX6_Clamp(p_cls, 0.01, 0.99));
               double delta = FX6_ClipSym(target_logit - m_trees[cls][t].nodes[n].leaf_value, m_cfg_max_delta);
               m_trees[cls][t].nodes[n].leaf_value += m_cfg_refresh_lr * delta;
               m_trees[cls][t].nodes[n].leaf_value = FX6_ClipSym(m_trees[cls][t].nodes[n].leaf_value, m_cfg_leaf_clip);
            }
         }
      }
   }

   void ClassMoveStats(const int cls,
                       const double &x[],
                       double &mean,
                       double &var,
                       double &q50) const
   {
      mean = 0.0;
      var = 0.0;
      q50 = 0.0;

      double wsum = 0.0;
      for(int t=0; t<m_tree_count[cls]; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[cls][t], x);
         if(leaf < 0 || leaf >= m_trees[cls][t].node_count) continue;

         double w = m_trees[cls][t].weight * (MathAbs(m_trees[cls][t].nodes[leaf].leaf_value) + 0.05);
         if(w <= 0.0) continue;

         mean += w * m_trees[cls][t].nodes[leaf].move_mean;
         var += w * MathMax(m_trees[cls][t].nodes[leaf].move_var, 0.0);
         q50 += w * MathMax(m_trees[cls][t].nodes[leaf].move_q50, 0.0);
         wsum += w;
      }

      if(wsum > 0.0)
      {
         mean /= wsum;
         var /= wsum;
         q50 /= wsum;
      }
   }

public:
   CFX6AIXGBFast(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_XGB_FAST; }
   virtual string AIName(void) const { return "xgb_fast"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_step = 0;

      for(int c=0; c<FX6_XGBF_CLASS_COUNT; c++)
      {
         m_bias[c] = 0.0;
         m_tree_count[c] = 0;
         for(int t=0; t<FX6_XGBF_MAX_TREES; t++)
            InitTree(m_trees[c][t]);
      }

      m_buf_head = 0;
      m_buf_size = 0;
      for(int i=0; i<FX6_XGBF_BUFFER; i++)
      {
         m_buf_class[i] = FX6_XGBF_SKIP;
         m_buf_move[i] = 0.0;
         m_buf_cost[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FX6_AI_WEIGHTS; k++)
            m_buf_x[i][k] = 0.0;
      }

      m_cfg_depth = 3;
      m_cfg_bins = 16;
      m_cfg_max_trees = 64;
      m_cfg_build_every = 64;
      m_cfg_min_buffer = 128;
      m_cfg_min_split_samples = 12;

      m_cfg_subsample = 0.8;
      m_cfg_colsample = 0.8;
      m_cfg_lambda = 0.2;
      m_cfg_alpha = 0.05;
      m_cfg_gamma = 0.01;
      m_cfg_min_child_weight = 0.10;
      m_cfg_max_delta = 1.5;
      m_cfg_leaf_clip = 4.0;

      m_cfg_tree_decay = 0.997;
      m_cfg_refresh_every = 192;
      m_cfg_refresh_lr = 0.06;
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized)
      {
         ConfigureRuntime(hp);
         m_initialized = true;
      }
   }

   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      ConfigureRuntime(hp);
      m_step++;

      int cls = MapUpdateClass(y, x, move_points);

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double sw = MoveSampleWeight(x, move_points);
      sw = FX6_Clamp(sw, 0.25, 4.00);

      // Plugin-level binary calibration still targets directional buy vs sell.
      double probs_now[FX6_XGBF_CLASS_COUNT];
      PredictRawClassProbs(x, probs_now);
      double den = probs_now[FX6_XGBF_BUY] + probs_now[FX6_XGBF_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir = probs_now[FX6_XGBF_BUY] / den;
      if(cls != FX6_XGBF_SKIP)
      {
         int y_dir = (cls == FX6_XGBF_BUY ? 1 : 0);
         UpdateCalibration(p_dir, y_dir, sw);
      }

      PushSample(cls, x, move_points, sw);

      // 8) Drift handling.
      ApplyTreeDecay();

      if(m_buf_size >= m_cfg_min_buffer && (m_step % m_cfg_build_every) == 0)
      {
         for(int c=0; c<FX6_XGBF_CLASS_COUNT; c++)
            BuildOneClassTree(c, h);
      }

      if(m_cfg_refresh_every > 0 && (m_step % m_cfg_refresh_every) == 0)
         RefreshLeaves();

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, sw);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FX6_XGBF_CLASS_COUNT];
      PredictRawClassProbs(x, probs);

      double dir_den = probs[FX6_XGBF_BUY] + probs[FX6_XGBF_SELL];
      if(dir_den < 1e-9) dir_den = 1e-9;
      double p_dir_raw = probs[FX6_XGBF_BUY] / dir_den;
      double p_dir_cal = CalibrateProb(p_dir_raw);

      // Fold skip confidence into directional up-probability.
      double p_up = p_dir_cal * FX6_Clamp(1.0 - probs[FX6_XGBF_SKIP], 0.0, 1.0);
      return FX6_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FX6_XGBF_CLASS_COUNT];
      PredictRawClassProbs(x, probs);

      // 7) Distributional leaf stats (mean/variance/quantile proxy).
      double mu_buy, var_buy, q_buy;
      double mu_sell, var_sell, q_sell;
      double mu_skip, var_skip, q_skip;
      ClassMoveStats(FX6_XGBF_BUY, x, mu_buy, var_buy, q_buy);
      ClassMoveStats(FX6_XGBF_SELL, x, mu_sell, var_sell, q_sell);
      ClassMoveStats(FX6_XGBF_SKIP, x, mu_skip, var_skip, q_skip);

      double buy_abs = MathMax(0.0, mu_buy);
      if(buy_abs <= 0.0) buy_abs = q_buy;

      double sell_abs = MathMax(0.0, -mu_sell);
      if(sell_abs <= 0.0) sell_abs = q_sell;

      double base = probs[FX6_XGBF_BUY] * buy_abs + probs[FX6_XGBF_SELL] * sell_abs;
      double unc = probs[FX6_XGBF_BUY] * var_buy + probs[FX6_XGBF_SELL] * var_sell;
      if(unc < 0.0) unc = 0.0;
      unc = MathSqrt(unc);

      double qmix = probs[FX6_XGBF_BUY] * q_buy + probs[FX6_XGBF_SELL] * q_sell;
      double edge = base + 0.15 * unc + 0.10 * qmix;

      if(edge > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.70 * edge + 0.30 * m_move_ema_abs;
      if(edge > 0.0) return edge;
      return CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
   }
};

#endif // __FX6_AI_XGB_FAST_MQH__
