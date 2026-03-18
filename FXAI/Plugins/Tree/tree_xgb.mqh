#ifndef __FXAI_AI_XGBOOST_MQH__
#define __FXAI_AI_XGBOOST_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_XGB_MAX_TREES 64
#define FXAI_XGB_MAX_DEPTH 3
#define FXAI_XGB_MAX_NODES 31
#define FXAI_XGB_BINS 16
#define FXAI_XGB_BUFFER 2048
#define FXAI_XGB_MIN_SPLIT_SAMPLES 12
#define FXAI_XGB_MIN_CHILD_WEIGHT 0.10
#define FXAI_XGB_GAMMA 0.01
#define FXAI_XGB_BUILD_EVERY 64
#define FXAI_XGB_MIN_BUFFER 128

struct FXAIXGBNode
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
   double class_mass[3];
   int    sample_count;
};

struct FXAIXGBTree
{
   int node_count;
   FXAIXGBNode nodes[FXAI_XGB_MAX_NODES];
};

class CFXAIAIXGBoost : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   double m_bias;
   CFXAINativeQualityHeads m_quality_heads;

   FXAIXGBTree m_trees[FXAI_XGB_MAX_TREES];
   int        m_tree_count;

   double m_buf_x[FXAI_XGB_BUFFER][FXAI_AI_WEIGHTS];
   int    m_buf_y[FXAI_XGB_BUFFER];
   int    m_buf_cls[FXAI_XGB_BUFFER];
   double m_buf_move[FXAI_XGB_BUFFER];
   double m_buf_w[FXAI_XGB_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   void InitTree(FXAIXGBTree &tree) const
   {
      tree.node_count = 1;
      for(int n=0; n<FXAI_XGB_MAX_NODES; n++)
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
         for(int c=0; c<3; c++)
            tree.nodes[n].class_mass[c] = (c == (int)FXAI_LABEL_SKIP ? 1.0 : 0.5);
         tree.nodes[n].sample_count = 0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FXAI_XGB_BUFFER;
      int p = start + logical_idx;
      while(p >= FXAI_XGB_BUFFER) p -= FXAI_XGB_BUFFER;
      return p;
   }

   void PushSample(const int y,
                   const int cls,
                   const double &x[],
                   const double move_points,
                   const double sample_w)
   {
      int pos = m_buf_head;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_buf_x[pos][i] = x[i];
      m_buf_y[pos] = y;
      m_buf_cls[pos] = cls;
      m_buf_move[pos] = move_points;
      m_buf_w[pos] = sample_w;

      m_buf_head++;
      if(m_buf_head >= FXAI_XGB_BUFFER) m_buf_head = 0;
      if(m_buf_size < FXAI_XGB_BUFFER) m_buf_size++;
   }

   int TraverseLeafIndex(const FXAIXGBTree &tree, const double &x[]) const
   {
      int node = 0;
      int guard = 0;
      while(node >= 0 && node < tree.node_count && guard < FXAI_XGB_MAX_NODES)
      {
         if(tree.nodes[node].is_leaf) return node;

         int f = tree.nodes[node].feature;
         if(f < 1 || f >= FXAI_AI_WEIGHTS) return node;
         double xv = x[f];
         bool go_left;
         if(!MathIsValidNumber(xv))
            go_left = tree.nodes[node].default_left;
         else
            go_left = (xv <= tree.nodes[node].threshold);

         int nxt = (go_left ? tree.nodes[node].left : tree.nodes[node].right);
         if(nxt < 0 || nxt >= tree.node_count) return node;
         node = nxt;
         guard++;
      }
      return 0;
   }

   double TreeOutput(const FXAIXGBTree &tree, const double &x[]) const
   {
      int leaf = TraverseLeafIndex(tree, x);
      if(leaf < 0 || leaf >= tree.node_count) return 0.0;
      return tree.nodes[leaf].leaf_value;
   }

   double ModelMargin(const double &x[]) const
   {
      double s = m_bias;
      for(int t=0; t<m_tree_count; t++)
         s += TreeOutput(m_trees[t], x);
      return s;
   }

   void SetLeaf(FXAIXGBTree &tree,
                const int node_idx,
                const int &sidx[],
                const double &g[],
                const double &h[],
                const double &mv[],
                const int &cls_all[],
                const double &w_all[],
                const double lambda,
                const double eta) const
   {
      double G = 0.0, H = 0.0, sum_mv = 0.0, sum_mv2 = 0.0;
      int n = ArraySize(sidx);
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         G += g[id];
         H += h[id];
         double amv = MathAbs(mv[id]);
         sum_mv += amv;
         sum_mv2 += amv * amv;
      }

      tree.nodes[node_idx].is_leaf = true;
      tree.nodes[node_idx].feature = -1;
      tree.nodes[node_idx].threshold = 0.0;
      tree.nodes[node_idx].default_left = true;
      tree.nodes[node_idx].left = -1;
      tree.nodes[node_idx].right = -1;
      tree.nodes[node_idx].sample_count = n;
      tree.nodes[node_idx].move_mean = (n > 0 ? sum_mv / (double)n : 0.0);
      tree.nodes[node_idx].move_var = (n > 0 ? MathMax(0.0, sum_mv2 / (double)n - tree.nodes[node_idx].move_mean * tree.nodes[node_idx].move_mean) : 0.0);
      for(int c=0; c<3; c++)
         tree.nodes[node_idx].class_mass[c] = 1e-3;
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         int cls = cls_all[id];
         if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
            cls = (mv[id] >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
         tree.nodes[node_idx].class_mass[cls] += MathMax(0.05, w_all[id]);
      }

      double leaf = 0.0;
      if(H > 1e-9)
         leaf = eta * FXAI_ClipSym(G / (H + lambda), 5.0);
      tree.nodes[node_idx].leaf_value = leaf;
   }

   bool FindBestSplit(const int &sidx[],
                      const double &g[],
                      const double &h[],
                      const double &x_all[][FXAI_AI_WEIGHTS],
                      const double lambda,
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
      if(n < 2 * FXAI_XGB_MIN_SPLIT_SAMPLES) return false;

      double Gtot = 0.0, Htot = 0.0;
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         Gtot += g[id];
         Htot += h[id];
      }
      if(Htot <= FXAI_XGB_MIN_CHILD_WEIGHT * 2.0) return false;

      double parent_score = (Gtot * Gtot) / (Htot + lambda + 1e-9);

      for(int f=1; f<FXAI_AI_WEIGHTS; f++)
      {
         double minv = DBL_MAX;
         double maxv = -DBL_MAX;
         double Gmiss = 0.0, Hmiss = 0.0;
         int Cmiss = 0, Cvalid = 0;

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
            if(xv < minv) minv = xv;
            if(xv > maxv) maxv = xv;
            Cvalid++;
         }

         if(Cvalid < 2 * FXAI_XGB_MIN_SPLIT_SAMPLES) continue;
         if(maxv - minv < 1e-9) continue;

         for(int b=1; b<FXAI_XGB_BINS; b++)
         {
            double thr = minv + (maxv - minv) * ((double)b / (double)FXAI_XGB_BINS);
            double GL = 0.0, HL = 0.0, GR = 0.0, HR = 0.0;
            int CL = 0, CR = 0;

            for(int i=0; i<n; i++)
            {
               int id = sidx[i];
               double xv = x_all[id][f];
               if(!MathIsValidNumber(xv)) continue;
               if(xv <= thr)
               {
                  GL += g[id];
                  HL += h[id];
                  CL++;
               }
               else
               {
                  GR += g[id];
                  HR += h[id];
                  CR++;
               }
            }

            // missing -> left
            {
               double gL = GL + Gmiss;
               double hL = HL + Hmiss;
               int cL = CL + Cmiss;
               double gR = GR;
               double hR = HR;
               int cR = CR;

               if(cL >= FXAI_XGB_MIN_SPLIT_SAMPLES && cR >= FXAI_XGB_MIN_SPLIT_SAMPLES &&
                  hL >= FXAI_XGB_MIN_CHILD_WEIGHT && hR >= FXAI_XGB_MIN_CHILD_WEIGHT)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FXAI_XGB_GAMMA;
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
               double gR = GR + Gmiss;
               double hR = HR + Hmiss;
               int cR = CR + Cmiss;

               if(cL >= FXAI_XGB_MIN_SPLIT_SAMPLES && cR >= FXAI_XGB_MIN_SPLIT_SAMPLES &&
                  hL >= FXAI_XGB_MIN_CHILD_WEIGHT && hR >= FXAI_XGB_MIN_CHILD_WEIGHT)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FXAI_XGB_GAMMA;
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

   void BuildNode(FXAIXGBTree &tree,
                  const int node_idx,
                  const int depth,
                  const int &sidx[],
                  const double &g[],
                  const double &h[],
                  const double &x_all[][FXAI_AI_WEIGHTS],
                  const double &mv[],
                  const int &cls_all[],
                  const double &w_all[],
                  const double lambda,
                  const double eta) const
   {
      if(node_idx < 0 || node_idx >= FXAI_XGB_MAX_NODES) return;
      if(ArraySize(sidx) < FXAI_XGB_MIN_SPLIT_SAMPLES || depth >= FXAI_XGB_MAX_DEPTH)
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, cls_all, w_all, lambda, eta);
         return;
      }

      int best_f = -1;
      double best_thr = 0.0;
      bool best_def_left = true;
      double best_gain = 0.0;
      if(!FindBestSplit(sidx, g, h, x_all, lambda, best_f, best_thr, best_def_left, best_gain))
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, cls_all, w_all, lambda, eta);
         return;
      }

      int left_idx[];
      int right_idx[];
      ArrayResize(left_idx, 0);
      ArrayResize(right_idx, 0);

      int n = ArraySize(sidx);
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         double xv = x_all[id][best_f];
         bool go_left = (!MathIsValidNumber(xv) ? best_def_left : (xv <= best_thr));
         if(go_left)
         {
            int sz = ArraySize(left_idx);
            ArrayResize(left_idx, sz + 1);
            left_idx[sz] = id;
         }
         else
         {
            int sz = ArraySize(right_idx);
            ArrayResize(right_idx, sz + 1);
            right_idx[sz] = id;
         }
      }

      if(ArraySize(left_idx) < FXAI_XGB_MIN_SPLIT_SAMPLES ||
         ArraySize(right_idx) < FXAI_XGB_MIN_SPLIT_SAMPLES ||
         tree.node_count + 2 > FXAI_XGB_MAX_NODES)
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, cls_all, w_all, lambda, eta);
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
      tree.nodes[node_idx].move_var = 0.0;
      tree.nodes[node_idx].sample_count = n;
      for(int c=0; c<3; c++)
         tree.nodes[node_idx].class_mass[c] = 0.0;

      BuildNode(tree, left_node, depth + 1, left_idx, g, h, x_all, mv, cls_all, w_all, lambda, eta);
      BuildNode(tree, right_node, depth + 1, right_idx, g, h, x_all, mv, cls_all, w_all, lambda, eta);
   }

   bool BuildOneTree(const FXAIAIHyperParams &hp)
   {
      if(m_buf_size < FXAI_XGB_MIN_BUFFER) return false;

      int n = m_buf_size;
      double x_all[][FXAI_AI_WEIGHTS];
      int y_all[];
      int cls_all[];
      double mv_all[];
      double w_all[];
      ArrayResize(x_all, n);
      ArrayResize(y_all, n);
      ArrayResize(cls_all, n);
      ArrayResize(mv_all, n);
      ArrayResize(w_all, n);

      for(int i=0; i<n; i++)
      {
         int p = BufPos(i);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) x_all[i][k] = m_buf_x[p][k];
         y_all[i] = m_buf_y[p];
         cls_all[i] = m_buf_cls[p];
         mv_all[i] = m_buf_move[p];
         w_all[i] = m_buf_w[p];
      }

      double g[];
      double h[];
      ArrayResize(g, n);
      ArrayResize(h, n);

      for(int i=0; i<n; i++)
      {
         double xloc[FXAI_AI_WEIGHTS];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) xloc[k] = x_all[i][k];
         double margin = ModelMargin(xloc);
         double p = FXAI_Sigmoid(margin);
         double wi = FXAI_Clamp(w_all[i], 0.25, 4.00);
         if(cls_all[i] == (int)FXAI_LABEL_SKIP) wi *= 0.05;
         g[i] = ((double)y_all[i] - p) * wi;
         h[i] = FXAI_Clamp(p * (1.0 - p) * wi, 0.02, 4.00);
      }

      // Small bias refinement from batch gradient statistics.
      double G = 0.0, H = 0.0;
      for(int i=0; i<n; i++) { G += g[i]; H += h[i]; }
      double lambda = FXAI_Clamp(hp.xgb_l2, 0.0001, 10.0000);
      double eta = FXAI_Clamp(hp.xgb_lr, 0.0001, 1.0000);
      if(H > 1e-9)
         m_bias += 0.20 * eta * FXAI_ClipSym(G / (H + lambda), 5.0);

      int root_idx[];
      ArrayResize(root_idx, n);
      for(int i=0; i<n; i++) root_idx[i] = i;

      FXAIXGBTree tree;
      InitTree(tree);
      BuildNode(tree, 0, 0, root_idx, g, h, x_all, mv_all, cls_all, w_all, lambda, eta);
      if(tree.node_count <= 0) return false;

      if(m_tree_count < FXAI_XGB_MAX_TREES)
      {
         m_trees[m_tree_count] = tree;
         m_tree_count++;
      }
      else
      {
         for(int t=1; t<FXAI_XGB_MAX_TREES; t++)
            m_trees[t - 1] = m_trees[t];
         m_trees[FXAI_XGB_MAX_TREES - 1] = tree;
         m_tree_count = FXAI_XGB_MAX_TREES;
      }

      return true;
   }

public:
   CFXAIAIXGBoost(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_XGBOOST; }
   virtual string AIName(void) const { return "tree_xgb"; }


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
      double margin = ModelMargin(x);
      double p_dir = CalibrateProb(FXAI_Sigmoid(margin));
      double cls_mass[3] = {1e-3, 1e-3, 1e-3};
      double sum = 0.0, wsum = 0.0, mvsum = 0.0, mv2sum = 0.0;
      for(int t=0; t<m_tree_count; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[t], x);
         if(leaf < 0 || leaf >= m_trees[t].node_count) continue;
         double lw = MathAbs(m_trees[t].nodes[leaf].leaf_value) + 0.20;
         for(int c=0; c<3; c++)
            cls_mass[c] += lw * m_trees[t].nodes[leaf].class_mass[c];
         mvsum += lw * m_trees[t].nodes[leaf].move_mean;
         mv2sum += lw * (m_trees[t].nodes[leaf].move_var + m_trees[t].nodes[leaf].move_mean * m_trees[t].nodes[leaf].move_mean);
         wsum += lw;
      }
      sum = cls_mass[0] + cls_mass[1] + cls_mass[2];
      if(sum <= 0.0) sum = 1.0;
      double leaf_buy = cls_mass[(int)FXAI_LABEL_BUY] / sum;
      double leaf_sell = cls_mass[(int)FXAI_LABEL_SELL] / sum;
      double leaf_skip = cls_mass[(int)FXAI_LABEL_SKIP] / sum;
      double active = FXAI_Clamp(1.0 - leaf_skip, 0.0, 1.0);
      double dir_mix = (leaf_buy + leaf_sell > 1e-9 ? leaf_buy / (leaf_buy + leaf_sell) : p_dir);
      double p_up = FXAI_Clamp(0.55 * p_dir + 0.45 * dir_mix, 0.001, 0.999);
      class_probs[(int)FXAI_LABEL_BUY] = active * p_up;
      class_probs[(int)FXAI_LABEL_SELL] = active * (1.0 - p_up);
      class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;
      NormalizeClassDistribution(class_probs);
      if(wsum > 0.0)
      {
         double mean_mv = MathMax(0.0, mvsum / wsum);
         double var_mv = MathMax(0.0, mv2sum / wsum - mean_mv * mean_mv);
         expected_move_points = mean_mv + 0.10 * MathSqrt(var_mv + 1e-6);
      }
      else
         expected_move_points = MathMax(PredictMoveHeadRaw(x), m_move_ema_abs);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double ev = 0.0;
      if(!PredictModelCore(x, hp, out.class_probs, ev))
         return false;
      double mvsum = 0.0, mv2sum = 0.0, wsum = 0.0;
      for(int t=0; t<m_tree_count; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[t], x);
         if(leaf < 0 || leaf >= m_trees[t].node_count) continue;
         double lw = MathAbs(m_trees[t].nodes[leaf].leaf_value) + 0.20;
         mvsum += lw * m_trees[t].nodes[leaf].move_mean;
         mv2sum += lw * (m_trees[t].nodes[leaf].move_var + m_trees[t].nodes[leaf].move_mean * m_trees[t].nodes[leaf].move_mean);
         wsum += lw;
      }
      double mean_mv = (wsum > 0.0 ? MathMax(0.0, mvsum / wsum) : MathMax(0.0, ev));
      double var_mv = (wsum > 0.0 ? MathMax(0.0, mv2sum / wsum - mean_mv * mean_mv) : 0.25 * mean_mv * mean_mv);
      double sigma = MathSqrt(var_mv + 1e-6);
      out.move_mean_points = mean_mv;
      out.move_q25_points = MathMax(0.0, mean_mv - 0.674 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, mean_mv);
      out.move_q75_points = MathMax(out.move_q50_points, mean_mv + 0.674 * sigma);
      double dir_mass = MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]);
      out.confidence = FXAI_Clamp(dir_mass, 0.0, 1.0);
      out.reliability = FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(x,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }


   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_quality_heads.Reset();
      m_step = 0;
      m_bias = 0.0;
      m_tree_count = 0;
      m_buf_head = 0;
      m_buf_size = 0;

      for(int i=0; i<FXAI_XGB_BUFFER; i++)
      {
         m_buf_y[i] = 0;
         m_buf_cls[i] = (int)FXAI_LABEL_SKIP;
         m_buf_move[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_buf_x[i][k] = 0.0;
      }
      for(int t=0; t<FXAI_XGB_MAX_TREES; t++)
         InitTree(m_trees[t]);
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         m_initialized = true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      int cls = NormalizeClassLabel(y, x, move_points);
      int y_dir = (cls == (int)FXAI_LABEL_BUY ? 1 : 0);

      EnsureInitialized(hp);
      m_step++;

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cls_w = 1.0;
      double w = FXAI_Clamp(MoveSampleWeight(x, move_points) * cls_w, 0.10, 4.00);
      UpdateNativeQualityHeads(x, w, h.lr, h.l2);

      double margin = ModelMargin(x);
      double p_raw = FXAI_Sigmoid(margin);

      // Keep bias adaptive between tree builds.
      if(cls != (int)FXAI_LABEL_SKIP)
      {
         double g_now = ((double)y_dir - p_raw) * w;
         m_bias += 0.01 * FXAI_ClipSym(g_now, 2.0);
      }

      PushSample(y_dir, cls, x, move_points, w);

      if(m_buf_size >= FXAI_XGB_MIN_BUFFER && (m_step % FXAI_XGB_BUILD_EVERY) == 0)
         BuildOneTree(h);

      UpdateCalibration(p_raw, y_dir, w);
      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, w);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double p_raw = FXAI_Sigmoid(ModelMargin(x));
      return CalibrateProb(p_raw);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double sum = 0.0;
      double sum2 = 0.0;
      double wsum = 0.0;
      for(int t=0; t<m_tree_count; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[t], x);
         if(leaf < 0 || leaf >= m_trees[t].node_count) continue;
         double mv = m_trees[t].nodes[leaf].move_mean;
         double lw = MathAbs(m_trees[t].nodes[leaf].leaf_value) + 0.10;
         if(mv <= 0.0) continue;
         sum += lw * mv;
         sum2 += lw * (m_trees[t].nodes[leaf].move_var + mv * mv);
         wsum += lw;
      }

      if(wsum > 0.0)
      {
         double mean_mv = sum / wsum;
         double var_mv = MathMax(0.0, sum2 / wsum - mean_mv * mean_mv);
         return MathMax(0.0, mean_mv + 0.10 * MathSqrt(var_mv + 1e-6));
      }
      return MathMax(PredictMoveHeadRaw(x), m_move_ema_abs);
   }
};

#endif // __FXAI_AI_XGBOOST_MQH__
