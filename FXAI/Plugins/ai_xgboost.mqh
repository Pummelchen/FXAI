#ifndef __FX6_AI_XGBOOST_MQH__
#define __FX6_AI_XGBOOST_MQH__

#include "..\plugin_base.mqh"

#define FX6_XGB_MAX_TREES 64
#define FX6_XGB_MAX_DEPTH 3
#define FX6_XGB_MAX_NODES 31
#define FX6_XGB_BINS 16
#define FX6_XGB_BUFFER 2048
#define FX6_XGB_MIN_SPLIT_SAMPLES 12
#define FX6_XGB_MIN_CHILD_WEIGHT 0.10
#define FX6_XGB_GAMMA 0.01
#define FX6_XGB_BUILD_EVERY 64
#define FX6_XGB_MIN_BUFFER 128

struct FX6XGBNode
{
   bool   is_leaf;
   int    feature;
   double threshold;
   bool   default_left;
   int    left;
   int    right;
   double leaf_value;
   double move_mean;
   int    sample_count;
};

struct FX6XGBTree
{
   int node_count;
   FX6XGBNode nodes[FX6_XGB_MAX_NODES];
};

class CFX6AIXGBoost : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   double m_bias;

   FX6XGBTree m_trees[FX6_XGB_MAX_TREES];
   int        m_tree_count;

   double m_buf_x[FX6_XGB_BUFFER][FX6_AI_WEIGHTS];
   int    m_buf_y[FX6_XGB_BUFFER];
   double m_buf_move[FX6_XGB_BUFFER];
   double m_buf_w[FX6_XGB_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   void InitTree(FX6XGBTree &tree) const
   {
      tree.node_count = 1;
      for(int n=0; n<FX6_XGB_MAX_NODES; n++)
      {
         tree.nodes[n].is_leaf = true;
         tree.nodes[n].feature = -1;
         tree.nodes[n].threshold = 0.0;
         tree.nodes[n].default_left = true;
         tree.nodes[n].left = -1;
         tree.nodes[n].right = -1;
         tree.nodes[n].leaf_value = 0.0;
         tree.nodes[n].move_mean = 0.0;
         tree.nodes[n].sample_count = 0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FX6_XGB_BUFFER;
      int p = start + logical_idx;
      while(p >= FX6_XGB_BUFFER) p -= FX6_XGB_BUFFER;
      return p;
   }

   void PushSample(const int y,
                   const double &x[],
                   const double move_points,
                   const double sample_w)
   {
      int pos = m_buf_head;
      for(int i=0; i<FX6_AI_WEIGHTS; i++) m_buf_x[pos][i] = x[i];
      m_buf_y[pos] = y;
      m_buf_move[pos] = move_points;
      m_buf_w[pos] = sample_w;

      m_buf_head++;
      if(m_buf_head >= FX6_XGB_BUFFER) m_buf_head = 0;
      if(m_buf_size < FX6_XGB_BUFFER) m_buf_size++;
   }

   int TraverseLeafIndex(const FX6XGBTree &tree, const double &x[]) const
   {
      int node = 0;
      int guard = 0;
      while(node >= 0 && node < tree.node_count && guard < FX6_XGB_MAX_NODES)
      {
         if(tree.nodes[node].is_leaf) return node;

         int f = tree.nodes[node].feature;
         if(f < 1 || f >= FX6_AI_WEIGHTS) return node;
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

   double TreeOutput(const FX6XGBTree &tree, const double &x[]) const
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

   void SetLeaf(FX6XGBTree &tree,
                const int node_idx,
                const int &sidx[],
                const double &g[],
                const double &h[],
                const double &mv[],
                const double lambda,
                const double eta) const
   {
      double G = 0.0, H = 0.0, sum_mv = 0.0;
      int n = ArraySize(sidx);
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         G += g[id];
         H += h[id];
         sum_mv += MathAbs(mv[id]);
      }

      tree.nodes[node_idx].is_leaf = true;
      tree.nodes[node_idx].feature = -1;
      tree.nodes[node_idx].threshold = 0.0;
      tree.nodes[node_idx].default_left = true;
      tree.nodes[node_idx].left = -1;
      tree.nodes[node_idx].right = -1;
      tree.nodes[node_idx].sample_count = n;
      tree.nodes[node_idx].move_mean = (n > 0 ? sum_mv / (double)n : 0.0);

      double leaf = 0.0;
      if(H > 1e-9)
         leaf = eta * FX6_ClipSym(G / (H + lambda), 5.0);
      tree.nodes[node_idx].leaf_value = leaf;
   }

   bool FindBestSplit(const int &sidx[],
                      const double &g[],
                      const double &h[],
                      const double &x_all[][FX6_AI_WEIGHTS],
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
      if(n < 2 * FX6_XGB_MIN_SPLIT_SAMPLES) return false;

      double Gtot = 0.0, Htot = 0.0;
      for(int i=0; i<n; i++)
      {
         int id = sidx[i];
         Gtot += g[id];
         Htot += h[id];
      }
      if(Htot <= FX6_XGB_MIN_CHILD_WEIGHT * 2.0) return false;

      double parent_score = (Gtot * Gtot) / (Htot + lambda + 1e-9);

      for(int f=1; f<FX6_AI_WEIGHTS; f++)
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

         if(Cvalid < 2 * FX6_XGB_MIN_SPLIT_SAMPLES) continue;
         if(maxv - minv < 1e-9) continue;

         for(int b=1; b<FX6_XGB_BINS; b++)
         {
            double thr = minv + (maxv - minv) * ((double)b / (double)FX6_XGB_BINS);
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

               if(cL >= FX6_XGB_MIN_SPLIT_SAMPLES && cR >= FX6_XGB_MIN_SPLIT_SAMPLES &&
                  hL >= FX6_XGB_MIN_CHILD_WEIGHT && hR >= FX6_XGB_MIN_CHILD_WEIGHT)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FX6_XGB_GAMMA;
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

               if(cL >= FX6_XGB_MIN_SPLIT_SAMPLES && cR >= FX6_XGB_MIN_SPLIT_SAMPLES &&
                  hL >= FX6_XGB_MIN_CHILD_WEIGHT && hR >= FX6_XGB_MIN_CHILD_WEIGHT)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FX6_XGB_GAMMA;
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

   void BuildNode(FX6XGBTree &tree,
                  const int node_idx,
                  const int depth,
                  const int &sidx[],
                  const double &g[],
                  const double &h[],
                  const double &x_all[][FX6_AI_WEIGHTS],
                  const double &mv[],
                  const double lambda,
                  const double eta) const
   {
      if(node_idx < 0 || node_idx >= FX6_XGB_MAX_NODES) return;
      if(ArraySize(sidx) < FX6_XGB_MIN_SPLIT_SAMPLES || depth >= FX6_XGB_MAX_DEPTH)
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, lambda, eta);
         return;
      }

      int best_f = -1;
      double best_thr = 0.0;
      bool best_def_left = true;
      double best_gain = 0.0;
      if(!FindBestSplit(sidx, g, h, x_all, lambda, best_f, best_thr, best_def_left, best_gain))
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, lambda, eta);
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

      if(ArraySize(left_idx) < FX6_XGB_MIN_SPLIT_SAMPLES ||
         ArraySize(right_idx) < FX6_XGB_MIN_SPLIT_SAMPLES ||
         tree.node_count + 2 > FX6_XGB_MAX_NODES)
      {
         SetLeaf(tree, node_idx, sidx, g, h, mv, lambda, eta);
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

      BuildNode(tree, left_node, depth + 1, left_idx, g, h, x_all, mv, lambda, eta);
      BuildNode(tree, right_node, depth + 1, right_idx, g, h, x_all, mv, lambda, eta);
   }

   bool BuildOneTree(const FX6AIHyperParams &hp)
   {
      if(m_buf_size < FX6_XGB_MIN_BUFFER) return false;

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
         for(int k=0; k<FX6_AI_WEIGHTS; k++) x_all[i][k] = m_buf_x[p][k];
         y_all[i] = m_buf_y[p];
         mv_all[i] = m_buf_move[p];
         w_all[i] = m_buf_w[p];
      }

      double g[];
      double h[];
      ArrayResize(g, n);
      ArrayResize(h, n);

      for(int i=0; i<n; i++)
      {
         double xloc[FX6_AI_WEIGHTS];
         for(int k=0; k<FX6_AI_WEIGHTS; k++) xloc[k] = x_all[i][k];
         double margin = ModelMargin(xloc);
         double p = FX6_Sigmoid(margin);
         double wi = FX6_Clamp(w_all[i], 0.25, 4.00);
         g[i] = ((double)y_all[i] - p) * wi;
         h[i] = FX6_Clamp(p * (1.0 - p) * wi, 0.02, 4.00);
      }

      // Small bias refinement from batch gradient statistics.
      double G = 0.0, H = 0.0;
      for(int i=0; i<n; i++) { G += g[i]; H += h[i]; }
      double lambda = FX6_Clamp(hp.xgb_l2, 0.0001, 10.0000);
      double eta = FX6_Clamp(hp.xgb_lr, 0.0001, 1.0000);
      if(H > 1e-9)
         m_bias += 0.20 * eta * FX6_ClipSym(G / (H + lambda), 5.0);

      int root_idx[];
      ArrayResize(root_idx, n);
      for(int i=0; i<n; i++) root_idx[i] = i;

      FX6XGBTree tree;
      InitTree(tree);
      BuildNode(tree, 0, 0, root_idx, g, h, x_all, mv_all, lambda, eta);
      if(tree.node_count <= 0) return false;

      if(m_tree_count < FX6_XGB_MAX_TREES)
      {
         m_trees[m_tree_count] = tree;
         m_tree_count++;
      }
      else
      {
         for(int t=1; t<FX6_XGB_MAX_TREES; t++)
            m_trees[t - 1] = m_trees[t];
         m_trees[FX6_XGB_MAX_TREES - 1] = tree;
         m_tree_count = FX6_XGB_MAX_TREES;
      }

      return true;
   }

public:
   CFX6AIXGBoost(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_XGBOOST; }
   virtual string AIName(void) const { return "xgboost"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_bias = 0.0;
      m_tree_count = 0;
      m_buf_head = 0;
      m_buf_size = 0;

      for(int i=0; i<FX6_XGB_BUFFER; i++)
      {
         m_buf_y[i] = 0;
         m_buf_move[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FX6_AI_WEIGHTS; k++) m_buf_x[i][k] = 0.0;
      }
      for(int t=0; t<FX6_XGB_MAX_TREES; t++)
         InitTree(m_trees[t]);
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized)
         m_initialized = true;
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
      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls == (int)FX6_LABEL_SKIP) return;
      int y_dir = (cls == (int)FX6_LABEL_BUY ? 1 : 0);

      EnsureInitialized(hp);
      m_step++;

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cls_w = 1.0;
      double w = FX6_Clamp(MoveSampleWeight(x, move_points) * cls_w, 0.10, 4.00);

      double margin = ModelMargin(x);
      double p_raw = FX6_Sigmoid(margin);

      // Keep bias adaptive between tree builds.
      double g_now = ((double)y_dir - p_raw) * w;
      m_bias += 0.01 * FX6_ClipSym(g_now, 2.0);

      PushSample(y_dir, x, move_points, w);

      if(m_buf_size >= FX6_XGB_MIN_BUFFER && (m_step % FX6_XGB_BUILD_EVERY) == 0)
         BuildOneTree(h);

      UpdateCalibration(p_raw, y_dir, w);
      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, w);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double p_raw = FX6_Sigmoid(ModelMargin(x));
      return CalibrateProb(p_raw);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double sum = 0.0;
      double wsum = 0.0;
      for(int t=0; t<m_tree_count; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[t], x);
         if(leaf < 0 || leaf >= m_trees[t].node_count) continue;
         double mv = m_trees[t].nodes[leaf].move_mean;
         double lw = MathAbs(m_trees[t].nodes[leaf].leaf_value) + 0.10;
         if(mv <= 0.0) continue;
         sum += lw * mv;
         wsum += lw;
      }

      if(wsum > 0.0) return sum / wsum;
      return CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
   }
};

#endif // __FX6_AI_XGBOOST_MQH__
