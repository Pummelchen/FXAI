#ifndef __FX6_AI_LIGHTGBM_MQH__
#define __FX6_AI_LIGHTGBM_MQH__

#include "..\plugin_base.mqh"

#define FX6_LGB_BINS 16
#define FX6_LGB_MAX_LEAVES 16
#define FX6_LGB_MAX_DEPTH 6
#define FX6_LGB_MAX_NODES (2 * FX6_LGB_MAX_LEAVES - 1)
#define FX6_LGB_MAX_TREES 64
#define FX6_LGB_BUFFER 2048
#define FX6_LGB_MIN_DATA 12
#define FX6_LGB_MIN_CHILD_HESS 0.10
#define FX6_LGB_GAMMA 0.01
#define FX6_LGB_BUILD_EVERY 64
#define FX6_LGB_MIN_BUFFER 128

struct FX6LGBNode
{
   bool   is_leaf;
   int    feature;
   double threshold;
   bool   default_left;
   int    left;
   int    right;
   int    depth;
   double leaf_value;
   double move_mean;
   int    sample_count;
};

struct FX6LGBTree
{
   int node_count;
   FX6LGBNode nodes[FX6_LGB_MAX_NODES];
};

class CFX6AILightGBM : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   double m_bias;

   FX6LGBTree m_trees[FX6_LGB_MAX_TREES];
   int        m_tree_count;

   double m_buf_x[FX6_LGB_BUFFER][FX6_AI_WEIGHTS];
   int    m_buf_y[FX6_LGB_BUFFER];
   double m_buf_move[FX6_LGB_BUFFER];
   double m_buf_w[FX6_LGB_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   void InitTree(FX6LGBTree &tree) const
   {
      tree.node_count = 1;
      for(int n=0; n<FX6_LGB_MAX_NODES; n++)
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
         tree.nodes[n].sample_count = 0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FX6_LGB_BUFFER;
      int p = start + logical_idx;
      while(p >= FX6_LGB_BUFFER) p -= FX6_LGB_BUFFER;
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
      if(m_buf_head >= FX6_LGB_BUFFER) m_buf_head = 0;
      if(m_buf_size < FX6_LGB_BUFFER) m_buf_size++;
   }

   int TraverseLeafIndex(const FX6LGBTree &tree, const double &x[]) const
   {
      int node = 0;
      int guard = 0;
      while(node >= 0 && node < tree.node_count && guard < FX6_LGB_MAX_NODES)
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

   double TreeOutput(const FX6LGBTree &tree, const double &x[]) const
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

   int BinByRange(const double x, const double minv, const double maxv) const
   {
      if(!MathIsValidNumber(x)) return -1;
      double range = maxv - minv;
      if(range <= 1e-12) return 0;
      double q = (x - minv) / range;
      if(q <= 0.0) return 0;
      if(q >= 1.0) return FX6_LGB_BINS - 1;
      int b = (int)MathFloor(q * (double)FX6_LGB_BINS);
      if(b < 0) b = 0;
      if(b >= FX6_LGB_BINS) b = FX6_LGB_BINS - 1;
      return b;
   }

   void SetLeafFromAssign(FX6LGBTree &tree,
                          const int node_idx,
                          const int &assign[],
                          const int tag,
                          const double &g[],
                          const double &h[],
                          const double &mv[],
                          const double eta,
                          const double lambda) const
   {
      double G = 0.0, H = 0.0, sum_mv = 0.0;
      int n = ArraySize(assign);
      int cnt = 0;
      for(int i=0; i<n; i++)
      {
         if(assign[i] != tag) continue;
         G += g[i];
         H += h[i];
         sum_mv += MathAbs(mv[i]);
         cnt++;
      }

      tree.nodes[node_idx].is_leaf = true;
      tree.nodes[node_idx].feature = -1;
      tree.nodes[node_idx].threshold = 0.0;
      tree.nodes[node_idx].default_left = true;
      tree.nodes[node_idx].left = -1;
      tree.nodes[node_idx].right = -1;
      tree.nodes[node_idx].sample_count = cnt;
      tree.nodes[node_idx].move_mean = (cnt > 0 ? sum_mv / (double)cnt : 0.0);

      double leaf = 0.0;
      if(H > 1e-9)
         leaf = eta * FX6_ClipSym(G / (H + lambda), 5.0);
      tree.nodes[node_idx].leaf_value = leaf;
   }

   bool FindBestSplitForLeaf(const int &assign[],
                             const int leaf_tag,
                             const int depth,
                             const double &x_all[][FX6_AI_WEIGHTS],
                             const double &g[],
                             const double &h[],
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

      if(depth >= FX6_LGB_MAX_DEPTH) return false;

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

      if(Ctot < 2 * FX6_LGB_MIN_DATA) return false;
      if(Htot < 2.0 * FX6_LGB_MIN_CHILD_HESS) return false;

      double parent_score = (Gtot * Gtot) / (Htot + lambda + 1e-9);

      for(int f=1; f<FX6_AI_WEIGHTS; f++)
      {
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

         if(c_valid < 2 * FX6_LGB_MIN_DATA) continue;
         if(maxv - minv < 1e-9) continue;

         double gbin[FX6_LGB_BINS];
         double hbin[FX6_LGB_BINS];
         int cbin[FX6_LGB_BINS];
         for(int b=0; b<FX6_LGB_BINS; b++)
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
         for(int sb=0; sb<FX6_LGB_BINS - 1; sb++)
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

               if(cL >= FX6_LGB_MIN_DATA && cR >= FX6_LGB_MIN_DATA &&
                  hL >= FX6_LGB_MIN_CHILD_HESS && hR >= FX6_LGB_MIN_CHILD_HESS)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FX6_LGB_GAMMA;
                  if(gain > best_gain)
                  {
                     best_gain = gain;
                     best_feature = f;
                     best_default_left = true;
                     best_thr = minv + (maxv - minv) * ((double)(sb + 1) / (double)FX6_LGB_BINS);
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

               if(cL >= FX6_LGB_MIN_DATA && cR >= FX6_LGB_MIN_DATA &&
                  hL >= FX6_LGB_MIN_CHILD_HESS && hR >= FX6_LGB_MIN_CHILD_HESS)
               {
                  double gain = 0.5 * ((gL * gL) / (hL + lambda + 1e-9) +
                                       (gR * gR) / (hR + lambda + 1e-9) -
                                       parent_score) - FX6_LGB_GAMMA;
                  if(gain > best_gain)
                  {
                     best_gain = gain;
                     best_feature = f;
                     best_default_left = false;
                     best_thr = minv + (maxv - minv) * ((double)(sb + 1) / (double)FX6_LGB_BINS);
                  }
               }
            }
         }
      }

      return (best_feature >= 1 && best_gain > 0.0);
   }

   bool BuildOneTree(const FX6AIHyperParams &hp)
   {
      if(m_buf_size < FX6_LGB_MIN_BUFFER) return false;

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

      double G = 0.0, H = 0.0;
      for(int i=0; i<n; i++) { G += g[i]; H += h[i]; }

      double lambda = FX6_Clamp(hp.xgb_l2, 0.0001, 10.0000);
      double eta = FX6_Clamp(hp.xgb_lr, 0.0001, 1.0000);
      if(H > 1e-9)
         m_bias += 0.20 * eta * FX6_ClipSym(G / (H + lambda), 5.0);

      FX6LGBTree tree;
      InitTree(tree);

      int assign[];
      ArrayResize(assign, n);
      for(int i=0; i<n; i++) assign[i] = 0;
      SetLeafFromAssign(tree, 0, assign, 0, g, h, mv_all, eta, lambda);

      int leaves = 1;
      while(leaves < FX6_LGB_MAX_LEAVES)
      {
         int best_leaf = -1;
         int best_feature = -1;
         double best_thr = 0.0;
         bool best_default_left = true;
         double best_gain = 0.0;

         for(int node=0; node<tree.node_count; node++)
         {
            if(!tree.nodes[node].is_leaf) continue;
            if(tree.nodes[node].depth >= FX6_LGB_MAX_DEPTH) continue;
            if(tree.nodes[node].sample_count < 2 * FX6_LGB_MIN_DATA) continue;

            int f = -1;
            double thr = 0.0;
            bool def_left = true;
            double gain = 0.0;
            if(!FindBestSplitForLeaf(assign,
                                     node,
                                     tree.nodes[node].depth,
                                     x_all,
                                     g,
                                     h,
                                     lambda,
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
         if(tree.node_count + 2 > FX6_LGB_MAX_NODES) break;

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

         for(int i=0; i<n; i++)
         {
            if(assign[i] != best_leaf) continue;
            double xv = x_all[i][best_feature];
            bool go_left = (!MathIsValidNumber(xv) ? best_default_left : (xv <= best_thr));
            assign[i] = (go_left ? left : right);
         }

         SetLeafFromAssign(tree, left, assign, left, g, h, mv_all, eta, lambda);
         SetLeafFromAssign(tree, right, assign, right, g, h, mv_all, eta, lambda);
         leaves++;
      }

      if(m_tree_count < FX6_LGB_MAX_TREES)
      {
         m_trees[m_tree_count] = tree;
         m_tree_count++;
      }
      else
      {
         for(int t=1; t<FX6_LGB_MAX_TREES; t++)
            m_trees[t - 1] = m_trees[t];
         m_trees[FX6_LGB_MAX_TREES - 1] = tree;
         m_tree_count = FX6_LGB_MAX_TREES;
      }

      return true;
   }

   void UpdateWeighted(const int y,
                       const double &x[],
                       const FX6AIHyperParams &hp,
                       const double sample_w,
                       const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;

      double w = FX6_Clamp(sample_w, 0.25, 4.00);
      double l2 = FX6_Clamp(hp.xgb_l2, 0.0001, 10.0000);

      double margin = ModelMargin(x);
      double p_raw = FX6_Sigmoid(margin);
      double g_now = ((double)y - p_raw) * w;
      double h_now = FX6_Clamp(p_raw * (1.0 - p_raw) * w, 0.02, 4.00);
      m_bias += 0.01 * FX6_ClipSym(g_now / (h_now + l2), 2.0);

      PushSample(y, x, move_points, w);

      if(m_buf_size >= FX6_LGB_MIN_BUFFER && (m_step % FX6_LGB_BUILD_EVERY) == 0)
         BuildOneTree(hp);

      UpdateCalibration(p_raw, y, w);
      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, hp, w);
   }

public:
   CFX6AILightGBM(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_LIGHTGBM; }
   virtual string AIName(void) const { return "lightgbm"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_bias = 0.0;
      m_tree_count = 0;
      m_buf_head = 0;
      m_buf_size = 0;

      for(int i=0; i<FX6_LGB_BUFFER; i++)
      {
         m_buf_y[i] = 0;
         m_buf_move[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FX6_AI_WEIGHTS; k++) m_buf_x[i][k] = 0.0;
      }

      for(int t=0; t<FX6_LGB_MAX_TREES; t++)
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

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cls_w = 1.0;
      double w = FX6_Clamp(MoveSampleWeight(x, move_points) * cls_w, 0.10, 4.00);
      UpdateWeighted(y_dir, x, h, w, move_points);
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

#endif // __FX6_AI_LIGHTGBM_MQH__
