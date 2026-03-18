#ifndef __FXAI_AI_CATBOOST_MQH__
#define __FXAI_AI_CATBOOST_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_CAT_CLASS_COUNT 3
#define FXAI_CAT_MAX_TREES 320
#define FXAI_CAT_MAX_DEPTH 6
#define FXAI_CAT_MAX_LEAVES (1 << FXAI_CAT_MAX_DEPTH)
#define FXAI_CAT_BINS 64
#define FXAI_CAT_MAX_BORDERS (FXAI_CAT_BINS - 1)
#define FXAI_CAT_BUFFER 6144
#define FXAI_CAT_MIN_BUFFER 256
#define FXAI_CAT_BUILD_EVERY 48
#define FXAI_CAT_MIN_DATA 12
#define FXAI_CAT_MIN_CHILD_HESS 0.08
#define FXAI_CAT_GAMMA 0.015
#define FXAI_CAT_CAL_BINS 16
#define FXAI_CAT_ORDER_PERMS 6
#define FXAI_CAT_CTR_BASE 10
#define FXAI_CAT_CTR_FEAT_PER_BASE 3
#define FXAI_CAT_CTR_PAIR_COUNT 6
#define FXAI_CAT_CTR_PAIR_HASH 257
#define FXAI_CAT_CTR_FEATURES ((FXAI_CAT_CTR_BASE * FXAI_CAT_CTR_FEAT_PER_BASE) + FXAI_CAT_CTR_PAIR_COUNT)
#define FXAI_CAT_EXT_WEIGHTS (FXAI_AI_WEIGHTS + FXAI_CAT_CTR_FEATURES)
#define FXAI_CAT_LEAF_NEWTON_STEPS 5
#define FXAI_CAT_ECE_BINS 12
#define FXAI_CAT_TRACK_FEATS 6

struct FXAICatLevelSplit
{
   int    feature;
   double threshold;
   bool   default_left;
};

struct FXAICatTree
{
   int depth;
   FXAICatLevelSplit levels[FXAI_CAT_MAX_DEPTH];
   double leaf_value[FXAI_CAT_MAX_LEAVES][FXAI_CAT_CLASS_COUNT];
   double leaf_move_mean[FXAI_CAT_MAX_LEAVES];
   int    leaf_count[FXAI_CAT_MAX_LEAVES];
};

class CFXAIAICatBoost : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   int    m_tree_count;
   double m_bias[FXAI_CAT_CLASS_COUNT];
   CFXAINativeQualityHeads m_quality_heads;

   FXAICatTree m_trees[FXAI_CAT_MAX_TREES];

   // Ordered online sample buffer.
   double m_buf_x[FXAI_CAT_BUFFER][FXAI_AI_WEIGHTS];
   int    m_buf_y[FXAI_CAT_BUFFER];
   double m_buf_move[FXAI_CAT_BUFFER];
   double m_buf_w[FXAI_CAT_BUFFER];
   int    m_buf_head;
   int    m_buf_size;

   // Categorical/CTR settings and runtime stats.
   int    m_ctr_base_feat[FXAI_CAT_CTR_BASE];
   int    m_ctr_pair_a[FXAI_CAT_CTR_PAIR_COUNT];
   int    m_ctr_pair_b[FXAI_CAT_CTR_PAIR_COUNT];
   bool   m_ctr_ready;
   double m_ctr_global_class[FXAI_CAT_CLASS_COUNT];
   double m_ctr_bin_total[FXAI_CAT_CTR_BASE][FXAI_CAT_BINS];
   double m_ctr_bin_class[FXAI_CAT_CTR_BASE][FXAI_CAT_BINS][FXAI_CAT_CLASS_COUNT];
   double m_ctr_pair_total[FXAI_CAT_CTR_PAIR_COUNT][FXAI_CAT_CTR_PAIR_HASH];
   double m_ctr_pair_buy[FXAI_CAT_CTR_PAIR_COUNT][FXAI_CAT_CTR_PAIR_HASH];

   // Quantile borders for base feature quantization.
   int    m_base_border_count[FXAI_AI_WEIGHTS];
   double m_base_borders[FXAI_AI_WEIGHTS][FXAI_CAT_MAX_BORDERS];

   // Candidate borders for split search on extended feature space.
   int    m_split_border_count[FXAI_CAT_EXT_WEIGHTS];
   int    m_split_valid_count[FXAI_CAT_EXT_WEIGHTS];
   double m_split_borders[FXAI_CAT_EXT_WEIGHTS][FXAI_CAT_MAX_BORDERS];

   // Split reuse penalty / feature usage tracker.
   int    m_feature_use[FXAI_CAT_EXT_WEIGHTS];

   // Drift adaptation and class balance.
   double m_cls_ema[FXAI_CAT_CLASS_COUNT];
   bool   m_loss_ready;
   double m_loss_fast;
   double m_loss_slow;
   int    m_drift_cooldown;

   // Plugin-native multiclass calibration (vector scaling + isotonic-like bins).
   double m_cal_vs_w[FXAI_CAT_CLASS_COUNT][FXAI_CAT_CLASS_COUNT];
   double m_cal_vs_b[FXAI_CAT_CLASS_COUNT];
   double m_cal_session_b[4][FXAI_CAT_CLASS_COUNT];
   double m_cal_regime_b[2][FXAI_CAT_CLASS_COUNT];
   double m_cal3_iso_pos[FXAI_CAT_CLASS_COUNT][FXAI_CAT_CAL_BINS];
   double m_cal3_iso_cnt[FXAI_CAT_CLASS_COUNT][FXAI_CAT_CAL_BINS];
   int    m_cal3_steps;

   // Validation and drift harness (rolling CE/Brier/ECE + feature drift alarms).
   bool   m_val_ready;
   int    m_val_steps;
   double m_val_ce_fast;
   double m_val_ce_slow;
   double m_val_brier_fast;
   double m_val_brier_slow;
   double m_val_ece_fast;
   double m_val_ece_slow;
   double m_ece_mass[FXAI_CAT_ECE_BINS];
   double m_ece_acc[FXAI_CAT_ECE_BINS];
   double m_ece_conf[FXAI_CAT_ECE_BINS];

   bool   m_feat_stats_ready;
   double m_feat_ref_mean[FXAI_CAT_TRACK_FEATS];
   double m_feat_ref_var[FXAI_CAT_TRACK_FEATS];
   double m_feat_cur_mean[FXAI_CAT_TRACK_FEATS];
   double m_feat_cur_var[FXAI_CAT_TRACK_FEATS];
   double m_feat_drift_score;
   int    m_quality_alarm;

   int TrackFeatIndex(const int k) const
   {
      static const int ids[FXAI_CAT_TRACK_FEATS] = {1, 2, 3, 7, 12, 20};
      if(k < 0 || k >= FXAI_CAT_TRACK_FEATS) return 1;
      int f = ids[k];
      if(f < 1) f = 1;
      if(f >= FXAI_AI_WEIGHTS) f = FXAI_AI_WEIGHTS - 1;
      return f;
   }

   int SessionBucket(const datetime t) const
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      int h = dt.hour;
      if(h < 8) return 0;      // Asia session
      if(h < 13) return 1;     // Europe open/core
      if(h < 20) return 2;     // US overlap/core
      return 3;                // late/illiquid hours
   }

   int RegimeBucket(void) const
   {
      if(!m_move_ready) return 0;
      if(m_move_ema_abs > 5.0) return 1;
      return 0;
   }

   double Rand01(void)
   {
      return PluginRand01();
   }

   double RandGauss(void)
   {
      // Lightweight approx N(0,1) via CLT sum(U)-6.
      double s = 0.0;
      for(int i=0; i<12; i++) s += Rand01();
      return s - 6.0;
   }

   int PairHash(const int a, const int b, const int q) const
   {
      int aa = (a >= 0 ? a : 0);
      int bb = (b >= 0 ? b : 0);
      int h = (aa * 131 + bb * 17 + q * 53 + 97) % FXAI_CAT_CTR_PAIR_HASH;
      if(h < 0) h += FXAI_CAT_CTR_PAIR_HASH;
      return h;
   }

   void InitTree(FXAICatTree &tree) const
   {
      tree.depth = 0;
      for(int d=0; d<FXAI_CAT_MAX_DEPTH; d++)
      {
         tree.levels[d].feature = -1;
         tree.levels[d].threshold = 0.0;
         tree.levels[d].default_left = true;
      }

      for(int l=0; l<FXAI_CAT_MAX_LEAVES; l++)
      {
         tree.leaf_move_mean[l] = 0.0;
         tree.leaf_count[l] = 0;
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            tree.leaf_value[l][c] = 0.0;
      }
   }

   int BufPos(const int logical_idx) const
   {
      if(m_buf_size <= 0) return 0;
      int start = m_buf_head - m_buf_size;
      while(start < 0) start += FXAI_CAT_BUFFER;
      int p = start + logical_idx;
      while(p >= FXAI_CAT_BUFFER) p -= FXAI_CAT_BUFFER;
      return p;
   }

   void PushSample(const int y,
                   const double &x[],
                   const double move_points,
                   const double sample_w)
   {
      int pos = m_buf_head;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_buf_x[pos][i] = x[i];
      m_buf_y[pos] = y;
      m_buf_move[pos] = move_points;
      m_buf_w[pos] = sample_w;

      m_buf_head++;
      if(m_buf_head >= FXAI_CAT_BUFFER) m_buf_head = 0;
      if(m_buf_size < FXAI_CAT_BUFFER) m_buf_size++;
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
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

   void InitCTRLayout(void)
   {
      for(int b=0; b<FXAI_CAT_CTR_BASE; b++)
      {
         int f = 1 + b;
         if(f >= FXAI_AI_WEIGHTS) f = FXAI_AI_WEIGHTS - 1;
         m_ctr_base_feat[b] = f;
      }

      // Pair selected high-signal interactions from base list.
      m_ctr_pair_a[0] = 0; m_ctr_pair_b[0] = 1;
      m_ctr_pair_a[1] = 0; m_ctr_pair_b[1] = 2;
      m_ctr_pair_a[2] = 1; m_ctr_pair_b[2] = 3;
      m_ctr_pair_a[3] = 2; m_ctr_pair_b[3] = 4;
      m_ctr_pair_a[4] = 5; m_ctr_pair_b[4] = 6;
      m_ctr_pair_a[5] = 7; m_ctr_pair_b[5] = 8;
   }

   void ResetCTRState(void)
   {
      m_ctr_ready = false;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_ctr_global_class[c] = 1.0 / 3.0;

      for(int b=0; b<FXAI_CAT_CTR_BASE; b++)
      {
         for(int k=0; k<FXAI_CAT_BINS; k++)
         {
            m_ctr_bin_total[b][k] = 0.0;
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               m_ctr_bin_class[b][k][c] = 0.0;
         }
      }

      for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
      {
         for(int h=0; h<FXAI_CAT_CTR_PAIR_HASH; h++)
         {
            m_ctr_pair_total[q][h] = 0.0;
            m_ctr_pair_buy[q][h] = 0.0;
         }
      }
   }

   void ResetSplitState(void)
   {
      for(int f=0; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         m_feature_use[f] = 0;
         m_split_border_count[f] = 0;
         m_split_valid_count[f] = 0;
         for(int b=0; b<FXAI_CAT_MAX_BORDERS; b++)
            m_split_borders[f][b] = 0.0;
      }
      for(int f=0; f<FXAI_AI_WEIGHTS; f++)
      {
         m_base_border_count[f] = 0;
         for(int b=0; b<FXAI_CAT_MAX_BORDERS; b++)
            m_base_borders[f][b] = 0.0;
      }
   }

   void BuildBordersForFeature(const int n,
                               const double &x_all[][FXAI_AI_WEIGHTS],
                               const int feature)
   {
      if(feature < 1 || feature >= FXAI_AI_WEIGHTS)
      {
         m_base_border_count[feature] = 0;
         return;
      }

      double vals[];
      ArrayResize(vals, 0);
      for(int i=0; i<n; i++)
      {
         double v = x_all[i][feature];
         if(!MathIsValidNumber(v)) continue;
         int sz = ArraySize(vals);
         ArrayResize(vals, sz + 1);
         vals[sz] = v;
      }

      int valid = ArraySize(vals);
      if(valid < 2)
      {
         m_base_border_count[feature] = 0;
         return;
      }

      ArraySort(vals);
      int desired = FXAI_CAT_BINS - 1;
      if(desired > valid - 1) desired = valid - 1;

      int out = 0;
      double last = -DBL_MAX;
      for(int b=1; b<=desired; b++)
      {
         int qi = (int)MathFloor(((double)b / (double)(desired + 1)) * (double)(valid - 1));
         if(qi < 0) qi = 0;
         if(qi >= valid) qi = valid - 1;
         double thr = vals[qi];
         if(out == 0 || MathAbs(thr - last) > 1e-12)
         {
            m_base_borders[feature][out] = thr;
            last = thr;
            out++;
            if(out >= FXAI_CAT_MAX_BORDERS) break;
         }
      }
      m_base_border_count[feature] = out;
   }

   void BuildBaseBorders(const int n,
                         const double &x_all[][FXAI_AI_WEIGHTS])
   {
      for(int f=1; f<FXAI_AI_WEIGHTS; f++)
         BuildBordersForFeature(n, x_all, f);
   }

   int QuantizeBaseFeature(const int base_feature_idx,
                           const double v) const
   {
      int f = base_feature_idx;
      if(f < 1 || f >= FXAI_AI_WEIGHTS) return -1;
      if(!MathIsValidNumber(v)) return -1;
      int bc = m_base_border_count[f];
      if(bc <= 0) return 0;

      int lo = 0;
      int hi = bc - 1;
      int pos = bc;
      while(lo <= hi)
      {
         int mid = (lo + hi) >> 1;
         if(v <= m_base_borders[f][mid])
         {
            pos = mid;
            hi = mid - 1;
         }
         else
         {
            lo = mid + 1;
         }
      }

      if(pos < 0) pos = 0;
      if(pos >= FXAI_CAT_BINS) pos = FXAI_CAT_BINS - 1;
      return pos;
   }

   int TraverseLeafIndex(const FXAICatTree &tree,
                         const double &x_ext[]) const
   {
      if(tree.depth <= 0) return 0;
      int leaf = 0;
      for(int d=0; d<tree.depth; d++)
      {
         int f = tree.levels[d].feature;
         if(f < 1 || f >= FXAI_CAT_EXT_WEIGHTS) break;

         double xv = x_ext[f];
         bool go_left;
         if(!MathIsValidNumber(xv))
            go_left = tree.levels[d].default_left;
         else
            go_left = (xv <= tree.levels[d].threshold);

         leaf = (leaf << 1) | (go_left ? 0 : 1);
         if(leaf < 0 || leaf >= FXAI_CAT_MAX_LEAVES)
            return 0;
      }
      return leaf;
   }

   void TreeMargins(const FXAICatTree &tree,
                    const double &x_ext[],
                    double &margins[]) const
   {
      int leaf = TraverseLeafIndex(tree, x_ext);
      if(leaf < 0 || leaf >= FXAI_CAT_MAX_LEAVES) return;

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         margins[c] += tree.leaf_value[leaf][c];
   }

   void ModelMarginsExt(const double &x_ext[],
                        double &margins[]) const
   {
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         margins[c] = m_bias[c];

      for(int t=0; t<m_tree_count; t++)
         TreeMargins(m_trees[t], x_ext, margins);
   }

   void BuildSplitCandidates(const int n,
                             const double &x_ext[][FXAI_CAT_EXT_WEIGHTS])
   {
      for(int f=0; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         m_split_border_count[f] = 0;
         m_split_valid_count[f] = 0;
      }

      for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         double vals[];
         ArrayResize(vals, 0);
         for(int i=0; i<n; i++)
         {
            double v = x_ext[i][f];
            if(!MathIsValidNumber(v)) continue;
            int sz = ArraySize(vals);
            ArrayResize(vals, sz + 1);
            vals[sz] = v;
         }

         int valid = ArraySize(vals);
         m_split_valid_count[f] = valid;
         if(valid < 2)
         {
            m_split_border_count[f] = 0;
            continue;
         }

         ArraySort(vals);
         int desired = FXAI_CAT_BINS - 1;
         if(desired > valid - 1) desired = valid - 1;

         int out = 0;
         double last = -DBL_MAX;
         for(int b=1; b<=desired; b++)
         {
            int qi = (int)MathFloor(((double)b / (double)(desired + 1)) * (double)(valid - 1));
            if(qi < 0) qi = 0;
            if(qi >= valid) qi = valid - 1;
            double thr = vals[qi];
            if(out == 0 || MathAbs(thr - last) > 1e-12)
            {
               m_split_borders[f][out] = thr;
               last = thr;
               out++;
               if(out >= FXAI_CAT_MAX_BORDERS) break;
            }
         }
         m_split_border_count[f] = out;
      }
   }

   void BuildPermutations(const int n,
                          int &perm[][FXAI_CAT_ORDER_PERMS])
   {
      ArrayResize(perm, n);
      for(int i=0; i<n; i++)
      {
         for(int p=0; p<FXAI_CAT_ORDER_PERMS; p++)
            perm[i][p] = i;
      }

      for(int p=1; p<FXAI_CAT_ORDER_PERMS; p++)
      {
         for(int i=n-1; i>0; i--)
         {
            int j = (int)MathFloor(Rand01() * (double)(i + 1));
            if(j < 0) j = 0;
            if(j > i) j = i;
            int tmp = perm[i][p];
            perm[i][p] = perm[j][p];
            perm[j][p] = tmp;
         }
      }
   }

   void BuildGlobalCTRStats(const int n,
                            const int &y_all[],
                            const int &qbin[][FXAI_CAT_CTR_BASE])
   {
      ResetCTRState();

      double cls_cnt[FXAI_CAT_CLASS_COUNT] = {0.0, 0.0, 0.0};
      for(int i=0; i<n; i++)
      {
         int cls = y_all[i];
         if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;
         cls_cnt[cls] += 1.0;
      }

      double total = cls_cnt[0] + cls_cnt[1] + cls_cnt[2];
      if(total <= 0.0) total = 1.0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_ctr_global_class[c] = FXAI_Clamp(cls_cnt[c] / total, 0.001, 0.998);

      for(int i=0; i<n; i++)
      {
         int cls = y_all[i];
         if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;

         for(int bf=0; bf<FXAI_CAT_CTR_BASE; bf++)
         {
            int bin = qbin[i][bf];
            if(bin < 0 || bin >= FXAI_CAT_BINS) continue;
            m_ctr_bin_total[bf][bin] += 1.0;
            m_ctr_bin_class[bf][bin][cls] += 1.0;
         }

         for(int q=0; q<FXAI_CAT_CTR_PAIR_COUNT; q++)
         {
            int ba = qbin[i][m_ctr_pair_a[q]];
            int bb = qbin[i][m_ctr_pair_b[q]];
            if(ba < 0 || bb < 0) continue;
            int h = PairHash(ba, bb, q);
            m_ctr_pair_total[q][h] += 1.0;
            double ydir = (cls == (int)FXAI_LABEL_BUY ? 1.0 : (cls == (int)FXAI_LABEL_SKIP ? 0.5 : 0.0));
            m_ctr_pair_buy[q][h] += ydir;
         }
      }

      m_ctr_ready = true;
   }

   void BuildOrderedCTRFeatures(const int n,
                                const int &y_all[],
                                const int &qbin[][FXAI_CAT_CTR_BASE],
                                double &ctr_all[][FXAI_CAT_CTR_FEATURES],
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
               meanv += cand[l][c];
            }
            meanv /= (double)FXAI_CAT_CLASS_COUNT;
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               cand[l][c] -= meanv;
         }

         double loss_old = ComputeLeafLoss(n, leaf_idx, y_all, sample_w, base_margin, tree.leaf_value);
         double loss_new = ComputeLeafLoss(n, leaf_idx, y_all, sample_w, base_margin, cand);

         double step_scale = 1.0;
         if(loss_new > loss_old)
         {
            step_scale = 0.5;
            for(int l=0; l<leaf_count; l++)
            {
               for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
                  cand[l][c] = tree.leaf_value[l][c] + step_scale * (cand[l][c] - tree.leaf_value[l][c]);
            }
            loss_new = ComputeLeafLoss(n, leaf_idx, y_all, sample_w, base_margin, cand);
            if(loss_new > loss_old)
            {
               for(int l=0; l<leaf_count; l++)
                  for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
                     cand[l][c] = tree.leaf_value[l][c];
            }
         }

         for(int l=0; l<leaf_count; l++)
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               tree.leaf_value[l][c] = cand[l][c];
      }
   }

   void ApplyModelShrinkage(const double shrink)
   {
      double s = FXAI_Clamp(1.0 - shrink, 0.90, 1.00);
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_bias[c] *= s;

      for(int t=0; t<m_tree_count; t++)
      {
         int leaf_count = (m_trees[t].depth > 0 ? (1 << m_trees[t].depth) : 1);
         if(leaf_count < 1) leaf_count = 1;
         if(leaf_count > FXAI_CAT_MAX_LEAVES) leaf_count = FXAI_CAT_MAX_LEAVES;
         for(int l=0; l<leaf_count; l++)
         {
            for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
               m_trees[t].leaf_value[l][c] *= s;
         }
      }
   }

   void BuildRawLogits(const double &p_raw[],
                       const int session_bucket,
                       const int regime_bucket,
                       double &logits[]) const
   {
      double lraw[FXAI_CAT_CLASS_COUNT];
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double z = m_cal_vs_b[c] + m_cal_session_b[session_bucket][c] + m_cal_regime_b[regime_bucket][c];
         for(int j=0; j<FXAI_CAT_CLASS_COUNT; j++)
            z += m_cal_vs_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      int session = SessionBucket(ResolveContextTime());
      int regime = RegimeBucket();

      double logits[FXAI_CAT_CLASS_COUNT];
      BuildRawLogits(p_raw, session, regime, logits);
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_CAT_CLASS_COUNT];
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_CAT_CAL_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 50.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_CAT_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_CAT_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_CAT_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_CAT_CAL_BINS) bi = FXAI_CAT_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         p_cal[c] = FXAI_Clamp(0.80 * p_cal[c] + 0.20 * p_iso[c], 0.0005, 0.9990);

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
      int session = SessionBucket(ResolveContextTime());
      int regime = RegimeBucket();

      double logits[FXAI_CAT_CLASS_COUNT];
      BuildRawLogits(p_raw, session, regime, logits);

      double p_cal[FXAI_CAT_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double lraw[FXAI_CAT_CLASS_COUNT];
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.20, 8.00);
      double cal_lr = FXAI_Clamp(0.15 * lr * w, 0.0001, 0.0200);
      double reg = 0.0005;

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_vs_b[c] = FXAI_ClipSym(m_cal_vs_b[c] + cal_lr * e, 4.0);
         m_cal_session_b[session][c] = FXAI_ClipSym(m_cal_session_b[session][c] + 0.7 * cal_lr * e, 3.0);
         m_cal_regime_b[regime][c] = FXAI_ClipSym(m_cal_regime_b[regime][c] + 0.6 * cal_lr * e, 3.0);

         for(int j=0; j<FXAI_CAT_CLASS_COUNT; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double wij = m_cal_vs_w[c][j];
            double grad = e * lraw[j] - reg * (wij - target_w);
            m_cal_vs_w[c][j] = FXAI_ClipSym(wij + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_CAT_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_CAT_CAL_BINS) bi = FXAI_CAT_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

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

   void UpdateValidationHarness(const int cls,
                                const double &x[],
                                const double &p_cal[],
                                const double sample_w)
   {
      int y = cls;
      if(y < 0 || y >= FXAI_CAT_CLASS_COUNT) y = (int)FXAI_LABEL_SKIP;

      double ce = -MathLog(FXAI_Clamp(p_cal[y], 1e-6, 1.0));
      double brier = 0.0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double t = (c == y ? 1.0 : 0.0);
         double d = p_cal[c] - t;
         brier += d * d;
      }
      brier /= 3.0;

      double conf = p_cal[0];
      int pred = 0;
      for(int c=1; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         if(p_cal[c] > conf)
         {
            conf = p_cal[c];
            pred = c;
         }
      }
      double acc = (pred == y ? 1.0 : 0.0);

      double w = FXAI_Clamp(sample_w, 0.25, 4.0);
      if(!m_val_ready)
      {
         m_val_ce_fast = ce;
         m_val_ce_slow = ce;
         m_val_brier_fast = brier;
         m_val_brier_slow = brier;
         m_val_ece_fast = 0.0;
         m_val_ece_slow = 0.0;
         m_val_ready = true;
      }
      else
      {
         m_val_ce_fast = 0.92 * m_val_ce_fast + 0.08 * ce;
         m_val_ce_slow = 0.995 * m_val_ce_slow + 0.005 * ce;
         m_val_brier_fast = 0.92 * m_val_brier_fast + 0.08 * brier;
         m_val_brier_slow = 0.995 * m_val_brier_slow + 0.005 * brier;
      }

      for(int b=0; b<FXAI_CAT_ECE_BINS; b++)
      {
         m_ece_mass[b] *= 0.997;
         m_ece_acc[b] *= 0.997;
         m_ece_conf[b] *= 0.997;
      }
      int bi = (int)MathFloor(conf * (double)FXAI_CAT_ECE_BINS);
      if(bi < 0) bi = 0;
      if(bi >= FXAI_CAT_ECE_BINS) bi = FXAI_CAT_ECE_BINS - 1;
      m_ece_mass[bi] += w;
      m_ece_acc[bi] += w * acc;
      m_ece_conf[bi] += w * conf;

      double ece_num = 0.0;
      double ece_den = 0.0;
      for(int b=0; b<FXAI_CAT_ECE_BINS; b++)
      {
         if(m_ece_mass[b] <= 1e-9) continue;
         double ba = m_ece_acc[b] / m_ece_mass[b];
         double bc = m_ece_conf[b] / m_ece_mass[b];
         ece_num += m_ece_mass[b] * MathAbs(ba - bc);
         ece_den += m_ece_mass[b];
      }
      double ece = (ece_den > 0.0 ? ece_num / ece_den : 0.0);
      m_val_ece_fast = 0.92 * m_val_ece_fast + 0.08 * ece;
      m_val_ece_slow = 0.995 * m_val_ece_slow + 0.005 * ece;

      if(!m_feat_stats_ready)
      {
         for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
         {
            int f = TrackFeatIndex(k);
            double v = x[f];
            m_feat_ref_mean[k] = v;
            m_feat_cur_mean[k] = v;
            m_feat_ref_var[k] = 1.0;
            m_feat_cur_var[k] = 1.0;
         }
         m_feat_stats_ready = true;
      }
      else
      {
         for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
         {
            int f = TrackFeatIndex(k);
            double v = x[f];

            double d_ref = v - m_feat_ref_mean[k];
            m_feat_ref_mean[k] += 0.001 * d_ref;
            m_feat_ref_var[k] = 0.999 * m_feat_ref_var[k] + 0.001 * d_ref * d_ref;

            double d_cur = v - m_feat_cur_mean[k];
            m_feat_cur_mean[k] += 0.025 * d_cur;
            m_feat_cur_var[k] = 0.975 * m_feat_cur_var[k] + 0.025 * d_cur * d_cur;
         }
      }

      double drift = 0.0;
      for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
      {
         double sd = MathSqrt(MathMax(m_feat_ref_var[k], 1e-6));
         drift += MathAbs(m_feat_cur_mean[k] - m_feat_ref_mean[k]) / sd;
      }
      drift /= (double)FXAI_CAT_TRACK_FEATS;
      m_feat_drift_score = drift;

      bool bad = false;
      if(m_val_ce_fast > 1.45 * MathMax(m_val_ce_slow, 0.10) && m_val_ce_fast > 0.85) bad = true;
      if(m_val_brier_fast > 1.40 * MathMax(m_val_brier_slow, 0.02) && m_val_brier_fast > 0.18) bad = true;
      if(m_val_ece_fast > 0.18) bad = true;
      if(m_feat_drift_score > 4.0) bad = true;

      if(bad) m_quality_alarm++;
      else if(m_quality_alarm > 0) m_quality_alarm--;

      if(m_quality_alarm > 16)
      {
         m_quality_alarm = 16;
         if(m_drift_cooldown < 120) m_drift_cooldown = 120;
      }

      m_val_steps++;
   }

   bool BuildOneTree(const FXAIAIHyperParams &hp)
   {
      if(m_buf_size < FXAI_CAT_MIN_BUFFER) return false;

      int n = m_buf_size;
      double x_all[][FXAI_AI_WEIGHTS];
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
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            x_all[i][k] = m_buf_x[p][k];
         int cls = m_buf_y[p];
         if(cls < 0 || cls >= FXAI_CAT_CLASS_COUNT) cls = (int)FXAI_LABEL_SKIP;
         y_all[i] = cls;
         mv_all[i] = m_buf_move[p];
         w_all[i] = m_buf_w[p];
      }

      // Recommended defaults: lr 0.02..0.05, l2 3..8.
      double eta_base = FXAI_Clamp(hp.xgb_lr, 0.0200, 0.0500);
      double lambda = FXAI_Clamp(hp.xgb_l2, 3.0000, 8.0000);

      // LR schedule + drift adaptation.
      double schedule = MathPow(0.985, (double)m_tree_count / 16.0);
      double eta = FXAI_Clamp(eta_base * schedule, 0.0020, 0.0500);
      if(m_drift_cooldown > 0)
      {
         eta = FXAI_Clamp(eta * 0.75, 0.0020, 0.0500);
         lambda = FXAI_Clamp(lambda * 1.20, 2.0, 12.0);
      }

      // Bootstrap policy + row subsampling.
      int bs_mode = (m_step % 3); // 0 Bayesian, 1 Bernoulli, 2 MVS-like.
      double subsample = (m_quality_alarm > 8 ? 0.90 : 0.75);
      double boot_w[];
      ArrayResize(boot_w, n);
      BuildBootstrapWeights(n, bs_mode, subsample, w_all, boot_w);

      // Ordered/CTR-expanded feature matrix and ordered bias.
      double x_ext[][FXAI_CAT_EXT_WEIGHTS];
      double ordered_bias[][FXAI_CAT_CLASS_COUNT];
      ArrayResize(x_ext, n);
      ArrayResize(ordered_bias, n);
      BuildExtendedTraining(n, x_all, y_all, x_ext, ordered_bias);

      // Build candidate split borders in extended space.
      BuildSplitCandidates(n, x_ext);

      // Optional feature subsampling (RSM).
      double rsm = 0.65;
      if(rsm < 0.20) rsm = 0.20;
      if(rsm > 1.00) rsm = 1.00;
      bool feat_use[];
      ArrayResize(feat_use, FXAI_CAT_EXT_WEIGHTS);
      feat_use[0] = false;
      int avail = 0;
      for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
      {
         if(m_split_border_count[f] <= 0 || m_split_valid_count[f] < 2 * FXAI_CAT_MIN_DATA)
         {
            feat_use[f] = false;
            continue;
         }
         feat_use[f] = (Rand01() <= rsm);
         if(feat_use[f]) avail++;
      }
      if(avail <= 0)
      {
         for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
         {
            if(m_split_border_count[f] > 0) { feat_use[f] = true; break; }
         }
      }

      // Base margins and ordered-safe gradients/hessians.
      double base_margin[][FXAI_CAT_CLASS_COUNT];
      double g[][FXAI_CAT_CLASS_COUNT];
      double h[][FXAI_CAT_CLASS_COUNT];
      ArrayResize(base_margin, n);
      ArrayResize(g, n);
      ArrayResize(h, n);

      double noise_sigma = FXAI_Clamp(0.010 * MathSqrt(eta), 0.0, 0.020);
      for(int i=0; i<n; i++)
      {
         double xloc[FXAI_CAT_EXT_WEIGHTS];
         for(int k=0; k<FXAI_CAT_EXT_WEIGHTS; k++) xloc[k] = x_ext[i][k];
         double mloc[FXAI_CAT_CLASS_COUNT];
         ModelMarginsExt(xloc, mloc);
         for(int c0=0; c0<FXAI_CAT_CLASS_COUNT; c0++) base_margin[i][c0] = mloc[c0];

         double z[FXAI_CAT_CLASS_COUNT];
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            z[c] = base_margin[i][c] + ordered_bias[i][c];

         double p[FXAI_CAT_CLASS_COUNT];
         Softmax3(z, p);

         double wi = FXAI_Clamp(boot_w[i], 0.0, 12.0);
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         {
            double target = (c == y_all[i] ? 1.0 : 0.0);
            double gc = (target - p[c]) * wi;
            if(noise_sigma > 0.0)
               gc += noise_sigma * RandGauss();
            double hc = FXAI_Clamp(p[c] * (1.0 - p[c]) * wi, 0.001, 8.0);
            g[i][c] = gc;
            h[i][c] = hc;
         }
      }

      // Bias Newton refinement.
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         double G = 0.0, H = 0.0;
         for(int i=0; i<n; i++)
         {
            G += g[i][c];
            H += h[i][c];
         }
         if(H > 1e-9)
            m_bias[c] += 0.15 * eta * FXAI_ClipSym(G / (H + lambda), 3.0);
      }
      double mean_bias = (m_bias[0] + m_bias[1] + m_bias[2]) / 3.0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++) m_bias[c] -= mean_bias;

      FXAICatTree tree;
      InitTree(tree);

      int leaf_idx[];
      ArrayResize(leaf_idx, n);
      for(int i=0; i<n; i++) leaf_idx[i] = 0;

      int depth_used = 0;
      for(int d=0; d<FXAI_CAT_MAX_DEPTH; d++)
      {
         int best_f = -1;
         double best_thr = 0.0;
         bool best_default_left = true;
         double best_gain = 0.0;

         for(int f=1; f<FXAI_CAT_EXT_WEIGHTS; f++)
         {
            if(!feat_use[f]) continue;
            int bc = m_split_border_count[f];
            if(bc <= 0) continue;

            for(int b=0; b<bc; b++)
            {
               double thr = m_split_borders[f][b];
               for(int pass=0; pass<2; pass++)
               {
                  bool default_left = (pass == 0);
                  double gain = 0.0;
                  if(!EvaluateSplit(d, n, leaf_idx, x_ext, g, h, boot_w, f, thr, default_left, lambda, gain))
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
         m_feature_use[best_f]++;

         for(int i=0; i<n; i++)
         {
            double xv = x_ext[i][best_f];
            bool go_left;
            if(!MathIsValidNumber(xv)) go_left = best_default_left;
            else go_left = (xv <= best_thr);
            leaf_idx[i] = (leaf_idx[i] << 1) | (go_left ? 0 : 1);
         }

         depth_used = d + 1;
      }

      tree.depth = depth_used;
      BuildLeafValuesIterative(tree, n, leaf_idx, y_all, mv_all, boot_w, base_margin, eta, lambda);

      // Model shrinkage before appending new tree.
      double shrink = FXAI_Clamp(0.0015 + 0.0100 * eta, 0.0010, 0.0100);
      ApplyModelShrinkage(shrink);

      if(m_tree_count < FXAI_CAT_MAX_TREES)
      {
         m_trees[m_tree_count] = tree;
         m_tree_count++;
      }
      else
      {
         for(int t=1; t<FXAI_CAT_MAX_TREES; t++)
            m_trees[t - 1] = m_trees[t];
         m_trees[FXAI_CAT_MAX_TREES - 1] = tree;
         m_tree_count = FXAI_CAT_MAX_TREES;
      }

      return true;
   }

public:
   CFXAIAICatBoost(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_CATBOOST; }
   virtual string AIName(void) const { return "tree_catboost"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TREE, caps, 1, 1);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_quality_heads.Reset();
      m_step = 0;
      m_tree_count = 0;
      m_buf_head = 0;
      m_buf_size = 0;

      m_loss_ready = false;
      m_loss_fast = 0.0;
      m_loss_slow = 0.0;
      m_drift_cooldown = 0;

      m_cal3_steps = 0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         m_bias[c] = 0.0;
         m_cls_ema[c] = 1.0;
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_CAT_CLASS_COUNT; j++)
            m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);

         for(int b=0; b<FXAI_CAT_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      for(int s=0; s<4; s++)
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            m_cal_session_b[s][c] = 0.0;
      for(int r=0; r<2; r++)
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            m_cal_regime_b[r][c] = 0.0;

      m_val_ready = false;
      m_val_steps = 0;
      m_val_ce_fast = 0.0;
      m_val_ce_slow = 0.0;
      m_val_brier_fast = 0.0;
      m_val_brier_slow = 0.0;
      m_val_ece_fast = 0.0;
      m_val_ece_slow = 0.0;
      for(int b=0; b<FXAI_CAT_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
      m_feat_stats_ready = false;
      m_feat_drift_score = 0.0;
      m_quality_alarm = 0;
      for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
      {
         m_feat_ref_mean[k] = 0.0;
         m_feat_ref_var[k] = 1.0;
         m_feat_cur_mean[k] = 0.0;
         m_feat_cur_var[k] = 1.0;
      }

      for(int i=0; i<FXAI_CAT_BUFFER; i++)
      {
         m_buf_y[i] = (int)FXAI_LABEL_SKIP;
         m_buf_move[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_buf_x[i][k] = 0.0;
      }

      InitCTRLayout();
      ResetCTRState();
      ResetSplitState();

      for(int t=0; t<FXAI_CAT_MAX_TREES; t++)
         InitTree(m_trees[t]);
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_initialized) return;
      m_initialized = true;
      // Slight skip prior reduces early over-trading before first trees are built.
      m_bias[(int)FXAI_LABEL_SKIP] = 0.10;
   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double margins[FXAI_CAT_CLASS_COUNT];
      double p_raw[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, class_probs);

      expected_move_points = PredictExpectedMovePoints(x, hp);
      if(expected_move_points < 0.0)
         expected_move_points = 0.0;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);
      double margins[FXAI_CAT_CLASS_COUNT], p_raw[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double pred = PredictExpectedMovePoints(x, hp);
      out.move_mean_points = MathMax(0.0, pred);
      double sigma = MathMax(0.10, 0.30 * out.move_mean_points + 0.25 * (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * (m_move_ready ? 1.0 : 0.0) + 0.30 * MathMin((double)m_tree_count / 32.0, 1.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(x,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

protected:
   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (int)FXAI_LABEL_SKIP;

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FXAI_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cost = InputCostProxyPoints(x);
      double abs_move = MathAbs(move_points);
      double edge = MathMax(0.0, abs_move - cost);
      double ev_w = FXAI_Clamp(0.35 + (edge / MathMax(cost, 0.50)), 0.10, 6.00);
      if(cls == (int)FXAI_LABEL_SKIP) ev_w *= 0.85;
      double w = FXAI_Clamp(ev_w * cls_bal, 0.10, 6.00);
      UpdateNativeQualityHeads(x, w, h.lr, h.l2);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double margins[FXAI_CAT_CLASS_COUNT];
      double p_raw[FXAI_CAT_CLASS_COUNT];
      double p_cal[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, p_cal);

      double ce = -MathLog(FXAI_Clamp(p_cal[cls], 1e-6, 1.0));
      UpdateLossDrift(ce);
      UpdateValidationHarness(cls, x, p_cal, w);

      double cal_lr = FXAI_Clamp(0.01 + 0.12 * FXAI_Clamp(h.xgb_lr, 0.0005, 0.3000), 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, w, cal_lr);

      // Keep legacy binary calibrator aligned for compatibility paths.
      double den_dir = p_raw[(int)FXAI_LABEL_BUY] + p_raw[(int)FXAI_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FXAI_LABEL_BUY] / den_dir;
      if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, w);
      else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, w);

      PushSample(cls, x, move_points, w);

      int build_every = FXAI_CAT_BUILD_EVERY;
      if(m_drift_cooldown > 0) build_every = FXAI_CAT_BUILD_EVERY / 2;
      if(m_quality_alarm > 8) build_every = MathMax(24, build_every / 2);
      if(build_every < 16) build_every = 16;
      if(m_buf_size >= FXAI_CAT_MIN_BUFFER && (m_step % build_every) == 0)
         BuildOneTree(h);

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, w);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double margins[FXAI_CAT_CLASS_COUNT];
      double p_raw[FXAI_CAT_CLASS_COUNT];
      double p_cal[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, p_cal);

      double den = p_cal[(int)FXAI_LABEL_BUY] + p_cal[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FXAI_Clamp(p_cal[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double sum = 0.0;
      double wsum = 0.0;
      for(int t=0; t<m_tree_count; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[t], x_ext);
         if(leaf < 0 || leaf >= FXAI_CAT_MAX_LEAVES) continue;

         double mv = m_trees[t].leaf_move_mean[leaf];
         if(mv <= 0.0) continue;

         double conf = MathAbs(m_trees[t].leaf_value[leaf][(int)FXAI_LABEL_BUY] -
                               m_trees[t].leaf_value[leaf][(int)FXAI_LABEL_SELL]) + 0.15;
         sum += conf * mv;
         wsum += conf;
      }

      double tree_est = (wsum > 0.0 ? sum / wsum : -1.0);
      if(tree_est > 0.0 && m_move_ready && m_move_ema_abs > 0.0) return 0.70 * tree_est + 0.30 * m_move_ema_abs;
      if(tree_est > 0.0) return tree_est;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
};

#endif // __FXAI_AI_CATBOOST_MQH__
