#ifndef __FXAI_AI_LIGHTGBM_MQH__
#define __FXAI_AI_LIGHTGBM_MQH__

#include "..\\..\\API\\plugin_base.mqh"

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
   #include "tree_lgbm\\tree_lgbm_private.mqh"
public:
   #include "tree_lgbm\\tree_lgbm_public.mqh"
};

#endif // __FXAI_AI_LIGHTGBM_MQH__
