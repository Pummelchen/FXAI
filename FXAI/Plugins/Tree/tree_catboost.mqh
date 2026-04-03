#ifndef __FXAI_AI_CATBOOST_MQH__
#define __FXAI_AI_CATBOOST_MQH__

#include "..\\..\\API\\plugin_base.mqh"

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
   #include "tree_catboost\\tree_catboost_private.mqh"
public:
   #include "tree_catboost\\tree_catboost_public.mqh"
protected:
   #include "tree_catboost\\tree_catboost_protected.mqh"
};

#endif // __FXAI_AI_CATBOOST_MQH__
