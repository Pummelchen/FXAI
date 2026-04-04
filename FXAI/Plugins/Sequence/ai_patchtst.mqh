#ifndef __FXAI_AI_PATCHTST_MQH__
#define __FXAI_AI_PATCHTST_MQH__

#include "..\..\API\plugin_base.mqh"

// Multivariate PatchTST reference plugin for FXAI.
// Design: patch embedding -> positional encoding -> transformer encoder stack
// -> 3-class probabilities + move-distribution heads (mu/logvar/q25/q75).
#define FXAI_PTST_CLASS_COUNT 3
#define FXAI_PTST_SEQ 96
#define FXAI_PTST_PATCH_LEN 8
#define FXAI_PTST_STRIDE 4
#define FXAI_PTST_MAX_PATCHES 24
#define FXAI_PTST_LAYERS 2
#define FXAI_PTST_HEADS 2
#define FXAI_PTST_D_MODEL FXAI_AI_MLP_HIDDEN
#define FXAI_PTST_D_HEAD (FXAI_PTST_D_MODEL / FXAI_PTST_HEADS)
#define FXAI_PTST_D_FF 16
#define FXAI_PTST_CAL_BINS 12

class CFXAIAIPatchTST : public CFXAIAIPlugin
{
private:
#include "ai_patchtst\ai_patchtst_private.mqh"

public:
#include "ai_patchtst\ai_patchtst_public.mqh"

protected:
#include "ai_patchtst\ai_patchtst_training.mqh"
};

#endif // __FXAI_AI_PATCHTST_MQH__
