#ifndef __FXAI_AUDIT_CORE_MQH__
#define __FXAI_AUDIT_CORE_MQH__

#include "..\Engine\core.mqh"
#include "audit_utils.mqh"
#include "..\Engine\data_pipeline.mqh"
#include "..\API\api.mqh"

#include "audit_defs.mqh"

#define FXAI_AUDIT_RUNTIME_DIR "FXAI\\Audit\\Runtime"
#define FXAI_AUDIT_RUNTIME_MAGIC 1179874889
#define FXAI_AUDIT_RUNTIME_VERSION 1
#define FXAI_AUDIT_REL_MAX_PENDING 2048
#define FXAI_AUDIT_REGIME_COUNT FXAI_PLUGIN_REGIME_BUCKETS
#define FXAI_AUDIT_MAX_HORIZONS 8

double   g_conf_class_score[FXAI_AI_COUNT][FXAI_AUDIT_REGIME_COUNT][FXAI_AUDIT_MAX_HORIZONS][FXAI_CONFORMAL_DEPTH];
double   g_conf_move_score[FXAI_AI_COUNT][FXAI_AUDIT_REGIME_COUNT][FXAI_AUDIT_MAX_HORIZONS][FXAI_CONFORMAL_DEPTH];
double   g_conf_path_score[FXAI_AI_COUNT][FXAI_AUDIT_REGIME_COUNT][FXAI_AUDIT_MAX_HORIZONS][FXAI_CONFORMAL_DEPTH];
int      g_conf_count[FXAI_AI_COUNT][FXAI_AUDIT_REGIME_COUNT][FXAI_AUDIT_MAX_HORIZONS];
int      g_conf_head[FXAI_AI_COUNT][FXAI_AUDIT_REGIME_COUNT][FXAI_AUDIT_MAX_HORIZONS];
int      g_conf_pending_seq[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING];
int      g_conf_pending_regime[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING];
int      g_conf_pending_horizon[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING];
double   g_conf_pending_prob[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING][3];
double   g_conf_pending_move_q25[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING];
double   g_conf_pending_move_q50[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING];
double   g_conf_pending_move_q75[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING];
double   g_conf_pending_path_risk[FXAI_AI_COUNT][FXAI_AUDIT_REL_MAX_PENDING];
int      g_conf_pending_head[FXAI_AI_COUNT];
int      g_conf_pending_tail[FXAI_AI_COUNT];
bool     g_runtime_artifacts_dirty = false;
datetime g_runtime_last_save_time = 0;

CFXAIAIPlugin *g_audit_runtime_plugin = NULL;

string FXAI_AuditRuntimeSafeKey(const string raw)
{
   string key = raw;
   if(StringLen(key) <= 0)
      key = "audit";
   StringReplace(key, "\\", "_");
   StringReplace(key, "/", "_");
   StringReplace(key, ":", "_");
   StringReplace(key, "*", "_");
   StringReplace(key, "?", "_");
   StringReplace(key, "\"", "_");
   StringReplace(key, "<", "_");
   StringReplace(key, ">", "_");
   StringReplace(key, "|", "_");
   StringReplace(key, " ", "_");
   return key;
}

string FXAI_AuditRuntimeArtifactFile(const string symbol)
{
   return FXAI_AUDIT_RUNTIME_DIR + "\\fxai_audit_runtime_" + FXAI_AuditRuntimeSafeKey(symbol) + ".bin";
}

void FXAI_AuditBindRuntimePlugin(CFXAIAIPlugin *plugin)
{
   g_audit_runtime_plugin = plugin;
}

void FXAI_MarkRuntimeArtifactsDirty(void)
{
   g_runtime_artifacts_dirty = true;
}

double FXAI_AuditConformalQuantile(const int ai_idx,
                                   const int regime_id,
                                   const int hslot,
                                   const int score_kind,
                                   const double fallback)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT)
      return fallback;
   if(regime_id < 0 || regime_id >= FXAI_AUDIT_REGIME_COUNT)
      return fallback;
   if(hslot < 0 || hslot >= FXAI_AUDIT_MAX_HORIZONS)
      return fallback;

   int count = g_conf_count[ai_idx][regime_id][hslot];
   if(count <= 0)
      return fallback;
   if(count > FXAI_CONFORMAL_DEPTH)
      count = FXAI_CONFORMAL_DEPTH;

   double tmp[];
   ArrayResize(tmp, count);
   for(int i=0; i<count; i++)
   {
      if(score_kind == 0)
         tmp[i] = g_conf_class_score[ai_idx][regime_id][hslot][i];
      else if(score_kind == 1)
         tmp[i] = g_conf_move_score[ai_idx][regime_id][hslot][i];
      else
         tmp[i] = g_conf_path_score[ai_idx][regime_id][hslot][i];
   }
   ArraySort(tmp);

   int qi = (int)MathFloor(0.90 * (double)(count - 1));
   if(qi < 0)
      qi = 0;
   if(qi >= count)
      qi = count - 1;
   return tmp[qi];
}

void FXAI_ResetConformalState(void)
{
   int default_h = FXAI_AuditClampHorizon(PredictionTargetMinutes);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      g_conf_pending_head[ai] = 0;
      g_conf_pending_tail[ai] = 0;
      for(int k=0; k<FXAI_AUDIT_REL_MAX_PENDING; k++)
      {
         g_conf_pending_seq[ai][k] = -1;
         g_conf_pending_regime[ai][k] = 0;
         g_conf_pending_horizon[ai][k] = default_h;
         g_conf_pending_prob[ai][k][0] = 0.0;
         g_conf_pending_prob[ai][k][1] = 0.0;
         g_conf_pending_prob[ai][k][2] = 1.0;
         g_conf_pending_move_q25[ai][k] = 0.0;
         g_conf_pending_move_q50[ai][k] = 0.0;
         g_conf_pending_move_q75[ai][k] = 0.0;
         g_conf_pending_path_risk[ai][k] = 0.5;
      }

      for(int r=0; r<FXAI_AUDIT_REGIME_COUNT; r++)
      {
         for(int h=0; h<FXAI_AUDIT_MAX_HORIZONS; h++)
         {
            g_conf_count[ai][r][h] = 0;
            g_conf_head[ai][r][h] = 0;
            for(int i=0; i<FXAI_CONFORMAL_DEPTH; i++)
            {
               g_conf_class_score[ai][r][h][i] = 0.35;
               g_conf_move_score[ai][r][h][i] = 0.20;
               g_conf_path_score[ai][r][h][i] = 0.10;
            }
         }
      }
   }
}

void FXAI_AuditConformalPushScore(const int ai_idx,
                                  const int regime_id,
                                  const int hslot,
                                  const double class_score,
                                  const double move_score,
                                  const double path_score)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT)
      return;

   int r = regime_id;
   if(r < 0)
      r = 0;
   if(r >= FXAI_AUDIT_REGIME_COUNT)
      r = FXAI_AUDIT_REGIME_COUNT - 1;
   if(hslot < 0 || hslot >= FXAI_AUDIT_MAX_HORIZONS)
      return;

   int head = g_conf_head[ai_idx][r][hslot];
   if(head < 0 || head >= FXAI_CONFORMAL_DEPTH)
      head = 0;
   g_conf_class_score[ai_idx][r][hslot][head] = FXAI_Clamp(class_score, 0.0, 1.0);
   g_conf_move_score[ai_idx][r][hslot][head] = FXAI_Clamp(move_score, 0.0, 6.0);
   g_conf_path_score[ai_idx][r][hslot][head] = FXAI_Clamp(path_score, 0.0, 1.0);
   head++;
   if(head >= FXAI_CONFORMAL_DEPTH)
      head = 0;
   g_conf_head[ai_idx][r][hslot] = head;
   if(g_conf_count[ai_idx][r][hslot] < FXAI_CONFORMAL_DEPTH)
      g_conf_count[ai_idx][r][hslot]++;
   FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_EnqueueConformalPending(const int ai_idx,
                                  const int signal_seq,
                                  const int regime_id,
                                  const int horizon_minutes,
                                  const FXAIAIPredictionV4 &pred)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT || signal_seq < 0)
      return;

   int h = FXAI_AuditClampHorizon(horizon_minutes);
   int head = g_conf_pending_head[ai_idx];
   int tail = g_conf_pending_tail[ai_idx];

   int prev = tail - 1;
   if(prev < 0)
      prev += FXAI_AUDIT_REL_MAX_PENDING;
   if(head != tail && g_conf_pending_seq[ai_idx][prev] == signal_seq)
   {
      g_conf_pending_regime[ai_idx][prev] = regime_id;
      g_conf_pending_horizon[ai_idx][prev] = h;
      g_conf_pending_prob[ai_idx][prev][0] = pred.class_probs[0];
      g_conf_pending_prob[ai_idx][prev][1] = pred.class_probs[1];
      g_conf_pending_prob[ai_idx][prev][2] = pred.class_probs[2];
      g_conf_pending_move_q25[ai_idx][prev] = pred.move_q25_points;
      g_conf_pending_move_q50[ai_idx][prev] = pred.move_q50_points;
      g_conf_pending_move_q75[ai_idx][prev] = pred.move_q75_points;
      g_conf_pending_path_risk[ai_idx][prev] = pred.path_risk;
      FXAI_MarkRuntimeArtifactsDirty();
      return;
   }

   g_conf_pending_seq[ai_idx][tail] = signal_seq;
   g_conf_pending_regime[ai_idx][tail] = regime_id;
   g_conf_pending_horizon[ai_idx][tail] = h;
   g_conf_pending_prob[ai_idx][tail][0] = pred.class_probs[0];
   g_conf_pending_prob[ai_idx][tail][1] = pred.class_probs[1];
   g_conf_pending_prob[ai_idx][tail][2] = pred.class_probs[2];
   g_conf_pending_move_q25[ai_idx][tail] = pred.move_q25_points;
   g_conf_pending_move_q50[ai_idx][tail] = pred.move_q50_points;
   g_conf_pending_move_q75[ai_idx][tail] = pred.move_q75_points;
   g_conf_pending_path_risk[ai_idx][tail] = pred.path_risk;

   int next_tail = tail + 1;
   if(next_tail >= FXAI_AUDIT_REL_MAX_PENDING)
      next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_AUDIT_REL_MAX_PENDING)
         head = 0;
      g_conf_pending_head[ai_idx] = head;
   }
   g_conf_pending_tail[ai_idx] = next_tail;
   FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_ApplyConformalPredictionAdjustment(const int ai_idx,
                                             const int regime_id,
                                             const int horizon_minutes,
                                             const double min_move_points,
                                             FXAIAIPredictionV4 &pred)
{
   int hslot = FXAI_AuditGetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_AUDIT_MAX_HORIZONS)
      return;

   int r = regime_id;
   if(r < 0)
      r = 0;
   if(r >= FXAI_AUDIT_REGIME_COUNT)
      r = FXAI_AUDIT_REGIME_COUNT - 1;

   double q_class = FXAI_AuditConformalQuantile(ai_idx, r, hslot, 0, 0.35);
   double q_move = FXAI_AuditConformalQuantile(ai_idx, r, hslot, 1, 0.20);
   double q_path = FXAI_AuditConformalQuantile(ai_idx, r, hslot, 2, 0.10);

   double uncertainty = FXAI_Clamp(q_class, 0.0, 0.55);
   double skip_boost = FXAI_Clamp(0.32 * uncertainty + 0.14 * q_path, 0.0, 0.45);
   double sell = pred.class_probs[(int)FXAI_LABEL_SELL] * (1.0 - skip_boost);
   double buy = pred.class_probs[(int)FXAI_LABEL_BUY] * (1.0 - skip_boost);
   double skip = pred.class_probs[(int)FXAI_LABEL_SKIP] +
                 skip_boost * (1.0 - pred.class_probs[(int)FXAI_LABEL_SKIP]);
   double denom = sell + buy + skip;
   if(denom <= 0.0)
      denom = 1.0;
   pred.class_probs[(int)FXAI_LABEL_SELL] = sell / denom;
   pred.class_probs[(int)FXAI_LABEL_BUY] = buy / denom;
   pred.class_probs[(int)FXAI_LABEL_SKIP] = skip / denom;

   double move_width = MathMax(pred.move_q75_points - pred.move_q25_points,
                               MathMax(min_move_points, 0.25));
   double extra = FXAI_Clamp(q_move, 0.0, 3.0) * MathMax(0.50 * move_width, min_move_points);
   pred.move_mean_points = MathMax(0.0, pred.move_mean_points * (1.0 - 0.12 * uncertainty));
   pred.move_q25_points = MathMax(0.0, pred.move_q25_points - 0.50 * extra);
   pred.move_q50_points = MathMax(pred.move_q25_points, pred.move_q50_points);
   pred.move_q75_points = MathMax(pred.move_q50_points, pred.move_q75_points + 0.50 * extra);
   pred.path_risk = FXAI_Clamp(pred.path_risk + 0.28 * q_path + 0.10 * uncertainty, 0.0, 1.0);
   pred.fill_risk = FXAI_Clamp(pred.fill_risk + 0.18 * q_path + 0.12 * uncertainty, 0.0, 1.0);
   pred.confidence = FXAI_Clamp(MathMax(pred.class_probs[(int)FXAI_LABEL_BUY],
                                        pred.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
   pred.reliability = FXAI_Clamp(pred.reliability * (1.0 - 0.35 * uncertainty), 0.0, 1.0);
}

bool FXAI_AuditWriteConformalState(const int handle)
{
   FileWriteInteger(handle, FXAI_AUDIT_RUNTIME_MAGIC);
   FileWriteInteger(handle, FXAI_AUDIT_RUNTIME_VERSION);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_AUDIT_REGIME_COUNT; r++)
      {
         for(int h=0; h<FXAI_AUDIT_MAX_HORIZONS; h++)
         {
            FileWriteInteger(handle, g_conf_count[ai][r][h]);
            FileWriteInteger(handle, g_conf_head[ai][r][h]);
            for(int i=0; i<FXAI_CONFORMAL_DEPTH; i++)
            {
               FileWriteDouble(handle, g_conf_class_score[ai][r][h][i]);
               FileWriteDouble(handle, g_conf_move_score[ai][r][h][i]);
               FileWriteDouble(handle, g_conf_path_score[ai][r][h][i]);
            }
         }
      }

      FileWriteInteger(handle, g_conf_pending_head[ai]);
      FileWriteInteger(handle, g_conf_pending_tail[ai]);
      for(int k=0; k<FXAI_AUDIT_REL_MAX_PENDING; k++)
      {
         FileWriteInteger(handle, g_conf_pending_seq[ai][k]);
         FileWriteInteger(handle, g_conf_pending_regime[ai][k]);
         FileWriteInteger(handle, g_conf_pending_horizon[ai][k]);
         FileWriteDouble(handle, g_conf_pending_prob[ai][k][0]);
         FileWriteDouble(handle, g_conf_pending_prob[ai][k][1]);
         FileWriteDouble(handle, g_conf_pending_prob[ai][k][2]);
         FileWriteDouble(handle, g_conf_pending_move_q25[ai][k]);
         FileWriteDouble(handle, g_conf_pending_move_q50[ai][k]);
         FileWriteDouble(handle, g_conf_pending_move_q75[ai][k]);
         FileWriteDouble(handle, g_conf_pending_path_risk[ai][k]);
      }
   }
   return true;
}

bool FXAI_AuditReadConformalState(const int handle)
{
   if(handle == INVALID_HANDLE)
      return false;

   int magic = FileReadInteger(handle);
   int version = FileReadInteger(handle);
   if(magic != FXAI_AUDIT_RUNTIME_MAGIC || version != FXAI_AUDIT_RUNTIME_VERSION)
      return false;

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_AUDIT_REGIME_COUNT; r++)
      {
         for(int h=0; h<FXAI_AUDIT_MAX_HORIZONS; h++)
         {
            g_conf_count[ai][r][h] = FileReadInteger(handle);
            g_conf_head[ai][r][h] = FileReadInteger(handle);
            for(int i=0; i<FXAI_CONFORMAL_DEPTH; i++)
            {
               g_conf_class_score[ai][r][h][i] = FileReadDouble(handle);
               g_conf_move_score[ai][r][h][i] = FileReadDouble(handle);
               g_conf_path_score[ai][r][h][i] = FileReadDouble(handle);
            }
         }
      }

      g_conf_pending_head[ai] = FileReadInteger(handle);
      g_conf_pending_tail[ai] = FileReadInteger(handle);
      for(int k=0; k<FXAI_AUDIT_REL_MAX_PENDING; k++)
      {
         g_conf_pending_seq[ai][k] = FileReadInteger(handle);
         g_conf_pending_regime[ai][k] = FileReadInteger(handle);
         g_conf_pending_horizon[ai][k] = FileReadInteger(handle);
         g_conf_pending_prob[ai][k][0] = FileReadDouble(handle);
         g_conf_pending_prob[ai][k][1] = FileReadDouble(handle);
         g_conf_pending_prob[ai][k][2] = FileReadDouble(handle);
         g_conf_pending_move_q25[ai][k] = FileReadDouble(handle);
         g_conf_pending_move_q50[ai][k] = FileReadDouble(handle);
         g_conf_pending_move_q75[ai][k] = FileReadDouble(handle);
         g_conf_pending_path_risk[ai][k] = FileReadDouble(handle);
      }
   }
   return true;
}

void FXAI_UpdateConformalFromPending(const int ai_idx,
                                     const long signal_seq,
                                     const int regime_id,
                                     const int horizon_minutes,
                                     const int label_class,
                                     const double realized_move_points,
                                     const double mfe_points,
                                     const double mae_points,
                                     const double time_to_hit_frac,
                                     const int path_flags,
                                     const double spread_stress,
                                     const double commission_points,
                                     const double cost_buffer_points,
                                     const double min_move_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT)
      return;

   int head = g_conf_pending_head[ai_idx];
   int tail = g_conf_pending_tail[ai_idx];
   if(head == tail)
      return;

   int keep_seq[];
   int keep_regime[];
   int keep_horizon[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   double keep_q25[];
   double keep_q50[];
   double keep_q75[];
   double keep_path[];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);
   ArrayResize(keep_q25, 0);
   ArrayResize(keep_q50, 0);
   ArrayResize(keep_q75, 0);
   ArrayResize(keep_path, 0);

   bool matched = false;
   int matched_regime = regime_id;
   int matched_horizon = FXAI_AuditClampHorizon(horizon_minutes);
   double matched_prob0 = 0.0;
   double matched_prob1 = 0.0;
   double matched_prob2 = 1.0;
   double matched_q25 = 0.0;
   double matched_q50 = 0.0;
   double matched_q75 = 0.0;
   double matched_path = 0.5;

   int idx = head;
   while(idx != tail)
   {
      bool consume = ((long)g_conf_pending_seq[ai_idx][idx] == signal_seq && !matched);
      if(consume)
      {
         matched = true;
         matched_regime = g_conf_pending_regime[ai_idx][idx];
         matched_horizon = g_conf_pending_horizon[ai_idx][idx];
         matched_prob0 = g_conf_pending_prob[ai_idx][idx][0];
         matched_prob1 = g_conf_pending_prob[ai_idx][idx][1];
         matched_prob2 = g_conf_pending_prob[ai_idx][idx][2];
         matched_q25 = g_conf_pending_move_q25[ai_idx][idx];
         matched_q50 = g_conf_pending_move_q50[ai_idx][idx];
         matched_q75 = g_conf_pending_move_q75[ai_idx][idx];
         matched_path = g_conf_pending_path_risk[ai_idx][idx];
      }
      else
      {
         int k = ArraySize(keep_seq);
         if(k < FXAI_AUDIT_REL_MAX_PENDING)
         {
            ArrayResize(keep_seq, k + 1);
            ArrayResize(keep_regime, k + 1);
            ArrayResize(keep_horizon, k + 1);
            ArrayResize(keep_prob0, k + 1);
            ArrayResize(keep_prob1, k + 1);
            ArrayResize(keep_prob2, k + 1);
            ArrayResize(keep_q25, k + 1);
            ArrayResize(keep_q50, k + 1);
            ArrayResize(keep_q75, k + 1);
            ArrayResize(keep_path, k + 1);
            keep_seq[k] = g_conf_pending_seq[ai_idx][idx];
            keep_regime[k] = g_conf_pending_regime[ai_idx][idx];
            keep_horizon[k] = g_conf_pending_horizon[ai_idx][idx];
            keep_prob0[k] = g_conf_pending_prob[ai_idx][idx][0];
            keep_prob1[k] = g_conf_pending_prob[ai_idx][idx][1];
            keep_prob2[k] = g_conf_pending_prob[ai_idx][idx][2];
            keep_q25[k] = g_conf_pending_move_q25[ai_idx][idx];
            keep_q50[k] = g_conf_pending_move_q50[ai_idx][idx];
            keep_q75[k] = g_conf_pending_move_q75[ai_idx][idx];
            keep_path[k] = g_conf_pending_path_risk[ai_idx][idx];
         }
      }

      idx++;
      if(idx >= FXAI_AUDIT_REL_MAX_PENDING)
         idx = 0;
   }

   for(int k=0; k<FXAI_AUDIT_REL_MAX_PENDING; k++)
   {
      g_conf_pending_seq[ai_idx][k] = -1;
      g_conf_pending_regime[ai_idx][k] = 0;
      g_conf_pending_horizon[ai_idx][k] = FXAI_AuditClampHorizon(PredictionTargetMinutes);
      g_conf_pending_prob[ai_idx][k][0] = 0.0;
      g_conf_pending_prob[ai_idx][k][1] = 0.0;
      g_conf_pending_prob[ai_idx][k][2] = 1.0;
      g_conf_pending_move_q25[ai_idx][k] = 0.0;
      g_conf_pending_move_q50[ai_idx][k] = 0.0;
      g_conf_pending_move_q75[ai_idx][k] = 0.0;
      g_conf_pending_path_risk[ai_idx][k] = 0.5;
   }

   int keep_n = ArraySize(keep_seq);
   for(int k=0; k<keep_n; k++)
   {
      g_conf_pending_seq[ai_idx][k] = keep_seq[k];
      g_conf_pending_regime[ai_idx][k] = keep_regime[k];
      g_conf_pending_horizon[ai_idx][k] = keep_horizon[k];
      g_conf_pending_prob[ai_idx][k][0] = keep_prob0[k];
      g_conf_pending_prob[ai_idx][k][1] = keep_prob1[k];
      g_conf_pending_prob[ai_idx][k][2] = keep_prob2[k];
      g_conf_pending_move_q25[ai_idx][k] = keep_q25[k];
      g_conf_pending_move_q50[ai_idx][k] = keep_q50[k];
      g_conf_pending_move_q75[ai_idx][k] = keep_q75[k];
      g_conf_pending_path_risk[ai_idx][k] = keep_path[k];
   }
   g_conf_pending_head[ai_idx] = 0;
   g_conf_pending_tail[ai_idx] = keep_n;

   if(!matched)
      return;

   double probs_eval[3];
   probs_eval[0] = matched_prob0;
   probs_eval[1] = matched_prob1;
   probs_eval[2] = matched_prob2;
   int cls_idx = label_class;
   if(cls_idx < (int)FXAI_LABEL_SELL || cls_idx > (int)FXAI_LABEL_SKIP)
      cls_idx = (realized_move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
   double p_true = probs_eval[cls_idx];
   if(cls_idx < 0 || cls_idx > 2)
      p_true = MathMax(probs_eval[(int)FXAI_LABEL_BUY], probs_eval[(int)FXAI_LABEL_SELL]);

   double min_move = MathMax(min_move_points, 0.10);
   double realized_abs = MathAbs(realized_move_points);
   double width = MathMax(matched_q75 - matched_q25, MathMax(min_move, 0.25));
   double move_score = MathAbs(realized_abs - MathMax(matched_q50, 0.0)) / width;
   double spread_points_est = FXAI_Clamp(spread_stress, 0.0, 4.0) * min_move;
   double cost_points_est = MathMax(min_move - spread_points_est - MathMax(commission_points, 0.0) - MathMax(cost_buffer_points, 0.0), 0.0);
   double path_actual = FXAI_PathRiskFromTargets(mfe_points,
                                                mae_points,
                                                min_move,
                                                time_to_hit_frac,
                                                path_flags);
   double fill_actual = FXAI_FillRiskFromTargets(spread_stress,
                                                 min_move,
                                                 cost_points_est);
   double path_score = 0.70 * MathAbs(path_actual - matched_path) +
                       0.30 * MathAbs(fill_actual - matched_path);

   FXAI_AuditConformalPushScore(ai_idx,
                                matched_regime,
                                FXAI_AuditGetHorizonSlot(matched_horizon),
                                1.0 - FXAI_Clamp(p_true, 0.0, 1.0),
                                FXAI_Clamp(move_score, 0.0, 6.0),
                                FXAI_Clamp(path_score, 0.0, 1.0));
}

bool FXAI_SaveRuntimeArtifacts(const string symbol)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_AUDIT_REPORT_DIR, FILE_COMMON);
   FolderCreate(FXAI_AUDIT_RUNTIME_DIR, FILE_COMMON);

   string file_name = FXAI_AuditRuntimeArtifactFile(symbol);
   int handle = FileOpen(file_name, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   bool saved_global = FXAI_AuditWriteConformalState(handle);
   FileClose(handle);

   bool saved_plugin = false;
   if(g_audit_runtime_plugin != NULL && g_audit_runtime_plugin.SupportsPersistentState())
      saved_plugin = g_audit_runtime_plugin.SaveStateFile(g_audit_runtime_plugin.PersistentStateFile(symbol));

   if(saved_global || saved_plugin)
   {
      g_runtime_artifacts_dirty = false;
      g_runtime_last_save_time = TimeCurrent();
   }
   return (saved_global || saved_plugin);
}

bool FXAI_LoadRuntimeArtifacts(const string symbol)
{
   bool loaded_global = false;
   string file_name = FXAI_AuditRuntimeArtifactFile(symbol);
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      loaded_global = FXAI_AuditReadConformalState(handle);
      FileClose(handle);
   }

   bool loaded_plugin = false;
   if(g_audit_runtime_plugin != NULL && g_audit_runtime_plugin.SupportsPersistentState())
      loaded_plugin = g_audit_runtime_plugin.LoadStateFile(g_audit_runtime_plugin.PersistentStateFile(symbol));

   if(loaded_global || loaded_plugin)
   {
      g_runtime_artifacts_dirty = false;
      g_runtime_last_save_time = TimeCurrent();
   }
   return (loaded_global || loaded_plugin);
}

void FXAI_MaybeSaveRuntimeArtifacts(const string symbol,
                                    const datetime bar_time)
{
   if(!g_runtime_artifacts_dirty)
      return;
   datetime now = bar_time;
   if(now <= 0)
      now = TimeCurrent();
   if(g_runtime_last_save_time > 0 &&
      now > g_runtime_last_save_time &&
      (now - g_runtime_last_save_time) < 900)
      return;
   if(FXAI_SaveRuntimeArtifacts(symbol))
      g_runtime_last_save_time = now;
}

#include "audit_scenarios.mqh"
#include "audit_samples.mqh"
#include "audit_scoring.mqh"
#include "audit_report.mqh"
#include "audit_tensor.mqh"

#endif // __FXAI_AUDIT_CORE_MQH__
