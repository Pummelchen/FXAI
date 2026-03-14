#ifndef __FXAI_AUDIT_DEFS_MQH__
#define __FXAI_AUDIT_DEFS_MQH__

#include "..\Engine\core.mqh"
#include "..\Engine\data_pipeline.mqh"
#include "..\API\api.mqh"

#ifndef FXAI_PATHFLAG_DUAL_HIT
#define FXAI_PATHFLAG_DUAL_HIT 1
#endif
#ifndef FXAI_PATHFLAG_KILLED_EARLY
#define FXAI_PATHFLAG_KILLED_EARLY 2
#endif
#ifndef FXAI_PATHFLAG_SPREAD_STRESS
#define FXAI_PATHFLAG_SPREAD_STRESS 4
#endif
#ifndef FXAI_PATHFLAG_SLOW_HIT
#define FXAI_PATHFLAG_SLOW_HIT 8
#endif

#define FXAI_AUDIT_REPORT_DIR "FXAI\\Audit"
#define FXAI_AUDIT_REPORT_FILE "FXAI\\Audit\\fxai_audit_report.tsv"
#define FXAI_AUDIT_ISSUE_INVALID_PRED     1
#define FXAI_AUDIT_ISSUE_OVERTRADES_NOISE 2
#define FXAI_AUDIT_ISSUE_MISSES_TREND     4
#define FXAI_AUDIT_ISSUE_CALIBRATION_DRIFT 8
#define FXAI_AUDIT_ISSUE_RESET_DRIFT      16
#define FXAI_AUDIT_ISSUE_SEQUENCE_WEAK    32
#define FXAI_AUDIT_ISSUE_DEAD_OUTPUT      64
#define FXAI_AUDIT_ISSUE_SIDE_COLLAPSE    128

int FXAI_AuditGetSequenceBarsOverride(void);
int FXAI_AuditGetSchemaOverride(void);
double FXAI_AuditGetCommissionPerLotSide(void);
double FXAI_AuditGetCostBufferPoints(void);
double FXAI_AuditGetSlippagePoints(void);
double FXAI_AuditGetFillPenaltyPoints(void);
int FXAI_AuditGetWalkForwardTrainBars(void);
int FXAI_AuditGetWalkForwardTestBars(void);
ulong FXAI_AuditGetFeatureGroupsMaskOverride(void);

void FXAI_AuditAssignExcursionsForRealizedMove(const double realized_move_points,
                                               const double best_up_points,
                                               const double best_dn_points,
                                               double &mfe_points,
                                               double &mae_points)
{
   if(realized_move_points > 0.0)
   {
      mfe_points = best_up_points;
      mae_points = best_dn_points;
      return;
   }
   if(realized_move_points < 0.0)
   {
      mfe_points = best_dn_points;
      mae_points = best_up_points;
      return;
   }

   mfe_points = MathMax(best_up_points, best_dn_points);
   mae_points = MathMax(best_up_points, best_dn_points);
}

int FXAI_AuditBuildTripleBarrierLabelEx(const int i,
                                        const int H,
                                        const double roundtrip_cost_points,
                                        const double ev_threshold_points,
                                        const FXAIDataSnapshot &snapshot,
                                        const double &high_arr[],
                                        const double &low_arr[],
                                        const double &close_arr[],
                                        double &realized_move_points,
                                        double &mfe_points,
                                        double &mae_points,
                                        double &time_to_hit_frac,
                                        int &path_flags)
{
   realized_move_points = 0.0;
   mfe_points = 0.0;
   mae_points = 0.0;
   time_to_hit_frac = 1.0;
   path_flags = 0;
   if(i < 0 || H < 1) return (int)FXAI_LABEL_SKIP;
   if(i >= ArraySize(close_arr) || i >= ArraySize(high_arr) || i >= ArraySize(low_arr))
      return (int)FXAI_LABEL_SKIP;

   double entry = close_arr[i];
   if(entry <= 0.0 || snapshot.point <= 0.0)
      return (int)FXAI_LABEL_SKIP;

   int max_step = H;
   if(i - max_step < 0) max_step = i;
   if(max_step < 1) return (int)FXAI_LABEL_SKIP;

   double ev_min = (ev_threshold_points > 0.0 ? ev_threshold_points : 0.0);
   double barrier = roundtrip_cost_points + ev_min;
   if(barrier < 0.10) barrier = 0.10;

   double mom = 0.0;
   if(i + 5 < ArraySize(close_arr) && snapshot.point > 0.0)
      mom = FXAI_MovePoints(close_arr[i + 5], close_arr[i], snapshot.point);
   double drift = FXAI_Clamp(mom / MathMax(barrier, 0.10), -1.0, 1.0);

   double range_sum = 0.0;
   int range_n = 0;
   for(int k=0; k<10; k++)
   {
      int ik = i + k;
      if(ik < 0 || ik >= ArraySize(high_arr) || ik >= ArraySize(low_arr)) break;
      range_sum += MathMax(0.0, (high_arr[ik] - low_arr[ik]) / snapshot.point);
      range_n++;
   }
   double range_avg = (range_n > 0 ? (range_sum / (double)range_n) : barrier);
   double vol_scale = FXAI_Clamp(range_avg / MathMax(barrier, 0.10), 0.7, 1.8);

   double buy_barrier = barrier * vol_scale * (1.0 - 0.10 * drift);
   double sell_barrier = barrier * vol_scale * (1.0 + 0.10 * drift);
   if(buy_barrier < 0.10) buy_barrier = 0.10;
   if(sell_barrier < 0.10) sell_barrier = 0.10;

   double best_up = 0.0;
   double best_dn = 0.0;
   for(int step=1; step<=max_step; step++)
   {
      int idx = i - step;
      if(idx < 0) break;
      if(idx >= ArraySize(high_arr) || idx >= ArraySize(low_arr)) break;

      double up_mv = FXAI_MovePoints(entry, high_arr[idx], snapshot.point);
      double dn_mv = FXAI_MovePoints(entry, low_arr[idx], snapshot.point);
      if(up_mv > best_up) best_up = up_mv;
      double dn_abs = MathAbs(dn_mv);
      if(dn_abs > best_dn) best_dn = dn_abs;
      bool hit_up = (up_mv >= buy_barrier);
      bool hit_dn = (dn_mv <= -sell_barrier);

      if(hit_up && !hit_dn)
      {
         realized_move_points = MathMax(up_mv, buy_barrier);
         mfe_points = best_up;
         mae_points = best_dn;
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         if(time_to_hit_frac > 0.75) path_flags |= FXAI_PATHFLAG_SLOW_HIT;
         return (int)FXAI_LABEL_BUY;
      }
      if(hit_dn && !hit_up)
      {
         realized_move_points = MathMin(dn_mv, -sell_barrier);
         mfe_points = best_dn;
         mae_points = best_up;
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         if(time_to_hit_frac > 0.75) path_flags |= FXAI_PATHFLAG_SLOW_HIT;
         return (int)FXAI_LABEL_SELL;
      }
      if(hit_up && hit_dn)
      {
         path_flags |= FXAI_PATHFLAG_DUAL_HIT;
         double close_mv = FXAI_MovePoints(entry, close_arr[idx], snapshot.point);
         double up_excess = up_mv - buy_barrier;
         double dn_excess = -dn_mv - sell_barrier;
         if(close_mv > 0.0 && up_excess >= dn_excess)
         {
            realized_move_points = MathMax(close_mv, buy_barrier);
            mfe_points = best_up;
            mae_points = best_dn;
            time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
            return (int)FXAI_LABEL_BUY;
         }
         if(close_mv < 0.0 && dn_excess >= up_excess)
         {
            realized_move_points = MathMin(close_mv, -sell_barrier);
            mfe_points = best_dn;
            mae_points = best_up;
            time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
            return (int)FXAI_LABEL_SELL;
         }
         realized_move_points = close_mv;
         FXAI_AuditAssignExcursionsForRealizedMove(realized_move_points, best_up, best_dn, mfe_points, mae_points);
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         return FXAI_BuildEVClassLabel(realized_move_points, roundtrip_cost_points, ev_threshold_points);
      }
   }

   int idx_term = i - max_step;
   if(idx_term < 0) idx_term = 0;
   realized_move_points = FXAI_MovePoints(entry, close_arr[idx_term], snapshot.point);
   FXAI_AuditAssignExcursionsForRealizedMove(realized_move_points, best_up, best_dn, mfe_points, mae_points);
   if(TradeKiller > 0 && H > TradeKiller)
      path_flags |= FXAI_PATHFLAG_KILLED_EARLY;
   return FXAI_BuildEVClassLabel(realized_move_points, roundtrip_cost_points, ev_threshold_points);
}

struct FXAIAuditScenarioSpec
{
   int id;
   string name;
   double drift_per_bar;
   double sigma_per_bar;
   double mean_revert_strength;
   double vol_cluster;
   double spike_prob;
   double spike_scale;
   double spread_points;
};

struct FXAIAuditScenarioMetrics
{
   int ai_id;
   string ai_name;
   int family;
   string scenario;
   int bars_total;
   int samples_total;
   int valid_preds;
   int invalid_preds;
   int buy_count;
   int sell_count;
   int skip_count;
   int true_buy_count;
   int true_sell_count;
   int true_skip_count;
   int exact_match_count;
   int directional_eval_count;
   int directional_correct_count;
   double trend_alignment_sum;
   int trend_alignment_count;
   double conf_sum;
   double rel_sum;
   double move_sum;
   double dir_conf_sum;
   double dir_hit_sum;
   double skip_ratio;
   double active_ratio;
   double bias_abs;
   double conf_drift;
   double reset_delta;
   double sequence_delta;
   double score;
   int issue_flags;
};

class CFXAIAuditRng
{
private:
   ulong m_state;

public:
   void Seed(const ulong seed)
   {
      m_state = (seed == 0 ? (ulong)881726454 : seed);
   }

   ulong NextU64(void)
   {
      m_state ^= (m_state << 13);
      m_state ^= (m_state >> 7);
      m_state ^= (m_state << 17);
      return m_state;
   }

   double NextUnit(void)
   {
      ulong lo31 = (ulong)(NextU64() % 2147483647);
      return (double)lo31 / 2147483646.0;
   }

   double NextSigned(void)
   {
      return 2.0 * NextUnit() - 1.0;
   }

   double NextNormal(void)
   {
      double u1 = NextUnit();
      double u2 = NextUnit();
      if(u1 < 1e-9) u1 = 1e-9;
      return MathSqrt(-2.0 * MathLog(u1)) * MathCos(2.0 * M_PI * u2);
   }
};


#endif // __FXAI_AUDIT_DEFS_MQH__
